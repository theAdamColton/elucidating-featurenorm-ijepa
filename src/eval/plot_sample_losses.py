import math
from tqdm import tqdm
import torch
import einx
import torchvision
import matplotlib.pyplot as plt
import tensorset as ts

from src.dataset import MASK_SAMPLE_ID, ContextTargetDatasetConfig, get_repeated_data
from src.model import IJEPAModel, IJEPAOutput
from src.utils import get_viz_output_path
from src.eval.utils import scale_to_zero_one


def prepare_context_target_batch(batch, device, dtype):
    if not isinstance(batch, ts.TensorSet):
        raise ValueError()

    position_ids = batch.named_columns.pop("position_ids")
    sample_ids = batch.named_columns.pop("sample_ids")
    # Token ids contains along the channel dim (sample_ids, register id, height idx, width idx)
    token_ids = torch.cat((sample_ids.unsqueeze(-1), position_ids), -1)

    patches = batch.named_columns.pop("patches")

    patches = patches.to(device=device, dtype=dtype, non_blocking=True)
    token_ids = token_ids.to(device, non_blocking=True)

    # Scale from [0,255] to [-1,1]
    patches = (patches / 255) * 2 - 1

    return patches, token_ids


def plot_sample_losses(
    context_target_dataset: ContextTargetDatasetConfig,
    dataset_pattern: str,
    image_column_name: str,
    model: IJEPAModel,
    device: torch.device,
    dtype: torch.dtype,
    patch_size: int,
    num_image_channels: int,
    autocast_fn,
    batch_size: int = 32,
    num_unique_samples: int = 10,
    num_repeat_samples: int = 2,
    seed: int | None = None,
):
    """
    Plot the distribution of losses for each sample in a batch.

    This randomly resizes each image only ONCE,
    and then applies `num_repeat_samples` random context-target masking,
    to obtain repeated data of shape (`num_unique_samples`, `num_repeat_samples`, ...)

    We record the loss over all repeated images, saving the loss into a image
    """
    context_sequence_length = context_target_dataset.packer_context_sequence_length
    window_size = context_target_dataset.mask_window_size

    model.eval()

    # Data is (n q ...), where n is the number of unique samples and q is the number of repeats
    data: ts.TensorSet = get_repeated_data(
        config=context_target_dataset,
        seed=seed,
        dataset_pattern=dataset_pattern,
        num_unique_samples=num_unique_samples,
        num_repeat_samples=num_repeat_samples,
        image_column_name=image_column_name,
    )

    all_token_ids = torch.cat(
        (data["sample_ids"].unsqueeze(-1), data["position_ids"]), -1
    )

    # Take the first augmentation of each unique sample
    first_patches = data["patches"][:, 0].clone()
    first_token_ids = all_token_ids[:, 0].clone()

    element_sample_ids = first_token_ids[:, :, 0]
    # We can take the first token id of each
    # sample, and it will not be padding
    # n s -> n
    unique_ids = element_sample_ids[:, 0]
    # n q s k -> n s 2
    element_position_ids = first_token_ids[:, :, -2:]

    sample_mean_losses = {k.item(): 0 for k in unique_ids}
    sample_losses_variance = {k.item(): 0 for k in unique_ids}
    _sample_counts = {k.item(): 0 for k in unique_ids}

    # Initialize sample loss images
    loss_images = dict()
    loss_counts = dict()
    for i, id in enumerate(unique_ids):
        sample_target_mask = element_sample_ids[i] == id
        # get the max height and width over all augmentations
        aug_positions = all_token_ids[i, ..., -2:]
        # q s 2 -> 2
        height, width = aug_positions.amax((0, 1)) + 1
        loss_image = torch.zeros(height, width)
        loss_count = torch.zeros(height, width)
        loss_images[id.item()] = loss_image
        loss_counts[id.item()] = loss_count

    # Ungroup the data and encode it in batches

    def _ungroup(x):
        return einx.rearrange("n q s ... -> (n q) s ...", x)

    data = data.apply(_ungroup)

    num_batches = math.ceil(data.size(0) / batch_size)

    for batch_idx in tqdm(range(num_batches)):
        idx_start = batch_idx * batch_size
        batch = data.iloc[idx_start : idx_start + batch_size]

        patches, token_ids = prepare_context_target_batch(batch, device, dtype)

        with torch.inference_mode():
            with autocast_fn():
                result: IJEPAOutput = model(
                    patches,
                    token_ids,
                    context_sequence_length=context_sequence_length,
                    return_tokenwise_loss=True,
                    return_predictor_target_token_ids=True,
                    window_size=window_size,
                )

        # tokenwise loss is the batch repeated loss
        # from the predictor
        tokenwise_loss = result.tokenwise_loss.cpu().float()
        # target sample_ids is batch repeated token ids fed to the predictor
        target_token_ids = result.predictor_target_token_ids.cpu()
        # Mask where (True) indicates tokens to take loss on
        is_target_mask = result.is_target_mask.cpu()

        # Ungroup result tensors
        tokenwise_loss = einx.rearrange(
            "(r b) ys d -> r b ys d",
            tokenwise_loss,
            r=model.config.predictor_batch_repeat,
        )
        target_token_ids = einx.rearrange(
            "(r b) ys k -> r b ys k",
            target_token_ids,
            r=model.config.predictor_batch_repeat,
        )
        is_target_mask = einx.rearrange(
            "(r b) ys -> r b ys",
            is_target_mask,
            r=model.config.predictor_batch_repeat,
        )

        overall_loss = tokenwise_loss[is_target_mask].mean()
        tokenwise_loss = einx.mean("r b ys [d]", tokenwise_loss)

        predictor_batch_repeat = model.config.predictor_batch_repeat
        assert predictor_batch_repeat == target_token_ids.shape[0]
        b = target_token_ids.shape[1]

        # Compute the mean loss for each unique sample across the batch
        for i in range(model.config.predictor_batch_repeat):
            for j in range(b):
                # This ith,jth sequence contains the
                # predictors predictions from a single sample, and also possibly includes some padding
                pred_sample_ids = target_token_ids[i, j, :, 0]
                pred_sample_ids = torch.unique(pred_sample_ids).tolist()
                if MASK_SAMPLE_ID in pred_sample_ids:
                    pred_sample_ids.remove(MASK_SAMPLE_ID)
                id = pred_sample_ids[0]
                assert len(pred_sample_ids) == 1, (
                    f"Each batch element should contain 1 sample, not {len(pred_sample_ids)}!"
                )

                sample_target_mask = is_target_mask[i, j, :]
                assert sample_target_mask.sum() > 0

                sample_position_ids = target_token_ids[i, j, :, -2:]
                sample_position_ids = sample_position_ids[sample_target_mask]

                sample_losses = tokenwise_loss[i, j][sample_target_mask]

                # Update the loss image
                loss_image = loss_images[id]
                loss_count = loss_counts[id]
                for (hid, wid), loss in zip(sample_position_ids, sample_losses):
                    loss_image[hid, wid] += loss

                    loss_count[hid, wid] += 1

                # Measure the mean sample loss, and the variance of the sample loss
                sample_loss = sample_losses.mean()
                sample_loss_variance = sample_losses.var()

                # Take the mean of sample losses across iterations
                sample_mean_losses[id] += sample_loss / (
                    num_repeat_samples * predictor_batch_repeat
                )
                sample_losses_variance[id] += sample_loss_variance / (
                    num_repeat_samples * predictor_batch_repeat
                )
                _sample_counts[id] += 1

    # Save an image for each sample
    output_path = get_viz_output_path()

    for i in range(num_unique_samples):
        id = unique_ids[i].item()
        sample_target_mask = first_token_ids[i, :, 0] == id
        if not sample_target_mask.any():
            continue
        sample_patches = first_patches[i, sample_target_mask]
        element_sample_ids = first_token_ids[i, sample_target_mask]

        d = sample_patches.shape[-1]

        sample_position_ids = element_sample_ids[:, -2:]
        nph, npw = (sample_position_ids.amax(0) + 1).tolist()
        unpacked_patches = torch.zeros(nph, npw, d, dtype=torch.uint8)

        for (hid, wid), patch in zip(sample_position_ids, sample_patches):
            unpacked_patches[hid, wid] = patch

        image = einx.rearrange(
            "nph npw (ph pw c) -> c (nph ph) (npw pw)",
            unpacked_patches,
            ph=patch_size,
            pw=patch_size,
            c=num_image_channels,
        )

        # convert from uint8 to [0,1] float
        image = image / 255

        sample_loss = sample_mean_losses[id].item()
        sample_loss = round(sample_loss, 5)

        sample_loss_variance = sample_losses_variance[id].item()
        sample_loss_variance = round(sample_loss_variance, 5)

        sample_loss_image = loss_images[id]
        sample_loss_count = loss_counts[id]
        sample_loss_image = sample_loss_image / sample_loss_count.clip(1)
        sample_loss_image = sample_loss_image.unsqueeze(0)
        sample_loss_image = torchvision.transforms.Resize(
            (image.shape[1], image.shape[2])
        )(sample_loss_image)
        sample_loss_image = scale_to_zero_one(sample_loss_image)

        # Color multiply
        blend_image = image * sample_loss_image

        # Hot colormap
        sample_loss_image = sample_loss_image.squeeze(0)
        sample_loss_image = plt.cm.hot(sample_loss_image)[..., :3]
        sample_loss_image = einx.rearrange("h w c -> c h w", sample_loss_image)
        sample_loss_image = torch.from_numpy(sample_loss_image)

        image = torch.cat((image, blend_image, sample_loss_image), 2)
        image = (image.clip(0, 1) * 255).to(torch.uint8)

        image_save_path = (
            output_path
            / f"sample-{i:04}-{id:08} loss {sample_loss} variance {sample_loss_variance}.png"
        )

        torchvision.io.write_png(image, str(image_save_path))

        print("wrote to ", image_save_path)

    # Plot the distribution of the losses
    plt.hist(
        sample_mean_losses.values(),
        bins=20,
        density=True,
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("sample loss")
    plt.ylabel("frequency")
    plot_save_path = output_path / "histogram.png"
    plt.savefig(str(plot_save_path))
    print("saved to ", plot_save_path)
    plt.close()

    plt.hist(
        sample_mean_losses.values(),
        bins=20,
        cumulative=True,
        density=True,
        color="skyblue",
    )
    plt.xlabel("sample loss")
    plt.ylabel("frequency")
    plot_save_path = output_path / "cum_histogram.png"
    plt.savefig(str(plot_save_path))
    print("saved to ", plot_save_path)
    plt.close()

    plt.hist(
        sample_losses_variance.values(),
        bins=20,
        density=True,
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("sample loss variance")
    plt.ylabel("frequency")
    plot_save_path = output_path / "histogram_variance.png"
    plt.savefig(str(plot_save_path))
    print("saved to ", plot_save_path)
    plt.close()

    plt.hist(
        sample_losses_variance.values(),
        bins=20,
        cumulative=True,
        density=True,
        color="skyblue",
    )
    plt.xlabel("sample loss variance")
    plt.ylabel("frequency")
    plot_save_path = output_path / "cum_histogram_variance.png"
    plt.savefig(str(plot_save_path))
    print("saved to ", plot_save_path)
    plt.close()
