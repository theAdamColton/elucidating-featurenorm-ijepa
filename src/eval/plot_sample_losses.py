from tqdm import tqdm
import torch
import einx
import torchvision
import matplotlib.pyplot as plt
import tensorset as ts

from src.dataset import MASK_SAMPLE_ID
from src.utils import get_viz_output_path
from src.eval.utils import scale_to_zero_one


def prepare_context_target_batch(batch, device, dtype):
    packed_batch, *_ = batch

    if not isinstance(packed_batch, ts.TensorSet):
        raise ValueError()

    position_ids = packed_batch.named_columns.pop("position_ids")
    sample_ids = packed_batch.named_columns.pop("sample_ids")
    # Token ids contains along the channel dim (sample_ids, register id, height idx, width idx)
    token_ids = torch.cat((sample_ids.unsqueeze(-1), position_ids), -1)

    patches = packed_batch.named_columns.pop("patches")

    patches = patches.to(device=device, dtype=dtype, non_blocking=True)
    token_ids = token_ids.to(device, non_blocking=True)

    # Scale from [0,255] to [-1,1]
    patches = (patches / 255) * 2 - 1

    return patches, token_ids


def plot_sample_losses(
    dataloader,
    context_sequence_length,
    model,
    device,
    dtype,
    patch_size,
    num_image_channels,
    autocast_fn,
    window_size,
):
    """
    Plot the distribution of losses for each sample in a batch.
    """
    batch = next(iter(dataloader))

    patches, token_ids = prepare_context_target_batch(batch, device, dtype)

    x_patches = patches[:, :context_sequence_length]
    y_patches = patches[:, context_sequence_length:]

    x_token_ids = token_ids[:, :context_sequence_length]
    y_token_ids = token_ids[:, context_sequence_length:]

    b, x_patches_length, d = x_patches.shape

    element_sample_ids = torch.cat((x_token_ids, y_token_ids), 1)[..., 0]

    batch_unique_sample_ids = []
    for i in range(b):
        unique_sample_ids = torch.unique(element_sample_ids[i]).tolist()
        if MASK_SAMPLE_ID in unique_sample_ids:
            unique_sample_ids.remove(MASK_SAMPLE_ID)
        batch_unique_sample_ids.append(unique_sample_ids)

    sample_mean_losses = [
        torch.zeros(len(unique_sample_ids))
        for unique_sample_ids in batch_unique_sample_ids
    ]
    sample_losses_variance = [
        torch.zeros(len(unique_sample_ids))
        for unique_sample_ids in batch_unique_sample_ids
    ]

    # Initialize sample loss images
    loss_images = dict()
    loss_counts = dict()
    for i in range(b):
        for id in batch_unique_sample_ids[i]:
            mask = token_ids[i, :, 0] == id
            sample_position_ids = token_ids[i, :, -2:][mask]
            height, width = sample_position_ids.amax(0) + 1
            loss_image = torch.zeros(height, width)
            loss_count = torch.zeros(height, width)
            loss_images[id] = loss_image
            loss_counts[id] = loss_count

    all_patches = torch.cat((x_patches, y_patches), 1)
    all_token_ids = torch.cat((x_token_ids, y_token_ids), 1)

    # Compute the loss several times, each time using a different context and target
    iters = 500

    for i in tqdm(range(iters), "computing loss..."):
        # Shuffle the patches, to create new random sets of context
        # and target tokens
        # Note, that this isn't windowed masking like in the
        # official context-target dataset

        b, s, d = all_patches.shape
        random_indices = torch.rand(b, s).argsort(dim=-1)
        shuffled_patches = einx.get_at(
            "b [s] d, b n -> b n d", all_patches, random_indices
        )
        shuffled_token_ids = einx.get_at(
            "b [s] nd, b n -> b n nd", all_token_ids, random_indices
        )

        x_patches, y_patches = (
            shuffled_patches[:, :x_patches_length],
            shuffled_patches[:, x_patches_length:],
        )
        x_token_ids, y_token_ids = (
            shuffled_token_ids[:, :x_patches_length],
            shuffled_token_ids[:, x_patches_length:],
        )

        patches = torch.cat((x_patches, y_patches), 1)
        token_ids = torch.cat((x_token_ids, y_token_ids), 1)

        with torch.inference_mode():
            with autocast_fn():
                result_dict = model(
                    patches,
                    token_ids,
                    context_sequence_length=context_sequence_length,
                    return_tokenwise_loss=True,
                    return_predictor_target_token_ids=True,
                    window_size=window_size,
                )

        # tokenwise loss is the batch repeated loss
        # from the predictor
        tokenwise_loss = result_dict["tokenwise_loss"].cpu().float()
        # target sample_ids is batch repeated sequence ids fed to the predictor
        target_sample_ids = result_dict["predictor_target_token_ids"][..., 0].cpu()
        target_position_ids = result_dict["predictor_target_token_ids"][..., -2:].cpu()

        tokenwise_loss = einx.mean("rb ys [d]", tokenwise_loss)

        # Compute the mean loss for each unique sample across the batch
        for i in range(model.config.predictor_batch_repeat):
            for j in range(b):
                batch_index = i * b + j
                element_sample_ids = target_sample_ids[batch_index]
                element_position_ids = target_position_ids[batch_index]
                for k, sample_id in enumerate(batch_unique_sample_ids[j]):
                    mask = element_sample_ids == sample_id
                    if not mask.any():
                        continue

                    sample_position_ids = element_position_ids[mask]
                    sample_tokenwise_loss = tokenwise_loss[batch_index][mask]

                    # Update the loss image
                    loss_image = loss_images[sample_id]
                    loss_count = loss_counts[sample_id]
                    for (hid, wid), token_loss in zip(
                        sample_position_ids, sample_tokenwise_loss
                    ):
                        loss_image[hid, wid] += token_loss
                        loss_count[hid, wid] += 1

                    # Measure the mean sample loss, and the variance of the sample loss
                    sample_loss = tokenwise_loss[batch_index, mask].mean()
                    sample_loss_variance = tokenwise_loss[batch_index, mask].var()

                    # Take the mean of sample losses across iterations
                    # TODO! This doesnt handle the special case
                    # where sometimes a sample might not be included
                    # in the loss for a batch because it is randomly dropped
                    sample_mean_losses[j][k] += sample_loss / iters
                    sample_losses_variance[j][k] += sample_loss_variance / iters

    # Save an image for each sample
    output_path = get_viz_output_path()

    all_patches = all_patches.cpu().float()
    all_token_ids = all_token_ids.cpu()
    for i in range(b):
        for j, sample_id in enumerate(batch_unique_sample_ids[i]):
            mask = all_token_ids[i, :, 0] == sample_id
            if not mask.any():
                continue
            sample_patches = all_patches[i, mask]
            element_sample_ids = all_token_ids[i, mask]

            sample_patches = (sample_patches + 1) / 2

            sample_position_ids = element_sample_ids[:, -2:]
            nph, npw = (sample_position_ids.amax(0) + 1).tolist()
            unpacked_patches = torch.zeros(nph, npw, d)

            for (hid, wid), patch in zip(sample_position_ids, sample_patches):
                unpacked_patches[hid, wid] = patch

            image = einx.rearrange(
                "nph npw (ph pw c) -> c (nph ph) (npw pw)",
                unpacked_patches,
                ph=patch_size,
                pw=patch_size,
                c=num_image_channels,
            )

            sample_loss = sample_mean_losses[i][j].item()
            sample_loss = round(sample_loss, 5)

            sample_loss_variance = sample_losses_variance[i][j].item()
            sample_loss_variance = round(sample_loss_variance, 5)

            sample_loss_image = loss_images[sample_id]
            sample_loss_count = loss_counts[sample_id]
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
                / f"sample-{i:04}-{sample_id:08} loss {sample_loss} variance {sample_loss_variance}.png"
            )

            torchvision.io.write_png(image, str(image_save_path))

            print("wrote to ", image_save_path)

    # Plot the distribution of the losses
    all_losses = []
    for batch in sample_mean_losses:
        all_losses.extend(batch.tolist())

    all_losses_variance = []
    for batch in sample_losses_variance:
        all_losses_variance.extend(batch.tolist())

    plt.hist(all_losses, bins=20, density=True, color="skyblue", edgecolor="black")
    plt.xlabel("sample loss")
    plt.ylabel("frequency")
    plot_save_path = output_path / "histogram.png"
    plt.savefig(str(plot_save_path))
    print("saved to ", plot_save_path)
    plt.close()

    plt.hist(
        all_losses,
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
        all_losses_variance,
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
        all_losses_variance,
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
