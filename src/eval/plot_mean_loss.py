"""
creates a single loss image by averaging the loss from a bunch of different images
"""

import torch
import torchvision
import einx
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import tensorset as ts
import matplotlib.pyplot as plt

from src.dataset import (
    MASK_SAMPLE_ID,
    ContextTargetDatasetConfig,
    get_context_target_dataloader,
)
from src.eval.utils import scale_to_zero_one
from src.model import IJEPAModel, IJEPAOutput
from src.utils import get_viz_output_path


def prepare_context_target_batch(batch, dtype, device):
    if not isinstance(batch, ts.TensorSet):
        raise ValueError()

    position_ids = batch.named_columns.pop("position_ids")
    sample_ids = batch.named_columns.pop("sample_ids")
    # Token ids contains along the channel dim (sample_ids, register id, height idx, width idx)
    token_ids = torch.cat((sample_ids.unsqueeze(-1), position_ids), -1)

    patches = batch.named_columns.pop("patches")

    patches = patches.to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )
    token_ids = token_ids.to(device, non_blocking=True)

    # Scale from [0,255] to [-1,1]
    patches = (patches / 255) * 2 - 1

    return patches, token_ids


def plot_mean_loss(
    context_target_dataset: ContextTargetDatasetConfig,
    dataset_pattern: str,
    image_column_name: str,
    model: IJEPAModel,
    device: torch.device,
    dtype: torch.dtype,
    patch_size: int,
    autocast_fn,
    batch_size: int,
    num_image_channels: int = 3,
    num_total_batches: int = 50,
    image_size: int = 256,
    seed: int | None = None,
):
    assert (
        context_target_dataset.min_side_length
        == context_target_dataset.max_side_length
        == image_size
    ), (
        context_target_dataset.min_side_length,
        context_target_dataset.max_side_length,
        image_size,
    )

    context_sequence_length = context_target_dataset.packer_context_sequence_length
    window_size = context_target_dataset.mask_window_size

    nph = npw = image_size // patch_size
    mean_loss_image = torch.zeros(nph, npw, patch_size, patch_size, num_image_channels)
    mean_loss_tokens = torch.zeros(nph, npw)

    token_counts = torch.zeros(nph, npw, dtype=torch.long)

    dataloader = get_context_target_dataloader(
        context_target_dataset,
        dataset_pattern=dataset_pattern,
        seed=seed,
        image_column_name=image_column_name,
        batch_size=batch_size,
        num_workers=0,
    )

    dataloader_iter = iter(dataloader)

    num_total_samples = 0
    num_processed_batches = 0
    for batch, *_ in tqdm(
        dataloader_iter, total=num_total_batches, desc="computing losses"
    ):
        # Run in eval mode to allow dynamic diffmoe allocation
        model.eval()
        patches, token_ids = prepare_context_target_batch(batch, dtype, device)

        with torch.inference_mode():
            with autocast_fn():
                result: IJEPAOutput = model(
                    patches=patches,
                    token_ids=token_ids,
                    context_sequence_length=context_sequence_length,
                    window_size=window_size,
                    return_predictor_target_token_ids=True,
                    return_tokenwise_loss=True,
                )

        # br = Batch size * predictor repeat
        # ps = predictor sequence length
        predictor_token_ids = result.predictor_target_token_ids.cpu()
        # br ps
        loss_mask = result.is_target_mask.cpu()
        # br ps d
        tokenwise_loss = result.tokenwise_loss.cpu().float()

        # extract the loss for each unique sample in the batch
        sample_ids = predictor_token_ids[..., 0]
        unique_ids = torch.unique(sample_ids).tolist()

        position_ids = predictor_token_ids[..., -2:]

        if MASK_SAMPLE_ID in unique_ids:
            unique_ids.remove(MASK_SAMPLE_ID)

        for id in unique_ids:
            num_total_samples += 1
            mask = (sample_ids == id) & loss_mask

            sample_position_ids = position_ids[mask]
            sample_losses = tokenwise_loss[mask]

            # this is experimental! Unpatchify losses from d -> input_d
            # by unpatching the 'internal' patch size of the encoder
            # (Which can be different from the input patch size)
            # and the scaling using interpolation
            d = sample_losses.shape[-1]
            input_size = patch_size**2 * num_image_channels
            hidden_feature_scale = d / input_size
            hidden_patch_size = int(patch_size * hidden_feature_scale)
            if patch_size % hidden_feature_scale != 0:
                raise ValueError(
                    f"unable to resize hidden features into patches \
                     patch size {patch_size}  \
                     isn't divisible by hidden patch size {hidden_patch_size}"
                )

            loss_patches = einx.rearrange(
                "n (hph hpw c) -> n c hph hpw",
                sample_losses,
                hph=hidden_patch_size,
                npw=hidden_patch_size,
                c=num_image_channels,
            )
            loss_patches = torchvision.transforms.Resize(
                (patch_size, patch_size), interpolation=InterpolationMode.NEAREST_EXACT
            )(loss_patches)
            loss_patches = einx.rearrange("n c ph pw -> n ph pw c", loss_patches)

            mean_sample_losses = einx.mean("n [d]", sample_losses)

            for loss_patch, loss, (hid, wid) in zip(
                loss_patches,
                mean_sample_losses,
                sample_position_ids,
            ):
                # Add-in the loss to the mean loss image
                # at this position
                mean_loss_image[hid, wid] += loss_patch
                mean_loss_tokens[hid, wid] += loss
                token_counts[hid, wid] += 1

        num_processed_batches += 1
        if num_processed_batches >= num_total_batches:
            break

    # Take the mean of each patch, by the number of times
    # it occurred as a loss target
    #
    token_counts.clip_(1)
    mean_loss_image = einx.divide(
        "nph npw ph pw c, nph npw", mean_loss_image, token_counts
    )
    mean_loss_tokens = einx.divide("nph npw, nph npw", mean_loss_tokens, token_counts)

    scaled_mean_loss_image = scale_to_zero_one(mean_loss_image)
    scaled_mean_loss_image = einx.rearrange(
        "nph npw ph pw c -> (nph ph) (npw pw) c", scaled_mean_loss_image
    )

    mean_loss_tokens = scale_to_zero_one(mean_loss_tokens)
    # upscale
    # nph npw -> h w
    mean_loss_tokens = torchvision.transforms.Resize(
        (image_size, image_size), interpolation=InterpolationMode.NEAREST_EXACT
    )(mean_loss_tokens.unsqueeze(0)).squeeze(0)

    # Make heatmap
    # nph npw -> nph npw c
    loss_heatmap = plt.cm.hot(mean_loss_tokens)[..., :3]
    loss_heatmap = torch.from_numpy(loss_heatmap)
    loss_heatmap = einx.rearrange("h w c -> c h w", loss_heatmap)
    loss_heatmap = (loss_heatmap.clip(0, 1) * 255).to(torch.uint8)

    # Prepare projected loss image
    scaled_mean_loss_image = einx.rearrange("h w c -> c h w", scaled_mean_loss_image)
    scaled_mean_loss_image = (scaled_mean_loss_image.clip(0, 1) * 255).to(torch.uint8)

    output_path = get_viz_output_path()

    heatmap_save_path = output_path / "loss-heatmap.png"
    torchvision.io.write_png(loss_heatmap, str(heatmap_save_path))
    print("wrote", heatmap_save_path)

    loss_save_path = output_path / "loss-image.png"
    torchvision.io.write_png(scaled_mean_loss_image, str(loss_save_path))
    print("wrote", loss_save_path)

    info_str = "\n".join(
        (
            f"num_total_batches {num_total_batches}",
            f"image_size {image_size}",
            f"num_total_samples {num_total_samples}",
        )
    )
    print(info_str)
    info_path = output_path / "info.txt"
    with open(info_path, "w") as f:
        f.write(info_str)
    print("wrote", info_path)
