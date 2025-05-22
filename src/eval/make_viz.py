import torch
import einx
import torchvision

from src.dataset import MASK_SAMPLE_ID
from src.utils import get_viz_output_path


def make_viz(dataloader, context_sequence_length, patch_size, num_image_channels):
    """
    Create visualizations of the context and target patches.
    """
    sample = next(iter(dataloader))

    # Decode and save one batch of images
    viz_output_path = get_viz_output_path()

    # overlay a black X-mark line on for all context tokens
    patch_border = torch.ones(patch_size, patch_size, num_image_channels)
    patch_border.diagonal().zero_()
    patch_border = patch_border * patch_border.flip(0)
    patch_border = einx.rearrange("ph pw c -> (ph pw c)", patch_border)

    packed_batch, *_ = sample
    x_patches, y_patches = (
        packed_batch.iloc[:, :context_sequence_length],
        packed_batch.iloc[:, context_sequence_length:],
    )
    x_patches["patches"] = einx.multiply("... d, d", x_patches["patches"], patch_border)

    b = x_patches.size(0)
    for i in range(b):
        x_seq = x_patches.iloc[i]
        y_seq = y_patches.iloc[i]
        sequence_ids = x_seq["sequence_ids"].unique().tolist()
        for j in sequence_ids:
            if j == MASK_SAMPLE_ID:
                continue

            device = x_seq["patches"].device

            x_sample_mask = (x_seq["sequence_ids"] == j) & (
                x_seq["position_ids"][..., 0] == MASK_SAMPLE_ID
            )
            y_sample_mask = (y_seq["sequence_ids"] == j) & (
                y_seq["position_ids"][..., 0] == MASK_SAMPLE_ID
            )

            x_sample = x_seq.iloc[x_sample_mask]
            y_sample = y_seq.iloc[y_sample_mask]

            assert x_sample.size(0) > 0
            assert y_sample.size(0) > 0

            x_sample_position_ids = x_sample["position_ids"][:, -2:]
            y_sample_position_ids = y_sample["position_ids"][:, -2:]

            all_position_ids = torch.cat(
                (x_sample_position_ids, y_sample_position_ids), 0
            )

            min_ph_pw = all_position_ids.amin(0)
            max_ph_pw = all_position_ids.amax(0)

            ph, pw = (max_ph_pw - min_ph_pw + 1).tolist()

            input_size = x_patches["patches"].shape[-1]
            image = torch.zeros(ph, pw, input_size, dtype=torch.uint8, device=device)

            for k in range(y_sample.size(0)):
                token = y_sample.iloc[k]
                patch = token["patches"]
                hid, wid = token["position_ids"][-2:] - min_ph_pw
                image[hid, wid] = patch

            for k in range(x_sample.size(0)):
                token = x_sample.iloc[k]
                patch = token["patches"]
                hid, wid = token["position_ids"][-2:] - min_ph_pw
                image[hid, wid] = patch

            image = einx.rearrange(
                "nph npw (ph pw c) -> c (nph ph) (npw pw)",
                image,
                ph=patch_size,
                pw=patch_size,
            )

            image_save_path = viz_output_path / f"{i:04} {j:06}.png"
            torchvision.io.write_png(image, str(image_save_path))

            print("saved to ", image_save_path)
