import torch
import torch.nn.functional as F
import einx
import torchvision
import tensorset as ts
from src.dataset import MASK_SEQUENCE_ID
from src.utils import get_viz_output_path
from src.visualize_embeddings import features_to_rgb


def prepare_context_target_batch(batch, device, dtype):
    packed_batch, *_ = batch

    if not isinstance(packed_batch, ts.TensorSet):
        raise ValueError()

    position_ids = packed_batch.named_columns.pop("position_ids")
    sequence_ids = packed_batch.named_columns.pop("sequence_ids")
    # Token ids contains along the channel dim (sequence_ids, register id, height idx, width idx)
    token_ids = torch.cat((sequence_ids.unsqueeze(-1), position_ids), -1)

    patches = packed_batch.named_columns.pop("patches")

    patches = patches.to(device=device, dtype=dtype, non_blocking=True)
    token_ids = token_ids.to(device, non_blocking=True)

    # Scale from [0,255] to [-1,1]
    patches = (patches / 255) * 2 - 1

    return patches, token_ids


def visualize_embeddings(dataloader, context_sequence_length, model, device, dtype, patch_size, num_image_channels, autocast_fn):
    """
    Visualize the embeddings of the model.
    """
    batch = next(iter(dataloader))
    patches, token_ids = prepare_context_target_batch(batch, device, dtype)

    x_patches = patches[:, :context_sequence_length]
    y_patches = patches[:, context_sequence_length:]

    x_token_ids = token_ids[:, :context_sequence_length]
    y_token_ids = token_ids[:, context_sequence_length:]

    # cat context and target
    y_patches = torch.cat((x_patches, y_patches), 1)
    y_token_ids = torch.cat((x_token_ids, y_token_ids), 1)

    b, s, d = y_patches.shape

    with autocast_fn():
        with torch.inference_mode():
            embeddings, *_ = model.ema_encoder(x=y_patches, token_ids=y_token_ids)

    *_, hidden_d = embeddings.shape

    embeddings = embeddings.cpu().float()
    y_token_ids = y_token_ids.cpu()
    # Unscale pixel values in patches
    y_patches = (y_patches.cpu().float() + 1) / 2
    y_patches = y_patches.clip(0, 1) * 255
    y_patches = y_patches.to(torch.uint8)

    viz_output_path = get_viz_output_path()

    for i in range(b):
        sequence_ids, position_ids = y_token_ids[i, :, 0], y_token_ids[i, :, -2:]

        unique_sequence_ids = sequence_ids.unique().tolist()
        unique_sequence_ids.sort()

        if MASK_SEQUENCE_ID in unique_sequence_ids:
            unique_sequence_ids.remove(MASK_SEQUENCE_ID)
        for sequence_id in unique_sequence_ids:
            sequence_mask = sequence_ids == sequence_id

            sample_position_ids = position_ids[sequence_mask]
            sample_tokens = y_patches[i][sequence_mask]
            sample_embeddings = embeddings[i][sequence_mask]

            s, d = sample_tokens.shape

            nph, npw = (sample_position_ids + 1).amax(0).tolist()

            unpacked_pixel_image = torch.zeros(nph, npw, d, dtype=torch.uint8)
            unpacked_embedding_image = torch.zeros(
                nph, npw, hidden_d, dtype=torch.float32
            )

            for j in range(s):
                hid, wid = sample_position_ids[j]
                unpacked_pixel_image[hid, wid] = sample_tokens[j]
                unpacked_embedding_image[hid, wid] = sample_embeddings[j]

            unpacked_embedding_image = einx.rearrange(
                "nph npw d -> one d nph npw", unpacked_embedding_image, one=1
            )
            unpacked_embedding_image = F.interpolate(
                unpacked_embedding_image,
                scale_factor=(patch_size, patch_size),
                mode="bilinear",
                antialias=True,
            )
            unpacked_embedding_image = einx.rearrange(
                "one d h w -> one h w d", unpacked_embedding_image
            ).squeeze(0)

            unpacked_embedding_image = features_to_rgb(unpacked_embedding_image)

            unpacked_embedding_image = einx.rearrange(
                "h w c -> c h w", unpacked_embedding_image
            )
            unpacked_pixel_image = einx.rearrange(
                "nph npw (ph pw c) -> c (nph ph) (npw pw)",
                unpacked_pixel_image,
                ph=patch_size,
                pw=patch_size,
                c=num_image_channels,
            )

            image = torch.cat((unpacked_pixel_image, unpacked_embedding_image), -1)

            output_path = viz_output_path / f"{i:05} {sequence_id:08}.png"
            torchvision.io.write_png(image, str(output_path))
            print("Wrote", output_path)