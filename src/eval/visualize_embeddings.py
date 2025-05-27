"""
Uses the visualization method as explained in https://arxiv.org/abs/2504.13181

Perception Encoder: The best visual embeddings are not at the output of the network
Section B.3.2
"""

import einx
import torchvision
import torch_pca
import torch

import torch.nn.functional as F
import tensorset as ts
from src.dataset import MASK_SAMPLE_ID
from src.utils import get_viz_output_path


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


def gaussian_blur(x, kernel_size=3, sigma=0.5):
    x = einx.rearrange("... h w d -> ... d h w", x)

    x_blurred = torchvision.transforms.GaussianBlur(kernel_size, sigma=sigma)(x)

    x = (x + x_blurred) / 2

    x = einx.rearrange("... d h w -> ... h w d", x)

    return x


def scale_to_zero_one(x, q=0.99):
    max = torch.quantile(x, q)
    min = torch.quantile(x, 1 - q)
    return ((x - min) / (max - min)).clip(0, 1)


def hsl_to_rgb(x):
    """
    similar to https://github.com/matplotlib/matplotlib/blob/v3.10.3/lib/matplotlib/colors.py#L3139-L3218
    """
    og_shape = x.shape
    x = einx.rearrange("... d -> (...) d", x)

    h = x[:, 0]
    s = x[:, 1]
    v = x[:, 2]

    r = torch.empty_like(h)
    g = torch.empty_like(h)
    b = torch.empty_like(h)

    i = (h * 6.0).to(torch.uint8)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = torch.stack([r, g, b], -1)

    rgb = rgb.reshape(og_shape)

    return rgb


def features_to_rgb(x):
    x = gaussian_blur(x)
    *leading_shape, d = x.shape
    x = einx.rearrange("... d -> (...) d", x)
    pca_model = torch_pca.PCA(n_components=3)
    # hue, saturation, lightness
    x = pca_model.fit_transform(x)
    x = scale_to_zero_one(x)
    x = hsl_to_rgb(x)
    x = (x.clip(0, 1) * 255).to(torch.uint8)
    x = x.reshape(*leading_shape, 3)
    return x


def visualize_embeddings(
    dataloader,
    context_sequence_length,
    model,
    device,
    dtype,
    patch_size,
    num_image_channels,
    autocast_fn,
):
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
        sample_ids, position_ids = y_token_ids[i, :, 0], y_token_ids[i, :, -2:]

        unique_sample_ids = sample_ids.unique().tolist()
        unique_sample_ids.sort()

        if MASK_SAMPLE_ID in unique_sample_ids:
            unique_sample_ids.remove(MASK_SAMPLE_ID)
        for sequence_id in unique_sample_ids:
            sequence_mask = sample_ids == sequence_id

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
