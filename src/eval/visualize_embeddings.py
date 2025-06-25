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
from src.model import EncoderOutput
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


def gaussian_blur(x, kernel_size=3, sigma=0.5):
    x = einx.rearrange("... h w d -> ... d h w", x)

    x_blurred = torchvision.transforms.GaussianBlur(kernel_size, sigma=sigma)(x)

    x = (x + x_blurred) / 2

    x = einx.rearrange("... d h w -> ... h w d", x)

    return x


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
    """
    x: Channels-last features: (h w d), cpu float tensor
    """
    x = gaussian_blur(x)
    *leading_shape, d = x.shape
    x = einx.rearrange("... d -> (...) d", x)

    pca_model = torch_pca.PCA(n_components=3, whiten=True)
    # hue, saturation, lightness
    x = pca_model.fit_transform(x)

    x = scale_to_zero_one(x)

    # Artificially increase saturation and lightness
    scale = torch.tensor([1, 1.2, 1.2])
    x = einx.multiply("... c, c", x, scale)

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
    feature_depth=-4,
):
    """
    Visualize the embeddings of the model.
    """
    batch = next(iter(dataloader))
    patches, token_ids = prepare_context_target_batch(batch, device, dtype)

    b, s, d = patches.shape

    # Run in eval mode - This causes the DiffMoEMLP to use dynamic (learned) expert allocation
    model.eval()

    # Encode and visualize all patches (context+target)

    with autocast_fn():
        with torch.inference_mode():
            encoder_output: EncoderOutput = model.ema_encoder(
                x=patches, token_ids=token_ids, return_all_layer_features=True
            )

    all_layer_features = encoder_output.all_layer_features
    features = all_layer_features[feature_depth]

    *_, hidden_d = features.shape

    features = features.cpu().float()
    token_ids = token_ids.cpu()
    # Unscale pixel values in patches
    patches = (patches.cpu().float() + 1) / 2
    patches = patches.clip(0, 1) * 255
    patches = patches.to(torch.uint8)

    viz_output_path = get_viz_output_path()

    for i in range(b):
        sample_ids, position_ids = token_ids[i, :, 0], token_ids[i, :, -2:]

        unique_sample_ids = sample_ids.unique().tolist()
        unique_sample_ids.sort()

        if MASK_SAMPLE_ID in unique_sample_ids:
            unique_sample_ids.remove(MASK_SAMPLE_ID)
        for sequence_id in unique_sample_ids:
            sequence_mask = sample_ids == sequence_id

            sample_position_ids = position_ids[sequence_mask]
            sample_tokens = patches[i][sequence_mask]
            sample_features = features[i][sequence_mask]

            s, d = sample_tokens.shape

            nph, npw = (sample_position_ids + 1).amax(0).tolist()

            unpacked_pixel_image = torch.zeros(nph, npw, d, dtype=torch.uint8)
            unpacked_feature_image = torch.zeros(
                nph, npw, hidden_d, dtype=torch.float32
            )

            for j in range(s):
                hid, wid = sample_position_ids[j]
                unpacked_pixel_image[hid, wid] = sample_tokens[j]
                unpacked_feature_image[hid, wid] = sample_features[j]

            unpacked_feature_image = features_to_rgb(unpacked_feature_image)

            unpacked_feature_image = einx.rearrange(
                "nph npw c -> one c nph npw", unpacked_feature_image, one=1
            )

            unpacked_feature_image = F.interpolate(
                unpacked_feature_image,
                scale_factor=(patch_size, patch_size),
                mode="bilinear",
                antialias=True,
            )

            # one c h w -> c h w
            unpacked_feature_image = unpacked_feature_image.squeeze(0)

            unpacked_pixel_image = einx.rearrange(
                "nph npw (ph pw c) -> c (nph ph) (npw pw)",
                unpacked_pixel_image,
                ph=patch_size,
                pw=patch_size,
                c=num_image_channels,
            )

            image = torch.cat((unpacked_pixel_image, unpacked_feature_image), -1)

            output_path = viz_output_path / f"{i:05} {sequence_id:08}.png"
            torchvision.io.write_png(image, str(output_path))
            print("Wrote", output_path)
