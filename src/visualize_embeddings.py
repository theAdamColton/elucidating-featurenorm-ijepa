"""
Uses the visualization method as explained in https://arxiv.org/abs/2504.13181

Perception Encoder: The best visual embeddings are not at the output of the network
Section B.3.2
"""

import einx
import torchvision
import torch_pca
import torch


def gaussian_blur(x, kernel_size=3, sigma=1):
    x = einx.rearrange("... h w d -> ... d h w", x)

    x_blurred = torchvision.transforms.GaussianBlur(kernel_size, sigma=sigma)(x)

    x = (x + x_blurred) / 2

    x = einx.rearrange("... d h w -> ... h w d", x)

    return x


def scale_to_zero_one(x):
    return (x - x.min()) / (x.max() - x.min())


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
