from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import einx
from tqdm import tqdm

from src.dataset import TorchImageResizer, get_test_dataset
from src.model import IJEPADepthSmart


def scale_and_shift_depth(x, eps=1e-7):
    """
    Scale and shift depth, from:

    Towards Robust Monocular Depth Estimation:
    Mixing Datasets for
    Zero-shot Cross-dataset Transfer

    https://arxiv.org/abs/1907.01341

    Equations 5 and 6
    """

    x_flat = einx.rearrange("b h w -> b (h w)", x)

    shift = x_flat.median(-1, keepdim=True).values
    scale = (x_flat - shift).abs().mean(-1, keepdim=True)

    # b one -> b one one
    shift = shift.unsqueeze(-1)
    scale = scale.unsqueeze(-1)

    x = (x - shift) / (scale + eps)
    return x


def gradient_loss(x, y):
    """
    x: b h w
    y: b h w

    https://arxiv.org/abs/1907.01341
    Equation 11
    """
    diff = y - x
    grad_x = (diff[:, :, 1:] - diff[:, :, :-1]).abs()
    grad_y = (diff[:, 1:, :] - diff[:, :-1, :]).abs()
    loss = grad_x.mean() + grad_y.mean()
    return loss


def multiscale_gradient_loss(x, y, scales=4):
    """
    x: b h w
    y: b h w

    https://arxiv.org/abs/1907.01341
    Equation 11
    """
    loss = 0
    for scale in range(scales):
        step = 2**scale
        loss += gradient_loss(x[:, ::step, ::step], y[:, ::step, ::step])
    return loss


class DPTDepthModel(nn.Module):
    def __init__(self, input_feature_size=256, non_negative=True):
        super().__init__()

        self.conv1 = nn.Conv2d(
            input_feature_size,
            input_feature_size // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            input_feature_size // 2, 32, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU()
        self.depth_head = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.act_out = nn.ReLU() if non_negative else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv2(x)
        x = self.relu(x)
        x = self.depth_head(x)
        x = self.act_out(x)

        inv_depth = x
        return inv_depth


def validate_monocular_depth_prediction(
    model: IJEPADepthSmart,
    image_column_name: str = "jpg",
    depth_column_name: str = "depth.npy",
    patch_size: int = 16,
    validation_image_size: int = 256,
    batch_size: int = 256,
    num_workers: int = 4,
    train_dataset_pattern: str = "/nvme/nyu-depthv2-wds/nyu-depth-train-{00000..00013}.tar",
    val_dataset_pattern: str = "/nvme/nyu-depthv2-wds/nyu-depth-val-00000.tar",
    dtype: torch.dtype = torch.bfloat16,
    test_mode: bool = False,
    should_compile: bool = False,
    validation_probe_lr: float = 1e-3,
    validation_train_epochs: int = 10,
    # Extract features from the last layer of the encoder
    feature_depth: int = -1,
    num_register_tokens: int = 0,
):
    def _get_depth_dataloader(pattern, shuffle=True):
        ds = (
            get_test_dataset(
                pattern,
                shuffle=shuffle,
                image_column_name=image_column_name,
                batch_size=batch_size,
                image_size=validation_image_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
            )
            .rename(depth=depth_column_name)
            .map_dict(depth=torch.from_numpy)
            .map_dict(depth=TorchImageResizer(validation_image_size))
            .to_tuple("pixel_values", "token_ids", "depth")
        )
        dl = DataLoader(ds, num_workers=num_workers, batch_size=None)
        return dl

    encoder = model.ema_encoder

    device = next(p.device for p in encoder.parameters())

    dpt_head = DPTDepthModel(input_feature_size=encoder.hidden_size).to(device)
    optim = AdamW(dpt_head.parameters(), lr=validation_probe_lr, betas=(0.9, 0.95))

    @contextmanager
    def autocast_fn():
        with torch.autocast(device.type, dtype):
            yield

    def _compute_losses(pixel_values, token_ids, depth):
        # scale to [-1,1]
        pixel_values = (pixel_values / 255) * 2 - 1
        with torch.inference_mode():
            with autocast_fn():
                _, layer_features = encoder(
                    x=pixel_values,
                    t=None,
                    token_ids=token_ids,
                    return_all_layer_features=True,
                )

        features = layer_features[feature_depth].clone()

        features = einx.rearrange(
            "b (nph npw) d -> b d nph npw",
            features,
            nph=validation_image_size // patch_size,
            npw=validation_image_size // patch_size,
        )

        with autocast_fn():
            depth_hat = dpt_head(features)

        _, h, w = depth.shape
        depth_hat = F.interpolate(depth_hat, size=(h, w), mode="bilinear")

        # b one h w -> b h w
        depth_hat = depth_hat.squeeze(1)

        scaled_depth = scale_and_shift_depth(depth)
        scaled_depth_hat = scale_and_shift_depth(depth_hat)

        loss_ssi = F.mse_loss(scaled_depth, scaled_depth_hat)

        loss_reg = multiscale_gradient_loss(scaled_depth, scaled_depth_hat)

        alpha = 0.5

        loss = loss_ssi + alpha * loss_reg

        with torch.no_grad():
            # TODO
            # Is rmse_loss computed between scaled and shifted depth maps?
            rmse_loss = F.mse_loss(depth, depth_hat, reduction="none")
            rmse_loss = einx.mean("b [h] [w]", rmse_loss) ** 0.5

        return dict(
            loss=loss, loss_ssi=loss_ssi, loss_reg=loss_reg, rmse_loss=rmse_loss
        )

    if should_compile:
        _compute_losses = torch.compile(_compute_losses)

    def _train():
        train_dataloader = _get_depth_dataloader(train_dataset_pattern, shuffle=True)

        for epoch in tqdm(range(validation_train_epochs), desc="depth train epoch"):
            prog_bar = tqdm(train_dataloader, desc=f"epoch {epoch}")
            for pixel_values, token_ids, depth in prog_bar:
                pixel_values = pixel_values.to(
                    device=device, dtype=dtype, non_blocking=True
                )
                token_ids = token_ids.to(device=device, non_blocking=True)
                depth = depth.to(device=device, dtype=dtype, non_blocking=True)
                losses = _compute_losses(pixel_values, token_ids, depth)

                loss = losses["loss"]
                loss.backward()
                optim.step()
                optim.zero_grad()

                log_str = " ".join(
                    [f"{k}:{v.mean().item():.4f}" for k, v in losses.items()]
                )
                log_str = f"epoch {epoch} - " + log_str
                prog_bar.set_description(log_str)

                if test_mode:
                    break
            prog_bar.close()

            if test_mode:
                break

    _train()

    validation_dataloader = _get_depth_dataloader(val_dataset_pattern, shuffle=False)

    all_rmse_losses = []
    for pixel_values, token_ids, depth in tqdm(validation_dataloader):
        pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
        token_ids = token_ids.to(device=device, non_blocking=True)
        depth = depth.to(device=device, dtype=dtype, non_blocking=True)
        with torch.inference_mode():
            losses = _compute_losses(pixel_values, token_ids, depth)

        rmse_loss = losses["rmse_loss"]
        rmse_loss = rmse_loss.cpu().float()
        all_rmse_losses.append(rmse_loss)

        if test_mode:
            break

    all_rmse_losses = torch.cat(all_rmse_losses)
    mean_rmse_loss = all_rmse_losses.mean()
    return dict(mean_rmse_loss=mean_rmse_loss)
