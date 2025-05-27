from contextlib import contextmanager

import wandb
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
import einx
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import get_simple_dataloader, TorchImageResizer
from src.model import IJEPAModel
from src.utils import get_viz_output_path


def convert_depth_to_rgb(x, quantile=0.99):
    x_shape = x.shape

    # Scale x to [0,1]
    x = einx.rearrange("b ... -> b (...)", x)
    max = x.quantile(quantile, dim=-1, keepdim=True)
    min = x.quantile(1 - quantile, dim=-1, keepdim=True)
    x = (x - min) / (max - min)
    x = x.reshape(x_shape)
    x = x.clip(0, 1)

    # Convert to rgb using a colormap
    x = plt.cm.hot(x)
    x = torch.from_numpy(x)
    # discard alpha channel
    x = x[..., :3]
    # channels first
    x = einx.rearrange("... h w c -> ... c h w", x)
    # to uint8
    x = (x * 255).to(torch.uint8)
    return x


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
    return x, shift, scale


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
    def __init__(self, input_feature_size=256):
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

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv2(x)
        x = self.relu(x)
        x = self.depth_head(x)

        return x


def _unsqueeze_channels(x):
    return x.unsqueeze(0)


def _squeeze_channels(x):
    return x.squeeze(0)


def validate_monocular_depth_prediction(
    model: IJEPAModel,
    image_column_name: str = "jpg",
    depth_column_name: str = "depth.npy",
    patch_size: int = 16,
    validation_image_size: int = 256,
    batch_size: int = 256,
    num_workers: int = 0,
    train_dataset_length: int = None,
    train_dataset_pattern: str = "/nvme/nyu-depthv2-wds/nyu-depth-train-{00000..00047}.tar",
    val_dataset_pattern: str = "/nvme/nyu-depthv2-wds/nyu-depth-val-00000.tar",
    dtype: torch.dtype = torch.bfloat16,
    test_mode: bool = False,
    should_compile: bool = False,
    validation_probe_lr: float = 5e-4,
    validation_train_epochs: int = 10,
    # Extract features from the 4th to last layer of the encoder
    feature_depth: int = -4,
    num_register_tokens: int = 0,
    log_every_num_steps: int = 50,
):
    run = wandb.init(
        project="ijepa-monocular-depth-eval",
        mode="disabled" if test_mode else None,
        reinit="create_new",
    )

    def _get_depth_dataloader(pattern, is_training=True):
        shuffle_size_samples = 1000
        dl = (
            get_simple_dataloader(
                pattern,
                is_training=is_training,
                shuffle_size_samples=shuffle_size_samples,
                image_column_name=image_column_name,
                batch_size=batch_size,
                image_size=validation_image_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                num_workers=num_workers,
            )
            .rename(depth=depth_column_name)
            # .map_dict(depth=torch.from_numpy)
            .map_dict(depth=_unsqueeze_channels)
            .map_dict(depth=TorchImageResizer(validation_image_size))
            .map_dict(depth=_squeeze_channels)
            .to_tuple("pixel_values", "token_ids", "depth")
        )

        if is_training:
            dl = dl.with_length(train_dataset_length // batch_size).with_epoch(
                train_dataset_length // batch_size
            )

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
            _, layer_features = encoder(
                x=pixel_values,
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

        # depth hat is (b 1 lh lw)
        depth_hat = dpt_head(features)

        _, h, w = depth.shape
        depth_hat = F.interpolate(depth_hat, size=(h, w), mode="bilinear")

        # b one h w -> b h w
        depth_hat = depth_hat.squeeze(1)

        scaled_depth, gt_shift, gt_scale = scale_and_shift_depth(depth)

        loss_ssi = F.mse_loss(scaled_depth, depth_hat)

        loss_reg = multiscale_gradient_loss(scaled_depth, depth_hat)

        alpha = 0.5

        loss = loss_ssi + alpha * loss_reg

        with torch.inference_mode():
            # Scale the model's depth prediction using the shift and scale of the
            # ground truth, to convert the predictions into the units that the ground truth uses.
            # This means that we don't expect the model to predict in the units used by ground truth depth map.
            # Instead, we care about the model focusing on relative differences in depth in the image.
            depth_hat_gt_space = (depth_hat * gt_scale) + gt_shift
            rmse_loss = F.mse_loss(depth, depth_hat_gt_space, reduction="none")
            rmse_loss = einx.mean("b [h w]", rmse_loss) ** 0.5

        losses = dict(
            loss=loss, loss_ssi=loss_ssi, loss_reg=loss_reg, rmse_loss=rmse_loss
        )

        return losses, depth_hat, depth

    if should_compile:
        encoder = torch.compile(encoder)
        dpt_head = torch.compile(dpt_head)

    def _train():
        train_dataloader = _get_depth_dataloader(
            train_dataset_pattern, is_training=True
        )

        step = 0

        for epoch in tqdm(range(validation_train_epochs), desc="depth train epoch"):
            prog_bar = tqdm(train_dataloader, desc=f"epoch {epoch}")
            for pixel_values, token_ids, depth in prog_bar:
                pixel_values = pixel_values.to(
                    device=device, dtype=dtype, non_blocking=True
                )
                token_ids = token_ids.to(device=device, non_blocking=True)
                depth = depth.to(device=device, dtype=dtype, non_blocking=True)
                with autocast_fn():
                    losses, *_ = _compute_losses(pixel_values, token_ids, depth)

                loss = losses["loss"]
                loss.backward()
                optim.step()
                optim.zero_grad()

                log_dict = {k: v.mean().item() for k, v in losses.items()}

                if step % log_every_num_steps == 0:
                    run.log(log_dict, step=step)

                log_str = " ".join([f"{k}:{v:.4f}" for k, v in log_dict.items()])
                log_str = f"epoch {epoch} - " + log_str
                prog_bar.set_description(log_str)

                step += 1

                if test_mode:
                    break
            prog_bar.close()

            if test_mode:
                break

    _train()

    validation_dataloader = _get_depth_dataloader(
        val_dataset_pattern, is_training=False
    )

    is_first_batch = True
    viz_output_path = get_viz_output_path()

    all_rmse_losses = []
    for pixel_values, token_ids, depth in tqdm(validation_dataloader):
        pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
        token_ids = token_ids.to(device=device, non_blocking=True)
        depth = depth.to(device=device, dtype=dtype, non_blocking=True)

        with torch.inference_mode():
            with autocast_fn():
                losses, depth_hat, depth = _compute_losses(
                    pixel_values, token_ids, depth
                )

        rmse_loss = losses["rmse_loss"]
        rmse_loss = rmse_loss.cpu().float()
        all_rmse_losses.append(rmse_loss)

        if is_first_batch:
            # Save an image of the predicted depth map
            pixel_values = pixel_values.cpu().to(torch.uint8)
            depth_hat = depth_hat.cpu().float()
            depth = depth.cpu().float()

            depth_hat = convert_depth_to_rgb(depth_hat)
            depth = convert_depth_to_rgb(depth)
            pixel_values = einx.rearrange(
                "b (nph npw) (ph pw c) -> b c (nph ph) (npw pw)",
                pixel_values,
                ph=patch_size,
                pw=patch_size,
                nph=validation_image_size // patch_size,
                npw=validation_image_size // patch_size,
            )

            # Put a white border between the images
            b, c, h, _ = pixel_values.shape
            border = torch.full((b, c, h, 4), 255, dtype=torch.uint8)
            pixel_values = torch.cat((pixel_values, border), -1)
            depth = torch.cat((depth, border), -1)

            pixel_values = torch.cat((pixel_values, depth, depth_hat), -1)

            for i, image in enumerate(pixel_values):
                filename = viz_output_path / f"depth image {i:05}.png"
                torchvision.io.write_png(image, str(filename))
                print("Wrote", filename)

        is_first_batch = False

        if test_mode:
            break

    all_rmse_losses = torch.cat(all_rmse_losses)
    mean_rmse_loss = all_rmse_losses.mean()

    result = dict(mean_rmse_loss=mean_rmse_loss)
    run.log(result)

    run.finish()

    return result
