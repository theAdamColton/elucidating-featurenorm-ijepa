from typing import Literal
import gc
from contextlib import contextmanager
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import numpy as np
from torch import nn
import einx
import jsonargparse
from torch.utils.data import DataLoader
import torchvision
import random
import torch
from dataclasses import dataclass, field, asdict
import wandb
import cabbage_patch
import tensorset
import webdataset as wds
from transformers.optimization import get_constant_schedule_with_warmup

from src.model import IJEPADepthSmartConfig, IJEPADepthSmart, MASK_SEQUENCE_ID
from validate import validate


@dataclass
class ContextTargetPatcherConfig:
    patch_size: int = 16
    window_size: int = 2

    max_side_length: int = 256
    min_side_length: int = 64
    max_num_context_tokens: int = 128

    def __post_init__(self):
        assert self.max_side_length % self.patch_size == 0
        assert self.min_side_length % self.patch_size == 0
        assert self.max_num_context_tokens % self.window_size**2 == 0


def patch(x, patch_size=8):
    h, w, c = x.shape
    nph, npw = h // patch_size, w // patch_size
    x = x.reshape(nph, patch_size, npw, patch_size, c)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(nph, npw, patch_size, patch_size, c)
    return x


class ContextTargetPatcher(nn.Module):
    def __init__(self, config=ContextTargetPatcherConfig()):
        super().__init__()
        self.config = config

    def forward(self, row):
        """
        x: PIL image
        """
        config = self.config
        patch_size = config.patch_size
        c = 3

        x = row.pop("pixel_values")

        row["x_patches"] = None
        row["y_patches"] = None

        input_h, input_w = x.size

        input_side_length = (input_h + input_w) / 2
        max_side_length = min(config.max_side_length, int(input_side_length))

        if max_side_length < config.min_side_length:
            # print("Warning, image is too small to process!", input_h, input_w)
            return row

        sampled_side_length = random.randint(config.min_side_length, max_side_length)

        scale_factor = sampled_side_length / input_side_length

        image_crop_size = (input_h * scale_factor, input_w * scale_factor)
        multiple_of = config.patch_size * config.window_size
        image_crop_size = tuple(
            int(size // multiple_of) * multiple_of for size in image_crop_size
        )
        image_crop_size = tuple(max(size, multiple_of) for size in image_crop_size)

        # Hack to prevent too few windows
        num_windows = sum(size // multiple_of for size in image_crop_size)
        while num_windows < 2:
            image_crop_size = tuple(size + multiple_of for size in image_crop_size)
            num_windows = sum(size // multiple_of for size in image_crop_size)

        # nearest resampling
        x = x.convert("RGB").resize(image_crop_size, resample=0)
        x = torch.from_numpy(np.array(x))

        # (nph ph) (npw pw) c -> nph npw ph pw c
        x = patch(x, patch_size)
        # nph npw ph pw c -> nph npw (ph pw c)
        x = x.reshape(x.shape[0], x.shape[1], patch_size**2 * c)

        position_ids = torch.meshgrid(
            (torch.arange(x.shape[0]), torch.arange(x.shape[1])), indexing="ij"
        )
        position_ids = torch.stack(position_ids, -1)

        # (nwh wh) (nww ww) d -> nwh nww wh ww d
        x = patch(x, config.window_size)
        # nwh nww wh ww d -> (nwh nww) (wh ww) d
        x = x.reshape(-1, config.window_size**2, x.shape[-1])

        # (nwh wh) (nww ww) nd -> nwh nww wh ww nd
        position_ids = patch(position_ids, config.window_size)
        # nwh nww wh ww nd -> (nwh nww) (wh ww) nd
        position_ids = position_ids.reshape(
            -1, config.window_size**2, position_ids.shape[-1]
        )

        nw, ws, d = x.shape

        max_num_ctx_windows = config.max_num_context_tokens // ws
        max_num_ctx_windows = min(nw - 1, max_num_ctx_windows)

        if nw <= 2:
            num_ctx_windows = 1
        else:
            num_ctx_windows = random.randint(1, max_num_ctx_windows)

        random_ids = torch.randperm(nw)

        # [nw] ws d, nw -> nw ws d
        x = x[random_ids]
        position_ids = position_ids[random_ids]

        ctx, target = x[:num_ctx_windows], x[num_ctx_windows:]
        ctx_position_ids, target_position_ids = (
            position_ids[:num_ctx_windows],
            position_ids[num_ctx_windows:],
        )

        # nxw ws d -> (nxw ws) d
        ctx = ctx.reshape(-1, ctx.shape[-1])
        ctx_position_ids = ctx_position_ids.reshape(-1, ctx_position_ids.shape[-1])

        # nyw ws d -> (nyw ws) d
        target = target.reshape(-1, target.shape[-1])
        target_position_ids = target_position_ids.reshape(
            -1, target_position_ids.shape[-1]
        )

        row["x_patches"] = tensorset.TensorSet(
            patches=ctx, position_ids=ctx_position_ids
        )
        row["y_patches"] = tensorset.TensorSet(
            patches=target, position_ids=target_position_ids
        )

        return row


def filter_rows(row):
    return row["x_patches"] is not None and row["y_patches"] is not None


@dataclass
class MainConfig:
    should_compile: bool = False
    dtype: str = "bfloat16"
    device: str = "cuda"
    batch_size: int = 256
    packer_batch_size: int = 16
    num_workers: int = 0
    seed: int = 420
    num_warmup_steps: int = 5000
    lr: float = 5e-4
    num_epochs: int = 100

    log_every_num_steps: int = 50
    validate_every_num_epochs: int = 5

    validation_probe_lr: float = 1e-3
    validation_image_size: int = 256
    validation_train_epochs: int = 50
    validation_probe_batch_size: int = 2048

    test_mode: bool = False

    num_image_channels: int = 3

    ema_beta: float = 0.996
    ema_beta_warmup_steps: int = 1000
    interp_warmup_steps: int = 100000

    # Webdataset tars
    train_dataset_pattern: str = "/nvme/imagenet1k/imagenet1k-train-{0000..1023}.tar"
    val_dataset_pattern: str = "/nvme/imagenet1k/imagenet1k-validation-{00..63}.tar"
    image_column_name: str = "jpg"
    label_column_name: str = "cls"
    num_classes: int = 1000

    patcher: ContextTargetPatcherConfig = field(
        default_factory=lambda: ContextTargetPatcherConfig()
    )

    model: IJEPADepthSmartConfig = field(
        default_factory=lambda: IJEPADepthSmartConfig()
    )

    mode: Literal["make-viz", "train", "validate"] = "train"

    def __post_init__(self):
        assert self.batch_size % self.packer_batch_size == 0
        assert self.packer_batch_size <= self.batch_size


def main(conf: MainConfig = MainConfig()):
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)

    device = torch.device(conf.device)
    dtype = getattr(torch, conf.dtype)

    input_size = conf.model.encoder.input_size
    num_image_channels = conf.num_image_channels

    patch_size = conf.patcher.patch_size
    max_patch_side_length = conf.patcher.max_side_length // patch_size
    max_sequence_length = max_patch_side_length**2

    pad_value_dict = {"position_ids": 0, "patches": 0, "sequence_ids": MASK_SEQUENCE_ID}

    dataset = (
        cabbage_patch.CabbageDataset(
            conf.train_dataset_pattern,
            shardshuffle=True,
            detshuffle=True,
            seed=conf.seed,
            nodesplitter=wds.split_by_node,
        )
        .shuffle(1000)
        .decode("pil", handler=wds.handlers.warn_and_continue)
        .rename(pixel_values=conf.image_column_name)
        .map(ContextTargetPatcher(conf.patcher))
        .select(filter_rows)
        .packed_x_y(
            conf.patcher.max_num_context_tokens,
            max_sequence_length,
            conf.packer_batch_size,
            pad_value_dict=pad_value_dict,
        )
        .shuffle(16)
        .to_tuple("x_patches", "y_patches")
        .batched(conf.batch_size // conf.packer_batch_size, partial=False)
    )

    if conf.mode == "make-viz":
        sample = next(iter(dataset))

        # Decode and save one batch of images
        viz_output_path = (
            Path(".") / "viz-outputs" / datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        viz_output_path.mkdir(parents=True)

        # overlay a black X-mark line on for all context tokens
        patch_border = torch.ones(patch_size, patch_size, num_image_channels)
        patch_border.diagonal().zero_()
        patch_border = patch_border * patch_border.flip(0)
        patch_border = einx.rearrange("ph pw c -> (ph pw c)", patch_border)

        x_patches, y_patches = sample
        x_patches["patches"] = einx.multiply(
            "... d, d", x_patches["patches"], patch_border
        )

        b = x_patches.size(0)
        for i in range(b):
            x_seq = x_patches.iloc[i]
            y_seq = y_patches.iloc[i]
            sequence_ids = x_seq["sequence_ids"].unique().tolist()
            for j in sequence_ids:
                if j == MASK_SEQUENCE_ID:
                    continue

                device = x_seq["patches"].device

                x_sample_mask = x_seq["sequence_ids"] == j
                y_sample_mask = y_seq["sequence_ids"] == j

                x_sample = x_seq.iloc[x_sample_mask]
                y_sample = y_seq.iloc[y_sample_mask]

                assert x_sample.size(0) > 0
                assert y_sample.size(0) > 0

                all_position_ids = torch.cat(
                    (x_sample["position_ids"], y_sample["position_ids"]), 0
                )

                min_ph_pw = all_position_ids.amin(0)
                max_ph_pw = all_position_ids.amax(0)

                ph, pw = (max_ph_pw - min_ph_pw + 1).tolist()

                image = torch.zeros(
                    ph, pw, input_size, dtype=torch.uint8, device=device
                )

                for k in range(y_sample.size(0)):
                    token = y_sample.iloc[k]
                    patch = token["patches"]
                    hid, wid = token["position_ids"] - min_ph_pw
                    image[hid, wid] = patch

                for k in range(x_sample.size(0)):
                    token = x_sample.iloc[k]
                    patch = token["patches"]
                    hid, wid = token["position_ids"] - min_ph_pw
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

    elif conf.mode == "validate":
        # TODO load model
        accuracies = validate(
            model=model,
            image_column_name=conf.image_column_name,
            label_column_name=conf.label_column_name,
            num_classes=conf.num_classes,
            patch_size=patch_size,
            validation_image_size=conf.validation_image_size,
            batch_size=conf.batch_size,
            num_workers=conf.num_workers,
            train_dataset_pattern=conf.train_dataset_pattern,
            val_dataset_pattern=conf.val_dataset_pattern,
            dtype=dtype,
            should_compile=conf.should_compile,
            test_mode=conf.test_mode,
            validation_probe_lr=conf.validation_probe_lr,
            validation_probe_batch_size=conf.validation_probe_batch_size,
            validation_train_epochs=conf.validation_train_epochs,
        )
        print("ACCURACY", accuracy)

    elif conf.mode == "train":
        train_dataloader = DataLoader(
            dataset, num_workers=conf.num_workers, batch_size=None
        )
        model = IJEPADepthSmart(conf.model).to(device)
        trainable_params = tuple(p for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(
            trainable_params, lr=conf.lr, betas=(0.9, 0.95), weight_decay=0.05
        )
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=conf.num_warmup_steps
        )

        conf_d = asdict(conf)
        conf_d["num_params"] = sum(p.nelement() for p in trainable_params)

        wandb.init(
            project="ijepa-depthsmart",
            config=conf_d,
            mode="disabled" if conf.test_mode else None,
        )

        if conf.should_compile:
            model = torch.compile(model)

        @contextmanager
        def autocast_fn():
            with torch.autocast(device.type, dtype):
                yield

        training_state = dict(global_step=0, epoch=0)

        for epoch in range(training_state["epoch"], conf.num_epochs):

            def train_epoch():
                for batch in tqdm(train_dataloader, desc=f"training epoch {epoch}"):
                    x_patches_ts, y_patches_ts, *_ = batch

                    x_patches = x_patches_ts.named_columns.pop("patches")
                    x_position_ids = x_patches_ts.named_columns.pop("position_ids")
                    x_sequence_ids = x_patches_ts.named_columns.pop("sequence_ids")
                    x_token_ids = torch.cat(
                        (x_sequence_ids.unsqueeze(-1), x_position_ids), -1
                    )

                    y_patches = y_patches_ts.named_columns.pop("patches")
                    y_position_ids = y_patches_ts.named_columns.pop("position_ids")
                    y_sequence_ids = y_patches_ts.named_columns.pop("sequence_ids")
                    y_token_ids = torch.cat(
                        (y_sequence_ids.unsqueeze(-1), y_position_ids), -1
                    )

                    x_patches = x_patches.to(
                        device=device, dtype=dtype, non_blocking=True
                    )
                    x_patches = (x_patches / 255) * 2 - 1
                    y_patches = y_patches.to(
                        device=device, dtype=dtype, non_blocking=True
                    )
                    y_patches = (y_patches / 255) * 2 - 1

                    x_token_ids = x_token_ids.to(device, non_blocking=True)
                    y_token_ids = y_token_ids.to(device, non_blocking=True)

                    interp = min(
                        1, training_state["global_step"] / conf.interp_warmup_steps
                    )
                    interp = 0

                    ema_beta = (
                        min(
                            1,
                            training_state["global_step"] / conf.ema_beta_warmup_steps,
                        )
                        * conf.ema_beta
                    )

                    should_log = (
                        training_state["global_step"] % conf.log_every_num_steps
                    ) == 0

                    with autocast_fn():
                        result_dict = model(
                            x_patches,
                            y_patches,
                            x_token_ids,
                            y_token_ids,
                            interp=interp,
                            return_smooth_rank=should_log,
                        )

                    loss = result_dict["loss"]

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    for ema_p, p in zip(
                        model.ema_encoder.parameters(), model.encoder.parameters()
                    ):
                        ema_p.lerp_(p, 1 - ema_beta)

                    if should_log:
                        num_samples = 0
                        for ids_seq in x_token_ids[..., 0].cpu():
                            ids = torch.unique(ids_seq).tolist()
                            try:
                                ids.remove(MASK_SEQUENCE_ID)
                            except:
                                pass
                            num_samples += len(ids)

                        wandb.log(
                            dict(
                                epoch=epoch,
                                loss=loss,
                                num_samples=num_samples,
                                lr=lr_scheduler.get_last_lr()[-1],
                                ema_beta=ema_beta,
                                smooth_rank=result_dict["smooth_rank"],
                                interp=interp,
                            ),
                            step=training_state["global_step"],
                        )

                    training_state["global_step"] += 1

                    if conf.test_mode:
                        break

            train_epoch()

            is_last_epoch = epoch == conf.num_epochs - 1
            should_validate = (
                conf.test_mode
                or is_last_epoch
                or ((epoch > 0) and (epoch % conf.validate_every_num_epochs == 0))
            )

            if should_validate:
                gc.collect()
                torch.cuda.empty_cache()

                accuracies = validate(
                    model=model,
                    image_column_name=conf.image_column_name,
                    label_column_name=conf.label_column_name,
                    num_classes=conf.num_classes,
                    patch_size=patch_size,
                    validation_image_size=conf.validation_image_size,
                    batch_size=conf.batch_size,
                    num_workers=conf.num_workers,
                    train_dataset_pattern=conf.train_dataset_pattern,
                    val_dataset_pattern=conf.val_dataset_pattern,
                    dtype=dtype,
                    test_mode=conf.test_mode,
                    should_compile=conf.should_compile,
                    validation_probe_lr=conf.validation_probe_lr,
                    validation_probe_batch_size=conf.validation_probe_batch_size,
                    validation_train_epochs=conf.validation_train_epochs,
                )

                gc.collect()
                torch.cuda.empty_cache()

                depths = list(range(len(accuracies)))

                line_series = wandb.plot.line_series(
                    xs=depths,
                    ys=[accuracies],
                    keys=[f"Epoch {epoch:03}"],
                    title="Feature Depth VS Accuracy@1",
                    xname="Depth",
                )

                wandb.log(
                    {"depth vs acc@1": line_series, "epoch": epoch},
                    step=training_state["global_step"],
                )

                print("EPOCH", epoch, "accuracies", accuracies)

            training_state["epoch"] += 1

            if conf.test_mode:
                break

        checkpoint_path = (
            Path("checkpoints") / f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pt"
        )
        checkpoint_path.parent.mkdir(exist_ok=True)

        torch.save(
            {
                "training_state": training_state,
                "model": model,
                "optimizer": optimizer,
            },
            str(checkpoint_path),
        )
        print("Saved to ", checkpoint_path)


if __name__ == "__main__":
    jsonargparse.CLI(main)
