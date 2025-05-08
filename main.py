from typing import Literal
import gc
from contextlib import contextmanager
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import numpy as np
import einx
import jsonargparse
from torch.utils.data import DataLoader
import torchvision
import random
import torch
from dataclasses import dataclass, field, asdict
import wandb

from src.dataset import get_context_target_dataset
from src.model import IJEPADepthSmartConfig, IJEPADepthSmart, MASK_SEQUENCE_ID
from src.validate import validate


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
    start_lr: float = 1e-4
    lr: float = 5e-4
    num_epochs: int = 100

    patch_size: int = 16

    log_every_num_steps: int = 50
    validate_every_num_epochs: int = 10

    validation_probe_lr: float = 1e-3
    validation_image_size: int = 256
    validation_train_epochs: int = 50
    validation_probe_batch_size: int = 2048
    validation_depthsmart_mode: Literal["learned", "extract-layers", "lastlayer"] = (
        "extract-layers"
    )

    resume_path: str | None = None

    test_mode: bool = False

    num_image_channels: int = 3

    ema_beta: float = 0.996
    ema_beta_start: float = 0.2
    ema_beta_warmup_steps: int = 1000

    interp_warmup_steps: int = 100000

    # Webdataset tars
    train_dataset_pattern: str = "/nvme/imagenet1k/imagenet1k-train-{0000..1023}.tar"
    val_dataset_pattern: str = "/nvme/imagenet1k/imagenet1k-validation-{00..63}.tar"
    image_column_name: str = "jpg"
    label_column_name: str = "cls"
    num_classes: int = 1000

    num_register_tokens: int = 0
    min_context_capacity: float = 0.25
    max_context_capacity: float = 0.5

    model: IJEPADepthSmartConfig = field(
        default_factory=lambda: IJEPADepthSmartConfig()
    )

    mode: Literal["make-viz", "train", "validate"] = "train"

    def __post_init__(self):
        assert self.batch_size % self.packer_batch_size == 0
        assert self.packer_batch_size <= self.batch_size
        image_channels = 3
        assert self.model.encoder.input_size == image_channels * self.patch_size**2


def main(conf: MainConfig = MainConfig()):
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)

    device = torch.device(conf.device)
    dtype = getattr(torch, conf.dtype)

    input_size = conf.model.encoder.input_size
    num_image_channels = conf.num_image_channels

    patch_size = conf.patch_size

    dataset = get_context_target_dataset(
        dataset_pattern=conf.train_dataset_pattern,
        seed=conf.seed,
        image_column_name=conf.image_column_name,
        label_column_name=conf.label_column_name,
        batch_size=conf.batch_size,
        packer_batch_size=conf.packer_batch_size,
        num_register_tokens=conf.num_register_tokens,
        patch_size=patch_size,
        min_context_capacity=conf.min_context_capacity,
        max_context_capacity=conf.max_context_capacity,
    )

    training_state = dict(global_step=0, epoch=0)

    model = IJEPADepthSmart(conf.model).to(device)
    trainable_params = tuple(p for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=conf.start_lr, betas=(0.9, 0.95), weight_decay=0.05
    )

    if conf.resume_path is not None:

        def _load():
            d = torch.load(conf.resume_path, map_location=device, weights_only=False)
            model.load_state_dict(d["model"])
            training_state.update(d["training_state"])
            optimizer.load_state_dict(d["optimizer"])

        _load()

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

                x_sample_mask = (x_seq["sequence_ids"] == j) & (
                    x_seq["position_ids"][..., 0] == MASK_SEQUENCE_ID
                )
                y_sample_mask = (y_seq["sequence_ids"] == j) & (
                    y_seq["position_ids"][..., 0] == MASK_SEQUENCE_ID
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

                image = torch.zeros(
                    ph, pw, input_size, dtype=torch.uint8, device=device
                )

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

    elif conf.mode == "validate":
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
            validation_depthsmart_mode=conf.validation_depthsmart_mode,
            num_register_tokens=conf.num_register_tokens,
        )
        print("ACCURACIES", accuracies)

    elif conf.mode == "train":
        # accelerator = Accelerator()

        train_dataloader = DataLoader(
            dataset, num_workers=conf.num_workers, batch_size=None
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

        checkpoint_folder_path = (
            Path("checkpoints") / f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        def save():
            checkpoint_folder_path.mkdir(exist_ok=True, parents=True)

            existing_checkpoints = list(checkpoint_folder_path.iterdir())
            existing_checkpoints.sort()
            max_num_checkpoints = 5

            checkpoints_to_delete = existing_checkpoints[:-max_num_checkpoints]

            for existing_checkpoint in checkpoints_to_delete:
                print("Deleting checkpoint", existing_checkpoint)
                existing_checkpoint.unlink()

            save_path = checkpoint_folder_path / f"{training_state['epoch']:05}.pt"

            torch.save(
                {
                    "training_state": training_state,
                    "model": (
                        model._orig_mod.state_dict()
                        if hasattr(model, "_orig_mod")
                        else model.state_dict()
                    ),
                    "optimizer": optimizer.state_dict(),
                },
                str(save_path),
            )
            print("Saved to ", save_path)

        @contextmanager
        def autocast_fn():
            with torch.autocast(device.type, dtype):
                yield

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
                    y_patches = y_patches.to(
                        device=device, dtype=dtype, non_blocking=True
                    )
                    x_token_ids = x_token_ids.to(device, non_blocking=True)
                    y_token_ids = y_token_ids.to(device, non_blocking=True)

                    x_patches = (x_patches / 255) * 2 - 1
                    y_patches = (y_patches / 255) * 2 - 1

                    interp = min(
                        1, training_state["global_step"] / conf.interp_warmup_steps
                    )

                    ema_beta = (
                        min(
                            1,
                            training_state["global_step"] / conf.ema_beta_warmup_steps,
                        )
                        * (conf.ema_beta - conf.ema_beta_start)
                        + conf.ema_beta_start
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

                    lr = (
                        min(1, training_state["global_step"] / conf.num_warmup_steps)
                    ) * (conf.lr - conf.start_lr) + conf.start_lr
                    for g in optimizer.param_groups:
                        g["lr"] = lr

                    for ema_p, p in zip(
                        model.ema_encoder.parameters(), model.encoder.parameters()
                    ):
                        if p.is_floating_point():
                            ema_p.lerp_(p, 1 - ema_beta)
                        else:
                            ema_p.copy_(p)

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
                                lr=lr,
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
                    validation_depthsmart_mode=conf.validation_depthsmart_mode,
                    num_register_tokens=conf.num_register_tokens,
                )

                gc.collect()
                torch.cuda.empty_cache()

                depths = list(range(len(accuracies)))
                best_accuracy = max(accuracies)

                line_series = wandb.plot.line_series(
                    xs=depths,
                    ys=[accuracies],
                    keys=[f"Epoch {epoch:03}"],
                    title="Feature Depth VS Accuracy@1",
                    xname="Depth",
                )

                wandb.log(
                    {
                        "depth vs acc@1": line_series,
                        "epoch": epoch,
                        "acc@1": best_accuracy,
                    },
                    step=training_state["global_step"],
                )

                print("EPOCH", epoch, "accuracies", accuracies)

            training_state["epoch"] += 1

            if conf.test_mode:
                break

            save()

        save()


if __name__ == "__main__":
    jsonargparse.CLI(main)
