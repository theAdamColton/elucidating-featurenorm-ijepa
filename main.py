from typing import Literal
from dataclasses import dataclass, field, asdict
import yaml
import gc
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import random

from tqdm import tqdm
import numpy as np
import einx
import jsonargparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torch
import matplotlib.pyplot as plt
import wandb
import tensorset as ts

from src.dataset import get_context_target_dataset, MASK_SEQUENCE_ID
from src.model import IJEPADepthSmartConfig, IJEPADepthSmart
from src.validate import validate
from src.validate_monocular_depth import validate_monocular_depth_prediction
from src.visualize_embeddings import features_to_rgb


def get_viz_output_path():
    viz_output_path = (
        Path(".") / "viz-outputs" / datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    viz_output_path.mkdir(parents=True)
    return viz_output_path


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
    num_epochs: int = 800

    patch_size: int = 16

    log_every_num_steps: int = 50
    validate_every_num_epochs: int = 10
    max_num_save_checkpoints: int = 2

    validation_probe_lr: float = 1e-3
    validation_image_size: int = 256
    validation_train_epochs: int = 50
    validation_monocular_depth_train_epochs: int = 10
    # Extract features from the last layer of the encoder to perform monocular depth estimation
    validation_monocular_depth_feature_depth: int = -1
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

    should_interp: bool = False
    interp_warmup_steps: int = 100000

    # Webdataset tars
    train_dataset_pattern: str = "/nvme/imagenet1k/imagenet1k-train-{0000..1023}.tar"
    val_dataset_pattern: str = "/nvme/imagenet1k/imagenet1k-validation-{00..63}.tar"
    image_column_name: str = "jpg"
    label_column_name: str = "cls"
    num_classes: int = 1000

    # Webdataset tars
    monocular_depth_train_dataset_pattern: str = (
        "/nvme/nyu-depthv2-wds/nyu-depth-train-{00000..13}.tar"
    )
    monocular_depth_val_dataset_pattern: str = (
        "/nvme/nyu-depthv2-wds/nyu-depth-val-00000.tar"
    )
    depth_column_name: str = "depth.npy"

    num_register_tokens: int = 0
    min_context_capacity: float = 0.05
    max_context_capacity: float = 0.95
    absolute_max_context_capacity: float = 0.5

    model: IJEPADepthSmartConfig = field(
        default_factory=lambda: IJEPADepthSmartConfig()
    )

    mode: Literal[
        "make-viz",
        "train",
        "validate",
        "visualize-embeddings",
        "plot-sample-losses",
        "validate-monocular-depth",
    ] = "train"

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

    dataset, context_sequence_length, target_sequence_length = (
        get_context_target_dataset(
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
            absolute_max_context_capacity=conf.absolute_max_context_capacity,
        )
    )
    dataloader = DataLoader(dataset, num_workers=conf.num_workers, batch_size=None)

    training_state = dict(global_step=0, epoch=0)

    model = IJEPADepthSmart(conf.model).to(device)
    trainable_params = tuple(p for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=conf.start_lr, betas=(0.9, 0.95), weight_decay=0.05
    )

    @contextmanager
    def autocast_fn():
        with torch.autocast(device.type, dtype):
            yield

    if conf.resume_path is not None:

        def _load():
            d = torch.load(conf.resume_path, map_location=device, weights_only=False)
            model.load_state_dict(d["model"], strict=False)
            training_state.update(d["training_state"])
            try:
                optimizer.load_state_dict(d["optimizer"])
            except Exception as e:
                print("Could not load optimizer state dict! Error:", e)

        _load()

    if conf.mode == "make-viz":
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

    elif conf.mode == "validate-monocular-depth":
        result = validate_monocular_depth_prediction(
            model=model,
            image_column_name=conf.image_column_name,
            depth_column_name=conf.depth_column_name,
            patch_size=patch_size,
            validation_image_size=conf.validation_image_size,
            batch_size=conf.batch_size,
            num_workers=conf.num_workers,
            train_dataset_pattern=conf.monocular_depth_train_dataset_pattern,
            val_dataset_pattern=conf.monocular_depth_val_dataset_pattern,
            feature_depth=conf.validation_monocular_depth_feature_depth,
            dtype=dtype,
            test_mode=conf.test_mode,
            should_compile=conf.should_compile,
            validation_probe_lr=conf.validation_probe_lr,
            validation_probe_batch_size=conf.validation_probe_batch_size,
            validation_train_epochs=conf.validation_train_epochs,
            num_register_tokens=conf.num_register_tokens,
        )
        print("MONOCULAR DEPTH RESULT", result)

    elif conf.mode == "plot-sample-losses":
        # TODO test me!
        batch = next(iter(dataloader))

        patches, token_ids = prepare_context_target_batch(batch, device, dtype)

        x_patches = patches[:, :context_sequence_length]
        y_patches = patches[:, context_sequence_length:]

        x_token_ids = token_ids[:, :context_sequence_length]
        y_token_ids = token_ids[:, context_sequence_length:]

        b, x_patches_length, d = x_patches.shape

        sample_ids = torch.cat((x_token_ids, y_token_ids), 1)[..., 0]

        batch_unique_sample_ids = []
        for i in range(b):
            unique_sample_ids = torch.unique(sample_ids[i]).tolist()
            if MASK_SEQUENCE_ID in unique_sample_ids:
                unique_sample_ids.remove(MASK_SEQUENCE_ID)
            batch_unique_sample_ids.append(unique_sample_ids)

        sample_losses = [
            torch.zeros(len(unique_sample_ids))
            for unique_sample_ids in batch_unique_sample_ids
        ]
        sample_losses_variance = [
            torch.zeros(len(unique_sample_ids))
            for unique_sample_ids in batch_unique_sample_ids
        ]

        # Compute the loss several times, each time using a different context and target
        iters = 1

        for i in range(iters):
            all_patches = torch.cat((x_patches, y_patches), 1)
            all_token_ids = torch.cat((x_token_ids, y_token_ids), 1)

            # Shuffle the patches, to create new random sets of context
            # and target tokens
            # Note, that this isn't windowed masking like in the
            # official context-target dataset

            b, s, d = all_patches.shape
            indices = torch.rand(b, s).argsort(dim=-1)
            all_patches = einx.get_at("b [s] d, b n -> b n d", all_patches, indices)
            all_token_ids = einx.get_at(
                "b [s] nd, b n -> b n nd", all_token_ids, indices
            )

            x_patches, y_patches = (
                all_patches[:, :x_patches_length],
                all_patches[:, x_patches_length:],
            )
            x_token_ids, y_token_ids = (
                all_token_ids[:, :x_patches_length],
                all_token_ids[:, x_patches_length:],
            )

            patches = torch.cat((x_patches, y_patches), 1)
            token_ids = torch.cat((x_token_ids, y_token_ids), 1)

            with torch.inference_mode():
                with autocast_fn():
                    result_dict = model(
                        patches,
                        token_ids,
                        context_sequence_length=context_sequence_length,
                        return_tokenwise_loss=True,
                        return_predictor_target_token_ids=True,
                    )

            # tokenwise loss is the batch repeated loss
            # from the predictor
            tokenwise_loss = result_dict["tokenwise_loss"].cpu().float()
            # target sequence_ids is batch repeated sequence ids fed to the predictor
            target_sequence_ids = result_dict["predictor_target_token_ids"][
                ..., 0
            ].cpu()

            tokenwise_loss = einx.mean("rb ys [d]", tokenwise_loss)

            # Compute the mean loss for each unique sample across the batch
            for i in range(conf.model.predictor_batch_repeat):
                for j in range(b):
                    batch_index = i * b + j
                    sequence_ids = target_sequence_ids[batch_index]
                    for k, sample_id in enumerate(batch_unique_sample_ids[j]):
                        mask = sequence_ids == sample_id
                        if not mask.any():
                            continue

                        # Measure the mean sample loss, and the variance of the sample loss
                        sample_loss = tokenwise_loss[batch_index, mask].mean()
                        sample_loss_variance = tokenwise_loss[batch_index, mask].var()

                        # Take the mean of sample losses across iterations
                        # TODO! This doesnt handle the special case
                        # where sometimes a sample might not be included
                        # in the loss for a batch because it is randomly dropped
                        sample_losses[j][k] += sample_loss / iters
                        sample_losses_variance[j][k] += sample_loss_variance / iters

        # Save an image for each sample
        output_path = get_viz_output_path()

        all_patches = all_patches.cpu().float()
        all_token_ids = all_token_ids.cpu()
        for i in range(b):
            for j, sample_id in enumerate(batch_unique_sample_ids[i]):
                mask = all_token_ids[i, :, 0] == sample_id
                if not mask.any():
                    continue
                sample_patches = all_patches[i, mask]
                sample_ids = all_token_ids[i, mask]

                sample_patches = (sample_patches + 1) / 2

                sample_position_ids = sample_ids[:, -2:]
                nph, npw = (sample_position_ids.amax(0) + 1).tolist()
                unpacked_patches = torch.zeros(nph, npw, d)

                for (hid, wid), patch in zip(sample_position_ids, sample_patches):
                    unpacked_patches[hid, wid] = patch

                image = einx.rearrange(
                    "nph npw (ph pw c) -> c (nph ph) (npw pw)",
                    unpacked_patches,
                    ph=patch_size,
                    pw=patch_size,
                    c=conf.num_image_channels,
                )
                image = (image.clip(0, 1) * 255).to(torch.uint8)

                sample_loss = sample_losses[i][j].item()
                sample_loss = round(sample_loss, 5)

                sample_loss_variance = sample_losses_variance[i][j].item()
                sample_loss_variance = round(sample_loss_variance, 5)

                image_save_path = (
                    output_path
                    / f"sample-{i:04}-{sample_id:08} loss {sample_loss} variance {sample_loss_variance}.png"
                )

                torchvision.io.write_png(image, str(image_save_path))

                print("wrote to ", image_save_path)

        # Plot the distribution of the losses
        all_losses = []
        for batch in sample_losses:
            all_losses.extend(batch.tolist())

        all_losses_variance = []
        for batch in sample_losses_variance:
            all_losses_variance.extend(batch.tolist())

        plt.hist(all_losses, bins=20, density=True, color="skyblue", edgecolor="black")
        plt.xlabel("sample loss")
        plt.ylabel("frequency")
        plot_save_path = output_path / "histogram.png"
        plt.savefig(str(plot_save_path))
        print("saved to ", plot_save_path)
        plt.close()

        plt.hist(
            all_losses,
            bins=20,
            cumulative=True,
            density=True,
            color="skyblue",
        )
        plt.xlabel("sample loss")
        plt.ylabel("frequency")
        plot_save_path = output_path / "cum_histogram.png"
        plt.savefig(str(plot_save_path))
        print("saved to ", plot_save_path)
        plt.close()

        plt.hist(
            all_losses_variance,
            bins=20,
            density=True,
            color="skyblue",
            edgecolor="black",
        )
        plt.xlabel("sample loss variance")
        plt.ylabel("frequency")
        plot_save_path = output_path / "histogram_variance.png"
        plt.savefig(str(plot_save_path))
        print("saved to ", plot_save_path)
        plt.close()

        plt.hist(
            all_losses_variance,
            bins=20,
            cumulative=True,
            density=True,
            color="skyblue",
        )
        plt.xlabel("sample loss variance")
        plt.ylabel("frequency")
        plot_save_path = output_path / "cum_histogram_variance.png"
        plt.savefig(str(plot_save_path))
        print("saved to ", plot_save_path)
        plt.close()

    elif conf.mode == "visualize-embeddings":
        # TODO! test me!
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
        t = torch.full((b, s), conf.model.encoder.num_transformer_blocks, device=device)

        with autocast_fn():
            with torch.inference_mode():
                embeddings, *_ = model.ema_encoder(
                    x=y_patches, t=t, token_ids=y_token_ids
                )

        *_, hidden_d = embeddings.shape

        embeddings = embeddings.cpu().float()
        y_token_ids = y_token_ids.cpu()
        y_patches = (y_patches.cpu().float() + 1) / 2
        y_patches = y_patches.clip(0, 1) * 255
        y_patches = y_patches.to(torch.uint8)

        viz_output_path = get_viz_output_path()

        for i in range(b):
            sequence_ids, position_ids = y_token_ids[i, :, 0], y_token_ids[i, :, -2:]

            unique_sequence_ids = sequence_ids.unique().tolist()
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
                    c=conf.num_image_channels,
                )

                image = torch.cat((unpacked_pixel_image, unpacked_embedding_image), -1)

                output_path = viz_output_path / f"{i:05} {sequence_id:08}.png"
                torchvision.io.write_png(image, str(output_path))
                print("Wrote", output_path)

    elif conf.mode == "train":
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

            checkpoints_to_delete = existing_checkpoints[
                : -conf.max_num_save_checkpoints
            ]

            for existing_checkpoint in checkpoints_to_delete:
                print("Deleting checkpoint", existing_checkpoint)
                existing_checkpoint.unlink()

            checkpoint_save_path = (
                checkpoint_folder_path / f"{training_state['epoch']:05}.pt"
            )

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
                str(checkpoint_save_path),
            )
            print("Saved checkpoint to ", checkpoint_save_path)

            yaml_save_path = checkpoint_folder_path / "config.yaml"
            conf_dict = asdict(conf)
            # Hack to allow loading from jsonargparse
            conf_dict = dict(conf=conf_dict)
            with open(yaml_save_path, "w") as f:
                yaml.dump(conf_dict, f)

        for epoch in range(training_state["epoch"], conf.num_epochs):

            def train_epoch():
                for batch in tqdm(dataloader, desc=f"training epoch {epoch}"):
                    patches, token_ids = prepare_context_target_batch(
                        batch, device, dtype
                    )

                    interp = 0
                    if conf.should_interp:
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
                            patches=patches,
                            token_ids=token_ids,
                            context_sequence_length=context_sequence_length,
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
                        for ids_seq in token_ids[..., 0].cpu():
                            ids = torch.unique(ids_seq).tolist()
                            if MASK_SEQUENCE_ID in ids:
                                ids.remove(MASK_SEQUENCE_ID)
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
                    should_compile=conf.should_compile,
                    test_mode=conf.test_mode,
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
