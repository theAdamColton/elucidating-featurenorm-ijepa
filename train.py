from typing import Literal
from contextlib import contextmanager
import torch.nn.functional as F
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

from src.model import IJEPADepthSmartConfig, IJEPADepthSmart, MASK_SEQUENCE_ID


class SimplePatcher:
    def __init__(self, size=256, patch_size=16):
        self.resize = torchvision.transforms.Resize((size, size))
        self.size = size
        self.patch_size = patch_size

    def __call__(self, row):
        x = row.pop("pixel_values")
        x = self.resize(x)
        x = einx.rearrange("c (np ps)... -> (np...) (ps... c)", x, ps=self.patch_size)
        position_ids = torch.meshgrid(
            (
                torch.arange(self.size // self.patch_size),
                torch.arange(self.size // self.patch_size),
            ),
            indexing="ij",
        )
        position_ids = torch.stack(position_ids, -1)
        position_ids = einx.rearrange("np... nd -> (np...) nd", position_ids)
        sequence_ids = torch.full((x.shape[0],), 0)
        token_ids = torch.cat((sequence_ids.unsqueeze(-1), position_ids), -1)
        row["patches"] = x
        row["token_ids"] = token_ids
        return row


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


class ContextTargetPatcher:
    def __init__(self, config=ContextTargetPatcherConfig()):
        self.config = config

    def __call__(self, row):
        """
        x: pixel values of shape (c h w)
        """
        config = self.config
        x = row.pop("pixel_values")

        row["x_patches"] = None
        row["y_patches"] = None

        c, input_h, input_w = x.shape

        input_side_length = (input_h + input_w) / 2
        max_side_length = min(config.max_side_length, int(input_side_length))

        if max_side_length < config.min_side_length:
            print("Warning, image is too small to process!", input_h, input_w)
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

        x = torchvision.transforms.Resize(image_crop_size)(x)

        x = einx.rearrange("c (np ps)... -> np... (ps... c)", x, ps=config.patch_size)

        position_ids = torch.meshgrid(
            (torch.arange(x.shape[0]), torch.arange(x.shape[1])), indexing="ij"
        )
        position_ids = torch.stack(position_ids, -1)

        x = einx.rearrange(
            "(nw ws)... d -> (nw...) (ws...) d", x, ws=config.window_size
        )
        position_ids = einx.rearrange(
            "(nw ws)... nd -> (nw...) (ws...) nd", position_ids, ws=config.window_size
        )

        nw, ws, d = x.shape

        max_num_ctx_windows = config.max_num_context_tokens // ws
        max_num_ctx_windows = min(nw - 1, max_num_ctx_windows)

        if nw <= 2:
            num_ctx_windows = 1
        else:
            num_ctx_windows = random.randint(1, max_num_ctx_windows)

        num_target_windows = nw - num_ctx_windows
        random_ids = torch.randperm(nw, device=x.device)
        ctx_ids = random_ids[:num_ctx_windows]

        ctx = x[ctx_ids]
        ctx_position_ids = position_ids[ctx_ids]

        target = x

        ctx = einx.rearrange("nw ws d -> (nw ws) d", ctx)
        ctx_position_ids = einx.rearrange("nw ws nd -> (nw ws) nd", ctx_position_ids)

        target = einx.rearrange("nw ws d -> (nw ws) d", target)
        target_position_ids = einx.rearrange("nw ws nd -> (nw ws) nd", position_ids)

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
class TrainConfig:
    should_compile: bool = False
    dtype: str = "bfloat16"
    device: str = "cuda"
    batch_size: int = 8
    packer_batch_size: int = 8
    num_workers: int = 0
    seed: int = 42
    lr: float = 5e-4
    num_epochs: int = 10

    log_every_num_steps: int = 50

    validation_probe_lr: float = 1e-3
    validation_image_size: int = 256
    validation_train_epochs: int = 50
    validation_probe_batch_size: int = 2048

    test_mode: bool = False

    num_image_channels: int = 3

    ema_beta: float = 0.995
    interp_warmup_steps: int = 5000

    # Webdataset tars
    train_dataset_pattern: str = "/nvme/imagenet1k/imagenet1k-train-{0000..1023}.tar"
    val_dataset_pattern: str = "/nvme/imagenet1k/imagenet1k-validation-{00..63}.tar"
    image_column_name: str = "jpg"
    label_column_name: str = "cls"

    patcher: ContextTargetPatcherConfig = field(
        default_factory=lambda: ContextTargetPatcherConfig()
    )

    model: IJEPADepthSmartConfig = field(
        default_factory=lambda: IJEPADepthSmartConfig()
    )

    mode: Literal["make-viz", "train"] = "train"


def main(conf: TrainConfig = TrainConfig()):
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

    dataset = cabbage_patch.CabbageDataset(
        conf.train_dataset_pattern, shardshuffle=True
    )
    dataset = (
        dataset.shuffle(1000)
        .decode("torchrgb8")
        .rename(pixel_values=conf.image_column_name)
        .map(ContextTargetPatcher(conf.patcher))
        .select(filter_rows)
        .packed_x_y(
            conf.patcher.max_num_context_tokens,
            max_sequence_length,
            conf.packer_batch_size,
            pad_value_dict=pad_value_dict,
        )
        .shuffle(10)
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
        sample["x_patches"]["patches"] = einx.multiply(
            "... d, d", sample["x_patches"]["patches"], patch_border
        )

        b = sample["x_patches"].size(0)
        for i in range(b):
            x_seq = sample["x_patches"].iloc[i]
            y_seq = sample["y_patches"].iloc[i]
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

                min_ph_pw = y_sample["position_ids"].amin(0)
                max_ph_pw = y_sample["position_ids"].amax(0)

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

    elif conf.mode == "train":

        train_dataloader = DataLoader(
            dataset, num_workers=conf.num_workers, batch_size=None
        )
        model = IJEPADepthSmart(conf.model).to(device)
        trainable_params = tuple(p for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(
            trainable_params, lr=conf.lr, betas=(0.9, 0.95), weight_decay=0.05
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
            model.encoder = torch.compile(model.encoder)

        @contextmanager
        def autocast_fn():
            with torch.autocast(device.type, dtype):
                yield

        training_state = dict(global_step=0)

        for epoch in range(conf.num_epochs):
            for batch in tqdm(train_dataloader, desc=f"training epoch {epoch}"):
                x_patches_ts, y_patches_ts = batch["x_patches"], batch["y_patches"]

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

                x_patches = x_patches.to(device=device, dtype=dtype)
                x_patches = (x_patches / 255) * 2 - 1
                y_patches = y_patches.to(device=device, dtype=dtype)
                y_patches = (y_patches / 255) * 2 - 1

                x_token_ids = x_token_ids.to(device)
                y_token_ids = y_token_ids.to(device)

                interp = min(
                    1, training_state["global_step"] / conf.interp_warmup_steps
                )

                with autocast_fn():
                    result_dict = model(
                        x_patches, y_patches, x_token_ids, y_token_ids, interp=interp
                    )

                loss = result_dict["loss"]

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                for ema_p, p in zip(
                    model.ema_encoder.parameters(), model.encoder.parameters()
                ):
                    ema_p.lerp_(p, 1 - conf.ema_beta)

                lr = conf.lr
                for g in optimizer.param_groups:
                    g["lr"] = lr

                should_log = (
                    training_state["global_step"] % conf.log_every_num_steps
                ) == 0

                if should_log:
                    num_samples = 0
                    for ids_seq in x_token_ids.cpu():
                        ids = torch.unique(ids_seq).tolist()
                        try:
                            ids.remove(MASK_SEQUENCE_ID)
                        except:
                            pass
                        num_samples += len(ids)

                    wandb.log(
                        dict(epoch=epoch, loss=loss, num_samples=num_samples, lr=lr),
                        step=training_state["global_step"],
                    )

                training_state["global_step"] += 1

                if conf.test_mode:
                    break

            def validate():
                def _load_simple_dataloader(pattern):
                    ds = (
                        wds.WebDataset(pattern)
                        .decode("torchrgb8")
                        .rename(
                            pixel_values=conf.image_column_name,
                            label=conf.label_column_name,
                        )
                        .map(SimplePatcher(conf.validation_image_size, patch_size))
                        .batched(conf.batch_size)
                    )
                    dl = DataLoader(ds, batch_size=None, num_workers=conf.num_workers)
                    return dl

                val_train_dataloader = _load_simple_dataloader(
                    conf.train_dataset_pattern
                )
                val_test_dataloader = _load_simple_dataloader(conf.val_dataset_pattern)

                def _embed_dataset(dl):
                    embeddings = []
                    labels = []
                    for batch in tqdm(dl, desc="embedding val dataset"):
                        patches = batch["patches"]
                        label = batch["label"]
                        token_ids = batch["token_ids"]

                        b, s, d = patches.shape

                        patches = patches.to(device=device, dtype=dtype)
                        patches = (patches / 255) * 2 - 1

                        token_ids = token_ids.to(device)

                        # Full depth
                        t = torch.full(
                            (b, s),
                            conf.model.encoder.num_transformer_blocks + 1,
                            device=device,
                        )

                        with autocast_fn():
                            with torch.inference_mode():
                                emb, *_ = model.encoder(patches, t, token_ids)

                        emb = emb.mean(1).cpu().float()

                        embeddings.append(emb)
                        labels.append(label)

                        if conf.test_mode:
                            break

                    embeddings = torch.cat(embeddings, 0)
                    labels = torch.cat(labels, 0)

                    return embeddings, labels

                train_embeddings, train_labels = _embed_dataset(val_train_dataloader)
                test_embeddings, test_labels = _embed_dataset(val_test_dataloader)

                num_classes = int(train_labels.max()) + 1
                classifier = nn.Linear(train_embeddings.shape[-1], num_classes).to(
                    device
                )
                c_optim = torch.optim.AdamW(
                    classifier.parameters(), lr=conf.validation_probe_lr
                )

                for val_epoch in tqdm(
                    range(conf.validation_train_epochs), desc="training val classifier"
                ):
                    rand_indices = torch.randperm(train_embeddings.shape[0])
                    train_embeddings = train_embeddings[rand_indices]
                    train_labels = train_labels[rand_indices]

                    for emb, lab in zip(
                        train_embeddings.split(conf.validation_probe_batch_size),
                        train_labels.split(conf.validation_probe_batch_size),
                    ):

                        emb = emb.to(device)
                        lab = lab.to(device)
                        logits = classifier(emb)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.shape[-1]), lab.view(-1)
                        )
                        loss.backward()
                        c_optim.step()
                        c_optim.zero_grad()

                        if conf.test_mode:
                            break

                    if conf.test_mode:
                        break

                preds = []
                for emb in tqdm(
                    test_embeddings.split(conf.validation_probe_batch_size),
                    desc="testing probe",
                ):
                    emb = emb.to(device)
                    with torch.inference_mode():
                        logits = classifier(emb)
                    pred = logits.argmax(-1)
                    preds.append(pred.cpu())
                preds = torch.cat(preds, 0)
                accuracy = (preds == test_labels).float().mean()

                return accuracy

            accuracy = validate()
            wandb.log({"acc@1": accuracy}, step=training_state["global_step"])

            print("EPOCH", epoch, "accuracy", accuracy)

            if conf.test_mode:
                break


if __name__ == "__main__":
    jsonargparse.CLI(main)
