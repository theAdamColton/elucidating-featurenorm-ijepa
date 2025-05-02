import numpy as np
from glob import glob
import uuid
import tempfile
import torch
import einx
from contextlib import contextmanager
import torchvision
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F

from src.model import IJEPADepthSmart


class MultiDepthClassifier(nn.Module):
    def __init__(self, num_depth, num_classes, num_features):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(num_depth, num_classes, num_features))
        nn.init.trunc_normal_(self.weights, std=0.02)
        self.biases = nn.Parameter(torch.zeros(num_depth, num_classes))
        self.num_depth = num_depth

    def forward(self, emb, lab):
        logits = einx.dot("b n d, n c d -> b n c", emb, self.weights)
        logits = einx.add("b n c, n c", logits, self.biases)
        preds = logits.argmax(-1)

        logits = einx.rearrange("b n c -> (b n) c", logits)
        lab = einx.rearrange("b -> (b n)", lab, n=self.num_depth)
        loss = F.cross_entropy(logits, lab)

        return preds, loss


class SimplePatcher:
    def __init__(self, size=256, patch_size=16):
        self.size = size
        self.patch_size = patch_size

    def __call__(self, row):
        x = row.pop("pixel_values")
        _, og_h, og_w = x.shape
        crop_size = min(og_h, og_w)
        x = torchvision.transforms.CenterCrop(crop_size)(x)
        x = torchvision.transforms.Resize((self.size, self.size))(x)
        x = einx.rearrange("c (np ps)... -> (np...) (ps... c)", x, ps=self.patch_size)
        position_ids = torch.meshgrid(
            (
                torch.arange(self.size // self.patch_size),
                torch.arange(self.size // self.patch_size),
            ),
            indexing="ij",
        )
        position_ids = torch.stack(position_ids, -1)
        position_ids = einx.rearrange("s... nd -> (s...) nd", position_ids)
        sequence_ids = torch.full((x.shape[0],), 0)
        token_ids = torch.cat((sequence_ids.unsqueeze(-1), position_ids), -1)
        row["patches"] = x
        row["token_ids"] = token_ids
        return row


def validate(
    model: IJEPADepthSmart,
    image_column_name: str = "jpg",
    label_column_name: str = "cls",
    patch_size: int = 16,
    validation_image_size: int = 256,
    num_classes: int = 1000,
    batch_size: int = 256,
    num_workers: int = 4,
    train_dataset_pattern: str = "/nvme/imagenet1k/imagenet1k-train-{0000..1023}.tar",
    val_dataset_pattern: str = "/nvme/imagenet1k/imagenet1k-validation-{00..63}.tar",
    dtype: torch.dtype = torch.bfloat16,
    test_mode: bool = False,
    should_compile: bool = False,
    validation_probe_lr: float = 1e-3,
    validation_probe_batch_size: int = 2048,
    validation_train_epochs: int = 50,
):
    encoder = model.ema_encoder
    num_features = encoder.hidden_size
    device = next(iter(encoder.parameters())).device
    num_feature_depth = model.config.encoder.num_transformer_blocks + 1

    if should_compile:
        encoder = torch.compile(encoder)

    @contextmanager
    def autocast_fn():
        with torch.autocast(device.type, dtype):
            yield

    def _load_simple_dataloader(pattern):
        ds = (
            wds.WebDataset(pattern)
            .shuffle(1000)
            .decode("torchrgb8", handler=wds.handlers.warn_and_continue)
            .rename(
                pixel_values=image_column_name,
                label=label_column_name,
            )
            .map(SimplePatcher(validation_image_size, patch_size))
            .batched(batch_size)
            .shuffle(16)
        )
        dl = DataLoader(ds, batch_size=None, num_workers=num_workers)
        return dl

    val_train_dataloader = _load_simple_dataloader(train_dataset_pattern)
    val_test_dataloader = _load_simple_dataloader(val_dataset_pattern)

    with tempfile.TemporaryDirectory(dir=".") as tmpdir:

        def _embed_dataset(dl, prefix="train-"):
            with wds.ShardWriter(f"{tmpdir}/{prefix}%04d.tar") as writer:
                pass

                for batch in tqdm(dl, desc="embedding val dataset"):
                    patches = batch["patches"]
                    label = batch["label"]
                    token_ids = batch["token_ids"]

                    b, s, d = patches.shape

                    patches = patches.to(device=device, dtype=dtype)
                    patches = (patches / 255) * 2 - 1

                    token_ids = token_ids.to(device)

                    if model.config.depthsmart_mode == "disabled":
                        # Full depth
                        t = torch.full(
                            (b, s),
                            model.config.encoder.num_transformer_blocks,
                            device=device,
                        )

                        with autocast_fn():
                            with torch.inference_mode():
                                _, layer_features = model.ema_encoder(
                                    patches,
                                    t,
                                    token_ids,
                                    return_all_layer_features=True,
                                )

                        layer_features = einx.mean("n b s d -> b n d", layer_features)

                    elif model.config.depthsmart_mode == "random":
                        # Repeating batch to condition on all depths
                        # results in OOM!
                        #
                        layer_features = []
                        for layer_id in range(num_feature_depth):
                            t = torch.full((b, s), layer_id, device=device)

                            with autocast_fn():
                                with torch.inference_mode():
                                    features, *_ = model.ema_encoder(
                                        patches, t, token_ids
                                    )
                            features = einx.mean("b [s] d", features)
                            layer_features.append(features)

                        layer_features = torch.stack(layer_features, 1)

                    layer_features = layer_features.cpu().to(torch.float16).numpy()

                    for i in range(b):
                        writer.write(
                            {
                                "__key__": uuid.uuid4().hex,
                                "label.cls": label[i].item(),
                                "features.npy": layer_features[i],
                            }
                        )

                    if test_mode:
                        break

                urls = list(glob(f"{tmpdir}/{prefix}*.tar"))
                return urls

        train_tar_urls = _embed_dataset(val_train_dataloader, "train-")
        test_tar_urls = _embed_dataset(val_test_dataloader, "test-")

        classifier = MultiDepthClassifier(
            num_feature_depth, num_classes, num_features
        ).to(device)

        optim = torch.optim.AdamW(
            classifier.parameters(),
            lr=validation_probe_lr,
        )

        train_dataset = (
            wds.WebDataset(train_tar_urls, empty_check=False).shuffle(1000).decode()
        )
        train_dataloader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            batch_size=validation_probe_batch_size,
        )

        for val_epoch in tqdm(
            range(validation_train_epochs), desc="training val classifier"
        ):
            for batch in train_dataloader:
                emb, lab = batch.pop("features.npy"), batch.pop("label.cls")

                emb = emb.to(device, torch.float32)
                lab = lab.to(device)

                preds, loss = classifier(emb, lab)

                loss.backward()
                optim.step()
                optim.zero_grad()

                if test_mode:
                    break

            if test_mode:
                break

        test_dataset = (
            wds.WebDataset(test_tar_urls, empty_check=False)
            .decode()
            .batched(validation_probe_batch_size)
        )
        test_dataloader = DataLoader(
            test_dataset, num_workers=num_workers, batch_size=None
        )

        preds = []
        labs = []
        for batch in tqdm(
            test_dataloader,
            desc="testing probe",
        ):
            emb, lab = batch.pop("features.npy"), batch.pop("label.cls")

            emb = emb.to(device, torch.float32)
            lab = lab.to(device)

            with torch.inference_mode():
                pred, loss = classifier(emb, lab)

            preds.append(pred.cpu())
            labs.append(lab.cpu())

        preds = torch.cat(preds, 0)
        labs = torch.cat(labs, 0)
        correct_labels = einx.equal("b n, b", preds, labs)
        accuracies = einx.mean("[b] n", correct_labels.float())

    return accuracies
