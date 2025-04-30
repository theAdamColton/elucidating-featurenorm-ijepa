from typing import Literal
import os
import numpy as np
import tempfile
import zarr
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
        self.resize = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size),
                torchvision.transforms.CenterCrop(size),
            ]
        )
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
            .decode("torchrgb8", handler=wds.handlers.warn_and_continue)
            .rename(
                pixel_values=image_column_name,
                label=label_column_name,
            )
            .map(SimplePatcher(validation_image_size, patch_size))
            .batched(batch_size)
        )
        dl = DataLoader(ds, batch_size=None, num_workers=num_workers)
        return dl

    val_train_dataloader = _load_simple_dataloader(train_dataset_pattern)
    val_test_dataloader = _load_simple_dataloader(val_dataset_pattern)

    with tempfile.TemporaryDirectory() as tmpdir:

        def _embed_dataset(dl):
            embedding_file_name = tempfile.mktemp(suffix=".zarr", dir=tmpdir)
            embedding_store = zarr.create_array(
                embedding_file_name,
                shape=(0, num_feature_depth, num_features),
                dtype=np.float16,
            )

            labels = []

            for batch in tqdm(dl, desc="embedding val train dataset"):
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
                                patches, t, token_ids, return_all_layer_features=True
                            )

                    layer_features = einx.rearrange(
                        "n b s d -> (n b) s d", layer_features
                    )

                elif model.config.depthsmart_mode == "random":
                    # Repeat batch to condition on all depths
                    t = torch.arange(num_feature_depth, device=device)
                    t = einx.rearrange("n -> (n b) s", t, b=b, s=s)

                    patches = einx.rearrange(
                        "b s d -> (n b) s d", patches, n=num_feature_depth
                    )

                    token_ids = einx.rearrange(
                        "b s nd -> (n b) s nd", token_ids, n=num_feature_depth
                    )

                    with autocast_fn():
                        with torch.inference_mode():
                            layer_features, *_ = model.ema_encoder(
                                patches, t, token_ids
                            )

                layer_features = einx.mean("(n b) s d -> b n d", layer_features, b=b)
                layer_features = layer_features.to(
                    dtype=torch.float16, device="cpu"
                ).numpy()

                embedding_store.append(layer_features)
                labels.append(label)

                if test_mode:
                    break

            labels = torch.cat(labels, 0)

            return embedding_store, labels

        train_embedding_store, train_labels = _embed_dataset(val_train_dataloader)
        test_embedding_store, test_labels = _embed_dataset(val_test_dataloader)

        classifier = MultiDepthClassifier(
            num_feature_depth, num_classes, num_features
        ).to(device)

        optim = torch.optim.AdamW(
            classifier.parameters(),
            lr=validation_probe_lr,
        )

        for val_epoch in tqdm(
            range(validation_train_epochs), desc="training val classifier"
        ):
            rand_indices = torch.randperm(train_labels.shape[0])

            for indices_batch in rand_indices.split(validation_probe_batch_size):

                emb = train_embedding_store[indices_batch.numpy()]
                lab = train_labels[indices_batch]

                emb = torch.from_numpy(emb).to(device, torch.float32)
                lab = lab.to(device)

                preds, loss = classifier(emb, lab)

                loss.backward()
                optim.step()
                optim.zero_grad()

                if test_mode:
                    break

            if test_mode:
                break

        preds = []
        for indices_batch in tqdm(
            torch.arange(test_labels.shape[0]).split(validation_probe_batch_size),
            desc="testing probe",
        ):
            emb = test_embedding_store[indices_batch.numpy()]
            lab = test_labels[indices_batch]

            emb = torch.from_numpy(emb).to(device, torch.float32)
            lab = lab.to(device)

            with torch.inference_mode():
                pred, loss = classifier(emb, lab)

            preds.append(pred.cpu())

        preds = torch.cat(preds, 0)
        correct_labels = einx.equal("b n, b", preds, test_labels)
        accuracies = einx.mean("[b] n", correct_labels.float())

    return accuracies
