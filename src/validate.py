import random
from typing import Literal
from glob import glob
import uuid
import tempfile
import torch
import einx
from contextlib import contextmanager
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F

from src.dataset import get_test_dataset
from src.model import IJEPADepthSmart


class MultiDepthClassifier(nn.Module):
    def __init__(self, num_depth, num_classes, num_features):
        super().__init__()
        self.mod = nn.Linear(num_features, num_classes)
        self.weights = nn.Parameter(torch.empty(num_depth, num_classes, num_features))
        nn.init.uniform_(
            self.weights, -1 / (num_features) ** 0.5, 1 / (num_features) ** 0.5
        )
        self.biases = nn.Parameter(torch.zeros(num_depth, num_classes))
        self.num_depth = num_depth

    def forward(self, emb, lab):
        # logits = einx.dot("b n d, n c d -> b n c", emb, self.weights)
        # logits = einx.add("b n c, n c", logits, self.biases)
        logits = self.mod(emb)
        preds = logits.argmax(-1)

        logits = einx.rearrange("b n c -> (b n) c", logits)
        lab = einx.rearrange("b -> (b n)", lab, n=self.num_depth)
        loss = F.cross_entropy(logits, lab)

        return preds, loss


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
    validation_depthsmart_mode: Literal[
        "learned", "extract-layers", "lastlayer"
    ] = "extract-layers",
    num_register_tokens: int = 8,
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

    val_train_dataset = get_test_dataset(
        dataset_pattern=train_dataset_pattern,
        shuffle=True,
        image_column_name=image_column_name,
        label_column_name=label_column_name,
        batch_size=batch_size,
        image_size=validation_image_size,
        num_register_tokens=num_register_tokens,
        patch_size=patch_size,
    )
    val_test_dataset = get_test_dataset(
        dataset_pattern=val_dataset_pattern,
        shuffle=False,
        image_column_name=image_column_name,
        label_column_name=label_column_name,
        batch_size=batch_size,
        image_size=validation_image_size,
        num_register_tokens=num_register_tokens,
        patch_size=patch_size,
    )

    with tempfile.TemporaryDirectory(dir=".") as tmpdir:

        def _embed_dataset(ds, prefix="train-"):
            dl = DataLoader(ds, num_workers=num_workers, batch_size=None)

            with wds.ShardWriter(f"{tmpdir}/{prefix}%04d.tar") as writer:
                pass

                for batch in tqdm(dl, desc="embedding val dataset"):
                    pixel_values = batch["pixel_values"]
                    labels = batch["labels"]
                    token_ids = batch["token_ids"]

                    b, s, d = pixel_values.shape

                    pixel_values = pixel_values.to(device=device, dtype=dtype)
                    pixel_values = (pixel_values / 255) * 2 - 1

                    token_ids = token_ids.to(device)

                    if validation_depthsmart_mode == "extract-layers":
                        # Full depth
                        t = torch.full(
                            (b, s),
                            num_feature_depth - 1,
                            device=device,
                        )

                        with autocast_fn():
                            with torch.inference_mode():
                                _, layer_features = encoder(
                                    pixel_values,
                                    t,
                                    token_ids,
                                    return_all_layer_features=True,
                                )

                        layer_features = einx.mean("n b s d -> b n d", layer_features)

                    elif validation_depthsmart_mode == "lastlayer":
                        # Full depth
                        t = torch.full(
                            (b, s),
                            num_feature_depth - 1,
                            device=device,
                        )

                        with autocast_fn():
                            with torch.inference_mode():
                                layer_features, *_ = encoder(
                                    pixel_values,
                                    t,
                                    token_ids,
                                    return_all_layer_features=True,
                                )

                        layer_features = einx.mean(
                            "b s d -> b one d", layer_features, one=1
                        )

                    elif validation_depthsmart_mode == "learned":
                        # Repeating batch to condition on all depths
                        # results in OOM!
                        #
                        layer_features = []
                        for layer_id in range(num_feature_depth):
                            t = torch.full((b, s), layer_id, device=device)

                            with autocast_fn():
                                with torch.inference_mode():
                                    features, *_ = encoder(pixel_values, t, token_ids)
                            features = einx.mean("b [s] d", features)
                            layer_features.append(features)

                        layer_features = torch.stack(layer_features, 1)

                    layer_features = layer_features.cpu().to(torch.float16).numpy()

                    for i in range(b):
                        writer.write(
                            {
                                "__key__": uuid.uuid4().hex,
                                "label.cls": labels[i].item(),
                                "features.npy": layer_features[i],
                            }
                        )

                    if test_mode:
                        break

                urls = list(glob(f"{tmpdir}/{prefix}*.tar"))
                return urls

        train_tar_urls = _embed_dataset(val_train_dataset, "train-")
        test_tar_urls = _embed_dataset(val_test_dataset, "test-")

        classifier = MultiDepthClassifier(
            1 if validation_depthsmart_mode == "lastlayer" else num_feature_depth,
            num_classes,
            num_features,
        ).to(device)

        optim = torch.optim.AdamW(
            classifier.parameters(),
            lr=validation_probe_lr,
        )

        if should_compile:
            classifier = torch.compile(classifier)

        train_dataset = (
            wds.WebDataset(
                train_tar_urls,
                empty_check=False,
                shardshuffle=100,
                detshuffle=True,
                seed=random.randint(0, 2**30),
            )
            .shuffle(1000)
            .decode()
            .batched(validation_probe_batch_size)
        )
        train_dataloader = (
            wds.WebLoader(
                train_dataset,
                num_workers=num_workers,
                batch_size=None,
            )
            .unbatched()
            .shuffle(1000)
            .batched(validation_probe_batch_size)
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
