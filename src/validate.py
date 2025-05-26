from typing import Literal
from glob import glob
import uuid
import tempfile
from contextlib import contextmanager

import torch
import einx
import webdataset as wds
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import tensorset as ts

from src.dataset import get_simple_dataloader
from src.model import IJEPAModel


class MultiDepthClassifier(nn.Module):
    def __init__(self, num_depth, num_classes, num_features):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(num_depth, num_classes, num_features))
        nn.init.uniform_(
            self.weights, -1 / (num_features) ** 0.5, 1 / (num_features) ** 0.5
        )
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


def _batch_embeddings(data, batch_size=2048):
    collated_batches = []
    for sample in data:
        embeddings, labels, *_ = sample

        embeddings = torch.from_numpy(embeddings)
        labels = torch.from_numpy(labels)

        new_batch = ts.TensorSet(embeddings, labels)

        collated_batches.append(new_batch)

        collated_batch_size = sum(x.size(0) for x in collated_batches)

        if collated_batch_size >= batch_size:
            batch = collated_batches[:-1]
            remainder = batch_size - collated_batch_size
            batch.append(collated_batches[-1].iloc[:remainder])
            batch = ts.cat(batch, 0)
            batch = batch.columns
            collated_batches = [collated_batches[-1].iloc[remainder:]]
            yield batch

    if len(collated_batches) > 0:
        batch = ts.cat(collated_batches, 0)
        batch = batch.columns
        yield batch


batch_embeddings = wds.pipelinefilter(_batch_embeddings)


def validate(
    model: IJEPAModel,
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
    validation_extraction_mode: Literal[
        "extract-layers", "lastlayer"
    ] = "extract-layers",
    num_register_tokens: int = 0,
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

    # Don't need to shuffle the train dataloader,
    # because the embeddings will be shuffled after
    # the dataset is embedded
    val_train_dataloader = get_simple_dataloader(
        dataset_pattern=train_dataset_pattern,
        is_training=False,
        image_column_name=image_column_name,
        label_column_name=label_column_name,
        batch_size=batch_size,
        image_size=validation_image_size,
        num_register_tokens=num_register_tokens,
        patch_size=patch_size,
        num_workers=num_workers,
    )

    val_test_dataloader = get_simple_dataloader(
        dataset_pattern=val_dataset_pattern,
        is_training=False,
        image_column_name=image_column_name,
        label_column_name=label_column_name,
        batch_size=batch_size,
        image_size=validation_image_size,
        num_register_tokens=num_register_tokens,
        patch_size=patch_size,
        num_workers=num_workers,
    )

    with tempfile.TemporaryDirectory(dir=".") as tmpdir:

        def _embed_dataset(dl, prefix="train-"):
            max_size_bytes = 1e9
            with wds.ShardWriter(
                f"{tmpdir}/{prefix}%04d.tar", maxsize=max_size_bytes
            ) as writer:
                pass

                for batch in tqdm(dl, desc="embedding val dataset"):
                    pixel_values = batch["pixel_values"]
                    labels = batch["labels"]
                    token_ids = batch["token_ids"]

                    b, s, d = pixel_values.shape

                    pixel_values = pixel_values.to(device=device, dtype=dtype)
                    pixel_values = (pixel_values / 255) * 2 - 1

                    token_ids = token_ids.to(device)

                    if validation_extraction_mode == "extract-layers":
                        with autocast_fn():
                            with torch.inference_mode():
                                _, layer_features = encoder(
                                    pixel_values,
                                    token_ids,
                                    return_all_layer_features=True,
                                )

                        layer_features = einx.mean("n b s d -> b n d", layer_features)

                    elif validation_extraction_mode == "lastlayer":
                        with autocast_fn():
                            with torch.inference_mode():
                                layer_features, *_ = encoder(
                                    pixel_values,
                                    token_ids,
                                    return_all_layer_features=True,
                                )

                        layer_features = einx.mean(
                            "b s d -> b one d", layer_features, one=1
                        )

                    layer_features = layer_features.to(torch.float16).cpu().numpy()
                    labels = labels.numpy()

                    # Write an entire batch to the tar file
                    writer.write(
                        {
                            "__key__": uuid.uuid4().hex,
                            "label.npy": labels,
                            "features.npy": layer_features,
                        }
                    )

                    if test_mode:
                        break

                urls = list(glob(f"{tmpdir}/{prefix}*.tar"))
                return urls

        train_tar_urls = _embed_dataset(val_train_dataloader, "train-")
        test_tar_urls = _embed_dataset(val_test_dataloader, "test-")

        classifier = MultiDepthClassifier(
            1 if validation_extraction_mode == "lastlayer" else num_feature_depth,
            num_classes,
            num_features,
        ).to(device)

        optim = torch.optim.AdamW(
            classifier.parameters(),
            lr=validation_probe_lr,
        )

        if should_compile:
            classifier = torch.compile(classifier)

        minibatch_shuffle_size = 16
        sample_shuffle_size = 1000

        train_dataset = (
            wds.WebDataset(
                train_tar_urls,
                empty_check=False,
                shardshuffle=100,
            )
            .shuffle(size=minibatch_shuffle_size, initial=minibatch_shuffle_size)
            .decode()
            .to_tuple("features.npy", "label.npy")
            .compose(batch_embeddings(validation_probe_batch_size))
        )
        train_dataloader = (
            wds.WebLoader(
                train_dataset,
                num_workers=num_workers,
                batch_size=None,
            )
            .unbatched()
            .shuffle(size=sample_shuffle_size, initial=sample_shuffle_size)
            .batched(validation_probe_batch_size)
        )

        for val_epoch in tqdm(
            range(validation_train_epochs), desc="training val classifier"
        ):
            for batch in train_dataloader:
                emb, lab, *_ = batch

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
            .to_tuple("features.npy", "label.npy")
            .compose(batch_embeddings(validation_probe_batch_size))
        )
        test_dataloader = wds.WebLoader(
            test_dataset, num_workers=num_workers, batch_size=None
        )

        preds = []
        labs = []
        for batch in tqdm(
            test_dataloader,
            desc="testing probe",
        ):
            emb, lab, *_ = batch

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

    accuracies = accuracies.tolist()

    return accuracies
