import random
import numpy as np
import torch
import webdataset as wds
import tensorset as ts
import PIL.Image

from src.packer import PairPacker

MASK_SEQUENCE_ID = -100


def collation_fn(samples, combine_tensors=True, combine_scalars=True):
    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], ts.TensorSet):
            b = ts.cat(b, 0)
        elif isinstance(b[0], list):
            # list summation
            b = [x for y in b for x in y]
        else:
            b = list(b)
        result.append(b)
    return result


def _get_image_dataset(
    dataset_pattern: str = "",
    shuffle=True,
    seed: int = 42,
    shuffle_size_samples: int = 1000,
    image_column_name: str = "jpg",
    label_column_name: str | None = None,
):
    dataset = wds.WebDataset(
        urls=dataset_pattern,
        shardshuffle=100 if shuffle else None,
        detshuffle=shuffle,
        seed=seed,
        nodesplitter=wds.split_by_node,
    )

    if shuffle:
        dataset = dataset.shuffle(shuffle_size_samples)

    dataset = dataset.decode("pil", handler=wds.handlers.warn_and_continue).rename(
        pixel_values=image_column_name
    )

    if label_column_name is not None:
        dataset = dataset.rename(labels=label_column_name)

    return dataset


class ImageResizer:
    def __init__(self, size=256):
        self.size = size

    def __call__(self, row):
        x = row.pop("pixel_values")
        og_w, og_h = x.size
        crop_size = min(og_h, og_w)
        if og_h > og_w:
            amount_to_crop = og_h - crop_size
            box = (0, amount_to_crop // 2, og_w, amount_to_crop // 2 + crop_size)
        else:
            amount_to_crop = og_w - crop_size
            box = (amount_to_crop // 2, 0, amount_to_crop // 2 + crop_size, og_h)

        x = x.convert("RGB").resize(
            size=(self.size, self.size),
            box=box,
            resample=PIL.Image.Resampling.BICUBIC,
        )
        x = np.array(x)
        x = torch.from_numpy(x)
        row["pixel_values"] = x
        return row


def patch(x, patch_size=8):
    h, w, c = x.shape
    nph, npw = h // patch_size, w // patch_size
    x = x.reshape(nph, patch_size, npw, patch_size, c)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(nph, npw, patch_size, patch_size, c)
    return x


class ImagePatcher:
    def __init__(self, patch_size=16, image_channels=3):
        self.patch_size = patch_size
        self.image_channels = image_channels

    def __call__(self, row):
        x = row.pop("pixel_values")
        # (nph ph) (npw pw) c -> nph npw ph pw c
        x = patch(x, self.patch_size)
        # nph npw ph pw c -> nph npw (ph pw c)
        d = self.patch_size**2 * self.image_channels

        x = x.reshape(x.shape[0], x.shape[1], d)

        position_ids = torch.meshgrid(
            (torch.arange(x.shape[0]), torch.arange(x.shape[1])), indexing="ij"
        )
        position_ids = torch.stack(position_ids, -1)

        row["pixel_values"] = x
        row["position_ids"] = position_ids

        return row


class TokenFlattener:
    def __call__(self, row):
        x = row.pop("pixel_values")
        position_ids = row.pop("position_ids")

        x = x.reshape(-1, x.shape[-1])
        position_ids = position_ids.reshape(-1, position_ids.shape[-1])

        row["pixel_values"] = x
        row["position_ids"] = position_ids

        return row


class RegisterTokenAdder:
    def __init__(
        self,
        num_register_tokens=8,
        token_column_name="pixel_values",
        position_id_column_name="position_ids",
    ):
        self.num_register_tokens = num_register_tokens
        self.token_column_name = token_column_name
        self.position_id_column_name = position_id_column_name

    def __call__(self, row):
        num_register_tokens = self.num_register_tokens

        x = row.pop(self.token_column_name)
        position_ids = row.pop(self.position_id_column_name)

        s, d = x.shape
        s, nd = position_ids.shape

        padding = torch.zeros(num_register_tokens, d, dtype=x.dtype, device=x.device)
        x = torch.cat((padding, x))

        padding = torch.zeros(
            num_register_tokens,
            nd,
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        position_ids = torch.cat((padding, position_ids))

        register_ids = torch.full((num_register_tokens + s,), MASK_SEQUENCE_ID)
        register_ids[:num_register_tokens] = torch.arange(
            num_register_tokens, dtype=position_ids.dtype, device=position_ids.device
        )

        token_ids = torch.cat((register_ids.unsqueeze(-1), position_ids), 1)

        row[self.token_column_name] = x
        row[self.position_id_column_name] = token_ids

        return row


class SequenceIDAdder:
    def __call__(self, row):
        token_ids = row.pop("position_ids")

        s, _ = token_ids.shape

        sequence_ids = torch.zeros(s, 1, dtype=token_ids.dtype, device=token_ids.device)

        token_ids = torch.cat((sequence_ids, token_ids), 1)

        row["token_ids"] = token_ids

        return row


def get_test_dataset(
    dataset_pattern: str = "",
    shuffle: bool = True,
    seed: int = 42,
    shuffle_size_samples: int = 1000,
    image_column_name: str = "jpg",
    label_column_name: str | None = None,
    batch_size: int = 256,
    image_size: int = 256,
    patch_size: int = 16,
    num_register_tokens: int = 8,
    shuffle_size_batches: int = 16,
):
    dataset = (
        _get_image_dataset(
            dataset_pattern=dataset_pattern,
            shuffle=shuffle,
            seed=seed,
            shuffle_size_samples=shuffle_size_samples,
            image_column_name=image_column_name,
            label_column_name=label_column_name,
        )
        .map(ImageResizer(image_size))
        .map(ImagePatcher(patch_size))
        .map(TokenFlattener())
        .map(RegisterTokenAdder(num_register_tokens))
        .map(SequenceIDAdder())
        .batched(batch_size)
    )

    if shuffle:
        dataset = dataset.shuffle(shuffle_size_batches)

    return dataset


class ImageSizeFilter:
    def __init__(self, min_side_length=64):
        self.min_side_length = min_side_length

    def __call__(self, row):
        w, h = row["pixel_values"].size

        side_length = int((h + w) / 2)
        return side_length > self.min_side_length


class RandomImageResizer:
    def __init__(
        self,
        max_side_length=256,
        min_side_length=64,
        multiple_of=32,
        min_mult_of_factor=2,
    ):
        self.max_side_length = max_side_length
        self.min_side_length = min_side_length
        self.multiple_of = multiple_of
        self.min_mult_of_factor = min_mult_of_factor

    def __call__(self, row):
        min_side_length = self.min_side_length
        max_side_length = self.max_side_length
        multiple_of = self.multiple_of
        min_mult_of_factor = self.min_mult_of_factor

        x = row.pop("pixel_values")

        input_w, input_h = x.size
        input_side_length = (input_h + input_w) / 2

        sampled_side_length = random.randint(min_side_length, max_side_length)
        scale_factor = sampled_side_length / input_side_length
        image_crop_size = (input_w * scale_factor, input_h * scale_factor)

        image_crop_size = tuple(
            int(size // multiple_of) * multiple_of for size in image_crop_size
        )
        image_crop_size = tuple(max(size, multiple_of) for size in image_crop_size)

        # Hack to prevent too few windows
        factor = sum(size // multiple_of for size in image_crop_size)
        while factor < min_mult_of_factor:
            image_crop_size = tuple(size + multiple_of for size in image_crop_size)
            factor = sum(size // multiple_of for size in image_crop_size)

        x = x.convert("RGB").resize(
            image_crop_size, resample=PIL.Image.Resampling.BICUBIC
        )
        x = torch.from_numpy(np.array(x))

        row["pixel_values"] = x

        return row


class ContextTargetSplitter:
    def __init__(
        self,
        max_context_sequence_length=128,
        window_size: int = 2,
    ):
        self.max_context_sequence_length = max_context_sequence_length
        self.window_size = window_size

    def __call__(self, row):
        window_size = self.window_size

        x = row.pop("pixel_values")
        position_ids = row.pop("position_ids")

        nph, npw, d = x.shape
        nph, npw, nd = position_ids.shape

        nwh, nww = nph // window_size, npw // window_size

        num_total_windows = nwh * nww

        tokens_per_window = window_size**2

        max_num_context_windows = self.max_context_sequence_length // tokens_per_window
        max_num_context_windows = min(num_total_windows - 1, max_num_context_windows)

        min_num_context_windows = 1

        if num_total_windows == 2:
            num_context_windows = 1
        else:
            num_context_windows = random.randint(
                min_num_context_windows, max_num_context_windows
            )

        idx = torch.arange(nph * npw)
        idx = idx.reshape(nph, npw)
        # (nwh wh) (nww ww) -> nwh nww wh ww
        idx = patch(idx.unsqueeze(-1), window_size).squeeze(-1)
        # nwh nww wh ww -> (nwh nww) (wh ww)
        idx = idx.reshape(-1, tokens_per_window)
        shuffle_idx = torch.randperm(num_total_windows)
        idx = idx[shuffle_idx]
        # n (wh ww) -> (n wh ww)
        idx = idx.reshape(-1)

        context_idx, target_idx = (
            idx[: num_context_windows * tokens_per_window],
            idx[num_context_windows * tokens_per_window :],
        )

        # nph npw d -> (nph npw) d
        x = x.reshape(-1, d)
        x, y = x[context_idx], x[target_idx]

        # nph npw nd -> (nph npw) nd
        position_ids = position_ids.reshape(-1, nd)
        x_position_ids, y_position_ids = (
            position_ids[context_idx],
            position_ids[target_idx],
        )

        row["x_patches"] = x
        row["x_position_ids"] = x_position_ids
        row["y_patches"] = y
        row["y_position_ids"] = y_position_ids

        return row


def verify_patches(sample, name="patches"):
    assert name in sample
    patches = sample.get(name)
    assert isinstance(patches, ts.TensorSet)


def _addin_ids(x: ts.TensorSet, id):
    x_length = x.size(0)
    ids = torch.full(
        size=(x_length,),
        fill_value=id,
        dtype=torch.long,
        device=x.all_columns[0].device,
    )
    x.named_columns["sequence_ids"] = ids


def _packed_x_y(
    data,
    sequence_length_x=256,
    sequence_length_y=256,
    batch_size=16,
    pad_value_dict=dict(),
):
    """
    Packs x,y pairs into two batches
    an x,y sample will have x and y put in the same sequence of each batch
    """
    packer = PairPacker(
        sequence_length_x, sequence_length_y, batch_size, pad_value_dict
    )
    id = 0
    for sample in data:
        verify_patches(sample, "x_patches")
        verify_patches(sample, "y_patches")
        x, y = sample.pop("x_patches"), sample.pop("y_patches")
        _addin_ids(x, id)
        _addin_ids(y, id)
        packer.append(x, y, id, sample)
        if packer.can_pop_batch():
            x, y, metadata = packer.pop_batch()
            yield {"x_patches": x, "y_patches": y, "metadata": metadata}

        id += 1


packed_x_y = wds.pipelinefilter(_packed_x_y)


class ToTensorSet:
    def __call__(self, row):
        x = row.pop("x_patches")
        x_position_ids = row.pop("x_position_ids")
        row["x_patches"] = ts.TensorSet(patches=x, position_ids=x_position_ids)
        y = row.pop("y_patches")
        y_position_ids = row.pop("y_position_ids")
        row["y_patches"] = ts.TensorSet(patches=y, position_ids=y_position_ids)
        return row


def get_context_target_dataset(
    dataset_pattern: str = "",
    seed: int = 42,
    shuffle_size_samples: int = 1000,
    image_column_name: str = "jpg",
    label_column_name: str | None = None,
    batch_size: int = 256,
    packer_batch_size: int = 16,
    shuffle_size_packer: int = 16,
    max_side_length: int = 256,
    min_side_length: int = 64,
    patch_size: int = 16,
    mask_window_size: int = 2,
    num_register_tokens: int = 8,
):

    resize_multiple_of = patch_size * mask_window_size

    max_sequence_length = (max_side_length // patch_size) ** 2
    # Between 1 and half are context tokens
    max_target_sequence_length = max_sequence_length

    max_context_sequence_length = int(round(max_target_sequence_length / 2))
    # Register tokens are added to the context after being patched
    packer_context_sequence_length = max_context_sequence_length + num_register_tokens

    tensorset_pad_value_dict = {
        "position_ids": 0,
        "patches": 0,
        "sequence_ids": MASK_SEQUENCE_ID,
    }

    dataset = (
        _get_image_dataset(
            dataset_pattern=dataset_pattern,
            shuffle=True,
            seed=seed,
            shuffle_size_samples=shuffle_size_samples,
            image_column_name=image_column_name,
            label_column_name=label_column_name,
        )
        .select(ImageSizeFilter(min_side_length))
        .map(
            RandomImageResizer(
                max_side_length=max_side_length,
                min_side_length=min_side_length,
                multiple_of=resize_multiple_of,
            )
        )
        .map(ImagePatcher(patch_size))
        .map(
            ContextTargetSplitter(
                max_context_sequence_length=max_context_sequence_length,
                window_size=mask_window_size,
            )
        )
        # Add register tokens only to the context
        .map(
            RegisterTokenAdder(
                num_register_tokens=num_register_tokens,
                token_column_name="x_patches",
                position_id_column_name="x_position_ids",
            )
        )
        .map(
            RegisterTokenAdder(
                num_register_tokens=0,
                token_column_name="y_patches",
                position_id_column_name="y_position_ids",
            )
        )
        .map(ToTensorSet())
        .compose(
            packed_x_y(
                packer_context_sequence_length,
                max_target_sequence_length,
                packer_batch_size,
                pad_value_dict=tensorset_pad_value_dict,
            )
        )
        .to_tuple("x_patches", "y_patches")
        .shuffle(shuffle_size_packer)
        .batched(
            batch_size // packer_batch_size, collation_fn=collation_fn, partial=False
        )
    )

    return dataset
