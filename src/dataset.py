import os
import time
import math
from dataclasses import dataclass
import random

import torch
import webdataset as wds
import tensorset as ts
import PIL.Image
import torchvision

from src.packer import PairPacker

MASK_SAMPLE_ID = -100
MAX_SAMPLE_ID = torch.iinfo(torch.long).max
MIN_SAMPLE_ID = torch.iinfo(torch.long).min


def identity(x):
    return x


def get_tensorset_collation_fn(mode="cat"):
    def tensorset_collation_fn(samples, combine_tensors=True, combine_scalars=True):
        batched = list(zip(*samples))
        result = []
        for b in batched:
            if isinstance(b[0], ts.TensorSet):
                if mode == "cat":
                    b = ts.cat(b, 0)
                elif mode == "stack":
                    b = ts.stack(b, 0)
                else:
                    raise ValueError(mode)
            elif isinstance(b[0], list):
                # list summation
                b = [x for y in b for x in y]
            else:
                b = list(b)
            result.append(b)
        return result

    return tensorset_collation_fn


def _unbatched_tensorset(data):
    for sample in data:
        assert isinstance(sample, (tuple, list))

        for x in sample:
            assert isinstance(x, ts.TensorSet)

        b = sample[0].size(0)

        for i in range(b):
            yield tuple(x.iloc[i] for x in sample)


unbatched_tensorset = wds.pipelinefilter(_unbatched_tensorset)


def _get_image_dataset(
    dataset_pattern: str = "",
    is_training: bool = True,
    seed: int | None = None,
    shuffle_size_samples: int = 1000,
    image_column_name: str = "jpg",
    label_column_name: str | None = None,
    dataset_length: int | None = None,
):
    # Setup rng
    # Prefer not to use seeding
    shard_shuffle_seed = None
    shuffle_rng = None
    if seed is not None:
        rng = random.Random(seed)
        shard_shuffle_seed = rng.randint(0, 2**30)
        shuffle_rng = random.Random(rng.randbytes(16))

    if is_training:
        nodesplitter = identity
        workersplitter = identity
    else:
        nodesplitter = wds.shardlists.split_by_node
        workersplitter = wds.shardlists.split_by_worker

    dataset = wds.WebDataset(
        urls=dataset_pattern,
        resampled=is_training,
        detshuffle=seed is not None,
        seed=shard_shuffle_seed,
        nodesplitter=nodesplitter,
        workersplitter=workersplitter,
        empty_check=False,
    )

    if is_training:
        dataset = (
            dataset.repeat()
            .with_epoch(dataset_length if dataset_length is not None else -1)
            .shuffle(
                size=shuffle_size_samples, initial=shuffle_size_samples, rng=shuffle_rng
            )
        )

    dataset = dataset.decode("pil", handler=wds.handlers.warn_and_continue).rename(
        pixel_values=image_column_name
    )

    if label_column_name is not None:
        dataset = dataset.rename(labels=label_column_name)

    return dataset


def get_pil_center_crop_box(h, w):
    """
    returns a tuple of (int,int,int,int)
    indicating (start_width_idx, start_height_idx, end_width_idx, end_height_idx)
    """
    crop_size = min(h, w)
    if h > w:
        amount_to_crop = h - crop_size
        box = (0, amount_to_crop // 2, w, amount_to_crop // 2 + crop_size)
    else:
        amount_to_crop = w - crop_size
        box = (amount_to_crop // 2, 0, amount_to_crop // 2 + crop_size, h)
    return box


class TorchImageResizer:
    """
    Same behavior as PILImageResizer
    """

    def __init__(self, size=256):
        self.size = size

    def __call__(self, x):
        *_, h, w = x.shape
        pil_box = get_pil_center_crop_box(h, w)
        # Convert pil-like (w,h) into torch-like (h,w)
        box = (pil_box[1], pil_box[0], pil_box[3], pil_box[2])
        x = x[..., box[0] : box[2], box[1] : box[3]]

        x = torchvision.transforms.Resize((self.size, self.size))(x)

        return x


def pil_to_tensor(x):
    w, h = x.size
    x = x.tobytes()
    x = torch.frombuffer(x, dtype=torch.uint8).reshape(h, w, 3)
    return x


class PILImageResizer:
    """
    square crops to min(h,w)
    and then resizes to the target size using bilinear sampling
    """

    def __init__(self, size=256):
        self.size = size

    def __call__(self, row):
        x = row.pop("pixel_values")
        og_w, og_h = x.size

        box = get_pil_center_crop_box(og_h, og_w)

        x = x.convert("RGB").resize(
            size=(self.size, self.size),
            box=box,
            resample=PIL.Image.Resampling.BILINEAR,
        )
        x = pil_to_tensor(x)
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

        if num_register_tokens > 0:
            padding = torch.zeros(
                num_register_tokens, d, dtype=x.dtype, device=x.device
            )
            x = torch.cat((padding, x))

        padding = torch.zeros(
            num_register_tokens,
            nd,
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        position_ids = torch.cat((padding, position_ids))

        register_ids = torch.full((num_register_tokens + s,), MASK_SAMPLE_ID)

        if num_register_tokens > 0:
            register_ids[:num_register_tokens] = torch.arange(
                num_register_tokens,
                dtype=position_ids.dtype,
                device=position_ids.device,
            )

        token_ids = torch.cat((register_ids.unsqueeze(-1), position_ids), 1)

        row[self.token_column_name] = x
        row[self.position_id_column_name] = token_ids

        return row


class DummySampleIDAdder:
    def __call__(self, row):
        token_ids = row.pop("position_ids")

        s, _ = token_ids.shape

        sequence_ids = torch.zeros(s, 1, dtype=token_ids.dtype, device=token_ids.device)

        token_ids = torch.cat((sequence_ids, token_ids), 1)

        row["token_ids"] = token_ids

        return row


def get_simple_dataloader(
    dataset_pattern: str = "",
    is_training: bool = True,
    seed: int | None = None,
    shuffle_size_samples: int = 1000,
    image_column_name: str = "jpg",
    label_column_name: str | None = None,
    batch_size: int | None = 256,
    image_size: int = 256,
    patch_size: int = 16,
    num_register_tokens: int = 8,
    num_workers: int = 0,
):
    dataset = (
        _get_image_dataset(
            dataset_pattern=dataset_pattern,
            is_training=is_training,
            seed=seed,
            shuffle_size_samples=shuffle_size_samples,
            image_column_name=image_column_name,
            label_column_name=label_column_name,
        )
        .map(PILImageResizer(image_size))
        .map(ImagePatcher(patch_size))
        .map(TokenFlattener())
        .map(RegisterTokenAdder(num_register_tokens))
        .map(DummySampleIDAdder())
    )

    if batch_size is not None:
        dataset = dataset.batched(batch_size)

    shuffle_rng = None
    if seed is not None:
        shuffle_rng = random.Random(seed)

    dataloader = (
        wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)
        .unbatched()
        .shuffle(
            size=shuffle_size_samples, initial=shuffle_size_samples, rng=shuffle_rng
        )
        .batched(batch_size)
    )

    return dataloader


class ImageSizeFilter:
    def __init__(self, min_side_length=64):
        self.min_side_length = min_side_length

    def __call__(self, row):
        w, h = row["pixel_values"].size

        side_length = int((h + w) / 2)
        return side_length >= self.min_side_length


class RandomImageResizer:
    def __init__(
        self,
        max_side_length=256,
        min_side_length=64,
        multiple_of=32,
        min_mult_of_factor=2,
        max_num_pixels=256**2,
        resample_mode=PIL.Image.Resampling.BILINEAR,
        rng=None,
    ):
        self.max_side_length = max_side_length
        self.min_side_length = min_side_length
        self.multiple_of = multiple_of
        self.min_mult_of_factor = min_mult_of_factor
        self.max_num_pixels = max_num_pixels
        self.resample_mode = resample_mode

        if rng is None:
            rng = random.Random()
        self.rng = rng

    def __call__(self, row):
        min_side_length = self.min_side_length
        max_side_length = self.max_side_length
        multiple_of = self.multiple_of
        min_mult_of_factor = self.min_mult_of_factor

        x = row.pop("pixel_values")

        input_w, input_h = x.size
        input_side_length = (input_h + input_w) / 2

        if input_side_length < min_side_length:
            raise ValueError(
                f"Will not resize image with side length {input_side_length}, which is less than minimum \
                side length, {min_side_length}!"
            )

        max_side_length = min(int(round(input_side_length)), max_side_length)

        # Sample a random side length to obtain a scaling factor
        if min_side_length == max_side_length:
            sampled_side_length = min_side_length
        else:
            sampled_side_length = self.rng.randint(min_side_length, max_side_length)

        scale_factor = sampled_side_length / input_side_length
        new_image_size = (input_w * scale_factor, input_h * scale_factor)

        # Ensure that the new size doesnt exceed max_num_pixels
        if math.prod(new_image_size) > self.max_num_pixels:
            # max_num_pixels = (h * scale * w * scale)
            # scale = (max_num_pixels / (h * w)) ** 0.5
            scale = (self.max_num_pixels / math.prod(new_image_size)) ** 0.5
            new_image_size = tuple(int(size * scale) for size in new_image_size)

        # Ensure that the new size is a divisible by multiple of
        new_image_size = tuple(
            int(size // multiple_of) * multiple_of for size in new_image_size
        )
        new_image_size = tuple(max(size, multiple_of) for size in new_image_size)

        # Ensure that there are enough multiple_ofs in the new size
        factor = min(size // multiple_of for size in new_image_size)
        while factor < min_mult_of_factor:
            new_image_size = tuple(size + multiple_of for size in new_image_size)
            factor = min(size // multiple_of for size in new_image_size)

        x = x.convert("RGB").resize(new_image_size, resample=self.resample_mode)

        x = pil_to_tensor(x)

        row["pixel_values"] = x

        return row


class ContextTargetSplitter:
    def __init__(
        self,
        window_size: int = 2,
        min_context_capacity: float = 0.05,
        max_context_capacity: float = 0.95,
        max_context_sequence_length: int = 128,
        rng=None,
    ):
        self.window_size = window_size
        self.min_context_capacity = min_context_capacity
        self.max_context_capacity = max_context_capacity
        self.max_context_sequence_length = max_context_sequence_length

        if rng is None:
            rng = random.Random()
        self.rng = rng

        self.torch_rng = torch.Generator().manual_seed(rng.randint(0, 2**30))

    def __call__(self, row):
        window_size = self.window_size
        min_context_capacity = self.min_context_capacity
        max_context_capacity = self.max_context_capacity
        max_context_sequence_length = self.max_context_sequence_length

        x = row.pop("pixel_values")
        position_ids = row.pop("position_ids")

        nph, npw, d = x.shape
        nph, npw, nd = position_ids.shape

        nwh, nww = nph // window_size, npw // window_size

        num_total_windows = nwh * nww

        tokens_per_window = window_size**2

        min_num_context_windows = int(round(num_total_windows * min_context_capacity))
        min_num_context_windows = max(min_num_context_windows, 1)
        max_num_context_windows = int(round(num_total_windows * max_context_capacity))
        max_num_context_windows = min(max_num_context_windows, num_total_windows - 1)

        # TODO this is an attempt to repro good run, where the num of context tokens
        # is capped and thus reduced on large sequence lengths. So for smaller length
        # inputs, the capacity is from 0.05 to 0.95. But for inputs exceeding 128,
        # assuming the maximum total sequence length is 256, the maximum capacity becomes 0.5.
        absolute_max_num_context_windows = (
            max_context_sequence_length // tokens_per_window
        )
        max_num_context_windows = min(
            max_num_context_windows, absolute_max_num_context_windows
        )

        # Sample a number of context windows
        if num_total_windows == 2:
            num_context_windows = 1
        else:
            num_context_windows = self.rng.randint(
                min_num_context_windows, max_num_context_windows
            )

        # Create a flat idx into x, arrange it into square windows,
        # and randomly permute it
        idx = torch.arange(nph * npw)
        idx = idx.reshape(nph, npw)
        # (nwh wh) (nww ww) -> nwh nww wh ww
        idx = patch(idx.unsqueeze(-1), window_size).squeeze(-1)
        # nwh nww wh ww -> (nwh nww) (wh ww)
        idx = idx.reshape(-1, tokens_per_window)
        shuffle_idx = torch.randperm(num_total_windows, generator=self.torch_rng)
        idx = idx[shuffle_idx]
        # n (wh ww) -> (n wh ww)
        idx = idx.reshape(-1)

        num_context_tokens = num_context_windows * tokens_per_window

        # Permute windows of x, and assign them to context or target

        # nph npw d -> (nph npw) d
        x = x.reshape(-1, d)
        x = x[idx]
        x, y = x[:num_context_tokens], x[num_context_tokens:]

        # nph npw nd -> (nph npw) nd
        position_ids = position_ids.reshape(-1, nd)
        position_ids = position_ids[idx]
        x_position_ids, y_position_ids = (
            position_ids[:num_context_tokens],
            position_ids[num_context_tokens:],
        )

        row["x_patches"] = x
        row["x_position_ids"] = x_position_ids
        row["y_patches"] = y
        row["y_position_ids"] = y_position_ids

        return row


def _get_random_sample_id(rng):
    sample_id = MASK_SAMPLE_ID
    while sample_id == MASK_SAMPLE_ID:
        sample_id = rng.randint(MIN_SAMPLE_ID, MAX_SAMPLE_ID)
    return sample_id


def _assign_sample_ids(data, rng=None):
    if rng is None:
        rng = random.Random(int((os.getpid() + time.time()) * 1e9))

    for sample in data:
        sample_id = _get_random_sample_id(rng)

        sample["sample_id"] = sample_id
        yield sample


assign_sample_ids = wds.pipelinefilter(_assign_sample_ids)


class AddSampleIdsToTensorSet:
    def __init__(self, k):
        self.k = k

    def __call__(self, row):
        x = row.pop(self.k)
        sample_id = row["sample_id"]

        x_length = x.size(0)
        ids = torch.full(
            size=(x_length,),
            fill_value=sample_id,
            dtype=torch.long,
            device=x.all_columns[0].device,
        )
        x.named_columns["sequence_ids"] = ids

        row[self.k] = x
        return row


def _verify_patches(sample, name="patches"):
    assert name in sample
    patches = sample.get(name)
    assert isinstance(patches, ts.TensorSet)


def _packed_x_y(
    data,
    pack_size_x=256,
    pack_size_y=256,
    batch_size=16,
    pad_value_dict=dict(),
):
    """
    Uses a PairPacker to pack x_patches and y_patches into batches,
    also saving related sample metadata.

    yields a dict with 'packed_batch' which is a TensorSet and 'packed_metadata'
     which is a list of dicts, one for each batch element, mapping ids to sample metadata.
    """
    packer = PairPacker(
        pack_size_x=pack_size_x,
        pack_size_y=pack_size_y,
        batch_size=batch_size,
        pad_value_dict=pad_value_dict,
    )
    for sample in data:
        _verify_patches(sample, "x_patches")
        _verify_patches(sample, "y_patches")
        x_patches, y_patches = sample.pop("x_patches"), sample.pop("y_patches")
        sample_id = sample.pop("sample_id")

        # Yields once or zero times
        for packed_batch, packed_metadata in packer.append(
            x_patches, y_patches, sample_id, sample
        ):
            yield {"packed_batch": packed_batch, "packed_metadata": packed_metadata}


packed_x_y = wds.pipelinefilter(_packed_x_y)


class PatchesToTensorSet:
    def __call__(self, row):
        x = row.pop("x_patches")
        x_position_ids = row.pop("x_position_ids")
        row["x_patches"] = ts.TensorSet(patches=x, position_ids=x_position_ids)
        y = row.pop("y_patches")
        y_position_ids = row.pop("y_position_ids")
        row["y_patches"] = ts.TensorSet(patches=y, position_ids=y_position_ids)
        return row


@dataclass
class ContextTargetDatasetConfig:
    num_register_tokens: int = 0
    patch_size: int = 16
    packer_batch_size: int = 64
    max_side_length: int = 256
    min_side_length: int = 64
    mask_window_size: int = 2
    min_context_capacity: float = 0.25
    max_context_capacity: float = 0.5
    absolute_max_context_capacity: float = 0.5

    def __post_init__(self):
        assert self.absolute_max_context_capacity <= self.max_context_capacity
        assert self.min_context_capacity <= self.max_context_capacity
        assert self.max_side_length >= self.min_side_length

        # Resize images to this multiple
        self.resize_multiple_of = self.patch_size * self.mask_window_size

        max_num_patches = (self.max_side_length // self.patch_size) ** 2
        # Absolute maximum number of pixels for resized images
        self.max_num_pixels = max_num_patches * self.patch_size**2

        # For an input of max_sequence_length,
        # the context recieves a random sequence length between
        # min_context_sequence_length and max_context_sequence_length.
        # This means that there are a maximum of (max_sequence_length - min_context_sequence_length) tokens
        # that are exclusive to the target.
        self.max_num_context_patches = int(
            round(max_num_patches * self.absolute_max_context_capacity)
        )

        min_context_windows = int(
            (max_num_patches // self.mask_window_size**2) * self.min_context_capacity
        )
        min_num_context_patches = min_context_windows * self.mask_window_size**2

        self.max_num_target_patches = max_num_patches - min_num_context_patches

        # Register tokens are added to the context after being patched;
        # the context sequence length grows by num_register_tokens just before being packed
        # This is the final context sequence length of batches returned by the dataloader
        self.packer_context_sequence_length = (
            self.max_num_context_patches + self.num_register_tokens
        )


def get_context_target_dataloader(
    config: ContextTargetDatasetConfig = ContextTargetDatasetConfig(),
    dataset_pattern: str = "",
    dataset_length: int | None = None,
    seed: int | None = None,
    shuffle_size_samples: int = 1000,
    image_column_name: str = "jpg",
    batch_size: int = 256,
    num_workers: int = 0,
    num_repeat_samples: int | None = None,
    is_training: bool = True,
):
    """
    Loads a webdataset containing image files,
    Randomly resizes, patches, and splits patches into context and target.
    Then packs context-target pairs, returning packed batches

    dataset_pattern: str
        This is the pattern for the tarfiles of the webdataset
    dataset_length: int | None
        Number of total samples of all the tarfiles
    seed: int | None
        Optional seed
    shuffle_size_samples: int
        Number of samples to shuffle before packing, and number of batches to shuffle
        between workers
    image_column_name: str
        Name of the image column in the webdataset
    batch_size: int
        Batch size
    num_workers: int
        Num dataloader workers
    num_repeat_samples: int | None
        Optional, if an integer then individual samples are repeated this
        number of times
    is_training: bool
        If true, then shards are sampled with replacement and shuffling is enabled
    """

    assert batch_size % config.packer_batch_size == 0

    print(
        f"Creating context target dataset: packer_context_sequence_length: {config.packer_context_sequence_length} \
          teacher sequence length {config.packer_context_sequence_length + config.max_num_target_patches}"
    )

    tensorset_pad_value_dict = {
        "position_ids": 0,
        "patches": 0,
        "sequence_ids": MASK_SAMPLE_ID,
    }

    if seed is not None:
        print(
            "Warning! Seeding should only be used for testing purposes and not for pretraining!"
        )
        rng = random.Random()
        rng.seed(seed)
        resizer_rng = random.Random(rng.randbytes(16))
        splitter_rng = random.Random(rng.randbytes(16))
        shuffle_rng = random.Random(rng.randbytes(16))
    else:
        rng = None
        resizer_rng = None
        splitter_rng = None
        shuffle_rng = None

    dataset = (
        _get_image_dataset(
            dataset_pattern=dataset_pattern,
            is_training=is_training,
            dataset_length=dataset_length,
            seed=seed,
            shuffle_size_samples=shuffle_size_samples,
            image_column_name=image_column_name,
        )
        .select(ImageSizeFilter(config.min_side_length))
        # Assign a unique sample id to each row
        .compose(assign_sample_ids())
    )

    if num_repeat_samples is not None:
        dataset = dataset.compose(repeat_samples(num_repeat_samples))

    dataset = (
        dataset.map(
            RandomImageResizer(
                max_side_length=config.max_side_length,
                min_side_length=config.min_side_length,
                multiple_of=config.resize_multiple_of,
                max_num_pixels=config.max_num_pixels,
                rng=resizer_rng,
            )
        )
        .map(ImagePatcher(config.patch_size))
        .map(
            ContextTargetSplitter(
                window_size=config.mask_window_size,
                min_context_capacity=config.min_context_capacity,
                max_context_capacity=config.max_context_capacity,
                max_context_sequence_length=config.max_num_context_patches,
                rng=splitter_rng,
            )
        )
        # Add register tokens only to the context
        .map(
            RegisterTokenAdder(
                num_register_tokens=config.num_register_tokens,
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
        .map(PatchesToTensorSet())
        # Add in the sample ids to the tensorset
        .map(AddSampleIdsToTensorSet("x_patches"))
        .map(AddSampleIdsToTensorSet("y_patches"))
        # Drop all columns except for these
        .rename(
            x_patches="x_patches",
            y_patches="y_patches",
            sample_id="sample_id",
            keep=False,
        )
        # Pack samples into micro batches
        .compose(
            packed_x_y(
                config.packer_context_sequence_length,
                config.max_num_target_patches,
                config.packer_batch_size,
                pad_value_dict=tensorset_pad_value_dict,
            )
        )
        .to_tuple("packed_batch")
        .batched(
            batch_size // config.packer_batch_size,
            collation_fn=get_tensorset_collation_fn("cat"),
        )
    )

    dataloader = (
        wds.WebLoader(dataset, batch_size=None, num_workers=num_workers, in_order=False)
        .compose(unbatched_tensorset())
        # Shuffle samples from different worker shards if training
        .shuffle(
            size=shuffle_size_samples if is_training else 0,
            initial=shuffle_size_samples,
            rng=shuffle_rng,
        )
        .batched(batch_size, collation_fn=get_tensorset_collation_fn("stack"))
    )

    return dataloader


def _repeat_samples(data, amount):
    for sample in data:
        for _ in range(amount):
            # Create a copy for each yield
            # TODO
            # I think will fail if any of the sample values
            # are edited in place by pipeline functions after this repeat
            sample_copy = dict(sample)
            yield sample_copy


repeat_samples = wds.pipelinefilter(_repeat_samples)


def get_lidar_data(
    config: ContextTargetDatasetConfig = ContextTargetDatasetConfig(),
    dataset_pattern: str = "",
    seed: int | None = None,
    image_column_name: str = "jpg",
    num_batches: int = 4,
    batch_size: int = 256,
    num_unique_images: int = 1000,
    num_repeat_samples: int = 50,
):
    """
    similar to get_context_target_dataset,
    but repeats samples a number of times ~50

    each of the repeated samples is randomly masking using the
    ContextTargetSplitter.
    """

    dataloader = get_context_target_dataloader(
        config=config,
        dataset_pattern=dataset_pattern,
        dataset_length=None,
        image_column_name=image_column_name,
        batch_size=batch_size,
        num_workers=0,
        num_repeat_samples=num_repeat_samples,
        is_training=False,
    )

    batches = []
    dataloader_iter = iter(dataloader)
    for _ in range(num_batches):
        batches.append(next(dataloader_iter))

    import bpdb

    bpdb.set_trace()
