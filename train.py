import einx
import torchvision
import random
import torch
from dataclasses import dataclass, field
import wandb
import cabbage_patch

from src.model import IJEPADepthSmartConfig, IJEPADepthSmart


@dataclass
class ContextTargetPatcherConfig:
    patch_size: int = 16
    window_size: int = 4

    max_side_length: int = 256
    min_side_length: int = 64

    def __post_init__(self):
        assert self.max_side_length % self.patch_size == 0
        assert self.min_side_length % self.patch_size == 0


class ContextTargetPatcher:
    def __init__(self, config=ContextTargetPatcherConfig()):
        self.config = config

    def __call__(self, x):
        """
        x: pixel values of shape (c h w)
        """
        config = self.config

        c, input_h, input_w = x.shape

        input_side_length = (input_h + input_w) / 2
        max_side_length = min(config.max_side_length, input_side_length)

        sampled_side_length = random.randint(config.min_side_length, max_side_length)

        scale_factor = sampled_side_length / input_side_length

        image_crop_size = (input_h * scale_factor, input_w * scale_factor)
        image_crop_size = (
            (x // config.patch_size) * config.patch_size for x in image_crop_size
        )

        x = torchvision.transforms.Resize(image_crop_size)(x)

        x = einx.rearrange(
            "... c (np ps)... -> ... np... (ps... c)", x, ps=self.patch_size
        )
        x = einx.rearrange(
            "... (nw ws)... d -> ... (nw... ws...) d", x, ws=self.window_size
        )


@dataclass
class TrainConfig:
    should_compile: bool = False
    dtype: str = "bfloat16"
    device: str = "cuda"
    batch_size: int = 8
    num_workers: int = 0

    # Webdataset tars
    dataset_pattern: str = "/nvme/imagenet1k/imagenet1k-train-{0000..1023}.tar"
    image_column_name: str = "jpg"

    patcher: ContextTargetPatcherConfig = field(
        default_factory=lambda: ContextTargetPatcherConfig()
    )

    model: IJEPADepthSmartConfig = field(
        default_factory=lambda: IJEPADepthSmartConfig()
    )


def train(conf: TrainConfig = TrainConfig()):
    input_size = conf.model.encoder.input_size

    dataset = cabbage_patch.CabbageDataset(conf.dataset_pattern)
    dataset = (
        dataset.decode("torchrgb8")
        .rename(pixel_values=conf.image_column_name)
        .map(ContextTargetPatcher(conf.patcher))
    )

    sample = next(iter(dataset))
    import bpdb

    bpdb.set_trace()
