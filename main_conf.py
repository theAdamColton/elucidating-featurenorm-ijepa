from typing import Literal
from dataclasses import dataclass, field

import torch


from src.model import IJEPAConfig
from src.dataset import ContextTargetDatasetConfig


@dataclass
class MainConfig:
    should_compile: bool = False
    dtype: str = "bfloat16"
    device: str = "cuda"
    batch_size: int = 256
    num_workers: int = 0
    seed: int | None = None

    # lr starts at start_lr, and is warmed up to steady_lr for num_lr_warmup_steps
    # Then it remains at steady_lr for num_lr_steady_steps,
    # after which it is cooled down to lr_end for num_lr_cooldown_steps
    start_lr: float = 1e-4
    num_lr_warmup_steps: int = 10000
    steady_lr: float = 5e-4
    num_lr_steady_steps: int = 250000
    num_lr_cooldown_steps: int = 50000
    end_lr: float = 1e-5

    num_epochs: int = 800

    patch_size: int = 16

    log_every_num_steps: int = 50

    log_lidar_every_num_steps: int = 1000
    lidar_num_unique_samples: int = 1000
    lidar_num_augmentations: int = 50

    validate_every_num_epochs: int = 50
    max_num_save_checkpoints: int = 2

    context_target_dataset: ContextTargetDatasetConfig = field(
        default_factory=lambda: ContextTargetDatasetConfig()
    )

    validation_probe_lr: float = 1e-3
    validation_monocular_depth_lr: float = 5e-4
    validation_image_size: int = 256
    validation_train_epochs: int = 50
    validation_monocular_depth_train_epochs: int = 10
    # Extract features from the 4th to last layer of the encoder to perform monocular depth estimation
    validation_monocular_depth_feature_depth: int = -4
    validation_probe_batch_size: int = 2048
    validation_extraction_mode: Literal["extract-layers", "lastlayer"] = (
        "extract-layers"
    )

    resume_checkpoint_path: str | None = None

    test_mode: bool = False

    num_image_channels: int = 3

    ema_beta_start: float = 0.8
    ema_beta_warmup_steps: int = 1000
    ema_beta_steady: float = 0.996
    ema_beta_steady_steps: int = 300000
    ema_beta_end: float = 0.9999

    # Webdataset tars
    train_dataset_pattern: str = (
        "datasets/imagenet1k-256-wds/imagenet1k-train-{0000..1023}.tar"
    )
    val_dataset_pattern: str = (
        "datasets/imagenet1k-256-wds/imagenet1k-validation-{00..63}.tar"
    )
    train_dataset_length: int = 1281167
    image_column_name: str = "jpg"
    label_column_name: str = "cls"
    num_classes: int = 1000

    # Webdataset tars
    monocular_depth_train_dataset_pattern: str = (
        "datasets/nyu-depthv2-wds/nyu-depth-train-{00000..47}.tar"
    )
    monocular_depth_train_dataset_length: int = 47584
    monocular_depth_val_dataset_pattern: str = (
        "datasets/nyu-depthv2-wds/nyu-depth-val-00000.tar"
    )
    depth_column_name: str = "depth.npy"

    num_register_tokens: int = 0

    model: IJEPAConfig = field(default_factory=lambda: IJEPAConfig())

    mode: Literal[
        "make-viz",
        "train",
        "validate",
        "visualize-embeddings",
        "plot-sample-losses",
        "validate-monocular-depth",
    ] = "train"

    def __post_init__(self):
        assert (
            self.model.encoder.input_size
            == self.num_image_channels * self.patch_size**2
        )

        assert self.batch_size % self.context_target_dataset.packer_batch_size == 0
        assert self.context_target_dataset.packer_batch_size <= self.batch_size
        assert self.context_target_dataset.patch_size == self.patch_size
        assert (
            self.num_register_tokens == self.context_target_dataset.num_register_tokens
        )

        assert self.start_lr <= self.steady_lr
        assert self.steady_lr >= self.end_lr
        assert self.num_lr_warmup_steps >= 0
        assert self.num_lr_steady_steps >= 0
        assert self.num_lr_cooldown_steps >= 0

        self.torch_device = torch.device(self.device)
        self.torch_dtype = getattr(torch, self.dtype)
