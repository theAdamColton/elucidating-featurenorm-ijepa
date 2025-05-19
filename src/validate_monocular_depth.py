import torch
from torch import nn
import torch.nn.functional as F

from src.dataset import get_test_dataset
from src.model import IJEPADepthSmart


class DPTDepthModel(nn.Module):
    def __init__(self, input_feature_size=256, non_negative=True):
        super().__init__()

        self.conv1 = nn.Conv2d(
            input_feature_size,
            input_feature_size // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            input_feature_size // 2, 32, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU()
        self.depth_head = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.act_out = nn.ReLU() if non_negative else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv2(x)
        x = self.relu(x)
        x = self.depth_head(x)
        x = self.act_out(x)

        inv_depth = x
        return inv_depth


def validate_monocular_depth_prediction(
    model: IJEPADepthSmart,
    image_column_name: str = "jpg",
    depth_column_name: str = "depth.npy",
    patch_size: int = 16,
    validation_image_size: int = 256,
    batch_size: int = 256,
    num_workers: int = 4,
    # TODO add all tar files
    train_dataset_pattern: str = "/nvme/nyu-depthv2-wds/nyu-depth-train-00000.tar",
    val_dataset_pattern: str = "/nvme/nyu-depthv2-wds/nyu-depth-val-00000.tar",
    dtype: torch.dtype = torch.bfloat16,
    test_mode: bool = False,
    should_compile: bool = False,
    validation_probe_lr: float = 1e-3,
    validation_probe_batch_size: int = 2048,
    validation_train_epochs: int = 50,
    # Extract features from the 4th to last layer of the encoder
    feature_depth: int = -4,
    num_register_tokens: int = 0,
):
    train_dataset = get_test_dataset(
        train_dataset_pattern,
        shuffle=True,
        image_column_name=image_column_name,
        batch_size=batch_size,
        image_size=validation_image_size,
        patch_size=patch_size,
        num_register_tokens=num_register_tokens,
    )

    import bpdb

    bpdb.set_trace()
