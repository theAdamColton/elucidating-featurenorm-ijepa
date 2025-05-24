import huggingface_hub
import os

os.makedirs("datasets", exist_ok=True)

huggingface_hub.snapshot_download(
    "adams-story/imagenet1k-256-wds",
    repo_type="dataset",
    local_dir="datasets/imagenet1k-256-wds",
)
huggingface_hub.snapshot_download(
    "adams-story/nyu-depthv2-wds",
    repo_type="dataset",
    local_dir="datasets/nyu-depthv2-wds",
)
