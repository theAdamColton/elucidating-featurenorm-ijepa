import huggingface_hub

huggingface_hub.snapshot_download(
    "adams-story/imagenet1k-256-wds", repo_type="dataset", local_dir="dataset/"
)
