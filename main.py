from contextlib import contextmanager
import random

import numpy as np
import jsonargparse
import torch

from main_conf import MainConfig
from src.dataset import get_context_target_dataloader
from src.model import IJEPAModel
from src.validate import validate
from src.validate_monocular_depth import validate_monocular_depth_prediction
from src.trainer import Trainer
from src.scripts.make_viz import make_viz
from src.scripts.plot_sample_losses import plot_sample_losses
from src.scripts.visualize_embeddings import visualize_embeddings


def main(conf: MainConfig = MainConfig()):
    if conf.seed is not None:
        random.seed(conf.seed)
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)

    device = torch.device(conf.device)
    dtype = getattr(torch, conf.dtype)

    num_image_channels = conf.num_image_channels

    patch_size = conf.patch_size

    dataloader, context_sequence_length, target_sequence_length = (
        get_context_target_dataloader(
            dataset_pattern=conf.train_dataset_pattern,
            dataset_length=conf.train_dataset_length,
            seed=conf.seed,
            image_column_name=conf.image_column_name,
            label_column_name=conf.label_column_name,
            batch_size=conf.batch_size,
            packer_batch_size=conf.packer_batch_size,
            max_side_length=conf.context_target_max_side_length,
            min_side_length=conf.context_target_min_side_length,
            mask_window_size=conf.context_target_mask_window_size,
            num_register_tokens=conf.num_register_tokens,
            patch_size=patch_size,
            min_context_capacity=conf.min_context_capacity,
            max_context_capacity=conf.max_context_capacity,
            absolute_max_context_capacity=conf.absolute_max_context_capacity,
            num_workers=conf.num_workers,
        )
    )

    model = IJEPAModel(conf.model).to(device)

    @contextmanager
    def autocast_fn():
        with torch.autocast(device.type, dtype):
            yield

    if conf.resume_path is not None:

        def _load():
            d = torch.load(conf.resume_path, map_location=device, weights_only=False)
            model.load_state_dict(d["model"], strict=False)

        _load()

    if conf.mode == "make-viz":
        make_viz(dataloader, context_sequence_length, patch_size, num_image_channels)

    elif conf.mode == "validate":
        accuracies = validate(
            model=model,
            image_column_name=conf.image_column_name,
            label_column_name=conf.label_column_name,
            num_classes=conf.num_classes,
            patch_size=patch_size,
            validation_image_size=conf.validation_image_size,
            batch_size=conf.batch_size,
            num_workers=conf.num_workers,
            train_dataset_pattern=conf.train_dataset_pattern,
            val_dataset_pattern=conf.val_dataset_pattern,
            dtype=dtype,
            should_compile=conf.should_compile,
            test_mode=conf.test_mode,
            validation_probe_lr=conf.validation_probe_lr,
            validation_probe_batch_size=conf.validation_probe_batch_size,
            validation_train_epochs=conf.validation_train_epochs,
            validation_extraction_mode=conf.validation_extraction_mode,
            num_register_tokens=conf.num_register_tokens,
        )
        print("ACCURACIES", accuracies)

    elif conf.mode == "validate-monocular-depth":
        result = validate_monocular_depth_prediction(
            model=model,
            image_column_name=conf.image_column_name,
            depth_column_name=conf.depth_column_name,
            patch_size=patch_size,
            validation_image_size=conf.validation_image_size,
            batch_size=conf.batch_size,
            num_workers=conf.num_workers,
            train_dataset_pattern=conf.monocular_depth_train_dataset_pattern,
            train_dataset_length=conf.monocular_depth_train_dataset_length,
            val_dataset_pattern=conf.monocular_depth_val_dataset_pattern,
            feature_depth=conf.validation_monocular_depth_feature_depth,
            dtype=dtype,
            test_mode=conf.test_mode,
            should_compile=conf.should_compile,
            validation_probe_lr=conf.validation_monocular_depth_lr,
            validation_train_epochs=conf.validation_monocular_depth_train_epochs,
            log_every_num_steps=conf.log_every_num_steps,
            num_register_tokens=conf.num_register_tokens,
        )
        print("MONOCULAR DEPTH RESULT", result)

    elif conf.mode == "plot-sample-losses":
        plot_sample_losses(
            dataloader=dataloader,
            context_sequence_length=context_sequence_length,
            model=model,
            device=device,
            dtype=dtype,
            patch_size=patch_size,
            num_image_channels=num_image_channels,
            autocast_fn=autocast_fn,
        )

    elif conf.mode == "visualize-embeddings":
        visualize_embeddings(
            dataloader=dataloader,
            context_sequence_length=context_sequence_length,
            model=model,
            device=device,
            dtype=dtype,
            patch_size=patch_size,
            num_image_channels=num_image_channels,
            autocast_fn=autocast_fn,
        )

    elif conf.mode == "train":
        trainer = Trainer(
            model=model,
            conf=conf,
            dataloader=dataloader,
            context_sequence_length=context_sequence_length,
            target_sequence_length=target_sequence_length,
            device=device,
            dtype=dtype,
        )
        trainer.train()


if __name__ == "__main__":
    jsonargparse.CLI(main)
