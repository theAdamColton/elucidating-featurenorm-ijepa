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
from src.eval.make_viz import make_viz
from src.eval.plot_sample_losses import plot_sample_losses
from src.eval.visualize_embeddings import visualize_embeddings


def main(conf: MainConfig = MainConfig()):
    if conf.seed is not None:
        random.seed(conf.seed)
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)

    num_image_channels = conf.num_image_channels

    patch_size = conf.patch_size

    dataloader = get_context_target_dataloader(
        config=conf.context_target_dataset,
        dataset_pattern=conf.train_dataset_pattern,
        dataset_length=conf.train_dataset_length,
        seed=conf.seed,
        image_column_name=conf.image_column_name,
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
    )

    model = IJEPAModel(conf.model).to(conf.torch_device)

    trainable_params = tuple(p for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=conf.start_lr,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    grad_scaler = torch.GradScaler()

    training_state = dict(global_step=0, epoch=0, num_total_samples=0)

    @contextmanager
    def autocast_fn():
        with torch.autocast(conf.torch_device.type, conf.torch_dtype):
            yield

    if conf.resume_checkpoint_path is not None:

        def _load():
            d = torch.load(
                conf.resume_checkpoint_path,
                map_location=conf.torch_device,
                weights_only=False,
            )

            if "grad_scaler" in d:
                grad_scaler.load_state_dict(d["grad_scaler"])
            else:
                print("Warning! grad_scaler not found in checkpoint file!")

            model.load_state_dict(d["model"], strict=False)
            optimizer.load_state_dict(d["optimizer"])
            training_state.update(d["training_state"])

        _load()

    if conf.mode == "make-viz":
        make_viz(
            dataloader=dataloader,
            context_sequence_length=conf.context_target_dataset.packer_context_sequence_length,
            patch_size=patch_size,
            num_image_channels=num_image_channels,
        )

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
            dtype=conf.torch_dtype,
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
            dtype=conf.torch_dtype,
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
            context_target_dataset=conf.context_target_dataset,
            dataset_pattern=conf.train_dataset_pattern,
            image_column_name=conf.image_column_name,
            model=model,
            device=conf.torch_device,
            dtype=conf.torch_dtype,
            patch_size=patch_size,
            num_image_channels=num_image_channels,
            autocast_fn=autocast_fn,
            seed=conf.seed,
            batch_size=conf.batch_size,
        )

    elif conf.mode == "visualize-embeddings":
        visualize_embeddings(
            dataloader=dataloader,
            context_sequence_length=conf.context_target_dataset.packer_context_sequence_length,
            model=model,
            device=conf.torch_device,
            dtype=conf.torch_dtype,
            patch_size=patch_size,
            num_image_channels=num_image_channels,
            autocast_fn=autocast_fn,
            feature_depth=conf.visualize_features_depth,
        )

    elif conf.mode == "train":
        trainer = Trainer(
            model=model,
            grad_scaler=grad_scaler,
            optimizer=optimizer,
            training_state=training_state,
            conf=conf,
            dataloader=dataloader,
        )
        trainer.train()


if __name__ == "__main__":
    jsonargparse.CLI(main)
