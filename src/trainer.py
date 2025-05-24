import time
import gc
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import yaml
from dataclasses import asdict
from tqdm import tqdm
import torch
import wandb
import tensorset as ts

from main_conf import MainConfig
from src.dataset import MASK_SAMPLE_ID
from src.model import IJEPAModel
from src.validate import validate
from src.dataset import get_lidar_data


class Trainer:
    def __init__(
        self,
        model: IJEPAModel,
        optimizer: torch.optim.Optimizer,
        training_state: dict,
        conf: MainConfig,
        dataloader,
    ):
        self.model = model
        self.conf = conf
        self.dataloader = dataloader
        self.training_state = training_state
        self.patch_size = conf.patch_size

        self.optimizer = optimizer

        self.checkpoint_folder_path = (
            Path("checkpoints") / f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if conf.should_compile:
            self.model = torch.compile(self.model)

        self.lidar_data = None

    @contextmanager
    def autocast_fn(self):
        with torch.autocast(self.conf.torch_device.type, self.conf.torch_dtype):
            yield

    def prepare_context_target_batch(self, batch):
        packed_batch, *_ = batch

        if not isinstance(packed_batch, ts.TensorSet):
            raise ValueError()

        position_ids = packed_batch.named_columns.pop("position_ids")
        sequence_ids = packed_batch.named_columns.pop("sequence_ids")
        # Token ids contains along the channel dim (sequence_ids, register id, height idx, width idx)
        token_ids = torch.cat((sequence_ids.unsqueeze(-1), position_ids), -1)

        patches = packed_batch.named_columns.pop("patches")

        patches = patches.to(
            device=self.conf.torch_device,
            dtype=self.conf.torch_dtype,
            non_blocking=True,
        )
        token_ids = token_ids.to(self.conf.torch_device, non_blocking=True)

        # Scale from [0,255] to [-1,1]
        patches = (patches / 255) * 2 - 1

        return patches, token_ids

    def save_checkpoint(self):
        self.checkpoint_folder_path.mkdir(exist_ok=True, parents=True)

        existing_checkpoints = list(self.checkpoint_folder_path.iterdir())
        existing_checkpoints.sort()

        checkpoints_to_delete = existing_checkpoints[
            : -self.conf.max_num_save_checkpoints
        ]

        for existing_checkpoint in checkpoints_to_delete:
            print("Deleting checkpoint", existing_checkpoint)
            existing_checkpoint.unlink()

        checkpoint_save_path = (
            self.checkpoint_folder_path / f"{self.training_state['epoch']:05}.pt"
        )

        torch.save(
            {
                "training_state": self.training_state,
                "model": (
                    self.model._orig_mod.state_dict()
                    if hasattr(self.model, "_orig_mod")
                    else self.model.state_dict()
                ),
                "optimizer": self.optimizer.state_dict(),
            },
            str(checkpoint_save_path),
        )
        print("Saved checkpoint to ", checkpoint_save_path)

        yaml_save_path = self.checkpoint_folder_path / "config.yaml"
        conf_dict = asdict(self.conf)
        # Hack to allow loading from jsonargparse
        conf_dict = dict(conf=conf_dict)
        with open(yaml_save_path, "w") as f:
            yaml.dump(conf_dict, f)

    def compute_lidar_score(self):
        if self.lidar_data is None:
            self.lidar_data = get_lidar_data(
                config=self.conf.context_target_dataset,
                dataset_pattern=self.conf.val_dataset_pattern,
                image_column_name=self.conf.image_column_name,
            )

    def train_step(self, batch, start_time):
        patches, token_ids = self.prepare_context_target_batch(batch)

        interp = 0
        if self.conf.should_interp:
            interp = min(
                1, self.training_state["global_step"] / self.conf.interp_warmup_steps
            )

        ema_beta = (
            min(
                1,
                self.training_state["global_step"] / self.conf.ema_beta_warmup_steps,
            )
            * (self.conf.ema_beta - self.conf.ema_beta_start)
            + self.conf.ema_beta_start
        )

        should_log_lidar = (
            self.training_state["global_step"] % self.conf.log_every_num_steps == 0
        )

        should_log = (
            self.training_state["global_step"] % self.conf.log_every_num_steps == 0
        ) or should_log_lidar

        with self.autocast_fn():
            result_dict = self.model(
                patches=patches,
                token_ids=token_ids,
                context_sequence_length=self.conf.context_target_dataset.packer_context_sequence_length,
                interp=interp,
                return_smooth_rank=should_log,
            )

        if should_log_lidar:
            lidar_score = self.compute_lidar_score()

        loss = result_dict["loss"]

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        lr = (
            min(1, self.training_state["global_step"] / self.conf.num_warmup_steps)
        ) * (self.conf.lr - self.conf.start_lr) + self.conf.start_lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        for ema_p, p in zip(
            self.model.ema_encoder.parameters(), self.model.encoder.parameters()
        ):
            if p.is_floating_point():
                ema_p.lerp_(p, 1 - ema_beta)
            else:
                ema_p.copy_(p)

        # Count the total number of samples in this batch
        num_samples_in_batch = 0
        for ids_seq in token_ids[..., 0].cpu():
            ids = torch.unique(ids_seq).tolist()
            if MASK_SAMPLE_ID in ids:
                ids.remove(MASK_SAMPLE_ID)
            num_samples_in_batch += len(ids)

        elapsed = time.time() - start_time
        samples_per_second = num_samples_in_batch / elapsed

        if should_log:
            mask_rate = (token_ids[..., 0] == MASK_SAMPLE_ID).float().mean()

            wandb.log(
                dict(
                    epoch=self.training_state["epoch"],
                    loss=loss.item(),
                    num_samples=num_samples_in_batch,
                    lr=lr,
                    ema_beta=ema_beta,
                    smooth_rank=result_dict["smooth_rank"].item(),
                    mask_rate=mask_rate,
                    interp=interp,
                    num_total_samples=self.training_state["num_total_samples"],
                    samples_per_second=samples_per_second,
                    step_time=elapsed,
                ),
                step=self.training_state["global_step"],
            )

        self.training_state["global_step"] += 1
        self.training_state["num_total_samples"] += num_samples_in_batch

        return loss.item()

    def train_one_epoch(self, dataloader_stream, prog_bar):
        while True:
            start_time = time.time()
            batch = next(dataloader_stream)
            loss = self.train_step(batch, start_time)

            prog_bar.update(1)
            prog_bar.set_description(f"{self.training_state} loss {round(loss, 3)}")

            estimated_epoch = (
                self.training_state["num_total_samples"]
                // self.conf.train_dataset_length
            )

            if estimated_epoch > self.training_state["epoch"]:
                self.training_state["epoch"] += 1
                return

            if self.conf.test_mode:
                return

    def run_validation(self):
        gc.collect()
        torch.cuda.empty_cache()

        accuracies = validate(
            model=self.model,
            image_column_name=self.conf.image_column_name,
            label_column_name=self.conf.label_column_name,
            num_classes=self.conf.num_classes,
            patch_size=self.patch_size,
            validation_image_size=self.conf.validation_image_size,
            batch_size=self.conf.batch_size,
            num_workers=self.conf.num_workers,
            train_dataset_pattern=self.conf.train_dataset_pattern,
            val_dataset_pattern=self.conf.val_dataset_pattern,
            dtype=self.conf.torch_dtype,
            should_compile=self.conf.should_compile,
            test_mode=self.conf.test_mode,
            validation_probe_lr=self.conf.validation_probe_lr,
            validation_probe_batch_size=self.conf.validation_probe_batch_size,
            validation_train_epochs=self.conf.validation_train_epochs,
            validation_extraction_mode=self.conf.validation_extraction_mode,
            num_register_tokens=self.conf.num_register_tokens,
        )

        gc.collect()
        torch.cuda.empty_cache()

        depths = list(range(len(accuracies)))
        best_accuracy = max(accuracies)

        line_series = wandb.plot.line_series(
            xs=depths,
            ys=[accuracies],
            keys=[f"Epoch {self.training_state['epoch']:03}"],
            title="Feature Depth VS Accuracy@1",
            xname="Depth",
        )

        wandb.log(
            {
                "depth vs acc@1": line_series,
                "epoch": self.training_state["epoch"],
                "acc@1": best_accuracy,
            },
            step=self.training_state["global_step"],
        )

        print("EPOCH", self.training_state["epoch"], "accuracies", accuracies)
        return accuracies

    def train(self):
        conf_d = asdict(self.conf)
        trainable_params = (p for p in self.model.parameters() if p.requires_grad)
        conf_d["num_params"] = sum(p.nelement() for p in trainable_params)

        wandb.init(
            project="ijepa-depthsmart",
            config=conf_d,
            mode="disabled" if self.conf.test_mode else None,
        )

        prog_bar = tqdm(desc="training")
        dataloader_stream = iter(self.dataloader)

        for _ in range(self.conf.num_epochs):
            self.train_one_epoch(dataloader_stream, prog_bar)

            is_last_epoch = self.training_state["epoch"] == self.conf.num_epochs - 1
            should_validate = (
                self.conf.test_mode
                or is_last_epoch
                or (
                    (self.training_state["epoch"] > 0)
                    and (
                        self.training_state["epoch"]
                        % self.conf.validate_every_num_epochs
                        == 0
                    )
                )
            )

            if should_validate:
                self.run_validation()

            if self.conf.test_mode:
                break

            self.save_checkpoint()

        self.save_checkpoint()
        prog_bar.close()
