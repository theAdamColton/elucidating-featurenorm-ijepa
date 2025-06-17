import json
import math
import time
import gc
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import yaml
import dataclasses

from tqdm import tqdm
import torch
import wandb
import tensorset as ts
import einx

from main_conf import MainConfig
from src.dataset import MASK_SAMPLE_ID, get_context_target_dataloader
from src.model import IJEPAModel, IJEPAOutput
from src.validate import validate
from src.dataset import get_repeated_data
from src.lidar import compute_lidar_score
from src.ops import masked_mean_along_sequence_dim


class Trainer:
    def __init__(
        self,
        model: IJEPAModel,
        grad_scaler: torch.GradScaler,
        optimizer: torch.optim.Optimizer,
        training_state: dict,
        conf: MainConfig,
        dataloader,
    ):
        self.model = model
        self.grad_scaler = grad_scaler
        self.conf = conf
        self.training_state = training_state
        self.patch_size = conf.patch_size

        self.optimizer = optimizer

        self.checkpoint_folder_path = (
            Path("checkpoints") / f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.log_file_path = self.checkpoint_folder_path / "training_log.jsonl"

        self.student_encoder = self.model.encoder
        if conf.should_compile:
            self.model = torch.compile(self.model, fullgraph=True, dynamic=False)
            # Is possibly run with dynamic shapes
            self.student_encoder = torch.compile(self.student_encoder)

        self.lidar_data = None

    def get_dataloader(self):
        conf = self.conf
        return get_context_target_dataloader(
            config=conf.context_target_dataset,
            dataset_pattern=conf.train_dataset_pattern,
            dataset_length=conf.train_dataset_length,
            seed=conf.seed,
            image_column_name=conf.image_column_name,
            batch_size=conf.batch_size,
            num_workers=conf.num_workers,
        )

    @contextmanager
    def autocast_fn(self):
        with torch.autocast(self.conf.torch_device.type, self.conf.torch_dtype):
            yield

    def prepare_context_target_batch(self, batch):
        if not isinstance(batch, ts.TensorSet):
            raise ValueError()

        position_ids = batch.named_columns.pop("position_ids")
        sample_ids = batch.named_columns.pop("sample_ids")
        # Token ids contains along the channel dim (sample_ids, register id, height idx, width idx)
        token_ids = torch.cat((sample_ids.unsqueeze(-1), position_ids), -1)

        patches = batch.named_columns.pop("patches")

        patches = patches.to(
            device=self.conf.torch_device,
            dtype=self.conf.torch_dtype,
            non_blocking=True,
        )
        token_ids = token_ids.to(self.conf.torch_device, non_blocking=True)

        # Scale from [0,255] to [-1,1]
        patches = (patches / 255) * 2 - 1

        return patches, token_ids

    def ensure_checkpoint_folder(self):
        self.checkpoint_folder_path.mkdir(exist_ok=True, parents=True)

    def save_checkpoint(self):
        self.ensure_checkpoint_folder()

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
                "grad_scaler": self.grad_scaler.state_dict(),
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
        conf_dict = dataclasses.asdict(self.conf)
        # Hack to allow loading from jsonargparse
        conf_dict = dict(conf=conf_dict)
        with open(yaml_save_path, "w") as f:
            yaml.dump(conf_dict, f)

    def compute_lidar_score(self):
        # we use eval mode, which causes
        # diffmoe to use dynamic allocation
        self.student_encoder.eval()

        conf = self.conf
        if self.lidar_data is None:
            lidar_data = get_repeated_data(
                config=conf.context_target_dataset,
                dataset_pattern=conf.train_dataset_pattern,
                num_unique_samples=conf.lidar_num_unique_samples,
                num_repeat_samples=conf.lidar_num_augmentations,
                image_column_name=conf.image_column_name,
            )

            context_sequence_length = (
                self.conf.context_target_dataset.packer_context_sequence_length
            )
            # Take only the contexts
            # n q (sx sy) ... -> n q sx ...
            lidar_data = lidar_data.iloc[:, :, :context_sequence_length]

            def _ungroup(x):
                return einx.rearrange("n q s ... -> (n q) s ...", x)

            lidar_data = lidar_data.apply(_ungroup)

            self.lidar_data = lidar_data

        # Embed lidar data

        batch_size = conf.batch_size
        num_samples_to_embed = self.lidar_data.size(0)
        num_batches = math.ceil(num_samples_to_embed / batch_size)

        features = []

        for i in range(num_batches):
            idx_start = i * batch_size
            batch = self.lidar_data.iloc[idx_start : idx_start + batch_size]

            patches, token_ids = self.prepare_context_target_batch(batch)

            with torch.inference_mode():
                with self.autocast_fn():
                    encoder_hidden_states = self.student_encoder(
                        x=patches, token_ids=token_ids
                    ).hidden_states

            sample_ids = token_ids[..., 0]
            is_sample_mask = sample_ids != MASK_SAMPLE_ID

            encoder_hidden_states = encoder_hidden_states.float()
            # b [s] d
            encoder_hidden_states = masked_mean_along_sequence_dim(
                encoder_hidden_states, is_sample_mask
            )

            features.append(encoder_hidden_states)

        features = torch.cat(features)
        features = einx.rearrange(
            "(n q) d -> n q d", features, n=conf.lidar_num_unique_samples
        )

        lidar_score = compute_lidar_score(features).item()

        return lidar_score

    def write_log_row(self, log_dict):
        self.ensure_checkpoint_folder()

        with open(self.log_file_path, "a") as f:
            row = json.dumps(log_dict) + "\n"
            f.write(row)

    @property
    def ema_beta(self):
        conf = self.conf
        global_step = self.training_state["global_step"]
        if global_step <= conf.ema_beta_warmup_steps:
            scale = global_step / conf.ema_beta_warmup_steps
            beta = (
                scale * (conf.ema_beta_steady - conf.ema_beta_start)
                + conf.ema_beta_start
            )
            return beta

        steady_step = global_step - conf.ema_beta_warmup_steps
        scale = steady_step / conf.ema_beta_steady_steps
        beta = conf.ema_beta_steady - scale * (conf.ema_beta_steady - conf.ema_beta_end)
        beta = min(beta, conf.ema_beta_end)
        return beta

    @property
    def lr(self):
        conf = self.conf
        global_step = self.training_state["global_step"]

        if global_step <= conf.num_lr_warmup_steps:
            scale = global_step / conf.num_lr_warmup_steps
            lr = (conf.steady_lr - conf.start_lr) * scale + conf.start_lr
        elif global_step - conf.num_lr_warmup_steps <= conf.num_lr_steady_steps:
            lr = conf.steady_lr
        else:
            cooldown_step = (
                global_step - conf.num_lr_warmup_steps - conf.num_lr_steady_steps
            )
            scale = cooldown_step / conf.num_lr_cooldown_steps
            lr = conf.steady_lr - scale * (conf.steady_lr - conf.end_lr)
            lr = max(lr, conf.end_lr)

        return lr

    def ema_update(self, ema_beta: float):
        for (teacher_parameter_name, teacher_parameter), (
            student_parameter_name,
            student_parameter,
        ) in zip(
            self.model.ema_encoder.named_parameters(),
            self.model.encoder.named_parameters(),
        ):
            assert teacher_parameter_name == student_parameter_name, (
                f"{teacher_parameter_name}!={student_parameter_name}"
            )

            should_lerp = (
                student_parameter.is_floating_point()
                and student_parameter.requires_grad
                or "capacity_predictor_thresholds.parameter" in student_parameter_name
            )

            if should_lerp:
                # trainable parameters, and capacity thresholds
                teacher_parameter.lerp_(student_parameter, 1 - ema_beta)
            else:
                # rope parameters, bool is_initted
                # running batchnorm means and vars
                teacher_parameter.copy_(student_parameter)

    def step_model(
        self, patches: torch.Tensor, token_ids: torch.Tensor, log_smooth_rank=False
    ):
        self.model.train()

        with self.autocast_fn():
            result: IJEPAOutput = self.model(
                patches=patches,
                token_ids=token_ids,
                context_sequence_length=self.conf.context_target_dataset.packer_context_sequence_length,
                return_smooth_rank=log_smooth_rank,
                window_size=self.conf.context_target_dataset.mask_window_size,
                return_predictor_target_token_ids=True,
                return_tokenwise_loss=True,
            )

        # take the loss over non padding tokens
        loss_mask = result.is_target_mask
        # accurate masked mean
        loss = result.tokenwise_loss[loss_mask].mean()

        # Combine IJEPA loss and diffmoe_loss (which might be zero)
        total_loss = loss + result.diffmoe_loss

        # Step student model
        lr = self.lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad()

        # Step ema model
        ema_beta = self.ema_beta
        self.ema_update(ema_beta)

        log_dict = dict(
            epoch=self.training_state["epoch"],
            num_total_samples=self.training_state["num_total_samples"],
            global_step=self.training_state["global_step"],
            lr=lr,
            ema_beta=ema_beta,
        )

        def safe_add_to_log_dict(key, value):
            if value is None:
                return
            if isinstance(value, torch.Tensor):
                if value.ndim > 0:
                    return

                value = value.item()

            if not isinstance(value, (float, int)):
                return

            log_dict[key] = value

        for k, v in result._asdict().items():
            safe_add_to_log_dict(k, v)

        safe_add_to_log_dict("loss", loss)

        return log_dict

    def compute_additional_metrics(self, token_ids):
        log_dict = {}

        key_pad_mask = token_ids[..., 0] == MASK_SAMPLE_ID
        log_dict["mask_rate"] = key_pad_mask.float().mean().item()

        # Plus 1 is because we index from 0
        max_image_height = (token_ids[..., -2].amax().item() + 1) * self.patch_size
        max_image_width = (token_ids[..., -1].amax().item() + 1) * self.patch_size
        log_dict["max_image_height"] = max_image_height
        log_dict["max_image_width"] = max_image_width

        sub_batch_size = 8
        sub_token_ids = token_ids[:sub_batch_size]
        unique_ids = torch.unique(sub_token_ids[..., 0]).tolist()
        if MASK_SAMPLE_ID in unique_ids:
            unique_ids.remove(MASK_SAMPLE_ID)

        side_lengths = []
        for id in unique_ids:
            mask = sub_token_ids[..., 0] == id
            sample_token_ids = sub_token_ids[mask]
            side_length = sample_token_ids[..., -2:].amax(0).float().mean(-1).item()
            side_length = (side_length + 1) * self.patch_size
            side_lengths.append(side_length)
        avg_side_length = torch.tensor(side_lengths).mean().item()

        log_dict["avg_side_length"] = avg_side_length

        return log_dict

    def train_one_epoch(self, dataloader_stream, prog_bar):
        while True:
            should_log_lidar = (
                self.training_state["global_step"] % self.conf.log_lidar_every_num_steps
                == 0
            )
            should_log_additional_metrics = (
                self.training_state["global_step"] % self.conf.log_every_num_steps == 0
            ) or should_log_lidar

            # Forward-backward model

            start_time = time.time()

            batch, *_ = next(dataloader_stream)
            patches, token_ids = self.prepare_context_target_batch(batch)

            log_dict = self.step_model(
                patches, token_ids, log_smooth_rank=should_log_additional_metrics
            )

            elapsed = time.time() - start_time

            token_ids = token_ids.cpu()

            # Count the total number of samples in the batch
            num_samples_in_batch = 0
            for ids_seq in token_ids[..., 0]:
                ids = torch.unique(ids_seq).tolist()
                if MASK_SAMPLE_ID in ids:
                    ids.remove(MASK_SAMPLE_ID)
                num_samples_in_batch += len(ids)

            log_dict["num_samples_in_batch"] = num_samples_in_batch

            elapsed = time.time() - start_time
            log_dict["step_time"] = elapsed
            log_dict["samples_per_second"] = num_samples_in_batch / elapsed

            # Compute the lidar score
            if should_log_lidar:
                log_dict["lidar_score"] = self.compute_lidar_score()

            # Log some other metrics
            if should_log_additional_metrics:
                log_dict.update(self.compute_additional_metrics(token_ids))

                # Post to wandb
                wandb.log(
                    log_dict,
                    step=self.training_state["global_step"],
                )

            # Write to log file
            self.write_log_row(log_dict)

            # Update training state
            self.training_state["global_step"] += 1
            self.training_state["num_total_samples"] += log_dict["num_samples_in_batch"]

            prog_bar.update(1)
            training_state_str = "".join(
                f"{k}:{v} " for k, v in self.training_state.items()
            )
            prog_bar.set_description(
                f"{training_state_str}loss:{round(log_dict['loss'], 3)}"
            )

            # Estimate the current epoch based on the number of total samples processed
            estimated_epoch = (
                self.training_state["num_total_samples"]
                // self.conf.train_dataset_length
            )

            if estimated_epoch > self.training_state["epoch"]:
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

        log_dict = {
            "epoch": self.training_state["epoch"],
            "acc@1": best_accuracy,
        }

        for i, accuracy in enumerate(accuracies):
            log_dict[f"depth{i:03}_acc@1"] = accuracy

        self.write_log_row(log_dict)

        line_series = wandb.plot.line_series(
            xs=depths,
            ys=[accuracies],
            keys=[f"Epoch {self.training_state['epoch']:03}"],
            title="Feature Depth VS Accuracy@1",
            xname="Depth",
        )

        log_dict["depth vs acc@1"] = line_series

        wandb.log(
            log_dict,
            step=self.training_state["global_step"],
        )

        print("EPOCH", self.training_state["epoch"], "accuracies", accuracies)
        return accuracies

    def train(self):
        conf_d = dataclasses.asdict(self.conf)
        trainable_params = (p for p in self.model.parameters() if p.requires_grad)
        num_params = sum(p.nelement() for p in trainable_params)
        print("NUM TRAINABLE PARAMETERS", num_params)
        conf_d["num_params"] = num_params

        wandb.init(
            project="ijepa-tanh",
            config=conf_d,
            mode="disabled" if self.conf.test_mode else None,
        )

        prog_bar = tqdm(desc="training")

        dataloader_stream = None

        while True:
            if dataloader_stream is None:
                dataloader_stream = iter(self.get_dataloader())

            self.train_one_epoch(dataloader_stream, prog_bar)
            self.training_state["epoch"] += 1

            is_last_epoch = self.training_state["epoch"] == self.conf.num_epochs
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

            # TODO
            # This is a hack to avoid memory usage creeping up
            # from dataloader workers never closing tar files
            should_gc = (
                should_validate
                or (self.training_state["epoch"] % self.conf.gc_every_num_epochs) == 0
            )
            if should_gc:
                # My most vile and perverted HACK
                del dataloader_stream
                time.sleep(1)
                gc.collect()
                dataloader_stream = None

            if should_validate:
                self.run_validation()

            if self.conf.test_mode:
                break

            if is_last_epoch:
                break

            self.save_checkpoint()

        self.save_checkpoint()
        prog_bar.close()
