"""Flow Matching training on Starmen (synthetic) dataset.

Usage:
    python run_fm_starmen.py --config configs/fm_starmen.yaml
    python run_fm_starmen.py --config configs/fm_starmen.yaml --lr 5e-5
"""
import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import torchdiffeq
from tqdm import tqdm

import ml_collections
from utils.config_loader import get_arg_parser, load_config, parse_simple_overrides, deep_update, print_config

from datasets.starmen import create_dataloaders
from src.models.autoencoder.unet import UNetModel
from src.models.flow_matching.tfm import *
from src.models.flow_matching.multi_marginal_fm import MultiMarginalFlowMatcher
from utils.general_utils import *

DEFAULT_CONFIG = os.path.join(ROOT, "configs/fm_starmen.yaml")


class FlowLightningModule(pl.LightningModule):

    def __init__(self, config: ml_collections.ConfigDict, noise_prediction=False, implementation="ODE"):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config
        self.noise_prediction = noise_prediction
        self.implementation = implementation
        self.init_autoencoder()

        fm = config.flow_matching
        self.flow_module = MLP_Cond_Memory_Module(
            treatment_cond=fm.treatment_cond,
            memory=fm.memory,
            dim=fm.dim, w=fm.w,
            time_varying=True, conditional=False,
            lr=config.training.lr, sigma=0.1,
            loss_fn=mse_loss if fm.loss_fn == "mse" else l1_loss,
            metrics=['mse_loss'], implementation=self.implementation,
            sde_noise=fm.sde_noise, clip=fm.clip,
            depth=fm.depth,
            velocity_net=fm.velocity_net,
            time_prediction=fm.time_prediction,
            class_conditional=fm.class_conditional,
            starman=True,
            noise_prediction=self.noise_prediction
        )
        self.criterion = mse_loss if fm.loss_fn == "mse" else l1_loss
        self.MAX_TIME = fm.max_time

        if fm.learned_time_scaling:
            self.time_scaling = self.flow_module.time_scaling
        else:
            self.time_scaling = None

        self.s_diffusion = torch.tensor(fm.s_diffusion)
        self.FM = MultiMarginalFlowMatcher(
            sigma=fm.sigma_scheduler,
            interpolation=fm.interpolation,
            mix_coeff=fm.deriv_mix_coeff,
            time_scaler=self.time_scaling
        )

    def init_autoencoder(self):
        m = self.config.model
        hw = self.config.hardware
        self.model_ae = UNetModel(
            image_size=tuple(m.image_size),
            in_channels=m.in_channels,
            model_channels=m.num_channels,
            out_channels=m.in_channels if not m.learn_sigma else 2 * m.in_channels,
            num_res_blocks=m.num_res_blocks,
            channel_mult=tuple(m.channel_mult),
            num_classes=m.num_classes,
            use_checkpoint=hw.get('use_checkpoint', False),
            attention_resolutions=m.attention_resolutions.split(','),
            num_heads=m.num_heads,
            num_head_channels=m.num_head_channels,
            num_heads_upsample=m.num_heads_upsample,
            use_scale_shift_norm=m.use_scale_shift_norm,
            dropout=m.dropout,
            resblock_updown=m.resblock_updown,
            use_fp16=hw.get('use_fp16', False),
            use_new_attention_order=m.use_new_attention_order,
            use_skip_connection=m.use_skip_connection,
            use_time_embed=m.use_time_embed,
            dae=False
        )

        w_path = ROOT + "/logs/Starmen/starmen_model.pth"
        checkpoint = torch.load(w_path, map_location='cpu', weights_only=False)
        self.model_ae.load_state_dict(checkpoint)
        print(f'Autoencoder loaded from {w_path}')

    def forward(self, in_tensor):
        if self.noise_prediction:
            return self.flow_module.forward_train(in_tensor)
        else:
            return self.flow_module.forward_train(in_tensor)

    def training_step(self, batch, batch_idx):
        fm = self.config.flow_matching
        sigma_t, xt, t, ut, x0_classes, x0_cls_time, label, x1, x1_time, age, sex, curr_noise = self.train_process_batch(batch)

        xt = xt.squeeze()
        time_mult = fm.get('time_multiplier', 1)
        x0_cls_time = x0_cls_time.unsqueeze(1) * time_mult
        t = t.unsqueeze(1) * time_mult
        tau = torch.zeros_like(t)
        t1_time = torch.zeros_like(tau)
        age = age.repeat(xt.shape[0], 1)
        sex = sex.repeat(xt.shape[0], 1)

        in_tensor = torch.cat([xt, x0_classes, x0_cls_time, sex, age, label, t1_time, tau, t], dim=-1)
        target = x1 if fm.data_parametrization else ut

        if self.noise_prediction:
            flow_out, score_out, noise_out = self(in_tensor)
            if self.implementation == "SDE":
                if fm.uncertainty_method == 'heteroscedastic':
                    log_var = noise_out[:, :self.flow_module.dim]
                    loss = 0.5 * torch.mean(log_var + (target - flow_out[:, :self.flow_module.dim])**2 / torch.exp(log_var))
                    loss += 0.01 * torch.mean(torch.exp(log_var))
                else:
                    loss_flow = self.criterion(flow_out, target)
                    loss_score = torch.mean((score_out * sigma_t + curr_noise)**2)

                    n_steps = 4
                    step_size = (x1_time - t) / n_steps
                    current_x, current_t = xt.clone(), t.clone()
                    with torch.no_grad():
                        for _ in range(n_steps):
                            step_input = torch.cat([current_x, x0_classes, x0_cls_time, sex, age, label,
                                                    torch.zeros_like(t), torch.zeros_like(t), current_t], dim=-1)
                            velocity, score, _ = self(step_input)
                            current_x = current_x + step_size * (velocity + score * self.s_diffusion.to(self.device) / 2)
                            current_t = current_t + (x1_time - t) / n_steps

                    uncertainty = (current_x - x1)**2
                    loss_noise = self.criterion(noise_out, uncertainty).clip(0, 1e3)
                    loss = loss_flow + 0.001 * loss_noise + 0.1 * loss_score
            else:
                loss_flow = self.criterion(flow_out[:, :self.flow_module.dim], target)
                uncertainty = torch.abs(flow_out[:, :self.flow_module.dim] - target)
                loss_noise = self.criterion(noise_out[:, :self.flow_module.dim], uncertainty)
                loss = loss_flow + 0.1 * loss_noise
        else:
            flow_out = self(in_tensor)
            if self.implementation == "SDE":
                std = sigma_t * self.flow_module.sde_noise
                noise = torch.randn_like(flow_out[:, :self.flow_module.dim]) * std
                loss = self.criterion(flow_out[:, :self.flow_module.dim] + noise, target)
            else:
                loss = self.criterion(flow_out[:, :self.flow_module.dim], target)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log('train_score_loss', loss_score, prog_bar=False, on_epoch=True)
        self.log('train_uncertainty_loss', loss_noise, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        fm = self.config.flow_matching
        sigma_t, xt, t, ut, x0_classes, x0_cls_time, label, x1, x1_time, age, sex, curr_noise = self.train_process_batch(batch)

        xt = xt.squeeze()
        time_mult = fm.get('time_multiplier', 1)
        x0_cls_time = x0_cls_time.unsqueeze(1) * time_mult
        t = t.unsqueeze(1) * time_mult
        tau = torch.zeros_like(t)
        t1_time = torch.zeros_like(tau)
        age = age.repeat(xt.shape[0], 1)
        sex = sex.repeat(xt.shape[0], 1)
        in_tensor = torch.cat([xt, x0_classes, x0_cls_time, sex, age, label, t1_time, tau, t], dim=-1)
        target = x1 if fm.data_parametrization else ut

        if self.noise_prediction:
            flow_out, score_out, noise_out = self(in_tensor)
            if self.implementation == "SDE":
                if fm.uncertainty_method == 'heteroscedastic':
                    log_var = noise_out[:, :self.flow_module.dim]
                    loss = 0.5 * torch.mean(log_var + (target - flow_out[:, :self.flow_module.dim])**2 / torch.exp(log_var))
                    loss += 0.01 * torch.mean(torch.exp(log_var))
                else:
                    loss_flow = self.criterion(flow_out, target)
                    loss_score = torch.mean((score_out * sigma_t + curr_noise)**2)

                    n_steps = 4
                    step_size = (x1_time - t) / n_steps
                    current_x, current_t = xt.clone(), t.clone()
                    with torch.no_grad():
                        for _ in range(n_steps):
                            step_input = torch.cat([current_x, x0_classes, x0_cls_time, sex, age, label,
                                                    torch.zeros_like(t), torch.zeros_like(t), current_t], dim=-1)
                            velocity, score, _ = self(step_input)
                            current_x = current_x + step_size * (velocity + score * self.s_diffusion.to(self.device) / 2)
                            current_t = current_t + (x1_time - t) / n_steps

                    uncertainty = (current_x - x1)**2
                    loss_noise = self.criterion(noise_out, uncertainty).clip(0, 1e3)
                    loss = loss_flow + 0.001 * loss_noise + 0.1 * loss_score
            else:
                loss_flow = self.criterion(flow_out[:, :self.flow_module.dim], target)
                uncertainty = torch.abs(flow_out[:, :self.flow_module.dim] - target)
                loss_noise = self.criterion(noise_out[:, :self.flow_module.dim], uncertainty)
                loss = loss_flow + loss_noise
        else:
            flow_out = self(in_tensor)
            if self.implementation == "SDE":
                variance = torch.sqrt(sigma_t) * self.flow_module.sde_noise
                noise = torch.randn_like(flow_out[:, :self.flow_module.dim]) * torch.sqrt(variance)
                loss = self.criterion(flow_out[:, :self.flow_module.dim] + noise, target)
            else:
                loss = self.criterion(flow_out[:, :self.flow_module.dim], target)

        self.log("validation_velocity", self.criterion(flow_out[:, :self.flow_module.dim], target), prog_bar=False, on_epoch=True)
        self.log("validation_loss", loss, prog_bar=True, on_epoch=True)
        self.log('validation_score_loss', loss_score, prog_bar=False, on_epoch=True)
        self.log('validation_uncertainty_loss', loss_noise, prog_bar=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.flow_module.parameters(), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, self.config.training.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def process_batch_base(self, batch):
        imgs = batch['images'].float().to(self.device)
        times = batch['times'].float().to(self.device)
        label = batch['class'].to(self.device) if 'class' in batch else torch.zeros(imgs.shape[0], 4).to(self.device)

        self.model_ae.to(self.device)
        if len(imgs.shape) == 3:
            imgs_z = self.model_ae.encoder(imgs)[0].view(imgs.shape[0], -1)
        else:
            imgs_z = torch.stack([
                self.model_ae.encoder(imgs[:, b].float())[0].view(imgs.shape[0], -1)
                for b in range(imgs.shape[1])
            ], dim=1)
        return imgs_z, times, label

    def train_process_batch(self, batch):
        fm = self.config.flow_matching
        imgs_z, times, label = self.process_batch_base(batch)

        age = torch.tensor([0, 0], device=self.device, dtype=torch.float32)
        sex = torch.tensor([0, 0, 0, 0], device=self.device, dtype=torch.float32)

        if not fm.class_conditional:
            label = torch.zeros_like(label)

        if fm.max_time is not None:
            denom = (torch.tensor(fm.max_time).view(-1, 1).to(self.device) - torch.min(times, dim=1)[0].unsqueeze(1)).clamp_min(1e-8)
            times = times / denom
        else:
            denom = torch.max(times, dim=1)[0].unsqueeze(1).clamp_min(1e-8)
            times = times / denom

        t, xt, ut, _, _, sigma_t, score, curr_noise = self.FM.sample_location_and_conditional_flow(
            xs=imgs_z.squeeze(), timepoints=times.squeeze())
        ut = ut.squeeze()
        sigma_t = torch.tensor(sigma_t, device=self.device, dtype=torch.float32).squeeze(-1)
        curr_noise = torch.tensor(curr_noise, device=self.device, dtype=torch.float32)

        indices = []
        for i in range(t.shape[0]):
            mask = times[i] <= t[i]
            indices.append(torch.max(torch.where(mask)[0]).item() if mask.any() else 0)
        idx = torch.tensor(indices, device=t.device)

        rand_indices = torch.tensor([torch.randint(0, idx[i].item() + 1, (1,)).item() for i in range(idx.shape[0])], device=idx.device)
        x0_classes = torch.stack([imgs_z[i, rand_indices[i]] for i in range(imgs_z.shape[0])])
        x0_cls_time = torch.stack([times[i, rand_indices[i]] for i in range(imgs_z.shape[0])])
        x1 = torch.stack([imgs_z[i, idx[i]+1] for i in range(imgs_z.shape[0])]).unsqueeze(1)
        x1_time = torch.stack([times[i, idx[i]+1] for i in range(imgs_z.shape[0])]).unsqueeze(1)

        return sigma_t, xt, t, ut, x0_classes, x0_cls_time, label, x1, x1_time, age, sex, curr_noise

    def _process_test_sequence(self, imgs_z, times, label, lookback):
        fm = self.config.flow_matching
        x0_values, x0_classes, x1_values = [], [], []
        times_x0, times_x1, x0_cls_time = [], [], []

        for i in range(lookback, imgs_z.shape[1] - 1):
            x0_values.append(imgs_z[:, i])
            times_x0.append(times[:, i] / self.MAX_TIME)
            times_x1.append(times[:, i + 1] / self.MAX_TIME)
            x1_values.append(imgs_z[:, i + 1])
            x0_classes.append(imgs_z[:, i - lookback:i])
            x0_cls_time.append(times[:, i - lookback:i] / self.MAX_TIME)

        processed = {
            'x0_values': torch.stack(x0_values).unsqueeze(0),
            'x0_classes': torch.stack(x0_classes).unsqueeze(0),
            'x1_values': torch.stack(x1_values).unsqueeze(0),
            'times_x0': torch.stack(times_x0).reshape(-1, 1),
            'times_x1': torch.stack(times_x1).reshape(-1, 1),
            'x0_cls_time': torch.stack(x0_cls_time).reshape(-1, lookback).float(),
            'label': label,
            'age': torch.tensor([0, 0, 0, 0], device=self.device, dtype=torch.float32).view(1, -1).repeat(imgs_z.shape[1], 1),
            'sex': torch.tensor([0, 0], device=self.device, dtype=torch.float32)
        }
        return tuple(v.to(self.device) for v in processed.values())

    def test_process_batch_one(self, batch):
        return self._process_test_sequence(*self.process_batch_base(batch), lookback=1)

    @torch.no_grad()
    def predict_trajectory(self, data_source, save_figs=True, n_samples=100,
                           sequential_traj=True, random_history=False, model_type='ode'):
        self.init_autoencoder()
        self.flow_module.eval()
        mse_loss_list = []

        for batch_idx, batch in tqdm(enumerate(data_source)):
            processed_batch = self.test_process_batch_one(batch)
            x0_values, x0_classes, x1_values, times_x0, times_x1, x0_cls_time, label, age, sex = processed_batch
            times_x0 = times_x0.squeeze()
            times_x1 = times_x1.squeeze()

            full_traj = torch.cat([x0_values[0, 0, :self.flow_module.dim].unsqueeze(0), x1_values[0, :, :self.flow_module.dim]], dim=0)

            if self.implementation == "ODE":
                ind_loss, pred_traj, _, _ = test_trajectory_ode(processed_batch, self.flow_module, self.noise_prediction, sequential_traj=True, random_history=random_history)
            else:
                ind_loss, pred_traj, _, _ = test_trajectory_sde_new(processed_batch, self.flow_module, None, self.noise_prediction, sequential_traj=sequential_traj, random_history=random_history, run_unconditional=True, dim=256, model_type=model_type, uw=0.25)

            full_traj = full_traj.detach().cpu().numpy()
            pred_traj = pred_traj.detach().cpu().numpy()

            full_traj_img = self.model_ae.decoder(torch.tensor(full_traj).view(full_traj.shape[0], 256, 1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            pred_traj_img = self.model_ae.decoder(torch.tensor(pred_traj).view(pred_traj.shape[0], 256, 1, 1).float().to(self.device)).squeeze().detach().cpu().numpy()

            metricD = metrics_calculation(pred_traj_img, full_traj_img[1:], normalize=True)
            mse_loss_list.append(np.mean((full_traj_img - np.concatenate([full_traj_img[:2], pred_traj_img]))**2))

            if batch_idx >= n_samples:
                break

        print(f"MSE: {np.mean(mse_loss_list)} ± {np.std(mse_loss_list)}")


def main(config: ml_collections.ConfigDict):
    set_seeds(config.training.seed)
    pl.seed_everything(config.training.seed)

    fm = config.flow_matching
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        modes='sequence',
        n_sequences=config.dataset.get('n_sequences', 10),
        sequence_mode=fm.train_sampling
    )

    if config.hardware.gpus > 0 and torch.cuda.is_available():
        accelerator, devices = "gpu", config.hardware.gpus
    else:
        accelerator, devices = "cpu", "auto"

    if config.logging.enable:
        save_dir = os.path.join(ROOT, config.logging.log_dir)
        logger_name = f"{fm.velocity_net}_class_{fm.class_conditional}_dparam_{fm.data_parametrization}_{fm.implementation}_{fm.interpolation}_{fm.sigma_scheduler}"
        logger = pl.loggers.WandbLogger(project=config.logging.project_name, name=logger_name, config=config.to_dict(), save_dir=save_dir)
    else:
        logger = None

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='validation_loss', mode='min', every_n_epochs=1, save_last=True)
    callbacks = [checkpoint_callback]
    if config.logging.enable:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))

    trainer = pl.Trainer(logger=logger, max_epochs=config.training.epochs, callbacks=callbacks, gradient_clip_val=0.5, accelerator=accelerator, devices=devices, enable_progress_bar=config.training.enable_progress_bar)

    if config.checkpoint.get('test_ckpt') is None:
        model = FlowLightningModule(config=config, noise_prediction=fm.noise_prediction, implementation=fm.implementation)
        trainer.fit(model, train_loader, val_loader, ckpt_path=config.checkpoint.get('resume_ckpt'))
    else:
        model = FlowLightningModule.load_from_checkpoint(config.checkpoint.test_ckpt, config=config, implementation=fm.implementation, noise_prediction=fm.noise_prediction)
        model.predict_trajectory(test_loader, save_figs=False, n_samples=5000, sequential_traj=True, random_history=False)


if __name__ == "__main__":
    parser = get_arg_parser(default_config_path=DEFAULT_CONFIG)
    args, unknown = parser.parse_known_args()

    config = load_config(args.config)
    print(f"📄 Loaded config: {args.config}")

    if unknown:
        print(f"⚙️  Applying {len(unknown)} CLI overrides...")
        overrides = parse_simple_overrides(unknown, config)
        config_dict = config.to_dict()
        deep_update(config_dict, overrides)
        config = ml_collections.ConfigDict(config_dict)

    print_config(config, title="Starmen Flow Matching")
    main(config)
