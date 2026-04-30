"""Flow Matching training on Brain GBM dataset.

Usage:
    python run_fm_mri.py --config configs/fm_gbm.yaml
    python run_fm_mri.py --config configs/fm_gbm.yaml --lr 5e-5
"""
import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)

import monai
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm

import ml_collections
from utils.config_loader import get_arg_parser, load_config, parse_simple_overrides, deep_update, print_config

from src.models.flow_matching.multi_marginal_fm import MultiMarginalFlowMatcher
from src.models.autoencoder.unet import UNetModel
from src.models.flow_matching.tfm import (
    MLP_Cond_Memory_Module, l1_loss, metrics_calculation, mse_loss,
    test_trajectory_ode, test_trajectory_sde_new
)
from utils.general_utils import *
from utils.metrics import *
from utils.prepare_dataset import prepare_dataset_npt

DEFAULT_CONFIG = os.path.join(ROOT, "configs/fm_gbm.yaml")


class MRIFlowLightningModule(pl.LightningModule):

    def __init__(self, config: ml_collections.ConfigDict, noise_prediction=True, implementation="ODE"):
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
            demo_conditional=fm.demographic_cond,
            noise_prediction=self.noise_prediction,
            time_scaling_min_clip=1e-4,
            gbm=False
        )

        self.criterion = mse_loss if fm.loss_fn == "mse" else l1_loss
        self.MAX_TIME = fm.max_time
        self.segmentor = None

        self.time_scaling = self.flow_module.time_scaling
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

        w_path = ROOT + "/logs/AE-ISOMAP/gzsljqej/checkpoints/last.ckpt"
        checkpoint = torch.load(w_path, map_location='cpu', weights_only=False)
        new_state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model_ae.load_state_dict(new_state_dict)
        print(f"Autoencoder loaded from {w_path}")

    def forward(self, in_tensor):
        if self.noise_prediction:
            return self.flow_module.forward_train(in_tensor)
        else:
            return self.flow_module.forward_train(in_tensor)

    def training_step(self, batch, batch_idx):
        fm = self.config.flow_matching
        sigma_t, xt, t, ut, x0_classes, x0_cls_time, label, x1, x1_time, age, sex, curr_noise = self.train_process_batch(batch)

        xt = xt.squeeze()
        x0_cls_time = x0_cls_time.unsqueeze(1)
        t = t.unsqueeze(1)
        tau = torch.zeros_like(t)
        t1_time = torch.zeros_like(tau)
        if fm.demographic_cond:
            in_tensor = torch.cat([xt, x0_classes, x0_cls_time, sex, age, label, t1_time, tau, t], dim=-1)
        else:
            in_tensor = torch.cat([xt, x0_classes, x0_cls_time, label, t1_time, tau, t], dim=-1)

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
                    step_size = self.time_scaling((x1_time - t)).detach() / n_steps
                    current_x, current_t = xt.clone(), t.clone()
                    with torch.no_grad():
                        for _ in range(n_steps):
                            step_input = torch.cat([current_x, x0_classes, x0_cls_time, sex, age, label,
                                                    torch.zeros_like(t), torch.zeros_like(t), current_t], dim=-1)
                            velocity, score, _ = self(step_input)
                            current_x = current_x + step_size * (velocity + score * self.s_diffusion.to(self.device) / 2)
                            current_t = current_t + (x1_time - t) / n_steps

                    uncertainty = (current_x - x1)**2
                    loss_noise = self.criterion(noise_out, uncertainty).clip(0, 2e3)
                    loss = loss_flow + 0.01 * loss_noise + 0.1 * loss_score
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
        self.log('train_uncentainty_loss', loss_noise, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        fm = self.config.flow_matching
        sigma_t, xt, t, ut, x0_classes, x0_cls_time, label, x1, x1_time, age, sex, curr_noise = self.train_process_batch(batch)

        xt = xt.squeeze()
        x0_cls_time = x0_cls_time.unsqueeze(1)
        t = t.unsqueeze(1)
        tau = torch.zeros_like(t)
        t1_time = torch.zeros_like(tau)
        if fm.demographic_cond:
            in_tensor = torch.cat([xt, x0_classes, x0_cls_time, sex, age, label, t1_time, tau, t], dim=-1)
        else:
            in_tensor = torch.cat([xt, x0_classes, x0_cls_time, label, t1_time, tau, t], dim=-1)

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
                    step_size = self.time_scaling((x1_time - t)).detach() / n_steps
                    current_x, current_t = xt.clone(), t.clone()
                    with torch.no_grad():
                        for _ in range(n_steps):
                            step_input = torch.cat([current_x, x0_classes, x0_cls_time, sex, age, label,
                                                    torch.zeros_like(t), torch.zeros_like(t), current_t], dim=-1)
                            velocity, score, _ = self(step_input)
                            current_x = current_x + step_size * (velocity + score * self.s_diffusion.to(self.device) / 2)
                            current_t = current_t + (x1_time - t) / n_steps

                    uncertainty = (current_x - x1)**2
                    loss_noise = self.criterion(noise_out, uncertainty).clip(0, 2e3)
                    loss = loss_flow + 0.01 * loss_noise + 0.1 * loss_score
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
        self.log('validation_uncentainty_loss', loss_noise, prog_bar=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.flow_module.parameters(), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, self.config.training.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def process_batch_base(self, batch, old_format=True):
        if old_format:
            imgs, times, _, _, label, age, sex = batch
        else:
            imgs, times, label, age, sex = batch['images'], batch['times'], batch['label'], batch['age'], batch['sex']
        sex = sex[:, 0, :]
        imgs = imgs.to(self.device).float()
        self.model_ae.to(self.device)

        if len(imgs.shape) == 3:
            imgs_z = self.model_ae.encoder(imgs)[0].view(imgs.shape[0], -1)
        else:
            imgs_z = torch.stack([
                self.model_ae.encoder(imgs[:, b].float())[0].view(imgs.shape[0], -1)
                for b in range(imgs.shape[1])
            ], dim=1)
        return imgs_z, times, label, age, sex

    def train_process_batch(self, batch, old_format=True):
        fm = self.config.flow_matching
        imgs_z, times, label, age, sex = self.process_batch_base(batch, old_format)

        if not fm.demographic_cond:
            age = torch.zeros_like(age)
            sex = torch.zeros_like(sex)
        if not fm.class_conditional:
            label = torch.zeros_like(label)

        times = (times - torch.min(times, dim=1)[0].unsqueeze(1))
        if fm.max_time is not None:
            times = times / (fm.max_time.view(-1, 1).to(self.device) - torch.min(times, dim=1)[0].unsqueeze(1))
        else:
            times = times / torch.max(times, dim=1)[0].unsqueeze(1)

        tmp_imgs_z = imgs_z.squeeze() if imgs_z.dim() == 4 else imgs_z
        tmp_times = times.squeeze() if imgs_z.dim() == 4 else times

        t, xt, ut, _, _, sigma_t, score, curr_noise = self.FM.sample_location_and_conditional_flow(
            xs=tmp_imgs_z, timepoints=tmp_times)
        ut = ut.squeeze()
        t = t.to(tmp_times.device)
        xt = xt.to(tmp_times.device)
        ut = ut.to(tmp_times.device)
        if isinstance(curr_noise, torch.Tensor):
            curr_noise = curr_noise.to(tmp_times.device)

        indices = []
        for i in range(t.shape[0]):
            mask = times[i] <= t[i]
            indices.append(torch.max(torch.where(mask)[0]).item() if mask.any() else 0)
        idx = torch.tensor(indices, device=t.device)

        rand_indices = torch.tensor([torch.randint(0, idx[i].item() + 1, (1,)).item() for i in range(idx.shape[0])], device=idx.device)
        x0_classes = torch.stack([imgs_z[i, idx[i]] for i in range(imgs_z.shape[0])])
        x0_cls_time = torch.stack([times[i, idx[i]] for i in range(imgs_z.shape[0])])
        label = torch.stack([label[i, idx[i]+1] for i in range(label.shape[0])])
        x1 = torch.stack([imgs_z[i, idx[i]+1] for i in range(imgs_z.shape[0])]).unsqueeze(1)
        x1_time = torch.stack([times[i, idx[i]+1] for i in range(imgs_z.shape[0])]).unsqueeze(1)

        return sigma_t, xt, t, ut, x0_classes, x0_cls_time, label, x1, x1_time, age, sex, curr_noise

    def _process_test_sequence(self, imgs_z, times, label, age, sex, lookback):
        fm = self.config.flow_matching
        x0_values, x0_classes, x1_values = [], [], []
        times_x0, times_x1, x0_cls_time, age_x0 = [], [], [], []

        if fm.max_time is not None:
            times = times / (fm.max_time.view(-1, 1).cpu() - torch.min(times, dim=1)[0].unsqueeze(1))
        else:
            times = times / torch.max(times, dim=1)[0].unsqueeze(1)

        for i in range(lookback, imgs_z.shape[1] - 1):
            x0_values.append(imgs_z[:, i])
            times_x0.append(times[:, i])
            times_x1.append(times[:, i + 1])
            x1_values.append(imgs_z[:, i + 1])
            age_x0.append(age[:, i])
            x0_classes.append(imgs_z[:, i - lookback:i])
            x0_cls_time.append(times[:, i - lookback:i])

        processed = {
            'x0_values': torch.stack(x0_values).unsqueeze(0),
            'x0_classes': torch.stack(x0_classes).unsqueeze(0),
            'x1_values': torch.stack(x1_values).unsqueeze(0),
            'times_x0': torch.from_numpy(np.stack(times_x0).reshape(-1, 1)),
            'times_x1': torch.from_numpy(np.stack(times_x1).reshape(-1, 1)),
            'x0_cls_time': torch.stack(x0_cls_time).reshape(-1, lookback).float(),
            'label': label,
            'age': torch.stack(age_x0).reshape(-1, lookback).float(),
            'sex': sex
        }
        return tuple(v.to(self.device) for v in processed.values())

    def test_process_batch_one(self, batch, old_format=True):
        return self._process_test_sequence(*self.process_batch_base(batch, old_format), lookback=1)

    @torch.no_grad()
    def predict_trajectory(self, test_loader, save_figs=True, n_sample=5,
                           random_history=False, sequential_traj=False,
                           run_unconditional=False, model_type='ode', uw=0.1):
        self.init_autoencoder()
        self.flow_module.eval()
        mse_loss_list = []

        for batch_idx, batch_ori in tqdm(enumerate(test_loader)):
            imgs, times, _, _, label, age, sex = batch_ori
            if imgs.shape[1] < 4:
                continue
            if batch_idx >= n_sample:
                break

            batch = self.test_process_batch_one(batch_ori)
            x0_values, x0_classes, x1_values, times_x0, times_x1, x0_cls_time, label, age, sex = batch
            times_x0 = times_x0.squeeze()
            times_x1 = times_x1.squeeze()

            full_traj = torch.cat([x0_values[0, 0, :self.flow_module.dim].unsqueeze(0), x1_values[0, :, :self.flow_module.dim]], dim=0)

            if self.implementation == "ODE":
                ind_loss, pred_traj, _, _ = test_trajectory_ode(batch, self.flow_module, self.noise_prediction, sequential_traj=sequential_traj, random_history=random_history)
            else:
                ind_loss, pred_traj, _, _ = test_trajectory_sde_new(batch, self.flow_module, self.time_scaling, self.noise_prediction, sequential_traj=sequential_traj, random_history=random_history, run_unconditional=run_unconditional, model_type=model_type, uw=uw)

            full_traj = to_np(full_traj)
            pred_traj = to_np(pred_traj)

            full_traj_img = to_np(self.model_ae.decoder(torch.tensor(full_traj).view(full_traj.shape[0], 256, 4, 4).to(self.device)).squeeze())
            pred_traj_img = to_np(self.model_ae.decoder(torch.tensor(pred_traj).view(pred_traj.shape[0], 256, 4, 4).float().to(self.device)).squeeze())

            metricD = metrics_calculation(pred_traj_img, full_traj_img[1:])
            mse_loss_list.append(metricD['mse_loss'])

        print(f"MSE: {np.mean(mse_loss_list)} ± {np.std(mse_loss_list)}")


def main(config: ml_collections.ConfigDict):
    set_seeds(config.training.seed)
    pl.seed_everything(config.training.seed)

    fm = config.flow_matching
    ds = config.dataset

    train_loader, val_loader, test_loader, _, _, _, _, _ = prepare_dataset_npt(
        config=config, batch_size=config.training.batch_size,
        train_seq_len=3, test_seq_len=6, sequence_mode=fm.train_sampling
    )

    if config.hardware.gpus > 0 and torch.cuda.is_available():
        accelerator, devices = "gpu", config.hardware.gpus
    else:
        accelerator, devices = "cpu", "auto"

    if config.logging.enable:
        save_dir = os.path.join(ROOT, config.logging.log_dir)
        logger_name = f"{fm.velocity_net}_class_{fm.class_conditional}_dparam_{fm.data_parametrization}_{fm.implementation}_{fm.interpolation}_demoCond_{fm.demographic_cond}_{fm.sigma_scheduler}"
        logger = pl.loggers.WandbLogger(project=config.logging.project_name, name=logger_name, config=config.to_dict(), save_dir=save_dir)
    else:
        logger = None

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='validation_velocity', mode='min', every_n_epochs=1, save_last=True)
    callbacks = [checkpoint_callback]
    if config.logging.enable:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))

    trainer = pl.Trainer(logger=logger, max_epochs=config.training.epochs, callbacks=callbacks, gradient_clip_val=0.5, accelerator=accelerator, devices=devices, enable_progress_bar=config.training.enable_progress_bar)

    if config.checkpoint.get('test_ckpt') is None:
        model = MRIFlowLightningModule(config=config, noise_prediction=fm.noise_prediction, implementation=fm.implementation)
        trainer.fit(model, train_loader, val_loader, ckpt_path=config.checkpoint.get('resume_ckpt'))
    else:
        model = MRIFlowLightningModule.load_from_checkpoint(config.checkpoint.test_ckpt, config=config, implementation=fm.implementation, noise_prediction=fm.noise_prediction)
        model.predict_trajectory(test_loader, save_figs=True, n_sample=10)


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

    print_config(config, title="GBM Flow Matching")
    main(config)
