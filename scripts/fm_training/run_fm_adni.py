import os
import sys
from pathlib import Path

ROOT_P = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT_P)

import monai
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from tqdm import tqdm

from utils.config_loader import *
import ml_collections

from src.models.autoencoder.unet import UNetModel
from src.models.flow_matching.multi_marginal_fm import MultiMarginalFlowMatcher
from datasets.adni import get_class_name_from_one_hot
from datasets.adni_seq import create_uniform_length_dataloaders
from src.models.flow_matching.tfm import MLP_Cond_Memory_Module, test_trajectory_sde_new
from src.models.flow_matching.components.fm_utils import mse_loss, l1_loss, metrics_calculation

from utils.general_utils import *
from utils.metrics import *



class ADNIFlowLightningModule(pl.LightningModule):
    
    def __init__(self, config: ml_collections.ConfigDict, noise_prediction=False, implementation="ODE"):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config
        self.noise_prediction = noise_prediction
        self.implementation = implementation
        self.init_autoencoder()
        
        fm = config.flow_matching
        
        # Initialize flow matching module
        self.flow_module = MLP_Cond_Memory_Module(
            treatment_cond=fm.treatment_cond,
            memory=fm.memory,
            dim=fm.dim,
            w=fm.w,
            time_varying=True,
            conditional=False,
            lr=config.training.lr,
            sigma=0.1,
            loss_fn=mse_loss if fm.loss_fn == "mse" else l1_loss,
            metrics=['mse_loss'],
            implementation=self.implementation,
            sde_noise=fm.sde_noise,
            clip=fm.clip,
            depth=fm.depth,
            velocity_net=fm.velocity_net,
            time_prediction=fm.time_prediction,
            class_conditional=fm.class_conditional,
            noise_prediction=self.noise_prediction
        )
        
        # Set time scaling
        if fm.learned_time_scaling:
            self.time_scaling = self.flow_module.time_scaling
        else:
            self.time_scaling = None

        # Initialize flow matcher
        self.s_diffusion = torch.tensor(fm.s_diffusion)
        self.FM = MultiMarginalFlowMatcher(
            sigma=fm.sigma_scheduler,
            interpolation=fm.interpolation,
            mix_coeff=fm.deriv_mix_coeff,
            time_scaler=self.time_scaling,
            md=True
        )
        
        self.criterion = mse_loss if fm.loss_fn == "mse" else l1_loss
        self.MAX_TIME = fm.max_time
        self.segmentor = None

    def init_autoencoder(self):
        """Initialize the autoencoder model."""
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
      
        w_path = ROOT_P + "/logs/AE-ADNI/9krvatgt/checkpoints/last.ckpt"
        checkpoint = torch.load(w_path, map_location='cpu', weights_only=False)

        new_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            new_key = key.replace('net.', '')
            new_state_dict[new_key] = value

        self.model_ae.load_state_dict(new_state_dict)
        print(f'Autoencoder loaded successfully from {w_path}')

    def forward(self, in_tensor):
        if self.noise_prediction:
            mu, score, g = self.flow_module.forward_train(in_tensor)
            return mu, score, g
        else:
            flow_out = self.flow_module.forward_train(in_tensor)
            return flow_out

    def training_step(self, batch, batch_idx):
        fm = self.config.flow_matching
        (sigma_t, xt, t, ut, x0_classes, x0_cls_time, label, x1, x1_time, x0, age, sex, curr_noise
         ) = self.train_process_batch(batch)
        
        xt = xt.squeeze()
        x0_class = x0_classes
        x0_cls_time = x0_cls_time.unsqueeze(1)
        t = t.unsqueeze(1)
        tau = torch.zeros_like(t)
        t1_time = torch.zeros_like(tau)
        x1 = x1.squeeze()
        in_tensor = torch.cat([xt, x0_class, x0_cls_time, sex, age, label, t1_time, tau, t], dim=-1)
        
        target = x1 if fm.data_parametrization else ut
        
        if self.noise_prediction:
            flow_out, score_out, noise_out = self(in_tensor)
            if self.implementation == "SDE":
                if fm.uncertainty_method == 'heteroscedastic':
                    log_var = noise_out[:, :self.flow_module.dim]
                    loss = 0.5 * torch.mean(
                        log_var + (target - flow_out[:, :self.flow_module.dim])**2 / torch.exp(log_var)
                    )
                    loss += 0.01 * torch.mean(torch.exp(log_var))
                    self.log("predicted_uncertainty", torch.mean(torch.exp(log_var)), prog_bar=False, on_epoch=True)
                else:
                    loss_flow = self.criterion(flow_out, target)
                    loss_score = torch.mean((score_out * sigma_t + curr_noise)**2)
                    
                    n_steps = 4
                    step_size = self.time_scaling((x1_time - t)).detach() / n_steps
                    current_x = xt.clone()
                    current_t = t.clone()

                    with torch.no_grad():
                        for i in range(n_steps):
                            step_input = torch.cat([
                                current_x, x0_classes, x0_cls_time, sex, age, label,
                                torch.zeros_like(t), torch.zeros_like(t), current_t
                            ], dim=-1)
                            velocity, score, _ = self(step_input)
                            noise = torch.randn_like(score) * torch.sqrt(self.s_diffusion.to(self.device)) * torch.sqrt(step_size)
                            current_x = current_x + step_size * (velocity + score * (self.s_diffusion.to(self.device)) / 2)
                            current_t = current_t + (x1_time - t) / n_steps

                    predicted_x1 = current_x
                    uncertainty = (predicted_x1 - x1)**2
                    loss_noise = self.criterion(noise_out, uncertainty).clip(0, 2e3)
                    loss = loss_flow + 0.01 * loss_noise + 0.1 * loss_score
            else:
                if fm.uncertainty_method == 'heteroscedastic':
                    log_var = noise_out[:, :self.flow_module.dim]
                    loss = 0.5 * torch.mean(log_var + (target - flow_out[:, :self.flow_module.dim])**2 / torch.exp(log_var))
                    loss += 0.01 * torch.mean(torch.exp(log_var))
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
                if fm.get('combined_loss', False):
                    loss = self.criterion(flow_out[:, :self.flow_module.dim], target) + self.criterion(flow_out[:, :self.flow_module.dim] + x0, x1)
                else:
                    loss = self.criterion(flow_out[:, :self.flow_module.dim], target)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log('train_score_loss', loss_score, prog_bar=False, on_epoch=True)
        self.log('train_uncentainty_loss', loss_noise, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        fm = self.config.flow_matching
        sigma_t, xt, t, ut, x0_classes, x0_cls_time, label, x1, x1_time, x0, age, sex, curr_noise = self.train_process_batch(batch)
        
        xt = xt.squeeze()
        x0_class = x0_classes
        x0_cls_time = x0_cls_time.unsqueeze(1)
        t = t.unsqueeze(1)
        tau = torch.zeros_like(t)
        t1_time = torch.zeros_like(tau)
        x1 = x1.squeeze()

        in_tensor = torch.cat([xt, x0_class, x0_cls_time, sex, age, label, t1_time, tau, t], dim=-1)
        target = x1 if fm.data_parametrization else ut
        
        if self.noise_prediction:
            flow_out, score_out, noise_out = self(in_tensor)
            if self.implementation == "SDE":
                if fm.uncertainty_method == 'heteroscedastic':
                    log_var = noise_out[:, :self.flow_module.dim]
                    loss = 0.5 * torch.mean(
                        log_var + (target - flow_out[:, :self.flow_module.dim])**2 / torch.exp(log_var)
                    )
                    loss += 0.01 * torch.mean(torch.exp(log_var))
                else:
                    loss_flow = self.criterion(flow_out, target)
                    loss_score = torch.mean((score_out * sigma_t + curr_noise)**2)
                    
                    n_steps = 4
                    if fm.learned_time_scaling:
                        step_size = self.time_scaling((x1_time - t)).detach() / n_steps
                    else:
                        step_size = (x1_time - t) / n_steps

                    current_x = xt.clone()
                    current_t = t.clone()

                    with torch.no_grad():
                        for i in range(n_steps):
                            step_input = torch.cat([
                                current_x, x0_classes, x0_cls_time, sex, age, label,
                                torch.zeros_like(t), torch.zeros_like(t), current_t
                            ], dim=-1)
                            velocity, score, _ = self(step_input)
                            noise = torch.randn_like(score) * torch.sqrt(self.s_diffusion.to(self.device)) * torch.sqrt(step_size)
                            current_x = current_x + step_size * (velocity + score * (self.s_diffusion.to(self.device)) / 2)
                            current_t = current_t + (x1_time - t) / n_steps

                    predicted_x1 = current_x
                    uncertainty = (predicted_x1 - x1)**2
                    loss_noise = self.criterion(noise_out, uncertainty).clip(0, 2e3)
                    loss = loss_flow + 0.01 * loss_noise + 0.1 * loss_score
            else:
                if fm.uncertainty_method == 'heteroscedastic':
                    log_var = noise_out[:, :self.flow_module.dim]
                    loss = 0.5 * torch.mean(log_var + (target - flow_out[:, :self.flow_module.dim])**2 / torch.exp(log_var))
                    loss += 0.01 * torch.mean(torch.exp(log_var))
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
                if fm.get('combined_loss', False):
                    vfm_com_loss = (
                        self.criterion(flow_out[:, :self.flow_module.dim], target) +
                        self.criterion(flow_out[:, :self.flow_module.dim] + x0, x1)
                    )
                    self.log("validation_combo_loss", vfm_com_loss, prog_bar=False, on_epoch=True)
                    loss = self.criterion(flow_out[:, :self.flow_module.dim], target)
                else:
                    loss = self.criterion(flow_out[:, :self.flow_module.dim], target)

        self.log("validation_velocity", self.criterion(flow_out[:, :self.flow_module.dim], target), prog_bar=False, on_epoch=True)
        self.log("validation_loss", loss, prog_bar=True, on_epoch=True)
        self.log('validation_score_loss', loss_score, prog_bar=False, on_epoch=True)
        self.log('validation_uncentainty_loss', loss_noise, prog_bar=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.flow_module.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay
        )
        scheduler = CosineWarmupScheduler(optimizer, self.config.training.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def process_batch_base(self, batch):
        imgs = batch['images'].float().to(self.device)
        times = batch['times'].float().to(self.device)
        label = batch['class'].to(self.device)
        age = batch['age'].to(self.device)
        sex = batch['sex'].to(self.device)

        self.model_ae.to(self.device)

        if len(imgs.shape) == 3:
            imgs_z = self.model_ae.encoder(imgs)[0].view(imgs.shape[0], -1)
        else:
            imgs_z = torch.stack([
                self.model_ae.encoder(imgs[:, b].float())[0].view(imgs.shape[0], -1)
                for b in range(imgs.shape[1])
            ], dim=1)

        return imgs_z, times, label, age, sex

    def _process_test_sequence(self, imgs_z, times, label, ages, sex, lookback):
        fm = self.config.flow_matching
        x0_values, x0_classes, x1_values = [], [], []
        times_x0, times_x1, x0_cls_time = [], [], []
        age_x0 = []
        if fm.max_time is not None:
            times = times / (fm.max_time.view(-1, 1).to(self.device) - torch.min(times, dim=1)[0].unsqueeze(1))
        else:
            times = times / torch.max(times, dim=1)[0].unsqueeze(1)

        for i in range(lookback, imgs_z.shape[1] - 1):
            x0_values.append(imgs_z[:, i])
            times_x0.append(times[:, i])
            times_x1.append(times[:, i + 1])
            age_x0.append(ages[:, i])
            x1_values.append(imgs_z[:, i + 1])
            x0_classes.append(imgs_z[:, i - lookback:i])
            x0_cls_time.append(times[:, i - lookback:i])

        processed = {
            'x0_values': torch.stack(x0_values).unsqueeze(0),
            'x0_classes': torch.stack(x0_classes).unsqueeze(0),
            'x1_values': torch.stack(x1_values).unsqueeze(0),
            'times_x0': torch.stack(times_x0).reshape(-1, 1),
            'times_x1': torch.stack(times_x1).reshape(-1, 1),
            'x0_cls_time': torch.stack(x0_cls_time).reshape(-1, lookback).float(),
            'label': label,
            'age': torch.stack(age_x0).reshape(-1, lookback).float(),
            'sex': sex
        }
        return tuple(v.to(self.device) for v in processed.values())

    def train_process_batch(self, batch):
        fm = self.config.flow_matching
        imgs_z, times, label, age, sex = self.process_batch_base(batch)

        if not fm.demographic_cond:
            age = torch.zeros_like(age)
            sex = torch.zeros_like(sex)
        if not fm.class_conditional:
            label = torch.zeros_like(label)

        if fm.max_time is not None:
            times = times / (fm.max_time.view(-1, 1) - torch.min(times, dim=1)[0].unsqueeze(1))
        else:
            times = times / torch.max(times, dim=1)[0].unsqueeze(1)

        if imgs_z.dim() == 4:
            tmp_imgs_z = imgs_z.squeeze()
            tmp_times = times.squeeze()
        else:
            tmp_imgs_z = imgs_z
            tmp_times = times

        t, xt, ut, _, _, sigma_t, score, curr_noise = self.FM.sample_location_and_conditional_flow(
            xs=tmp_imgs_z, timepoints=tmp_times)

        ut = ut.squeeze()
        sigma_t = torch.tensor(sigma_t).squeeze(-1)
        indices = []
        for i in range(t.shape[0]):
            time_seq = times[i]
            t_val = t[i]
            mask = time_seq <= t_val
            if not mask.any():
                interval_idx = 0
            else:
                interval_idx = torch.max(torch.where(mask)[0]).item()
            indices.append(interval_idx)

        idx = torch.tensor(indices, device=t.device)

        rand_indices = torch.tensor([
            torch.randint(0, idx[i].item() + 1, (1,)).item()
            for i in range(idx.shape[0])
        ], device=idx.device)

        x0_classes = torch.stack([imgs_z[i, rand_indices[i]] for i in range(imgs_z.shape[0])])
        x0_cls_time = torch.stack([times[i, rand_indices[i]] for i in range(imgs_z.shape[0])])
        x0 = torch.stack([imgs_z[i, idx[i]] for i in range(imgs_z.shape[0])])
        x1 = torch.stack([imgs_z[i, idx[i] + 1] for i in range(imgs_z.shape[0])]).unsqueeze(1)
        x1_time = torch.stack([times[i, idx[i] + 1] for i in range(imgs_z.shape[0])]).unsqueeze(1)
        age = torch.stack([age[i, idx[i]] for i in range(age.shape[0])]).unsqueeze(1)

        return sigma_t, xt, t, ut, x0_classes, x0_cls_time, label, x1, x1_time, x0, age, sex, curr_noise

    def test_process_batch(self, batch):
        return self._process_test_sequence(*self.process_batch_base(batch), lookback=2)

    def test_process_batch_one(self, batch):
        return self._process_test_sequence(*self.process_batch_base(batch), lookback=1)

    def predict_trajectory(self, test_loader, save_figs=True, selected_class=None,
                           n_samples=None, random_history=False, sequential_traj=True,
                           run_unconditional=True, interp='linear_sde',
                           model_type='ode', uw=0.1, write_csv=False):
        """Predict trajectories for test data and calculate metrics."""
        self.init_autoencoder()
        self.flow_module.eval()

        dict_full_trajs, dict_pred_trajs = {}, {}
        dict_full_trajs_seg, dict_pred_trajs_seg = {}, {}
        dict_classes, data_dict = {}, {}
        list_of_times = []
        p_psnr, p_ssim = [], []
        p_dice, g_dice, rel_dice, p_hd, g_hd = [], [], [], [], []
        loss_sum = 0
        mse_loss_list = []
        s_idx = 0
        with torch.no_grad():
            for batch_idx, batch_ori in tqdm(enumerate(test_loader)):
                if selected_class is not None:
                    if not torch.equal(batch_ori['class'][0], selected_class):
                        continue
                if n_samples is not None:
                    if s_idx > n_samples:
                        break
                s_idx += 1
                batch = self.test_process_batch_one(batch_ori)
                x0_values, x0_classes, x1_values, times_x0, times_x1, x0_cls_time, label, x0_age, sex = batch

                times_x0 = times_x0.squeeze()
                times_x1 = times_x1.squeeze()

                full_traj = torch.cat([
                    x0_values[0, 0, :self.flow_module.dim].unsqueeze(0),
                    x1_values[0, :, :self.flow_module.dim]
                ], dim=0)
                full_time = torch.cat([times_x0[0].unsqueeze(0), times_x1], dim=0)

                ind_loss, pred_traj, noise_mse, noise_pred = test_trajectory_sde_new(
                    batch, self.flow_module, None, self.noise_prediction,
                    sequential_traj=sequential_traj, random_history=random_history,
                    run_unconditional=run_unconditional, model_type=model_type, uw=uw
                )

                full_traj = full_traj.detach().cpu().numpy()
                pred_traj = pred_traj.detach().cpu().numpy()
                full_time = full_time.detach().cpu().numpy()

                full_traj_img = self.model_ae.decoder(
                    torch.tensor(full_traj).view(full_traj.shape[0], 256, 4, 4).to(self.device)
                ).squeeze().detach().cpu().numpy()
                pred_traj_img = self.model_ae.decoder(
                    torch.tensor(pred_traj).view(pred_traj.shape[0], 256, 4, 4).float().to(self.device)
                ).squeeze().detach().cpu().numpy()

                x0_classes = x0_classes.squeeze(0)[0]
                init_cond = self.model_ae.decoder(
                    torch.tensor(x0_classes).view(x0_classes.shape[0], 256, 4, 4).to(self.device)
                ).squeeze(1).detach().cpu().numpy()

                metricD = metrics_calculation(pred_traj_img, full_traj_img[1:], normalize=True)

                full_traj = np.concatenate([init_cond, full_traj_img], axis=0)
                offset = 2
                pred_traj = np.concatenate([full_traj[:offset], pred_traj_img], axis=0)

                loss_sum += ind_loss
                mse_loss_list.append(metricD['mse_loss'])
                list_of_times.append(full_time)
                dict_classes[batch_idx] = get_class_name_from_one_hot(label[0].detach().cpu())

                if self.config.segmentor.ckpt:
                    (pred_psnr, pred_ssim, seg_dice_xT, seg_dice_gt,
                     seg_hd_xT, seg_hd_gt, rel_dice_xT,
                     full_traj_seg, pred_traj_seg) = self.segment_trajectory(full_traj, pred_traj, offset)

                    if full_traj_seg[0].sum() > 0:
                        data_dict[batch_idx] = batch_ori
                        dict_full_trajs_seg[batch_idx] = full_traj_seg
                        dict_pred_trajs_seg[batch_idx] = pred_traj_seg
                        dict_full_trajs[batch_idx] = full_traj
                        dict_pred_trajs[batch_idx] = pred_traj
                        p_psnr.append(pred_psnr)
                        p_ssim.append(pred_ssim)
                        p_dice.append(seg_dice_xT)
                        g_dice.append(seg_dice_gt)
                        p_hd.append(seg_hd_xT)
                        g_hd.append(seg_hd_gt)
                        rel_dice.append(rel_dice_xT)

            print(f"PSNR: {np.mean(p_psnr)} SSIM: {np.mean(p_ssim)}")
            print(f"Segmentation Dice xT: {np.mean(p_dice)} GT: {np.mean(g_dice)}")
        return dict_full_trajs, dict_pred_trajs, list_of_times, dict_full_trajs_seg, dict_pred_trajs_seg, dict_classes, data_dict

    @torch.no_grad()
    def segment_trajectory(self, full_traj, pred_traj, offset=2):
        if self.segmentor is None:
            seg_path = self.config.segmentor.ckpt
            if seg_path and os.path.isfile(seg_path):
                self.segmentor = torch.nn.Sequential(
                    monai.networks.nets.DynUNet(
                        spatial_dims=2, in_channels=1, out_channels=1,
                        kernel_size=[3, 3, 3, 3, 3], filters=[16, 32, 64, 128, 256],
                        strides=[1, 2, 2, 2, 1], upsample_kernel_size=[1, 2, 2, 2, 1]),
                    torch.nn.Sigmoid()).to(self.device)
                self.segmentor.load_state_dict(torch.load(seg_path, map_location=self.device))
                self.segmentor.eval()
            else:
                self.segmentor = torch.nn.Identity()

        for i in range(full_traj.shape[0]):
            full_traj[i] = (full_traj[i] - np.min(full_traj[i])) / (np.max(full_traj[i]) - np.min(full_traj[i]))
        for i in range(pred_traj.shape[0]):
            pred_traj[i] = (pred_traj[i] - np.min(pred_traj[i])) / (np.max(pred_traj[i]) - np.min(pred_traj[i]))

        full_traj_seg = to_np(self.segmentor(torch.from_numpy(full_traj).unsqueeze(1).to(self.device)) > 0.5)
        pred_traj_seg = to_np(self.segmentor(torch.from_numpy(pred_traj).unsqueeze(1).to(self.device)) > 0.5)

        pred_psnr = pred_ssim = seg_dice_xT = seg_dice_gt = seg_hd_xT = seg_hd_gt = rel_seg_dice_gt = 0
        if full_traj_seg.sum() == 0 and pred_traj_seg.sum() == 0:
            return 0, 0, 0, 0, 0, 0, 0, full_traj_seg, pred_traj_seg

        for i in range(offset, full_traj_seg.shape[0]):
            pred_psnr += psnr(full_traj[i], pred_traj[i])
            pred_ssim += ssim(full_traj[i], pred_traj[i])
            seg_dice_xT += dice_coeff(full_traj_seg[i].squeeze(), pred_traj_seg[i].squeeze())
            rel_seg_dice_gt += dice_coeff(full_traj_seg[i].squeeze(), full_traj_seg[1].squeeze())
            seg_hd_xT += hausdorff(full_traj_seg[i].squeeze(), pred_traj_seg[i].squeeze())

        n = full_traj_seg.shape[0] - offset
        return pred_psnr/n, pred_ssim/n, seg_dice_xT/n, seg_dice_gt/n, seg_hd_xT/n, seg_hd_gt/n, rel_seg_dice_gt/n, full_traj_seg, pred_traj_seg


def main(config: ml_collections.ConfigDict):
    """Main training and evaluation function."""
    set_seeds(config.training.seed)
    pl.seed_everything(config.training.seed)

    fm = config.flow_matching
    if fm.noise_prediction and fm.velocity_net != 'vrf':
        raise ValueError("Noise prediction is only supported for 'vrf' velocity network.")

    train_loader, val_loader, test_loader = create_uniform_length_dataloaders(
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        include_classes=['AD', 'CN'],
        view=config.dataset.get('view', 'axial')
    )

    # Hardware
    if config.hardware.gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = config.hardware.gpus
    else:
        accelerator = "cpu"
        devices = "auto"

    # Logger
    if config.logging.enable:
        save_dir = os.path.join(ROOT_P, config.logging.log_dir)
        logger_name = (
            f"{fm.velocity_net}_class_{fm.class_conditional}_"
            f"dparam_{fm.data_parametrization}_{fm.implementation}_"
            f"{fm.interpolation}_{config.dataset.get('view','axial')}_"
            f"demoCond_{fm.demographic_cond}_{fm.sigma_scheduler}_"
            f"uncertainty_{fm.uncertainty_method}"
        )
        logger = pl.loggers.WandbLogger(
            project=config.logging.project_name,
            name=logger_name,
            config=config.to_dict(),
            save_dir=save_dir
        )
    else:
        logger = None

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='validation_velocity', mode='min', every_n_epochs=1, save_last=True
    )
    callbacks = [checkpoint_callback]
    if config.logging.enable:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=config.training.enable_progress_bar
    )

    # Train or evaluate
    if config.checkpoint.get('test_ckpt') is None:
        model = ADNIFlowLightningModule(
            config=config,
            noise_prediction=fm.noise_prediction,
            implementation=fm.implementation
        )
        trainer.fit(model, train_loader, val_loader, ckpt_path=config.checkpoint.get('resume_ckpt'))
    else:
        model = ADNIFlowLightningModule.load_from_checkpoint(
            config.checkpoint.test_ckpt,
            config=config,
            implementation=fm.implementation,
            noise_prediction=fm.noise_prediction
        )
        model.predict_trajectory(test_loader, save_figs=True)


if __name__ == "__main__":
    dp= os.path.join(ROOT_P, "configs/fm_adni.yaml")
    parser = get_arg_parser(default_config_path=dp)
    args, unknown = parser.parse_known_args()

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")

    if unknown:
        overrides = parse_simple_overrides(unknown, config)
        config_dict = config.to_dict()
        deep_update(config_dict, overrides)
        config = ml_collections.ConfigDict(config_dict)

    print_config(config, title="ADNI Flow Matching")
    main(config)
