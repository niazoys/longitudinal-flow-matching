import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(os.path.join(ROOT))

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
import ml_collections
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure

from datasets.ms import create_uniform_length_dataloaders
from src.models.autoencoder.unet import UNetModel
from utils.config_loader import (deep_update, get_arg_parser, load_config,
                                 parse_simple_overrides, print_config)
from utils.general_utils import CosineWarmupScheduler

torch.set_float32_matmul_precision('medium')
DEFAULT_CONFIG = os.path.join(ROOT, "configs/ae_ms.yaml")

class AE(pl.LightningModule):
    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config
        print("Using the Device:",self.device)
        m = config.model
        hw = config.hardware

        self.net = UNetModel(
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
            dae=m.use_disentangle,
            make_vae=m.make_vae
        )

        self.psnr = torchmetrics.PeakSignalNoiseRatio(data_range=2.0)

    def reconstruction_loss(self, y_pred, images):
        mse_loss = torch.nn.functional.mse_loss(y_pred, images)
        ssim_value = structural_similarity_index_measure(y_pred, images, data_range=2.0)
        ssim_loss = 1 - ssim_value
        loss = self.config.loss.w_mse * mse_loss + self.config.loss.w_ssim * ssim_loss
        return loss, mse_loss, ssim_value

    def forward_reconstruction(self, images):
        output = self.net(images)
        return output[0] if isinstance(output, (tuple, list)) else output

    def training_step(self, batch, batch_idx):
        images = batch['images']
        images = images.float().to(self.device)
        y_pred = self.forward_reconstruction(images)
        loss, mse_loss, ssim_value = self.reconstruction_loss(y_pred, images)

        self.psnr.update(y_pred, images)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train MSE", mse_loss, prog_bar=True)
        self.log("train SSIM", ssim_value, prog_bar=True)
        self.log("train PSNR", self.psnr.compute(), prog_bar=True)
        
        return loss

    def on_train_epoch_end(self):
        # Reset metrics
        self.psnr.reset()

    def validation_step(self, batch, batch_idx):
        self.net.eval()
        images = batch['images']
        images = images.float().to(self.device)
        y_pred = self.forward_reconstruction(images)
        val_loss, mse_loss, ssim_value = self.reconstruction_loss(y_pred, images)

        self.psnr.update(y_pred, images)
        self.log("validation_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("validation MSE", mse_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("validation SSIM", ssim_value, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("validation PSNR", self.psnr.compute(), prog_bar=True, on_epoch=True, sync_dist=True)

        self.log_img_idx = np.linspace(0, self.trainer.num_val_batches, 7).astype(int)
        if batch_idx in self.log_img_idx:
            wandb_image = wandb.Image(np.concatenate((images[0].squeeze(0).cpu().numpy(), y_pred[0].squeeze(0).cpu().numpy()), 
                                                     axis=1), caption="Left: Input, Right: Reconstruction")
            wandb.log({"examples: " + str(batch_idx): wandb_image})

    def on_validation_epoch_end(self):
        self.psnr.reset()

    def test_step(self, batch, batch_idx):
        self.net.eval()
        images = batch['images']
        images = images.float().to(self.device)

        y_pred = self.forward_reconstruction(images)
        test_loss, mse_loss, ssim_value = self.reconstruction_loss(y_pred, images)

        self.log("test_loss", test_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test MSE", mse_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test SSIM", ssim_value, prog_bar=True, on_epoch=True, sync_dist=True)

        self.psnr.update(y_pred, images)

        self.log_test_img_idx = np.linspace(0, self.trainer.num_test_batches, 7).astype(int)
        if batch_idx in self.log_test_img_idx:
            # Rescale images from [-1, 1] to [0, 1] for visualization
            images_vis = (images + 1) / 2
            y_pred_vis = (y_pred + 1) / 2
            wandb_image = wandb.Image(
                np.concatenate((images_vis[0].squeeze(0).cpu().numpy(), y_pred_vis[0].squeeze(0).cpu().numpy()), axis=1),
                caption="Left: Input, Right: Reconstruction"
            )
            wandb.log({"examples: " + str(batch_idx): wandb_image})

        # Save test images if required
        if self.config.checkpoint.get('save_results', False):
            image_path = os.path.join(self.config.checkpoint.get('save_dir', 'checkpoints'), "test_images")
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            for i in range(images.shape[0]):
                image = images[i].squeeze(0).cpu().numpy()
                pred = y_pred[i].squeeze(0).cpu().numpy()
                # Rescale images from [-1, 1] to [0, 255] for saving
                image = ((image + 1) / 2 * 255).astype(np.uint8)
                pred = ((pred + 1) / 2 * 255).astype(np.uint8)
                concatenated_image = np.concatenate((image, pred), axis=1)
                image = Image.fromarray(concatenated_image).convert('L')
                image_name = os.path.join(image_path, f"image_{batch_idx * self.config.training.batch_size + i}.png")
                image.save(image_name)

    def on_test_epoch_end(self):
        # Reset metrics
        self.psnr.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay
        )
        scheduler = CosineWarmupScheduler(optimizer, self.config.training.warmup_epochs, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main(config: ml_collections.ConfigDict):
    
    # Seed everything
    pl.seed_everything(config.training.seed)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_uniform_length_dataloaders(
        image_size=tuple(config.model.image_size),
        modes='autoencoder',
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )

    # Hardware settings
    if config.hardware.gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = config.hardware.gpus
    else:
        accelerator = "cpu"
        devices = "auto"
    if config.training.num_workers == -1:
        config.training.num_workers = os.cpu_count()

    # Logging settings
    if config.logging.enable:
        save_dir = os.path.join(ROOT, config.logging.log_dir)
        logger = pl.loggers.WandbLogger(
            project=config.logging.project_name,
            name=config.logging.exp_name,
            config=config.to_dict(),
            save_dir=save_dir
        )
    else:
        logger = None

    # Pytorch Lightning callbacks and trainer
    # Update the checkpoint callback to monitor 'validation_loss'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=config.checkpoint.get('monitor', 'validation_loss'),
        mode=config.checkpoint.get('mode', 'min'),
        every_n_epochs=1,
        save_last=True,
        save_top_k=config.checkpoint.get('save_top_k', 1)
    )
    callbacks = [checkpoint_callback]
    if config.logging.enable:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    
    trainer = pl.Trainer(logger=logger, max_epochs=config.training.epochs, callbacks=callbacks, gradient_clip_val=0.5,
                         accelerator=accelerator, devices=devices, enable_progress_bar=config.training.enable_progress_bar) #debug: overfit_batches=0.01

    # Do the training or testing
    if config.checkpoint.get('test_ckpt') is None:
        model = AE(config)
        trainer.fit(model, train_loader, val_loader, ckpt_path=config.checkpoint.get('resume_ckpt'))
        # trainer.test(model, val_loader, ckpt_path=checkpoint_callback.best_model_path)
    else:
        model = AE.load_from_checkpoint(config.checkpoint.test_ckpt, config=config)
        trainer.test(model, val_loader)



if __name__ == "__main__":
    parser = get_arg_parser(default_config_path=DEFAULT_CONFIG)
    args, unknown = parser.parse_known_args()

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")

    if unknown:
        overrides = parse_simple_overrides(unknown, config)
        config_dict = config.to_dict()
        deep_update(config_dict, overrides)
        config = ml_collections.ConfigDict(config_dict)

    print_config(config, title="MS Autoencoder")
    main(config)
