# ---------------------------------------------------------------
# Taken from the following link as is from:
# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_GUIDED_DIFFUSION).
# ---------------------------------------------------------------



import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    AttentionBlock,
    Downsample,
    MiddleBlock,
    ResBlock,
    Stretch,
    TimestepEmbedSequential,
    Upsample,
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    count_flops_attn
)




class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_time_embed=False,
        use_skip_connection=False,
        dae=False,
        make_vae=False, 
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_time_embed = use_time_embed
        self.use_skip_connection = use_skip_connection
        self.dae=dae
        self.make_vae=make_vae

        if self.dae:
            self.fc_hidden=4096
            self.latent_dim=int(self.fc_hidden)
            self.en_fc= nn.Linear(self.fc_hidden, self.latent_dim)
            # self.en_fc = nn.Sequential(nn.Linear(self.fc_hidden, self.fc_hidden),nn.Linear(self.fc_hidden, self.latent_dim))
            self.strecth = Stretch(self.latent_dim, 2, None)
            # self.de_fc= nn.Sequential(nn.Linear(self.latent_dim*2, self.fc_hidden),nn.SiLU(),nn.Linear(self.fc_hidden, self.fc_hidden),nn.SiLU())
            self.de_fc= nn.Sequential(nn.Linear(self.latent_dim*2, self.fc_hidden),nn.SiLU())
        
        



        ########## Time Emb Blocks #############
        ########################################
        if self.use_time_embed:
            ## Time Embedding
            time_embed_dim = model_channels * 4
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = model_channels

        ########## Class Emb Blocks ############
        ########################################
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ########## Input Blocks ################
        ########################################
        ch = input_ch = int(channel_mult[0] * model_channels)
        if self.use_time_embed:
            self.input_blocks = nn.ModuleList(
                [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
            )
        else:
            self.input_blocks = nn.ModuleList(
                [conv_nd(dims, in_channels, ch, 3, padding=1)]
            )

        ########## Encoder Blocks ##############
        ########################################
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )

                if self.use_time_embed:
                    self.input_blocks.append(TimestepEmbedSequential(*layers))
                else:
                    self.input_blocks.append(nn.Sequential(*layers))

                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                if self.use_time_embed:
                    self.input_blocks.append(
                        TimestepEmbedSequential(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,)
                                if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                else:
                    self.input_blocks.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,) 
                            if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch))

                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        
        ########## BottleNeck Blocks ###########
        ########################################
        
        self.middle_block = MiddleBlock(ch=ch, time_embed_dim=time_embed_dim, dropout=dropout, dims=dims,
                                        use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                                        use_time_embed=self.use_time_embed, make_vae=self.make_vae,
                                        num_heads=num_heads, num_head_channels=num_head_channels,
                                        use_new_attention_order=use_new_attention_order)


        self._feature_size += ch


        ########## Decoder Blocks ##############
        ########################################
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                
                # Skip Connection off for autoencoder
                if self.use_skip_connection:
                    ich = input_block_chans.pop()
                else:
                    ich = 0

                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                
                if self.use_time_embed:
                    self.output_blocks.append(TimestepEmbedSequential(*layers))
                else:
                    self.output_blocks.append(nn.Sequential(*layers))

                self._feature_size += ch

        ## Readout Block
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def reparameterize_trick(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = th.exp(0.5 * logvar)
            eps = th.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean
        
    def sample_vae_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize_trick(*latent_dist)
        return latent_sample

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def reparameterize(self, z):
        epsilon = 1e-6
        diff = th.abs(z - z.unsqueeze(axis=1)) + epsilon
        none_zeros = th.where(diff == 0., th.tensor([100.]).to(z.device), diff)
        z_scores, _ = th.min(none_zeros, axis=1)
        z_scores = th.clamp(z_scores, min=1e-6)  # Ensuring a minimum for stability
        # Check for NaNs in z_scores and replace with epsilon
        if th.isnan(z_scores).any():
            z_scores = th.nan_to_num(z_scores, nan=epsilon)  # Replaces NaNs with epsilon
        
        std =  th.normal(mean = th.tensor(0.0).to(z.device), std = z_scores.to(z.device)).to(z.device) #0.1 was z_scores 
        s = z + std
        c = th.cat((th.cos(2*th.pi*s), th.sin(2*th.pi*s)), 0)
        c = c.T.reshape(self.latent_dim*2,-1).T
        return c

    def reconstruct(self, z):
        shape = z.shape
        z = th.flatten(z, start_dim=1)
        z = self.en_fc(z)
        z = self.strecth(z) 
        c = th.cat((th.cos(2*np.pi*z), th.sin(2*np.pi*z)), 0)
        c = c.T.reshape(self.latent_dim*2, -1).T
        c = self.de_fc(c)
        c = c.view(shape)
        reconstr = self.decoder(c)
        return reconstr

    def latent(self, x):
        h = x.type(self.dtype)
        h = self.encoder(h)[0]
        return h

    def encoder(self,h,hs=None,emb=None):
         # Encoder & Bottleneck Blocks
        if self.use_time_embed:
            for module in self.input_blocks:
                h = module(h, emb)
                hs.append(h)
            h = self.middle_block(h,emb)
        else:
            for module in self.input_blocks:
                h = module(h)
                if self.use_skip_connection:
                    hs.append(h)
            f_shape=h.shape
            h = self.middle_block(h)
        return h, hs, f_shape

    def decoder(self, h, hs=None,emb=None):
         # Decoder Blocks
        if self.use_skip_connection:
            for module in self.output_blocks:
                h = th.cat([h, hs.pop()], dim=1)
                if self.use_time_embed:
                    h = module(h, emb)
                else:
                    h = module(h)
        else:
            for module in self.output_blocks:
                if self.use_time_embed:
                    h = module(h, emb)
                else:
                    h = module(h)

        return self.out(h)

    def disentangle(self,h):
        l_shape = h.shape
        h = th.flatten(h, start_dim=1)
        h = self.en_fc(h)
        h = self.strecth(h) 
        
        # Latent
        z = h.clone()

        h = self.reparameterize(h)
        h = self.de_fc(h)
        h = h.reshape(l_shape)
        return h, z

    def forward(self, x, timesteps=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None and self.use_time_embed is False
        ), "must specify y if and only if the model is class-conditional and time embedding should be also True"

        assert (timesteps is not None) != (
            self.use_time_embed is False
        ), "must enable `use_time_embed` for time conditioning"


        hs = []
        h = x.type(self.dtype)

        # Time Embedding
        if self.use_time_embed:
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        else:
            emb = None

        # Class Embedding
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        if self.make_vae:
            h_dist, _,f_shape = self.encoder(h=h,emb=emb)
            h_sample= self.reparameterize_trick(*h_dist)
            h = self.decoder(h=h_sample.reshape(f_shape),emb=emb)
            return self.out(h),h_dist,h_sample
        
        else:
            h, hs,_ = self.encoder(h,hs,emb)

            if self.dae:
                h,z = self.disentangle(h)
            else:
                z = h.clone()
            
            h=self.decoder(h, hs,emb)

            return h, z



if __name__ == "__main__":
    model = UNetModel(
        image_size=256,
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.0,
        channel_mult=(1,2,4,4,4,4,4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_time_embed=False,
        use_skip_connection=False,
        dae=False,make_vae=True)
    print(model)
    x= th.randn(1, 1, 256, 256)
    timesteps = th.randn(1)
    y,z  = model(x)
    print(y.shape)