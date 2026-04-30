import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from src.models.velocity_field_models.layers.position_encoding import build_position_encoding


def timestamp_embedding(timesteps, dim, scale=200, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param scale: a premultiplier of timesteps
    :param max_period: controls the minimum frequency of the embeddings.
    :param repeat_only: whether to repeat only the values in timesteps along the 2nd dim
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = scale * timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(scale * timesteps, 'b -> b d', d=dim)
    return embedding

def parse_input(x_in_: torch.Tensor, dim: int,_reshape):
    x_in_    = x_in_.float().to(x_in_.device)
    t_interp = x_in_[:, -2]
    time_kp1 = x_in_[:, -3]
    label    = x_in_[:, -7:-3]
    age      = x_in_[:, -8]
    sex      = x_in_[:, -10:-8]
    x_in     = torch.cat([x_in_[:, :-10], x_in_[:, -1].unsqueeze(1)], dim=1)

    x_k = x_in[:, :dim].view(_reshape)
    time_k = x_in[:, -1]
    x_km1 = x_in[:, dim : 2*dim].view(_reshape)
    time_km1 = x_in[:, -2]
    return {
        "t_interp": t_interp,
        "time_kp1": time_kp1,
        "label": label,
        "age": age,
        "sex": sex,
        "x_k": x_k,
        "time_k": time_k,
        "x_km1": x_km1,
        "time_km1": time_km1,
    }


class VectorFieldRegressorSigma_BNC(nn.Module):
    def __init__(
            self,
            depth: int,
            mid_depth: int,
            state_size: int,
            state_res: Tuple[int, int],
            inner_dim: int,
            out_norm: str = "ln",
            reference: bool = False):
        super(VectorFieldRegressorSigma_BNC, self).__init__()
        print("VectorFieldRegressor w/ Demographic Conditioning and Sigma Prediction BNC")
        self.state_size = state_size
        self.state_height = state_res[0]
        self.state_width = state_res[1]
        self.inner_dim = inner_dim
        self.reference = reference
        self._dim = 4096
        self.position_encoding = build_position_encoding(self.inner_dim, position_embedding_name="learned")

        
        # Add learnable time scaling parameters
        self.alpha = nn.Parameter(torch.ones(1) * 2.0)  # Controls steepness of sigmoid
        self.beta = nn.Parameter(torch.ones(1) * 1.0)   # Controls magnitude of scaling

        self.project_in = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"),
            nn.Linear(3 * self.state_size if self.reference else 2 * self.state_size, self.inner_dim)
        )

        self.time_projection = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, self.inner_dim)
        )

        self.class_projection = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, self.inner_dim)
        )
        
        self.age_projection = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, self.inner_dim)
        )
        
        self.sex_projection = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, self.inner_dim)
        )
        
        # Add new secondary projection layers for conditioning
        self.secondary_time_projection = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.GELU()
        )

        self.secondary_class_projection = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.GELU()
        )
        
        self.secondary_age_projection = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.GELU()
        )
        
        self.secondary_sex_projection = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.GELU()
        )

        def build_layer(d_model: int):
            return nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=4 * d_model,
                dropout=0.05,
                activation="gelu",
                norm_first=True,
                batch_first=True)

        self.in_blocks = nn.ModuleList()
        self.mid_blocks = nn.Sequential(*[build_layer(self.inner_dim) for _ in range(mid_depth)])
        self.out_blocks = nn.ModuleList()
        for i in range(depth):
            self.in_blocks.append(build_layer(self.inner_dim))
            self.out_blocks.append(nn.ModuleList([
                nn.Linear(2 * self.inner_dim, self.inner_dim),
                build_layer(self.inner_dim)]))

        if out_norm == "ln":
            # Mean projection head with transformer
            self.project_out = nn.Sequential(
                build_layer(self.inner_dim),  # Add transformer layer that processes ALL tokens
                nn.Linear(self.inner_dim, self.inner_dim),
                nn.GELU(),
                nn.LayerNorm(self.inner_dim),
                DropFirstToken(),  # Replace lambda with proper module
                Rearrange("b (h w) c -> b c h w", h=self.state_height),
                nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1),
            )
            
            # Sigma projection head with transformer
            self.project_sigma = nn.Sequential(
                build_layer(self.inner_dim),  # Add transformer layer that processes ALL tokens
                nn.Linear(self.inner_dim, self.inner_dim),
                nn.GELU(),
                nn.LayerNorm(self.inner_dim),
                DropFirstToken(),  # Replace lambda with proper module
                Rearrange("b (h w) c -> b c h w", h=self.state_height),
                nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1),
                nn.Softplus(),  # Ensure positive sigma values
            )
            
            # Score projection head with transformer
            self.project_score = nn.Sequential(
                build_layer(self.inner_dim),  # Add transformer layer that processes ALL tokens
                nn.Linear(self.inner_dim, self.inner_dim),
                nn.GELU(),
                nn.LayerNorm(self.inner_dim),
                DropFirstToken(),  # Replace lambda with proper module
                Rearrange("b (h w) c -> b c h w", h=self.state_height),
                nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),  # Bound scores between -1 and 1
            )
        elif out_norm == "bn":
            # Mean projection head with transformer
            self.project_out = nn.Sequential(
                build_layer(self.inner_dim),  # Add transformer layer that processes ALL tokens
                nn.Linear(self.inner_dim, self.inner_dim),
                lambda x: x[:, 1:],  # Now drop the first token AFTER transformer processing
                Rearrange("b (h w) c -> b c h w", h=self.state_height),
                nn.GELU(),
                nn.BatchNorm2d(self.inner_dim),
                nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1),
            )
            
            # Sigma projection head with transformer
            self.project_sigma = nn.Sequential(
                build_layer(self.inner_dim),  # Add transformer layer that processes ALL tokens
                nn.Linear(self.inner_dim, self.inner_dim),
                lambda x: x[:, 1:],  # Now drop the first token AFTER transformer processing
                Rearrange("b (h w) c -> b c h w", h=self.state_height),
                nn.GELU(),
                nn.BatchNorm2d(self.inner_dim),
                nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1),
                nn.Softplus(),  # Ensure positive sigma values
            )
            
            # Score projection head with transformer
            self.project_score = nn.Sequential(
                build_layer(self.inner_dim),  # Add transformer layer that processes ALL tokens
                nn.Linear(self.inner_dim, self.inner_dim),
                lambda x: x[:, 1:],  # Now drop the first token AFTER transformer processing
                Rearrange("b (h w) c -> b c h w", h=self.state_height),
                nn.GELU(),
                nn.BatchNorm2d(self.inner_dim),
                nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),  # Bound scores between -1 and 1
            )
        else:
            raise NotImplementedError
   
    def forward(self, x_in_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x_in_: Input tensor with format [B, ...]

        Returns:
            Output tensor with format [B, ...], containing both mean and sigma predictions
        """
        # Map inputs to VectorFieldRegressor format
        input_latents, reference_latents, conditioning_latents, index_distances, timestamps, t_future, label, age, sex = \
            self.map_to_vector_field_regressor(x_in_)

        # Forward pass
        mean, sigma, score = self._forward(
            input_latents=input_latents,
            reference_latents=reference_latents,
            conditioning_latents=conditioning_latents,
            index_distances=index_distances,
            timestamps=timestamps,
            label=label,
            age=age,
            sex=sex)

        # Concatenate mean, sigma, and score along the channel dimension
        outputs = torch.cat([mean, sigma, score], dim=1)
        
        # Reshape and add time_future
        return torch.cat([outputs.reshape(x_in_.size(0), -1), t_future], dim=-1)

    def _forward(
            self,
            input_latents: torch.Tensor,
            reference_latents: torch.Tensor,
            conditioning_latents: torch.Tensor,
            index_distances: torch.Tensor,
            timestamps: torch.Tensor,
            label: torch.Tensor,
            age: torch.Tensor,
            sex: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal forward pass implementation.

        :param input_latents: [b, c, h, w]
        :param reference_latents: [b, c, h, w]
        :param conditioning_latents: [b, c, h, w]
        :param index_distances: [b]
        :param timestamps: [b]
        :param label: [b, 4]
        :param age: [b]
        :param sex: [b, 2]
        :return: Tuple of [b, c, h, w], [b, c, h, w] for mean and sigma
        """
        # Fetch timestamp tokens
        t = timestamp_embedding(timestamps, dim=self.inner_dim).unsqueeze(1)

        # Calculate position embedding
        pos = self.position_encoding(input_latents)
        pos = rearrange(pos, "b c h w -> b (h w) c")

        # Calculate embeddings
        dist = self.time_projection(torch.log(index_distances).unsqueeze(1)).unsqueeze(1)
        cls = self.class_projection(label).unsqueeze(1)
        
        age_emb = self.age_projection(age.unsqueeze(1)).unsqueeze(1)
        sex_emb = self.sex_projection(sex).unsqueeze(1)
        
        # Build input tokens
        if self.reference:
            x = self.project_in(torch.cat([input_latents, reference_latents, conditioning_latents], dim=1))
        else:
            x = self.project_in(torch.cat([input_latents, conditioning_latents], dim=1))
       
        x = x + pos + dist + cls + age_emb + sex_emb
        x = torch.cat([t, x], dim=1)

        # Propagate through the main network
        hs = []
        for block in self.in_blocks:
            x = block(x)
            hs.append(x.clone())
        x = self.mid_blocks(x)
        
        # Apply secondary conditioning
        secondary_dist = self.secondary_time_projection(dist.squeeze(1)).unsqueeze(1)
        secondary_cls = self.secondary_class_projection(cls.squeeze(1)).unsqueeze(1)
        secondary_age = self.secondary_age_projection(age_emb.squeeze(1)).unsqueeze(1)
        secondary_sex = self.secondary_sex_projection(sex_emb.squeeze(1)).unsqueeze(1)
        
        # Add the secondary conditioning to the tokens (excluding the timestamp token)
        x[:, 1:] = x[:, 1:] + secondary_dist + secondary_cls + secondary_age + secondary_sex
        
        # Continue with the rest of the forward pass
        for i, block in enumerate(self.out_blocks):
            x = block[1](block[0](torch.cat([hs[-i - 1], x], dim=-1)))
        
        # Project to output - mean, sigma, and score - KEEPING ALL TOKENS
        mean = self.project_out(x)
        sigma = self.project_sigma(x)
        score = self.project_score(x)

        return mean, sigma, score

    def map_to_vector_field_regressor(self, x_in_):
        """
        Maps FlowTransformer inputs to VectorFieldRegressor inputs.
        
        Args:
            x_in_: Input tensor from FlowTransformer with format [B, ...]
            
        Returns:
            Tuple of (input_latents, reference_latents, conditioning_latents, 
                    index_distances, timestamps, time_future, label, age, sex)
        """
        # Parse the input to extract components
        parsed = parse_input(x_in_, self._dim, 
                             (x_in_.size(0), self.state_size, self.state_height, self.state_width))
        
        # Extract frames
        x_k = parsed["x_k"]     # Current frame (reference)
        x_km1 = parsed["x_km1"] # Previous frame (conditioning)
        
        # Extract times
        time_k = parsed["time_k"]
        time_km1 = parsed["time_km1"]
        label = parsed["label"]
        
        age = parsed["age"]
        sex = parsed["sex"]
        # ipdb.set_trace()
        # Ensure correct shapes
        reference_latents = None      # Shape: [B, 256, 4, 4]
        conditioning_latents = x_km1  # Shape: [B, 256, 4, 4]
        
        # For input_latents,
        input_latents = x_k.clone()
        
        # Calculate time differences (index distances)
        index_distances = torch.clamp(time_k - time_km1, min=1e-5)
        
        # Use current timestamps
        timestamps = time_k
        
        time_future = time_k.view(x_in_.size(0), 1)
        
        return input_latents, reference_latents, conditioning_latents, index_distances, timestamps, time_future, label, age, sex




# def build_vector_field_regressor(config: Configuration, reference: bool = True):
#     return VectorFieldRegressor(
#         state_size=config["state_size"],
#         state_res=config["state_res"],
#         inner_dim=config["inner_dim"],
#         depth=config["depth"],
#         mid_depth=config["mid_depth"],
#         out_norm=config["out_norm"],
#         reference=reference,
#     )


class DropFirstToken(nn.Module):
    """Module to drop the first token in a sequence."""
    def forward(self, x):
        return x[:, 1:]
