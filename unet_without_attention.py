import torch
from torch import nn
from torch.nn import functional as F
from util import ResidualBlock, SinusoidalPositionEmbeddings
from typing import Optional
from torchinfo import summary
from rrdb import RRDB
    
# ------------------------------------------
# --- UNet with Conditioning ---
# ------------------------------------------
        
class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,              # Input image channels (e.g., 4 if VAE latent)
        out_channels=3,             # Output channels (usually same as input)
        base_dim=128,               # Base channel dimension
        dim_mults=(1, 2, 4),        # Channel multipliers for each resolution level
        num_resnet_blocks=2,        # Number of ResNet blocks per level
        context_dim=256,            # Dimension of CLIP context embeddings
        attn_heads=4,               # Number of attention heads
        dropout=0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_dim = context_dim
        self.num_resnet_blocks = num_resnet_blocks

        self.lr_encoder = RRDB(in_channels=in_channels)
        # --- Time Embedding ---
        time_proj_dim = base_dim * 4 # Dimension to project sinusoidal embedding to
        self.time_embeddings = SinusoidalPositionEmbeddings(base_dim) # Initial embedding dim = base_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(base_dim, time_proj_dim),
            nn.Mish(),
            nn.Linear(time_proj_dim, time_proj_dim) # This will be used by ResNet blocks
        )
        actual_time_emb_dim = time_proj_dim # This is what ResNetBlock expects

        # --- Initial Convolution ---
        self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        # --- UNet Layers ---
        dims = [base_dim] + [base_dim * m for m in dim_mults] # e.g., [256, 256, 512, 1024]
        in_out_dims = list(zip(dims[:-1], dims[1:])) # e.g., [(256, 256), (256, 512), (512, 1024)]
        num_resolutions = len(in_out_dims)

        def make_resnet_block(in_c, out_c, t_emb_dim):
            return ResidualBlock(in_channels=in_c, out_channels=out_c, time_emb_dim=t_emb_dim, dropout=dropout)

        def make_downsample():
            # Use Conv2d with stride 2 to downsample, mean resize image (not channels)
            return nn.Conv2d(dims[i+1], dims[i+1], kernel_size=4, stride=2, padding=1)

        def make_upsample(in_channels, out_channels):
             # Use ConvTranspose2d or Upsample + Conv
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )


        # -- Encoder -- 1 attention each block
        self.downs = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out_dims):
            is_last = (i == num_resolutions - 1)
            stage_modules = nn.ModuleList([])
            # Add ResNet blocks
            if i == 2:
                stage_modules.append(make_resnet_block(dim_in + 131, dim_out, actual_time_emb_dim))
            else:
                stage_modules.append(make_resnet_block(dim_in, dim_out, actual_time_emb_dim))
            for _ in range(num_resnet_blocks - 1):
                stage_modules.append(make_resnet_block(dim_out, dim_out, actual_time_emb_dim))

            # Add Downsample layer if not the last stage
            if not is_last:
                stage_modules.append(make_downsample())
            else: # Add Identity if last stage (optional, for consistent structure)
                stage_modules.append(nn.Identity())

            self.downs.append(stage_modules)

        # -- Bottleneck --
        mid_dim = dims[-1]
        self.bottleneck = nn.ModuleList([])
        self.bottleneck.append(make_resnet_block(mid_dim, mid_dim, actual_time_emb_dim))
        self.bottleneck.append(make_resnet_block(mid_dim, mid_dim, actual_time_emb_dim))

        # -- Decoder -- 3 attention per block
        self.ups = nn.ModuleList([])
        # Reverse dimensions for decoder, e.g., [(512, 1024), (256, 512), (256, 256)]
        for i, (dim_out, dim_in) in enumerate(reversed(in_out_dims)): # Careful: dim_in/out are reversed role here
            is_last = (i == num_resolutions - 1)
            stage_modules = nn.ModuleList([])

            # Add ResNet blocks (Input channels = dim_in + skip_channels)
            skip_channels = dim_in # Channels from corresponding encoder stage
            stage_modules.append(make_resnet_block(dim_in + skip_channels, dim_in, actual_time_emb_dim))
            for _ in range(num_resnet_blocks - 1):
                stage_modules.append(make_resnet_block(dim_in, dim_in, actual_time_emb_dim))

            # Add Upsample layer if not the last stage (output stage)
            if not is_last:
                 stage_modules.append(make_upsample(dim_in, dim_out))
            else:
                 stage_modules.append(nn.Identity())

            self.ups.append(stage_modules)


        # --- Final Layer ---
        self.final_norm = nn.GroupNorm(32, base_dim) # Norm before final conv
        self.final_act = nn.Mish()
        self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor, time: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input noisy tensor (B, C_in, H, W)
            time (torch.Tensor): Time steps (B,)
            context (torch.Tensor): Context embeddings (B, seq_len, C_ctx)

        Returns:
            torch.Tensor: Predicted noise (B, C_out, H, W)
        """

        if context is not None:
            lr_encode = self.lr_encoder(context)

        # 1. Initial Convolution
        x = self.init_conv(x) # (B, base_dim, H, W)

        # 2. Time Embedding
        t_emb = self.time_embeddings(time) # (B, base_dim)
        t_emb = self.time_mlp(t_emb)      # (B, actual_time_emb_dim)
        # 3. Encoder Path
        skip_connections = []
        first = True

        for i, stage in enumerate(self.downs):
            for block in stage:
                if isinstance(block, ResidualBlock):
                    if i == 2 and first:
                        x = torch.cat([lr_encode, x], dim=1)
                        x = block(x, t_emb)
                        first = False
                    elif i== 2:
                        x = block(x, t_emb)
                    else:
                        x = block(x, t_emb)
                else:
                    skip_connections.append(x)
                    x = block(x)

        # 4. Bottleneck
        for block in self.bottleneck:
            x = block(x, t_emb)


        # 5. Decoder Path
        # Iterate through decoder stages and corresponding skip connections in reverse
        for skip, stage in zip(reversed(skip_connections), self.ups):
            x = torch.cat([x, skip], dim=1)
            for block in stage:
                if isinstance(block, ResidualBlock):
                    x = block(x, t_emb)
                else:
                    x = block(x)

        # 6. Final Layers
        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_conv(x)

        return x