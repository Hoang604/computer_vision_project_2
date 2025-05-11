import torch
from torch import nn
from torch.nn import functional as F
from util import ResidualBlock, SinusoidalPositionEmbeddings
from typing import Optional
from torchinfo import summary

class BasicTransformerBlock(nn.Module):
    """
    Combines Self-Attention, Cross-Attention, and FeedForward using nn.MultiheadAttention.
    Operates on inputs of shape (B, C, H, W).
    Cross-Attention is applied conditionally based on context availability.
    """
    def __init__(self, dim: int, context_dim: int, n_head: int, dropout: float = 0.1):
        """
        Args:
            dim (int): Input dimension (channels)
            context_dim (int): Dimension of context embeddings (only used if context is provided)
            n_head (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super().__init__()
        self.dim = dim
        # LayerNorms
        self.norm_self_attn = nn.LayerNorm(dim)
        self.norm_cross_attn = nn.LayerNorm(dim) # Norm before cross-attention
        self.norm_ff = nn.LayerNorm(dim)

        # Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True # Expect input (B, N, C)
        )

        # Cross-Attention Layer (will be used conditionally)
        # We define it here, but only use it in forward if context is not None
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,        # Query dimension (from image features x)
            kdim=context_dim,     # Key dimension (from context)
            vdim=context_dim,     # Value dimension (from context)
            num_heads=n_head,
            dropout=dropout,
            batch_first=True # Expect query(B, N_img, C), key/value(B, N_ctx, C_ctx)
        )

        # FeedForward Layer
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, H, W) - Image features
        # context: Optional[(B, seq_len_ctx, C_context)] - Text context embeddings or None
        batch_size, channels, height, width = x.shape
        n_tokens_img = height * width
        # Note: No residual = x here, residuals are added after each block

        # --- Reshape for Sequence Processing ---
        # (B, C, H, W) -> (B, C, N) -> (B, N, C)
        x_seq = x.view(batch_size, channels, n_tokens_img).transpose(1, 2)

        # --- Self-Attention ---
        x_norm = self.norm_self_attn(x_seq)
        self_attn_out, _ = self.self_attn(query=x_norm, key=x_norm, value=x_norm, need_weights=False)
        x_seq = x_seq + self_attn_out # Add residual

        # --- Cross-Attention (Conditional) ---
        # Only perform cross-attention if context is provided
        if context is not None:
            x_norm = self.norm_cross_attn(x_seq)
            cross_attn_out, _ = self.cross_attn(query=x_norm, key=context, value=context, need_weights=False)
            x_seq = x_seq + cross_attn_out # Add residual
        # If context is None, this block is skipped

        # --- FeedForward ---
        x_norm = self.norm_ff(x_seq)
        ff_out = self.ff(x_norm)
        x_seq = x_seq + ff_out # Add residual

        # --- Reshape back to Image Format ---
        # (B, N, C) -> (B, C, N) -> (B, C, H, W)
        out = x_seq.transpose(1, 2).view(batch_size, channels, height, width)

        return out # Return shape (B, C, H, W)
    
class ImageContextExtractor(nn.Module):
    """
    A PyTorch module to extract context from a batch of images.
    Input: (b, c, h, w)
    Output: (b, h*w, context_dim) - suitable for cross-attention.
    """
    def __init__(self,
                 in_channels: int,
                 context_dim: int = 512,
                 hidden_dim_scale_factor: int = 2,
                 preferred_num_groups: int = 32):
        """
        Initializes the ImageContextExtractor.

        Args:
            in_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            context_dim (int): Dimensionality of the output context vector for each spatial token.
            hidden_dim_scale_factor (int, optional): Multiplication factor to determine
                                                     the number of channels in the intermediate hidden layer
                                                     (intermediate_dim = context_dim * hidden_dim_scale_factor).
                                                     Defaults to 2.
            preferred_num_groups (int, optional): Preferred number of groups for GroupNorm.
                                                 The actual value will be adjusted to ensure validity.
                                                 Defaults to 32.
        """
        super().__init__() # Call __init__ of parent class (nn.Module)
        self.in_channels = in_channels
        self.context_dim = context_dim

        intermediate_dim = context_dim * hidden_dim_scale_factor
        # Determine num_groups for GroupNorm in intermediate layers
        # This ensures that num_channels (intermediate_dim) is divisible by num_groups.
        num_groups_intermediate: int
        if intermediate_dim % preferred_num_groups == 0 and intermediate_dim >= preferred_num_groups:
            num_groups_intermediate = preferred_num_groups
        else:
            # Fallback strategy: try common smaller group sizes or 1
            found_divisor = False
            # Try common smaller group sizes that are divisors of intermediate_dim
            for ng in [16, 8, 4, 2]:
                if ng < preferred_num_groups and intermediate_dim % ng == 0 and intermediate_dim >= ng:
                    num_groups_intermediate = ng
                    found_divisor = True
                    break
            if not found_divisor:
                num_groups_intermediate = 1 # Default to 1 group (acts like LayerNorm across channels)

        # First convolutional block
        self.conv_block1 = nn.Sequential(
            # Conv2d: kernel_size=3, padding=1 to preserve spatial dimensions (h,w)
            # bias=False because GroupNorm has its own scale/shift parameters
            nn.Conv2d(in_channels, intermediate_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_intermediate, num_channels=intermediate_dim),
            nn.SiLU() # Swish activation function
        )

        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_intermediate, num_channels=intermediate_dim), # Use the same num_groups
            nn.SiLU()
        )

        # Final projection layer to get to context_dim
        # Use 1x1 convolution to change channel depth without altering spatial dimensions
        self.projection = nn.Conv2d(intermediate_dim, context_dim, kernel_size=1)
        # Typically, no activation or normalization after the final projection layer
        # if these features are used as context, as further processing might occur
        # within the attention mechanism or post-attention.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (b, c, h, w).

        Returns:
            torch.Tensor: Output context tensor of shape (b, h*w, context_dim).
        """
        # x: (b, c, h, w)
        # b: batch_size, c: in_channels, h: height, w: width

        # Pass through convolutional blocks
        x = self.conv_block1(x) # Output: (b, intermediate_dim, h, w)
        x = self.conv_block2(x) # Output: (b, intermediate_dim, h, w)

        # Apply projection layer
        features = self.projection(x) # Output: (b, context_dim, h, w)

        b, output_channels, h, w = features.shape

        # Check if the output channel dimension matches the expected context_dim
        assert output_channels == self.context_dim, \
            f"Output channel dimension mismatch. Expected {self.context_dim}, got {output_channels}."

        # Reshape for cross-attention context
        # Target: (b, h*w, context_dim)

        # 1. Reshape: (b, context_dim, h, w) -> (b, context_dim, h*w)
        # Flatten the spatial dimensions h and w into a single sequence dimension.
        context = features.view(b, self.context_dim, h * w)

        # 2. Permute: (b, context_dim, h*w) -> (b, h*w, context_dim)
        # Move the sequence dimension (h*w) to the second position,
        # which is a common format for attention modules.
        # .contiguous() ensures the tensor is stored contiguously in memory.
        context = context.permute(0, 2, 1).contiguous()

        return context

# ------------------------------------------
# --- UNet with Attention and Conditioning ---
# ------------------------------------------
        
class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,              # Input image channels (e.g., 4 if VAE latent)
        out_channels=3,             # Output channels (usually same as input)
        base_dim=128,               # Base channel dimension
        dim_mults=(1, 2, 4),        # Channel multipliers for each resolution level
        num_resnet_blocks=2,        # Number of ResNet blocks per level
        context_dim=512,            # Dimension of CLIP context embeddings
        attn_heads=4,               # Number of attention heads
        dropout=0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_dim = context_dim
        self.num_resnet_blocks = num_resnet_blocks

        # --- Context Extraction ---
        self.context_extractor = ImageContextExtractor(
            in_channels=in_channels,
            context_dim=context_dim,
            hidden_dim_scale_factor=2,
            preferred_num_groups=32
        )        

        # --- Time Embedding ---
        time_proj_dim = base_dim * 4 # Dimension to project sinusoidal embedding to
        self.time_embeddings = SinusoidalPositionEmbeddings(base_dim) # Initial embedding dim = base_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(base_dim, time_proj_dim),
            nn.SiLU(),
            nn.Linear(time_proj_dim, time_proj_dim) # This will be used by ResNet blocks
        )
        actual_time_emb_dim = time_proj_dim # This is what ResNetBlock expects

        # --- Initial Convolution ---
        self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        # --- UNet Layers ---
        dims = [base_dim] + [base_dim * m for m in dim_mults] # e.g., [256, 256, 512, 1024]
        in_out_dims = list(zip(dims[:-1], dims[1:])) # e.g., [(256, 256), (256, 512), (512, 1024)]
        num_resolutions = len(in_out_dims)

        # Helper modules
        def make_attn_block(dim, heads, ctx_dim):
            # return MultiHeadAttentionBlock(channels=dim, n_head=heads) # Only self-attention
            return BasicTransformerBlock(dim=dim, context_dim=ctx_dim, n_head=heads)

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
            stage_modules.append(make_resnet_block(dim_in, dim_out, actual_time_emb_dim))
            for _ in range(num_resnet_blocks - 1):
                stage_modules.append(make_resnet_block(dim_out, dim_out, actual_time_emb_dim))
            if is_last:
                stage_modules.append(make_attn_block(dim_out, attn_heads, context_dim))

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
        self.bottleneck.append(make_attn_block(mid_dim, attn_heads, context_dim))
        self.bottleneck.append(make_resnet_block(mid_dim, mid_dim, actual_time_emb_dim))
        # self.bottleneck.append(make_resnet_block(mid_dim, mid_dim, actual_time_emb_dim))

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
            if i == 0: # Last stage
                stage_modules.append(make_attn_block(dim_in, attn_heads, context_dim))

            # Add Upsample layer if not the last stage (output stage)
            if not is_last:
                 stage_modules.append(make_upsample(dim_in, dim_out))
            else:
                 stage_modules.append(nn.Identity())

            self.ups.append(stage_modules)


        # --- Final Layer ---
        self.final_norm = nn.GroupNorm(32, base_dim) # Norm before final conv
        self.final_act = nn.SiLU()
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
            # Extract context from images
            context = self.context_extractor(context)
        # 1. Initial Convolution
        x = self.init_conv(x) # (B, base_dim, H, W)

        # 2. Time Embedding
        t_emb = self.time_embeddings(time) # (B, base_dim)
        t_emb = self.time_mlp(t_emb)      # (B, actual_time_emb_dim)

        # 3. Encoder Path
        skip_connections = []
        for stage in self.downs:
            for block in stage:
                if isinstance(block, BasicTransformerBlock):
                    x = block(x, context)
                elif isinstance(block, ResidualBlock):
                    x = block(x, t_emb)
                else:
                    skip_connections.append(x)
                    x = block(x)

        # 4. Bottleneck
        for block in self.bottleneck:
            if isinstance(block, BasicTransformerBlock):
                x = block(x, context)
            elif isinstance(block, ResidualBlock):
                x = block(x, t_emb)

        # 5. Decoder Path
        # Iterate through decoder stages and corresponding skip connections in reverse
        for skip, stage in zip(reversed(skip_connections), self.ups):
            x = torch.cat([x, skip], dim=1)
            for block in stage:
                if isinstance(block, BasicTransformerBlock):
                    x = block(x, context)
                elif isinstance(block, ResidualBlock):
                    x = block(x, t_emb)
                else:
                    x = block(x)

        # 6. Final Layers
        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_conv(x)

        return x