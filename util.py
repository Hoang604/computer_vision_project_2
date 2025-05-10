import torch
from torch import nn
from torch.nn import functional as F
import math
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as TF
from bicubic import upscale_image
import numpy as np

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Module to generate sinusoidal position embeddings for time steps t.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time (torch.Tensor): Tensor containing time steps, shape (batch_size,).

        Returns:
            torch.Tensor: Positional embedding tensor, shape (batch_size, dim).
        """
        device = time.device
        half_dim = self.dim // 2
        # Correct calculation for frequencies: 1 / (10000^(2i/dim))
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000.0) / half_dim))
        # Unsqueeze time for broadcasting: time shape (B, 1), embeddings shape (half_dim,) -> (B, half_dim)
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
        # Concatenate sin and cos: shape (B, dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1: # Handle odd dimensions if necessary
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """
    Residual Block without explicit stride control (stride is always 1).
    Includes dropout and optional time embedding conditioning.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1, time_emb_dim: int = None, num_groups: int = 32):
        """
        Initializes the Residual Block. Stride is fixed to 1.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel (default is 3).
            dropout (float): Dropout rate to apply (default is 0.1).
            time_emb_dim (int, optional): Dimension of the time embedding. If provided, enables time conditioning. Defaults to None.
            num_groups (int): Number of groups for GroupNorm (default is 32).
        """
        super().__init__()

        # Function to get valid group number
        def get_valid_groups(channels, requested_groups):
            if channels <= 0: return 1 # Avoid division by zero or invalid groups
            # Ensure requested_groups is positive before proceeding
            if requested_groups <= 0: return 1
            # Find the largest divisor of channels that is less than or equal to requested_groups
            for g in range(min(channels, requested_groups), 0, -1):
                 if channels % g == 0:
                      return g
            return 1 # Fallback: should ideally not happen if channels > 0

        actual_in_groups = get_valid_groups(in_channels, num_groups)
        actual_out_groups = get_valid_groups(out_channels, num_groups)

        # Time embedding MLP (if enabled)
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2) # Project to get scale and shift for out_channels
            )

        # First normalization and convolution
        self.norm1 = nn.GroupNorm(actual_in_groups, in_channels)
        # Use bias=False since GroupNorm has affine params or we use time conditioning
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False)

        # Second normalization and convolution 
        self.norm2 = nn.GroupNorm(actual_out_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Residual connection: handles channel changes only
        if in_channels != out_channels:
            # Use stride=1 for the residual connection convolution
            self.residual_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.GroupNorm(actual_out_groups, out_channels))
        else:
            # If channels match, no transformation needed
            self.residual_connection = nn.Identity()


    def forward(self, x, time_emb = None):
        """
        Forward pass for the ResidualBlock with optional time conditioning. Stride is fixed to 1.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).
            time_emb (torch.Tensor, optional): Time embedding tensor of shape (B, time_emb_dim). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W). Height and Width remain unchanged.
        """
        residue = x # Save input for residual connection

        # --- First part ---
        # Pre-activation style: Norm -> Activation -> Conv
        x_norm1 = self.norm1(residue)
        x_act1 = F.silu(x_norm1)
        x_conv1 = self.conv1(x_act1) # x_conv1 now has out_channels, H/W unchanged

        # --- Second part ---
        # Pre-activation style: Norm -> Activation -> Modulation -> Conv
        x_norm2 = self.norm2(x_conv1) # Input to norm2 has out_channels
        x_act2 = F.silu(x_norm2)      # Output still has out_channels

        # Apply time conditioning (scale & shift)
        if self.time_mlp is not None and time_emb is not None:
            # time_mlp projects to out_channels * 2
            time_proj = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1) # (B, 2*out_channels, 1, 1)
            scale, shift = time_proj.chunk(2, dim=1) # scale/shift have out_channels (B, out_channels, 1, 1)

            # Modulate features (x_act2 has out_channels, scale/shift have out_channels)
            # FiLM operation: gamma * x + beta
            x_modulated = x_act2 * (scale + 1) + shift
        else:
            # If no time embedding, just pass through
            x_modulated = x_act2

        # Apply dropout and second convolution
        x_dropout = self.dropout(x_modulated) # Apply dropout after modulation
        x_conv2 = self.conv2(x_dropout)       # Final convolution in the main path, H/W unchanged

        # --- Residual connection ---
        # Add residual connection (after potentially transforming residue channels)
        # self.residual_connection handles the change from in_channels to out_channels if needed, keeps H/W same
        output = x_conv2 + self.residual_connection(residue)

        return output
    
class MultiHeadAttentionBlock(nn.Module):
    # --- Use the MultiHeadAttentionBlock you provided ---
    def __init__(self, channels: int, n_head: int = 4, dropout: float = 0.1):
        """
        Args:
            channels: the number of channels in the input tensor (embedding dimension)
            n_head: the number of heads
            dropout: the dropout value
        """
        super().__init__()
        assert channels % n_head == 0, "channels must be divisible by n_head"
        self.n_head = n_head
        self.channels = channels
        self.head_dim = channels // n_head

        # Use a single linear layer for Q, K, V projection for efficiency
        self.w_qkv = nn.Linear(channels, 3 * channels)
        self.dropout = nn.Dropout(dropout)
        # Output projection
        self.w_o = nn.Linear(channels, channels)

    @staticmethod
    def scaled_dot_product_attention(q, k, v, head_dim, dropout_layer: nn.Dropout, causal_mask: bool = False):
        """ Static method for scaled dot-product attention """
        # Matmul Q and K transpose: (B, H, N, D_h) x (B, H, D_h, N) -> (B, H, N, N)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

        if causal_mask:
            # Create a mask to prevent attending to future positions (upper triangle)
            seq_len = q.size(-2)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        if dropout_layer is not None:
            attn_weights = dropout_layer(attn_weights)

        # Matmul attention weights with V: (B, H, N, N) x (B, H, N, D_h) -> (B, H, N, D_h)
        output = torch.matmul(attn_weights, v)
        return output # Return output only, attn_weights if needed for visualization

    def forward(self, x: torch.Tensor, causal_mask: bool = False):
        """
        Args:
            x: the input tensor of shape (batch_size, channels, height, width)
            causal_mask: whether to apply a causal mask (for autoregressive tasks)
        Returns:
            the output tensor of shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.size()
        n_pixels = height * width # Number of tokens (pixels)

        # 1. Reshape input: (B, C, H, W) -> (B, H*W, C) = (B, N, C)
        x_reshape = x.view(batch_size, channels, n_pixels).transpose(1, 2)

        # 2. Project to Q, K, V
        qkv = self.w_qkv(x_reshape)  # (B, N, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # Each is (B, N, C)

        # 3. Split into multiple heads: (B, N, C) -> (B, N, H, D_h) -> (B, H, N, D_h)
        q = q.view(batch_size, n_pixels, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_pixels, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_pixels, self.n_head, self.head_dim).transpose(1, 2)

        # 4. Apply Scaled Dot-Product Attention
        attention_output = self.scaled_dot_product_attention(q, k, v, self.head_dim, self.dropout, causal_mask)

        # 5. Concatenate heads: (B, H, N, D_h) -> (B, N, H, D_h) -> (B, N, C)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, n_pixels, channels)

        # 6. Final linear projection
        projected_attention_output = self.w_o(attention_output) # (B, N, C)

        output_with_residual = x_reshape + projected_attention_output

        # 7. Reshape back to image format: (B, N, C) -> (B, C, N) -> (B, C, H, W)
        output = output_with_residual.transpose(1, 2).view(batch_size, channels, height, width)

        return output

class ImageDataset(Dataset):
        def __init__(self, folder_path:str, img_size:int, downscale_factor:int):
            """
            Initializes the dataset to load images and generate multiple versions.

            Args:
                folder_path (str): Path to the folder containing images.
                img_size (int): Target size for the 'original' processed image (height and width).
                downscale_factor (int): Factor by which to downscale the image for the low-resolution version.
                                       This is also the factor for upscaling the low-res image.
            """
            self.folder_path = folder_path
            # Find image files - consider adding more extensions if needed
            self.image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            self.original_len = len(self.image_files) # Store original number of files
            self.img_size = img_size
            self.downscale_factor = downscale_factor

            if not isinstance(self.downscale_factor, int) or self.downscale_factor < 1:
                raise ValueError("downscale_factor must be an integer and >= 1.") # write message on console
            if not isinstance(self.img_size, int) or self.img_size <= 0:
                raise ValueError("img_size must be a positive integer.") # write message on console
            if self.img_size % self.downscale_factor != 0 :
                 print(f"Warning: img_size ({self.img_size}) is not perfectly divisible by downscale_factor ({self.downscale_factor}). "
                       "This might lead to slight dimension mismatches if not handled carefully by the upscale_image function.") # write message on console


            print(f"Found {self.original_len} images in {folder_path}. Target original_img_size: {img_size}x{img_size}, downscale_factor: {downscale_factor}.") # write message on console

        def __len__(self):
            """ Returns the total size of the dataset (original + flipped). """
            return self.original_len * 2 # Report double the length for data augmentation

        def __getitem__(self, idx):
            """
            Retrieves a tuple of image tensors:
            (low_res_image, upscaled_image, original_image_resized, residual_image)
            All tensors are in the [0, 1] value range.

            Args:
                idx (int): Index of the item to retrieve (0 to 2*original_len - 1).

            Returns:
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    - low_res_image (C, H_low, W_low)
                    - upscaled_image (C, H_orig, W_orig)
                    - original_image_resized (C, H_orig, W_orig)
                    - residual_image_0_1 (C, H_orig, W_orig)
            """
            # Determine if this index corresponds to a flipped image
            should_flip = idx >= self.original_len

            # Calculate the index of the original image file
            original_idx = idx % self.original_len

            # Construct the image path
            img_path = os.path.join(self.folder_path, self.image_files[original_idx])

            try:
                # Load the PIL image (H, W, C), values in [0, 255]
                image_pil = Image.open(img_path).convert("RGB")

                # Apply horizontal flip if needed, *before* other transforms
                if should_flip:
                    image_pil = TF.hflip(image_pil)

                # 1. Original Image (target size, [0,1])
                # Convert PIL Image to Tensor (C, H, W), values in [0,1]
                # The original dimensions of image_pil are used here.
                original_image_as_tensor = TF.to_tensor(image_pil)

                # Resize to the target 'img_size' for the "original" reference image
                # Output is Tensor (C, self.img_size, self.img_size), range [0,1]
                original_image_resized = TF.resize(
                    original_image_as_tensor,
                    [self.img_size, self.img_size],
                    interpolation=TF.InterpolationMode.BICUBIC,
                    antialias=True # Use antialias for better quality, especially when downscaling
                )

                # 2. Low-Resolution Image ([0,1])
                # Calculate dimensions for the low-resolution image
                low_res_h = self.img_size // self.downscale_factor
                low_res_w = self.img_size // self.downscale_factor

                if low_res_h == 0 or low_res_w == 0: # Safety check
                    raise ValueError(
                        f"Calculated low_res dimension is zero ({low_res_h}x{low_res_w}) for img_size={self.img_size} "
                        f"and downscale_factor={self.downscale_factor}. Adjust parameters."
                    ) # write message on console

                # Create low-resolution image by downscaling the 'original_image_resized'
                # Output is Tensor (C, low_res_h, low_res_w), range [0,1]
                low_res_image = TF.resize(
                    original_image_resized.clone(), # Clone to avoid modifying original_image_resized if resize is in-place (though TF.resize is not)
                    [low_res_h, low_res_w],
                    interpolation=TF.InterpolationMode.BICUBIC,
                    antialias=True
                )

                # 3. Upscaled Image ([0,1])
                # The 'upscale_image' function is expected to take the low_res_image (Tensor, [0,1])
                # and scale_factor, returning a NumPy array (H_up, W_up, C) in [0,1].
                # The target upscaled dimensions should be (self.img_size, self.img_size).
                upscaled_image_np = upscale_image(
                    image_source=low_res_image.clone(), # Pass a clone if 'upscale_image' might modify its input
                    scale_factor=self.downscale_factor,
                    save_image=False # Assuming we don't need to save it here
                )

                if upscaled_image_np is None:
                    raise RuntimeError(f"The 'upscale_image' function returned None for image: {img_path}") # write message on console

                # Convert NumPy array (H, W, C) from upscale_image to Tensor (C, H, W)
                upscaled_image_tensor = torch.from_numpy(upscaled_image_np).permute(2, 0, 1).float()

                # Ensure the upscaled image tensor matches the dimensions of original_image_resized.
                # This is a safeguard; ideally, upscale_image handles this.
                if upscaled_image_tensor.shape[1:] != original_image_resized.shape[1:]:
                    # print(f"Warning: Dimensions of upscaled image ({upscaled_image_tensor.shape[1:]}) " # write message on console
                    #       f"do not match original_image_resized ({original_image_resized.shape[1:]}) for {img_path}. Resizing upscaled image.") # write message on console
                    upscaled_image_tensor = TF.resize(
                        upscaled_image_tensor,
                        [self.img_size, self.img_size],
                        interpolation=TF.InterpolationMode.BICUBIC,
                        antialias=True
                    )

                # 4. Residual Image ([0,1])
                # Both original_image_resized and upscaled_image_tensor are (C, self.img_size, self.img_size), range [0,1]
                # Their difference will be in range [-1, 1]
                residual_tensor = original_image_resized - upscaled_image_tensor

                # Normalize residual_tensor from [-1, 1] to [0, 1]
                residual_image_0_1 = (residual_tensor + 1.0) / 2.0

                return low_res_image, upscaled_image_tensor, original_image_resized, residual_image_0_1

            except Exception as e:
                print(f"Error loading or processing image at index {idx} (original file: {self.image_files[original_idx]}): {e}") # write message on console
                # Fallback: return dummy tensors of expected shapes to prevent dataloader from crashing.
                # However, this might hide underlying issues. Re-raising 'e' can be better for debugging.
                dummy_c = 3 # Assuming 3 color channels
                dummy_low_h = max(1, self.img_size // self.downscale_factor if self.downscale_factor > 0 else self.img_size)
                dummy_low_w = max(1, self.img_size // self.downscale_factor if self.downscale_factor > 0 else self.img_size)

                # Create zero tensors as placeholders
                _dummy_low_res = torch.zeros((dummy_c, dummy_low_h, dummy_low_w))
                _dummy_upscaled = torch.zeros((dummy_c, self.img_size, self.img_size))
                _dummy_original = torch.zeros((dummy_c, self.img_size, self.img_size))
                _dummy_residual = torch.full((dummy_c, self.img_size, self.img_size), 0.5) # Mid-gray for normalized residual

                # raise e # Option: re-raise the exception to halt on error
                return _dummy_low_res, _dummy_upscaled, _dummy_original, _dummy_residual
