import torch
import torch.nn as nn
import functools

class DenseLayer(nn.Module):
    """
    A single layer in a Residual Dense Block (RDB).
    This layer takes the input features, applies a convolution,
    and then concatenates the input features with the output of the convolution.
    """
    def __init__(self, in_channels, growth_rate, kernel_size=3, bias=True):
        """
        Args:
            in_channels (int): Number of input channels.
            growth_rate (int): Number of output channels from the convolution (how much the channels 'grow').
            kernel_size (int): Kernel size for the convolution.
            bias (bool): Whether to use bias in the convolution.
        """
        super(DenseLayer, self).__init__()
        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)
        # LeakyReLU activation function (using Mish as per original code)
        self.mish = nn.Mish(inplace=True) # Inplace was True in original, keeping it. Consider if this is always desired.

    def forward(self, x):
        """
        Forward pass of the DenseLayer.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after convolution, activation, and concatenation with input.
        """
        # Apply convolution and activation
        out = self.mish(self.conv(x))
        # Concatenate the input features with the output features along the channel dimension
        return torch.cat((x, out), 1)

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB).
    This block consists of multiple DenseLayers, where the output of each DenseLayer
    is concatenated with its input and fed to the next DenseLayer.
    Normally, a 1x1 convolution (LFF) is applied to reduce the number of channels,
    and a residual connection adds the input of the RDB to its output.
    Can optionally abandon the LFF and residual connection to output raw concatenated features.
    """
    def __init__(self, in_channels, growth_rate, num_layers, kernel_size=3, bias=True):
        """
        Args:
            in_channels (int): Number of input channels to the RDB.
            growth_rate (int): Growth rate for each DenseLayer within the RDB.
            num_layers (int): Number of DenseLayers in this RDB.
            kernel_size (int): Kernel size for convolutions in DenseLayers.
            bias (bool): Whether to use bias in convolutions.
        """
        super(ResidualDenseBlock, self).__init__()
        # Store layers in a ModuleList
        self.layers = nn.ModuleList()
        # Current number of channels, starts with the input channels
        current_channels = in_channels
        # Create num_layers DenseLayers
        for i in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate, kernel_size, bias))
            # Update current_channels by adding the growth_rate
            current_channels += growth_rate

        # Local Feature Fusion: 1x1 convolution to reduce channels back to in_channels
        # This is only used if not abandoning LFF.
        self.lff = nn.Conv2d(current_channels, in_channels, kernel_size=1, padding=0, bias=bias)
        # Scaling factor for the residual connection (often used in ESRGAN and similar models)
        self.residual_scale = 0.2

    def forward(self, x, abandon_lff_and_residual=False):
        """
        Forward pass of the ResidualDenseBlock.
        Args:
            x (torch.Tensor): Input tensor.
            abandon_lff_and_residual (bool): If True, skips the LFF and residual connection,
                                             returning the concatenated features from dense layers.
        Returns:
            torch.Tensor: Output tensor.
        """
        # Store the initial input for the potential residual connection
        residual = x
        # Pass the input through all DenseLayers, concatenating features
        concatenated_features = x
        for layer in self.layers:
            concatenated_features = layer(concatenated_features)

        if abandon_lff_and_residual:
            # Return the raw concatenated features without LFF or residual connection
            return concatenated_features
        else:
            # Apply Local Feature Fusion (1x1 convolution)
            fused_features = self.lff(concatenated_features)
            # Add the residual (scaled) to the output of LFF
            return residual + fused_features * self.residual_scale

class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB).
    This block consists of multiple ResidualDenseBlocks (RDBs).
    It can operate in a standard mode (outputting features processed by all RDBs with a final residual connection)
    or in an encoder mode (outputting richer features from the last RDB before its LFF and residual connection).
    """
    def __init__(self, in_channels, growth_rate=32, num_rdb_blocks=4, num_dense_layers_per_rdb=4,
                 kernel_size=3, bias=True, is_encoder_mode=True):
        """
        Args:
            in_channels (int): Number of input channels to the RRDB.
            growth_rate (int): Growth rate for DenseLayers within each RDB.
            num_rdb_blocks (int): Number of RDBs to stack in this RRDB.
            num_dense_layers_per_rdb (int): Number of DenseLayers within each RDB.
            kernel_size (int): Kernel size for convolutions.
            bias (bool): Whether to use bias in convolutions.
            is_encoder_mode (bool): If True, the RRDB outputs features from the last RDB
                                     before its LFF and residual, and skips the RRDB's own
                                     final residual connection. This is for extracting
                                     "hidden LR image information".
        """
        super(RRDB, self).__init__()
        self.in_channels = in_channels # Store the number of input channels
        self.growth_rate = growth_rate # Store the growth rate
        self.num_dense_layers_per_rdb = num_dense_layers_per_rdb # Store the number of dense layers per RDB
        self.is_encoder_mode = is_encoder_mode # Store the mode

        # Store RDBs in a ModuleList
        self.rdb_blocks = nn.ModuleList()
        # Create num_rdb_blocks RDBs
        # All RDBs will have the same in_channels because LFF in standard RDBs maps back to in_channels.
        for _ in range(num_rdb_blocks):
            self.rdb_blocks.append(
                ResidualDenseBlock(in_channels, growth_rate, num_dense_layers_per_rdb, kernel_size, bias)
            )
        # Scaling factor for the outer residual connection (only used if not in is_encoder_mode)
        self.residual_scale = 0.2

    def forward(self, x):
        """
        Forward pass of the RRDB.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor. If is_encoder_mode, these are richer features
                          from the last RDB. Otherwise, standard RRDB output.
        """
        # Store the initial input for the potential outer residual connection
        residual_outer = x
        current_x = x

        # Pass the input through all RDBs sequentially
        for i, rdb in enumerate(self.rdb_blocks):
            if self.is_encoder_mode and i == len(self.rdb_blocks) - 1:
                # Last RDB in encoder mode: abandon LFF and its inner residual
                current_x = rdb(current_x, abandon_lff_and_residual=True)
            else:
                # Normal operation for this RDB
                current_x = rdb(current_x, abandon_lff_and_residual=False)

        if self.is_encoder_mode:
            # In encoder mode, return the (richer) features directly from the last (modified) RDB
            # The number of channels will be in_channels + num_dense_layers_per_rdb * growth_rate
            return current_x
        else:
            # Standard RRDB operation: add the outer residual connection
            # This assumes current_x has the same number of channels as residual_outer (in_channels)
            return residual_outer + current_x * self.residual_scale