"""
U-Net Architecture for Diffusion Models

Implements a U-Net architecture suitable for DDPM, with optional
class/attribute conditioning for classifier-free guidance.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep, [optional condition]
    
    Encoder (Downsampling path)

    Middle
    
    Decoder (Upsampling path)
    
    Output: (batch_size, channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """
    U-Net architecture for diffusion models with optional conditioning.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks
        num_classes: Number of class labels for conditioning (0 = unconditional)
        cond_embed_dim: Dimension of condition embedding (if num_classes > 0)
    
    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3, 
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ...     num_classes=40,
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> cond = torch.randint(0, 2, (4, 40)).float()
        >>> out = model(x, t, cond=cond)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
        num_classes: int = 0,
        cond_embed_dim: int = 0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_classes = num_classes
        
        # Time embedding
        time_embed_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(time_embed_dim)
        
        # Condition embedding: maps attribute vector to time_embed_dim
        if num_classes > 0:
            cond_input_dim = cond_embed_dim if cond_embed_dim > 0 else num_classes
            self.cond_embed = nn.Sequential(
                nn.Linear(cond_input_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        else:
            self.cond_embed = None
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Build encoder
        # We'll store everything in a flat ModuleList but track structure
        self.input_blocks = nn.ModuleList()
        
        ch = base_channels
        input_block_chans = [ch]  # Track channels for skip connections
        current_resolution = 64
        num_levels = len(channel_mult)
        
        for level in range(num_levels):
            mult = channel_mult[level]
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                layers = nn.ModuleList([
                    ResBlock(ch, out_ch, time_embed_dim, dropout, use_scale_shift_norm)
                ])
                ch = out_ch
                
                if current_resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                
                self.input_blocks.append(layers)
                input_block_chans.append(ch)
            
            # Downsample (except at the last level)
            if level != num_levels - 1:
                self.input_blocks.append(nn.ModuleList([Downsample(ch)]))
                input_block_chans.append(ch)
                current_resolution //= 2
        
        # Middle block
        self.middle_block = nn.ModuleList([
            ResBlock(ch, ch, time_embed_dim, dropout, use_scale_shift_norm),
            AttentionBlock(ch, num_heads),
            ResBlock(ch, ch, time_embed_dim, dropout, use_scale_shift_norm),
        ])
        
        # Build decoder
        self.output_blocks = nn.ModuleList()
        
        for level in reversed(range(num_levels)):
            mult = channel_mult[level]
            out_ch = base_channels * mult
            
            for i in range(num_res_blocks + 1):
                # Pop the channel count for the skip connection
                skip_ch = input_block_chans.pop()
                
                layers = nn.ModuleList([
                    ResBlock(ch + skip_ch, out_ch, time_embed_dim, dropout, use_scale_shift_norm)
                ])
                ch = out_ch
                
                if current_resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                
                # Upsample at the end of each level (except the last level in original order)
                if level > 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    current_resolution *= 2
                
                self.output_blocks.append(layers)
        
        # Output
        self.out_norm = GroupNorm32(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights - using PyTorch defaults which work with AMP."""
        # PyTorch default initialization is reasonable for conv/linear
        # Just ensure GroupNorm is properly set
        for module in self.modules():
            if isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the U-Net.
        
        Args:
            x: Input tensor (batch_size, in_channels, height, width)
            t: Timestep tensor (batch_size,)
            cond: Optional condition tensor (batch_size, num_classes) for CFG

        Returns:
            Output tensor (batch_size, out_channels, height, width)
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Add condition embedding if provided
        if self.cond_embed is not None and cond is not None:
            cond_emb = self.cond_embed(cond.float())
            t_emb = t_emb + cond_emb
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Store skip connections
        hs = [h]
        
        # Encoder
        for block in self.input_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Decoder
        for block in self.output_blocks:
            # Pop skip connection and concatenate
            h = torch.cat([h, hs.pop()], dim=1)
            
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h


def create_model_from_config(config: dict) -> UNet:
    """
    Create a UNet from a configuration dictionary.
    
    Args:
        config: Dictionary with model configuration
    
    Returns:
        Instantiated UNet model
    """
    model_config = config['model']
    data_config = config['data']
    
    return UNet(
        in_channels=data_config['channels'],
        out_channels=data_config['channels'],
        base_channels=model_config['base_channels'],
        channel_mult=tuple(model_config['channel_mult']),
        num_res_blocks=model_config['num_res_blocks'],
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        use_scale_shift_norm=model_config['use_scale_shift_norm'],
        num_classes=model_config.get('num_classes', 0),
        cond_embed_dim=model_config.get('cond_embed_dim', 0),
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
