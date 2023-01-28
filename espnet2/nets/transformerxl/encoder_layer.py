"""Encoder self-attention layer definition"""

import torch

from torch import nn 
from espnet2.nets.transformer.layer_norm import LayerNorm


class EncoderLayer(nn.Module)
    """Encoder layer module.
    
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
        feed_forward (torch.nn.Module): Feed-forward module instance.
        conv_module (torch.nn.Module): Convolution module instance.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object"""
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
    
    def forward(self, x, mem, p, position, mask):
        x_mem = torch.cat([mem.to(x.device), x], dim=1)
        
        x = self.norm1(x + self.self_attn(x, x_mem, x_mem, mask))
        x = self.norm2(x + self.feed_forward(x))
        
        return x
    
    
    