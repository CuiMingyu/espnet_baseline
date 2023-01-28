#!/usr/bin/env python

# Copyright 2022 Jiawen Kang, CUHK
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""An Attentive pooling implementation for learnin fixed-length ASR memory."""

from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F
from espnet2.nets.transformer.attention import MultiHeadedAttention
from espnet2.nets.transformer.embedding import PositionalEncoding

class AttentiveStatisticsPooling(torch.nn.Module):
    """
    Multihead attentive pooling for memory repersentation.
    Layer1-Conv1d :
    Time-wisly repersent (up or down dim) the feature dim.

    Layer2 ~ Layer4:
    Add non-linearity and normalization

    Layer5-Conv1d:
    In ecapa mode: it 
    """
    def __init__(self, pooling_mode, fix_mem_length, adim, x_global=False):
        super().__init__()
        self.mode = pooling_mode
        self.eps = 1e-12
        self.nheads = fix_mem_length
        self.x_global = x_global
        if x_global:
            self.channels = adim*3 
        else:
            self.channels = adim

        if self.mode == "mhap":
            self.layers = nn.Sequential(
             nn.Conv1d(self.channels, self.nheads, kernel_size=1),
             nn.ReLU(),
             nn.BatchNorm1d(self.nheads),
             )
        elif self.mode == "ecapa":
            self.layers = nn.Sequential(
               nn.Conv1d(self.channels, 128, kernel_size=1),
               nn.ReLU(),
               nn.BatchNorm1d(128),
               nn.Tanh(), 
               nn.Conv1d(128, self.nheads, kernel_size=1),
               )
        else:
            raise KeyError
        
        self.selfattn_layer = MultiHeadedAttention(
            n_head=4,
            n_feat=self.nheads,
            dropout_rate=0.0
        )

        self.pos = PositionalEncoding(
            self.channels, dropout_rate = 0.1
        )
    

    def reverse_dim(self,x):
        """Conv1d process the last dim. So we transpose Length
        into the last dim.
        """
        return x.transpose(-2, -1)
    

    def switch_dims(self, x, mask):
        """When input is like [L,B,T,D], there is one additonal dim.
        This func switch [L,B,T,D] <-> [L*B,T,D] 
        E.g. [2,2,T,D]: (L0B0 -> B0, L1B0 -> B3, L1B1 -> B4, etc.)

        Similarly, for mask, [B,T] <-> [B*L, T]
        E.g. [2,T]: ()"""
        l, b, t, d = x.size()
        x = x.view(l*b, t, d)
        mask = mask.unsqueeze(0).repeat(l,1,1,1)
        mask = mask.view(l*b, 1, t)
        return x, mask


    def forward(self, x, mask):
        # x: [L, B, T, C] mask:[B,1,T], L for layers;B for batch 
        L, B, T, C = x.size()
        x, mask= self.switch_dims(x, mask)  # x:[L*B,T,C] mask:[LB,1,T]
        x = self.reverse_dim(x)  # x: [LB,C,T]

        if self.x_global:
            # cat global info into x
            tot = mask.sum(dim=2, keepdim=True).float()
            mean, std = self._compute_statistics(x, mask/tot)
            mean = mean.unsqueeze(2).repeat(1,1,T) # mean:[LB,C,T]
            std = std.unsqueeze(2).repeat(1, 1, T)
            x = torch.cat([x, mean, std], dim=1) # x:[LB,C*3,T]

        # apply layers
        pos_enc = self.pos(x.transpose(-2,-1)).transpose(-2,-1)
        attn = self.layers(x + pos_enc)  # x[LB, T, heads]
        attn = attn.masked_fill(mask==0, float("-inf"))
        attn = F.softmax(attn, dim=2).masked_fill(mask==0, 0.0)

        pooled_x = torch.matmul(attn, x.transpose(-2, -1))

        return pooled_x.view(L,B,self.nheads,C)

    
    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x).sum(dim)
        std = torch.sqrt(
            (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps)
        )
        return mean, std


if __name__ == "__main__":
    # This part is fo debug
    # inp: [L,B,T,D]
    # mask: [B,T]
    inp = torch.rand([1,2,3,4])

    asp = AttentiveStatisticsPooling(inp.size(-1), 64)
    mask = torch.tensor([[1,1,0],
                        [1,1,1]])
    mask = mask.unsqueeze(1)
    # mask : [B,T]
    out = asp(inp, mask)
    print(out.shape)