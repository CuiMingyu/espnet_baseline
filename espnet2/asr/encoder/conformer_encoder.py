# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

from typing import Optional
from typing import Tuple

import logging
import torch

from typeguard import check_argument_types

from espnet2.nets.conformer.convolution import ConvolutionModule
from espnet2.nets.conformer.encoder_layer import EncoderLayer
from espnet2.nets.nets_utils import get_activation
from espnet2.nets.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)

from espnet2.nets.transformer.layer_norm import LayerNorm
from espnet2.nets.transformer.multi_layer_conv import Conv1dLinear
from espnet2.nets.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet2.nets.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet2.nets.transformer.repeat import repeat
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class ConformerEncoder(AbsEncoder):
    """Conformer encoder module.

    Args:
        output_size (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
        self,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        activation = get_activation(activation_type)

        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            raise ValueError("Not used anymore")
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        xs_pad_lens: torch.Tensor,
        prev_states: torch.Tensor = None,
        masks: torch.Tensor = None,
        mem_masks: torch.Tensor = None,
        mems=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """
        # 12 encoder 
        # query key value pos_emb(query)
        # key = mem + query
        # 
        intermediate_outputs = []
        memory_list = []
        for layer_idx, encoder_layer in enumerate(self.encoders):
            if len(mems) != 0:
                xs_pad, mem_masks, cache, memory = encoder_layer(xs_pad, mem_masks, None, mems[layer_idx])
            else:
                xs_pad, mem_masks, cache, memory = encoder_layer(xs_pad, mem_masks, None, mems)
            encoder_output = xs_pad
            if isinstance(encoder_output, tuple):
                encoder_output = encoder_output[0]
                if self.normalize_before:
                    encoder_output = self.after_norm(encoder_output)
            intermediate_outputs.append(encoder_output)
            memory_list.append(encoder_output)

        memory = memory_list[0].unsqueeze(0)
        for i in range(len(memory_list) - 1):
            memory = torch.concat((memory, memory_list[i + 1].unsqueeze(0)), 0)

        xs_pad = intermediate_outputs[-1]

        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, memory, masks

        # xs_pad, mem_masks, cache, mems = self.encoders(xs_pad, mem_masks, None, mems)
        # if isinstance(xs_pad, tuple):
        #     xs_pad = xs_pad[0]
        # if self.normalize_before:
        #     xs_pad = self.after_norm(xs_pad)

        # olens = masks.squeeze(1).sum(1)
        # return xs_pad, olens, mems