#!/usr/bin/env python3
#  2021, Carnegie Mellon University;  Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear Projection."""

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from typeguard import check_argument_types
from typing import Tuple
from espnet2.nets.nets_utils import make_pad_mask

import torch
from espnet2.nets.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from espnet2.nets.transformer.subsampling import (
    SuperFrame,
    Conv2dSubsampling,
    CausalConv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    check_short_utt,
    TooShortUttError,
)
import logging


class DownSampling(AbsPreEncoder):
    """DownSampling Preencoder.
    
            positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_size (int): Input dimension.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        pos_enc_layer_type: str = "rel_pos",
        input_layer: str = "conv2d",
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        padding_idx: int = -1,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()
        self._input_layer = input_layer
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            raise ValueError("Not used anymore")
            # pos_enc_class = LegacyRelPositionalEncoding
            logging.warning("Using legacy_rel_pos and it will be deprecated in the future.")
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "causal_conv2d":
            self.embed = CausalConv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "superframe":
            self.embed = torch.nn.Sequential(
                SuperFrame(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(pos_enc_class(output_size, positional_dropout_rate))
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        
        # self.pos = pos_enc_class(output_size, positional_dropout_rate)

        # Compatability with encoder attention type checking
        self.pos_enc_type = pos_enc_layer_type

    def forward(
        self,
        xs_pad: torch.Tensor,
        xs_pad_lens: torch.Tensor,
        prev_states: torch.Tensor = None,
        mems=None,
    ) -> Tuple[Tuple[torch.Tensor,torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward."""
        masks = (~make_pad_mask(xs_pad_lens)[:, None, :]).to(xs_pad.device)
        shape0 = xs_pad.shape[1]

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks, mems)
        elif self._input_layer == "superframe":
            # floor_v = xs_pad_lens // 3
            floor_v = torch.div(xs_pad_lens, 3, rounding_mode="floor")
            masks = (~make_pad_mask(floor_v - 2)[:, None, :]).to(xs_pad.device)
            xs_pad = self.embed(xs_pad)
        elif isinstance(self.embed, CausalConv2dSubsampling):
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        if type(xs_pad) == tuple:
            shape1 = xs_pad[0].shape[1]
        else:
            shape1 = xs_pad.shape[1]
        # xs_pad_lens = xs_pad_lens // (shape0 // shape1) # // length, but not precise
        shape_temp = torch.div(shape0, shape1, rounding_mode='trunc')
        xs_pad_lens = torch.div(xs_pad_lens, shape_temp, rounding_mode='trunc')
        return xs_pad, xs_pad_lens, masks

    def output_size(self) -> int:
        """Get the output size."""
        return self._output_size
