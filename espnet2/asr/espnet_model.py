from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types
import logging
from espnet2.nets.nets_utils import pad_list
from espnet2.nets.transducer.error_calculator import ErrorCalculator as RNNT_ErrorCalculator
from espnet2.nets.nets_utils import th_accuracy
from espnet2.nets.transformer.add_sos_eos import add_sos_eos

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.asr.abs_espnet_model import AbsESPnetModel
from espnet2.asr.joint_network.abs_joint_network import AbsJointNetwork
from espnet2.utils.import_utils import autocast
from espnet2.asr.ctc import CTC
from espnet2.utils.pooling import AttentiveStatisticsPooling

class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        joint_network: AbsJointNetwork,
        pooling,
        ctc: CTC,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        loss_conf: dict = {"reduction": "mean"} ,

        ignore_id: int = -1,
        blank_id: int = 0,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",

        report_cer: bool = False,
        report_wer: bool = False,

        transducer_loss_type: str = "warp-transducer",
        *args, **kwargs,
    ):
        assert check_argument_types()
        if len(kwargs) != 0:
            logging.warning(f"args not used: args{args}, kwargs{kwargs}")


        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.blank_id = blank_id
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.decoder = decoder
        self.joint_network = joint_network
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.pooling = pooling

        if transducer_loss_type == "warp-transducer":
            from warprnnt_pytorch import RNNTLoss
        elif transducer_loss_type == "warp-transducer-jit":
            from transducer_loss.transducer_loss import RNNTLoss
        else:
            raise ValueError
        self.criterion = RNNTLoss(
            blank=blank_id,
            **loss_conf
        )


        if report_cer or report_wer:
            self.rnnt_error_calculator = RNNT_ErrorCalculator(
                    self.decoder, self.joint_network,
                    token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.rnnt_error_calculator = None

        # TODO: special init

    def forward(
        self,
        batch: dict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        speech = batch["speech"]
        speech_lengths = batch["speech_lengths"]
        text = batch["text"]
        text_lengths = batch["text_lengths"]
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        # encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        encoder_out, encoder_out_lens, memory, masks = self.encode(batch)

        from espnet2.nets.transducer.utils import get_decoder_input
        ys_in_pad = get_decoder_input(text, self.blank_id, self.ignore_id)

        # shape as batch, text_len, vocab
        history_state = batch['history_state']
        decoder_out, history_state = self.decoder(ys_in_pad, text_lengths, history_state)
        joint_out = self.joint_network(encoder_out.unsqueeze(2), decoder_out.unsqueeze(1))

        loss = self.criterion(joint_out, text, encoder_out_lens, text_lengths)
        # loss_transducer /= joint_out.size(0) # NOTE: ignored batch here, as rnnt_loss already computed

        cer, wer = None, None
        if not self.training and self.rnnt_error_calculator is not None:
            labels_unpad = [label[label != self.ignore_id] for label in text]
            target = pad_list(labels_unpad, self.blank_id).int().to(text.device)
            cer, wer = self.rnnt_error_calculator(encoder_out, encoder_out_lens, target)

        stats = dict(
            loss=loss.item(),
            cer=cer,
            wer=wer,
        )
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return loss, stats, weight, batch['feats'], batch['feats_lengths'], memory, history_state, masks

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, batch: dict
        # speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        speech = batch["speech"]
        speech_lengths = batch["speech_lengths"]
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        batch["feats"] = feats
        batch["feats_lengths"] = feats_lengths
        mems = batch["mems"]
        new_mask = batch['masks']

        end_point = batch['endpoint']
        complemented_version = 0

        if len(mems) != 0 and self.pooling is not None:
            mems = self.pooling(mems, new_mask)
            _, bs, length, _ = mems.size()
            new_mask = torch.ones(bs, 1, length).to(new_mask.device)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            if len(mems) == 0:
                feats, feats_lengths, masks = self.preencoder(feats, feats_lengths, None, tuple())
            else:
                feats, feats_lengths, masks = self.preencoder(feats, feats_lengths, None, mems[0])
        # new_mask = tuple()
        # if len(mems) != 0 :
        #     mems_number = mems.size(2)
        #     batchsize, d, number = masks.size()
        #     mems_tensor = torch.full([batchsize, 1, mems_number], 1).bool().to(masks.device)
        #     new_mask = torch.concat([mems_tensor, masks], 2).bool()
        # else:
        #     new_mask = masks
        if len(mems) != 0:
            # concat mems mask
            #new_mask = batch['masks']
            new_mask = torch.concat((new_mask, masks), 2)
        else:
            new_mask = masks

        # if complemented_version != 0 and len(mems) != 0:
        # #if len(mems) != 0:
        #     feats = (torch.concat((mems[0], feats[0]), 1), feats[1])
        #     feats_lengths = feats_lengths + mems.size(2)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, memory, masks = self.encoder(feats, feats_lengths, None, masks, new_mask, mems)


        #if self.pooling is not None:
        #    memory = self.pooling(memory, masks)
        #    _, bs, length, _ = memory.size()
        #    masks = torch.ones(bs, 1, length).to(masks.device)



        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens, memory, masks

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths
