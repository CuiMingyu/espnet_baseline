
"""Transducer score interface module."""

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch



@dataclass
class Hypothesis:
    """Default hypothesis definition for Transducer search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        torch.Tensor,
    ]
    lm_state: Union[Dict[str, Any], List[Any]] = None


@dataclass
class ExtendedHypothesis(Hypothesis):
    """Extended hypothesis definition for NSC beam search and mAES."""

    dec_out: List[torch.Tensor] = None
    lm_scores: torch.Tensor = None

class ScorerInterface:
    """Scorer interface for beam search.

    The scorer performs scoring of the all tokens in vocabulary.

    Examples:
        * Search heuristics
            * :class:`espnet.nets.scorers.length_bonus.LengthBonus`
        * Decoder networks of the sequence-to-sequence models
            * :class:`espnet.nets.pytorch_backend.nets.transformer.decoder.Decoder`
            * :class:`espnet.nets.pytorch_backend.nets.rnn.decoders.Decoder`
        * Neural language models
            * :class:`espnet.nets.pytorch_backend.lm.transformer.TransformerLM`
            * :class:`espnet.nets.pytorch_backend.lm.default.DefaultRNNLM`
            * :class:`espnet.nets.pytorch_backend.lm.seq_rnn.SequentialRNNLM`

    """

    def init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        return None

    def select_state(self, state: Any, i: int, new_id: int = None) -> Any:
        """Select state with relative ids in the main beam search.

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label index to select a state if necessary

        Returns:
            state: pruned state

        """
        return None if state is None else state[i]

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys

        """
        raise NotImplementedError

    def final_score(self, state: Any) -> float:
        """Score eos (optional).

        Args:
            state: Scorer state for prefix tokens

        Returns:
            float: final score

        """
        return 0.0


class BatchScorerInterface(ScorerInterface):
    """Batch scorer interface."""

    def batch_init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        return self.init_state(x)

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        warnings.warn(
            "{} batch score is implemented through for loop not parallelized".format(
                self.__class__.__name__
            )
        )
        scores = list()
        outstates = list()
        for i, (y, state, x) in enumerate(zip(ys, states, xs)):
            score, outstate = self.score(y, state, x)
            outstates.append(outstate)
            scores.append(score)
        scores = torch.cat(scores, 0).view(ys.shape[0], -1)
        return scores, outstates


class PartialScorerInterface(ScorerInterface):
    """Partial scorer interface for beam search.

    The partial scorer performs scoring when non-partial scorer finished scoring,
    and receives pre-pruned next tokens to score because it is too heavy to score
    all the tokens.

    Examples:
         * Prefix search for connectionist-temporal-classification models
             * :class:`espnet.nets.scorers.ctc.CTCPrefixScorer`

    """

    def score_partial(
        self, y: torch.Tensor, next_tokens: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            y (torch.Tensor): 1D prefix token
            next_tokens (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        raise NotImplementedError


class BatchPartialScorerInterface(BatchScorerInterface, PartialScorerInterface):
    """Batch partial scorer interface for beam search."""

    def batch_score_partial(
        self,
        ys: torch.Tensor,
        next_tokens: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            next_tokens (torch.Tensor): torch.int64 tokens to score (n_batch, n_token).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for ys that has a shape `(n_batch, n_vocab)`
                and next states for ys
        """
        raise NotImplementedError

    # """Decoder interface for Transducer models."""

    # def init_state(
    #     self,
    #     batch_size: int,
    #     device: torch.device,
    # ) -> Union[
    #     Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    # ]:
    #     """Initialize decoder states.

    #     Args:
    #         batch_size: Batch size.

    #     Returns:
    #         state: Initial decoder hidden states.

    #     """
    #     raise NotImplementedError("init_state(...) is not implemented")

    # def score(
    #     self,
    #     hyp: Hypothesis,
    #     cache: Dict[str, Any],
    # ) -> Tuple[
    #     torch.Tensor,
    #     Union[
    #         Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    #     ],
    #     torch.Tensor,
    # ]:
    #     """One-step forward hypothesis.

    #     Args:
    #         hyp: Hypothesis.
    #         cache: Pairs of (dec_out, dec_state) for each token sequence. (key)

    #     Returns:
    #         dec_out: Decoder output sequence.
    #         new_state: Decoder hidden states.
    #         lm_tokens: Label ID for LM.

    #     """
    #     raise NotImplementedError("score(...) is not implemented")

    # def batch_score(
    #     self,
    #     hyps: Union[List[Hypothesis], List[ExtendedHypothesis]],
    #     dec_states: Union[
    #         Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    #     ],
    #     cache: Dict[str, Any],
    #     use_lm: bool,
    # ) -> Tuple[
    #     torch.Tensor,
    #     Union[
    #         Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    #     ],
    #     torch.Tensor,
    # ]:
    #     """One-step forward hypotheses.

    #     Args:
    #         hyps: Hypotheses.
    #         dec_states: Decoder hidden states.
    #         cache: Pairs of (dec_out, dec_states) for each label sequence. (key)
    #         use_lm: Whether to compute label ID sequences for LM.

    #     Returns:
    #         dec_out: Decoder output sequences.
    #         dec_states: Decoder hidden states.
    #         lm_labels: Label ID sequences for LM.

    #     """
    #     raise NotImplementedError("batch_score(...) is not implemented")

    # def select_state(
    #     self,
    #     batch_states: Union[
    #         Tuple[torch.Tensor, Optional[torch.Tensor]], List[torch.Tensor]
    #     ],
    #     idx: int,
    # ) -> Union[
    #     Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    # ]:
    #     """Get specified ID state from decoder hidden states.

    #     Args:
    #         batch_states: Decoder hidden states.
    #         idx: State ID to extract.

    #     Returns:
    #         state_idx: Decoder hidden state for given ID.

    #     """
    #     raise NotImplementedError("select_state(...) is not implemented")

    # def create_batch_states(
    #     self,
    #     states: Union[
    #         Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    #     ],
    #     new_states: List[
    #         Union[
    #             Tuple[torch.Tensor, Optional[torch.Tensor]],
    #             List[Optional[torch.Tensor]],
    #         ]
    #     ],
    #     l_tokens: List[List[int]],
    # ) -> Union[
    #     Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    # ]:
    #     """Create decoder hidden states.

    #     Args:
    #         batch_states: Batch of decoder states
    #         l_states: List of decoder states
    #         l_tokens: List of token sequences for input batch

    #     Returns:
    #         batch_states: Batch of decoder states

    #     """
    #     raise NotImplementedError("create_batch_states(...) is not implemented")