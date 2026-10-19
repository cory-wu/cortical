"""Locality-masked RNN (Khona et al. 2023) with an optional hard input mask."""

from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from .sheet import locality_mask
from .wiring import N_INPUT


class MaskedRNN(nn.Module):
    """``h_{t+1} = relu(W h_t + W_in u_t + b_ih + b_hh)``, ``o = W_out h + b_out``.

    ``W`` is restricted to unit pairs within distance ``d`` on the LxL torus
    and ``W_in`` to ``input_mask`` (I, L²). Call ``apply_masks()`` after every
    optimizer step; the readout is dense.
    """

    rec_mask: Tensor
    in_mask: Tensor

    def __init__(self, L: int, d: float, input_mask: np.ndarray, n_output: int = 17) -> None:
        super().__init__()
        n_units = L * L
        self.rnn = nn.RNN(N_INPUT, n_units, nonlinearity="relu")
        self.readout = nn.Linear(n_units, n_output)
        self.register_buffer("rec_mask", torch.from_numpy(locality_mask(L, d)).float())
        # nn.RNN stores input weights as (hidden, input): transpose the (I, N) mask.
        self.register_buffer("in_mask", torch.from_numpy(input_mask.T.copy()).float())
        self.apply_masks()

    @torch.no_grad()
    def apply_masks(self) -> None:
        # nn.Module.__getattr__ is typed Tensor | Module; these are Parameters.
        recurrent = cast(Tensor, self.rnn.weight_hh_l0)
        inputs = cast(Tensor, self.rnn.weight_ih_l0)
        recurrent.mul_(self.rec_mask)
        inputs.mul_(self.in_mask)

    def forward(self, x: Tensor) -> Tensor:
        """(T, B, I) inputs -> (T, B, n_output) logits, from a zero hidden state."""
        hidden, _ = self.rnn(x)
        return self.readout(hidden)
