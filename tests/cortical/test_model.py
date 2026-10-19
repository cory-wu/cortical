"""Masked LM-RNN: masks hold through init and training, oriented correctly."""

import numpy as np
import torch

from cortical.model import MaskedRNN
from cortical.sheet import locality_mask, square_patch
from cortical.wiring import N_INPUT, input_mask

L, D = 10, 2
N = L * L


def _model(layout):
    torch.manual_seed(0)
    return MaskedRNN(L, D, input_mask(L, layout))


def _masked_out(model, layout):
    rec_off = ~torch.from_numpy(locality_mask(L, D))
    in_off = ~torch.from_numpy(input_mask(L, layout)).T  # nn.RNN stores (N, I)
    return model.rnn.weight_hh_l0[rec_off], model.rnn.weight_ih_l0[in_off]


def test_masked_entries_zero_after_init(small_layout):
    model = _model(small_layout)
    rec, inp = _masked_out(model, small_layout)
    assert (rec == 0).all() and (inp == 0).all()
    assert (model.rnn.weight_hh_l0 != 0).sum() == locality_mask(L, D).sum()
    assert (model.rnn.weight_ih_l0 != 0).sum() == input_mask(L, small_layout).sum()


def test_masks_survive_adam_step(small_layout):
    model = _model(small_layout)
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    x = torch.randn(5, 2, N_INPUT)
    model(x).square().mean().backward()
    opt.step()
    model.apply_masks()
    rec, inp = _masked_out(model, small_layout)
    assert (rec == 0).all() and (inp == 0).all()


def test_ring1_drives_only_its_patch(small_layout):
    model = _model(small_layout)
    patch = np.zeros(N, dtype=bool)
    patch[square_patch(L, *small_layout["ring1"])] = True
    # Orientation: ring-1 input 1 has weights exactly on its patch units.
    assert ((model.rnn.weight_ih_l0[:, 1] != 0).numpy() == patch).all()
    with torch.no_grad():
        for p in (model.rnn.bias_ih_l0, model.rnn.bias_hh_l0, model.rnn.weight_hh_l0):
            p.zero_()
        x = torch.zeros(1, 1, N_INPUT)
        x[0, 0, 1] = 1.0
        h, _ = model.rnn(x)
    assert (h[0, 0].numpy()[~patch] == 0).all()
    assert (h[0, 0].numpy()[patch] > 0).any()


def test_forward_shape(small_layout):
    assert _model(small_layout)(torch.randn(7, 3, N_INPUT)).shape == (7, 3, 17)


def test_parameter_count_matches_nn_rnn(small_layout):
    n_params = sum(p.numel() for p in _model(small_layout).parameters())
    assert n_params == N * N + N * 115 + 2 * N + N * 17 + 17
