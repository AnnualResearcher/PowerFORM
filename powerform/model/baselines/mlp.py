"""MLP row of Table 1: a structure-free backbone with the decision head.

It keeps the GNN row's bus features, decision head, recipe, evaluation and
parameter budget, and has no graph: no adjacency, no message passing and no line
constants. The network sees the bus feature matrix as a flat vector and has to
learn the grid's coupling from data.

The topology is fixed across the corpus, so per-bus parameters are legitimate. The
trunk state returns to a per-bus latent through a learned scale and bias per bus,
which act as a bus identity embedding:

  z     = Trunk(flatten(X))                       # (n*14) -> H, L residual blocks
  h_i   = LN( gamma_i * W z + beta_i + MLP_bus(x_i) )
  (P,V) = BoxSigmoidHead(h_i)

Sized to the budget of the other baselines: H = 1068, L = 3, d = 256, 15,066,558
parameters.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from powerform.model.baselines.common import N_BUS_FEATS, DecisionHead

CONFIG = {"hidden": 1068, "n_blocks": 3, "d": 256}


class ResidualBlock(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.norm = nn.LayerNorm(h)
        self.net = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))

    def forward(self, z):
        return z + self.net(self.norm(z))


class MLPBackbone(nn.Module):
    """Flatten -> wide trunk -> per-bus affine modulation -> shared head."""

    def __init__(self, ctx: Dict, hidden: int, n_blocks: int, d: int):
        super().__init__()
        self.ctx = ctx
        n = ctx["n"]
        self.n = n
        self.inp = nn.Linear(n * N_BUS_FEATS, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden)
                                    for _ in range(n_blocks))
        self.to_bus = nn.Linear(hidden, d)
        self.bus_scale = nn.Parameter(torch.ones(n, d))
        self.bus_bias = nn.Parameter(torch.zeros(n, d))
        self.enc_bus = nn.Sequential(
            nn.Linear(N_BUS_FEATS, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.head = DecisionHead(d)

    def forward(self, x: torch.Tensor):
        z = self.inp(x.reshape(x.shape[0], -1))
        for blk in self.blocks:
            z = blk(z)
        broadcast = self.to_bus(z).unsqueeze(1)          # (B, 1, d)
        h = self.norm(broadcast * self.bus_scale + self.bus_bias
                      + self.enc_bus(x))
        return self.head(h, self.ctx, self.ctx["m_free"])
