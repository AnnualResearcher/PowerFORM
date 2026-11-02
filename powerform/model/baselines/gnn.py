"""GNN row of Table 1: a generic message-passing backbone with the decision head.

A GraphSAGE-style network over the bus graph with edge-conditioned messages, mean
aggregation and a residual update, and the line constants as edge features. It
builds in no physics: no conservation routing, no voltage/flow alternation and no
unrolled solver state.

  h_i^0   = MLP_bus(x_i)                       # 14 bus features
  e_ij    = MLP_edge(c_ij)                     # 9 line constants, both directions
  m_ij^k  = MLP_msg([h_i^k, h_j^k, e_ij])
  h_j^k+1 = LN(h_j^k + MLP_upd([h_j^k, mean_i m_ij^k]))     # K = 8 rounds

Sized to the budget of the other baselines: d = 500, K = 8, 15,041,502 parameters.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from powerform.model.baselines.common import (
    N_BUS_FEATS, N_EDGE_FEATS, DecisionHead,
)

CONFIG = {"d": 500, "k": 8}


class MessagePassingLayer(nn.Module):
    """One GraphSAGE-style round with edge-conditioned messages."""

    def __init__(self, d: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.upd = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, e, src, dst, deg):
        b, _, d = h.shape
        m = self.msg(torch.cat(
            [h[:, src], h[:, dst], e.unsqueeze(0).expand(b, -1, -1)], dim=-1))
        agg = torch.zeros_like(h).index_add_(1, dst, m) / deg
        return self.norm(h + self.upd(torch.cat([h, agg], dim=-1)))


class GNNBackbone(nn.Module):
    """K rounds of message passing, then the shared decision head."""

    def __init__(self, ctx: Dict, d: int, k: int):
        super().__init__()
        self.ctx = ctx
        self.enc_bus = nn.Sequential(
            nn.Linear(N_BUS_FEATS, d), nn.GELU(), nn.Linear(d, d))
        self.enc_edge = nn.Sequential(
            nn.Linear(N_EDGE_FEATS, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList(MessagePassingLayer(d) for _ in range(k))
        self.head = DecisionHead(d)
        n = ctx["n"]
        deg = torch.zeros(n, dtype=torch.float32, device=ctx["dst"].device)
        deg.index_add_(0, ctx["dst"], torch.ones_like(ctx["dst"],
                                                      dtype=deg.dtype))
        self.register_buffer("deg", deg.clamp_min(1.0).view(1, n, 1))

    def forward(self, x: torch.Tensor):
        h = self.enc_bus(x)
        e = self.enc_edge(self.ctx["edge_feat"])
        for layer in self.layers:
            h = layer(h, e, self.ctx["src"], self.ctx["dst"], self.deg)
        return self.head(h, self.ctx, self.ctx["m_free"])
