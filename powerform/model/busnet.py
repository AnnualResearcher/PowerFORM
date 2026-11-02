"""BusNet: the bus streams V and S, one block per layer.

Every incident line proposes a voltage for a bus from its far-end voltage, its
flow and its constants, and attention combines the proposals as one Jacobi
sweep. A learned gate damps the voltage update. The injection stream is refined
residually by a structurally identical attention with separate weights, queried
from the fresh voltage, so the injection inputs stay in the stream the dispatch
readout reads.
"""
from __future__ import annotations

import torch
from torch import nn


class LineAttention(nn.Module):
    """Attention over a bus's incident lines, one slot per (line, endpoint).

    The gate z acts twice, as in FlowNet: an alpha * log z bias in the score
    takes the weight of an open line to 0, and a z factor on the value makes its
    contribution exactly 0. An open line is thereby a deleted line.
    """

    def __init__(self, d: int):
        super().__init__()
        self.tok = nn.Sequential(          # [V_far, +-F, E] -> token
            nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.score = nn.Sequential(        # [query, token] -> logit
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.val = nn.Linear(d, d)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float64))

    def forward(self, query, V_far, F_signed, E_tok, owner, n_nodes, z_slot):
        """query (N,d); V_far, F_signed, E_tok (M,d) per slot; owner (M,) the
        bus of each slot; z_slot (M,) the gate of the slot's line. Returns
        (N,d)."""
        t = self.tok(torch.cat([V_far, F_signed, E_tok], dim=-1))
        logit = self.score(torch.cat([query[owner], t], dim=-1)).squeeze(-1)
        # log(0) is kept finite by the clamp; the value gate below makes z = 0
        # exactly zero
        logit = logit + self.alpha * torch.log(z_slot.clamp_min(1e-6))
        # per-bus softmax over incident slots, shifted by the per-bus maximum
        m = torch.full((n_nodes,), -1e30, dtype=logit.dtype,
                       device=logit.device)
        m = m.scatter_reduce(0, owner, logit, reduce="amax",
                             include_self=True)
        w = torch.exp(logit - m[owner])
        z = torch.zeros(n_nodes, dtype=w.dtype, device=w.device)
        z.index_add_(0, owner, w)
        a = (w / z.clamp_min(1e-30)[owner]).unsqueeze(-1)
        val = self.val(t)
        val = z_slot.unsqueeze(-1) * val
        out = torch.zeros(n_nodes, t.shape[-1], dtype=t.dtype,
                          device=t.device)
        out.index_add_(0, owner, a * val)
        return out


class BusNet(nn.Module):
    """V_{k+1} = gated blend(V_k, neighbour attention);
    S_{k+1} = S_k + update from attention queried at V_{k+1}."""

    def __init__(self, d: int):
        super().__init__()
        self.att_v = LineAttention(d)
        self.att_s = LineAttention(d)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.up_v = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, d))
        self.up_s = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                  nn.Linear(d, d))

    def forward(self, V_lat, S_lat, V_eff, Fb, E_emb, uv, n_nodes, z):
        u, v = uv[:, 0], uv[:, 1]
        # Two slots per line: the u-side slot sees far-end v, the v-side slot
        # sees far-end u. The flow enters as +F at u and -F at v, so the token
        # carries the KCL mismatch. A slack neighbour presents its constant
        # V_eff, so the level anchor propagates locally from the slack.
        owner = torch.cat([u, v])
        V_far = torch.cat([V_eff[v], V_eff[u]])
        F_signed = torch.cat([Fb, -Fb])
        E_tok = torch.cat([E_emb, E_emb])
        z_slot = torch.cat([z, z])

        prop = self.att_v(V_lat, V_far, F_signed, E_tok, owner, n_nodes,
                          z_slot)
        g = self.gate(torch.cat([V_lat, prop], dim=-1))
        V_new = (1.0 - g) * V_lat + g * self.up_v(prop)

        ctx = self.att_s(V_new, V_far, F_signed, E_tok, owner, n_nodes,
                         z_slot)
        S_new = S_lat + self.up_s(torch.cat([ctx, V_new], dim=-1))
        return V_new, S_new
