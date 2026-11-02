"""FlowNet: the line stream F, one block per layer.

Each bus splits its injection over its incident lines with a voltage-driven
softmax, and the line variable is the signed sum of the contributions from its
two endpoints. A second stage lets lines that share a bus interact. The gate z
biases the routing and multiplies the output, so a line with z = 0 carries
exactly zero flow.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlowNet(nn.Module):
    """Conservation routing, then bus-mediated line-line attention.

        score_{i->e} = f(V_i, V_j, E_e) + alpha * log z_e
        rho_{i->e}   = softmax over the lines incident to i
        F1_e         = rho_{u->e} * val(S_u) - rho_{v->e} * val(S_v)
        F_e          = z_e * (mix(F1_e + F_prev) + F2_e)

    The softmax is the right normalisation for routing: the fractions of a
    conserved quantity sum to one. Voltage only routes and injection is the
    only value, which keeps the two roles separate. All of it runs per head on
    head slices of the d-wide latents.
    """

    def __init__(self, d: int, n_heads: int = 8):
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.dh = d // n_heads
        # Set by ReconfigurationNet. Plain attributes, so the state_dict does
        # not depend on whether a line-status track is attached.
        self.stash_inputs = False
        self.last_inputs = None
        self.ln_v = nn.LayerNorm(d)
        self.ln_s = nn.LayerNorm(d)
        self.score = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, n_heads))
        self.value = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.mix = nn.Linear(d, d)
        self.ln_f = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.ones(n_heads, dtype=torch.float64))
        # Stage 2: one token per (line, endpoint bus), referenced to that bus.
        # Each bus runs attention over its incident tokens with its voltage as
        # the query, and both bus contexts feed back into the line.
        self.ln_l = nn.LayerNorm(d)
        self.W_tok = nn.Linear(4 * d, d)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(2 * d, d)

    def forward(self, V_eff, S_eff, Fb, E_emb, uv, z):
        E = uv.shape[0]
        N = V_eff.shape[0]
        u, v = uv[:, 0], uv[:, 1]
        Vn = self.ln_v(V_eff)
        Sn = self.ln_s(S_eff)
        Fn = self.ln_f(Fb)

        # What the line-status track decides from: this block's normalised
        # inputs. The block's output is multiplied by z, so a decision read
        # from it would shrink its own evidence as a line closes.
        if self.stash_inputs:
            self.last_inputs = {"Vn": Vn, "Sn": Sn, "Fb": Fn,
                                "E_emb": E_emb, "uv": uv, "z": z}

        # directed incidences: (bus, line) pairs for both endpoints
        src = torch.cat([u, v])
        oth = torch.cat([v, u])
        eid = torch.arange(E, device=uv.device)
        eid2 = torch.cat([eid, eid])
        E2 = E_emb[eid2]
        z2 = z[eid2].clamp_min(1e-6)

        s = self.score(torch.cat([Vn[src], Vn[oth], E2], dim=-1))   # (2E,H)
        s = s + self.alpha.unsqueeze(0) * torch.log(z2).unsqueeze(-1)
        rho = segment_softmax(s, src, N)

        val = self.value(Sn)
        val = val.view(N, self.h, self.dh)
        contrib = rho.unsqueeze(-1) * val[src]                      # (2E,H,dh)
        sign = torch.cat([torch.ones(E), -torch.ones(E)]).to(
            dtype=V_eff.dtype, device=uv.device)
        contrib = contrib * sign.view(-1, 1, 1)

        F1 = torch.zeros(E, self.h, self.dh, dtype=V_eff.dtype,
                         device=uv.device)
        F1.index_add_(0, eid2, contrib)
        F1 = F1.reshape(E, -1)
        F1 = self.mix(F1 + Fn)

        Ln = self.ln_l(F1)
        tok = self.W_tok(torch.cat(
            [Ln[eid2], E_emb[eid2], Vn[src], Vn[src] - Vn[oth]], dim=-1))
        Q = self.W_q(Vn).view(N, self.h, self.dh)
        Kt = self.W_k(tok).view(-1, self.h, self.dh)
        Vt = self.W_v(tok).view(-1, self.h, self.dh)
        sc = (Q[src] * Kt).sum(-1) / self.dh ** 0.5                 # (2E,H)
        w = segment_softmax(sc, src, N)
        ctx = torch.zeros(N, self.h, self.dh, dtype=V_eff.dtype,
                          device=uv.device)
        ctx.index_add_(0, src, w.unsqueeze(-1) * Vt)
        ctx = ctx.reshape(N, -1)
        F2 = self.W_o(torch.cat([ctx[u], ctx[v]], dim=-1))
        F = F1 + F2
        return z.unsqueeze(-1) * F


def segment_softmax(values, index, n):
    """Softmax of (M, H) `values` over the rows sharing an `index`, per column.

    The per-segment maximum is subtracted first, so the exponential cannot
    overflow on a large score.
    """
    neg = torch.full((n, values.shape[1]), -1e30,
                     device=values.device, dtype=values.dtype)
    m = neg.scatter_reduce(0, index.unsqueeze(-1).expand_as(values), values,
                           reduce="amax", include_self=True)
    ex = torch.exp(values - m[index])
    den = torch.zeros(n, values.shape[1], device=values.device,
                      dtype=values.dtype)
    den.scatter_add_(0, index.unsqueeze(-1).expand_as(ex), ex)
    return ex / den[index].clamp_min(1e-30)
