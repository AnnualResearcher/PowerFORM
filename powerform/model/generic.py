"""Generic graph-attention block, the alternative to the physics maps.

One textbook pre-LN graph-transformer layer replaces both FlowNet and BusNet.
Everything else about PowerForm stays: the input embeddings, type masking, the
streams V, S, F, the per-block readouts and the line-status gate.

  x_b      = LN([V_eff_b, S_eff_b])                        bus token, 2d
  t_{e->b} = LN([x_far(e,b), F_e, E_e])                    neighbour token, 4d
  a        = softmax_{e in E(b)} <Wq x_b, Wk t> / sqrt(d_h)   per head
  m_b      = sum_e a * z_e * Wv t
  [dV, dS] = MLP_node([x_b, m_b]);   V += dV,  S += dS
  F_e      = z_e * (F_e + MLP_edge(LN([x_u, x_v, F_e, E_e])))

z enters as in FlowNet, as a log z score bias and a z factor on the value, so
an open line is disconnected in both families.
"""
from __future__ import annotations

import torch
from torch import nn

from powerform.model.flownet import segment_softmax


class GenericBlock(nn.Module):
    def __init__(self, d: int, H: int = 8, hidden: int | None = None):
        super().__init__()
        if d % H:
            raise ValueError(f"d={d} not divisible by H={H}")
        self.d, self.H, self.dh = d, H, d // H
        hid = hidden or 4 * d
        self.ln_x = nn.LayerNorm(2 * d)
        self.ln_t = nn.LayerNorm(4 * d)
        self.wq = nn.Linear(2 * d, d)
        self.wk = nn.Linear(4 * d, d)
        self.wv = nn.Linear(4 * d, d)
        self.wo = nn.Linear(d, d)
        self.alpha = nn.Parameter(torch.ones(H))
        self.node_mlp = nn.Sequential(nn.Linear(3 * d, hid), nn.GELU(),
                                      nn.Linear(hid, 2 * d))
        self.ln_e = nn.LayerNorm(6 * d)
        self.edge_mlp = nn.Sequential(nn.Linear(6 * d, hid), nn.GELU(),
                                      nn.Linear(hid, d))
        # Set by ReconfigurationNet, as on FlowNet.
        self.stash_inputs = False
        self.last_inputs = None

    def forward(self, V_lat, S_lat, V_eff, S_eff, Fb, E_emb, uv, n_nodes, z):
        u, v = uv[:, 0], uv[:, 1]
        E = uv.shape[0]
        x = self.ln_x(torch.cat([V_eff, S_eff], dim=-1))            # (N,2d)
        owner = torch.cat([u, v])
        far = torch.cat([v, u])
        eidx = torch.cat([torch.arange(E, device=uv.device)] * 2)
        t = self.ln_t(torch.cat([x[far], Fb[eidx], E_emb[eidx]], dim=-1))
        q = self.wq(x)[owner].view(-1, self.H, self.dh)
        k = self.wk(t).view(-1, self.H, self.dh)
        val = self.wv(t).view(-1, self.H, self.dh)
        zs = z[eidx].clamp_min(1e-6)
        score = (q * k).sum(-1) / self.dh ** 0.5 \
            + self.alpha * torch.log(zs).unsqueeze(-1)              # (2E,H)
        a = segment_softmax(score, owner, n_nodes)
        msg = torch.zeros(n_nodes, self.H, self.dh, dtype=x.dtype,
                          device=x.device)
        msg.index_add_(0, owner, a.unsqueeze(-1) * z[eidx].view(-1, 1, 1) * val)
        msg = self.wo(msg.reshape(n_nodes, self.d))
        dV, dS = self.node_mlp(torch.cat([x, msg], dim=-1)).split(self.d, dim=-1)
        e_in = self.ln_e(torch.cat([x[u], x[v], Fb, E_emb], dim=-1))
        # What the line-status track decides from: both endpoints' [V, S], the
        # incoming F and the line embedding, 6d wide like the FlowNet inputs.
        if self.stash_inputs:
            self.last_inputs = {"e_in": e_in, "uv": uv, "z": z}
        F_new = z.unsqueeze(-1) * (Fb + self.edge_mlp(e_in))
        return V_lat + dV, S_lat + dS, F_new
