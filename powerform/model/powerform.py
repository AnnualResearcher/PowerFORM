"""PowerForm: K untied blocks over three physical streams.

The hidden state is indexed by physical quantity: V bus voltage, S bus
injection, F line flow. Every block advances all three and every decision is
read from the stream it belongs to. ``maps`` selects what a block computes: the
physics maps (FlowNet, then BusNet) or a generic graph-attention block with the
same streams. See ../ARCHITECTURE.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from powerform.data.batch import standardize_segments
from powerform.model.busnet import BusNet
from powerform.model.flownet import FlowNet
from powerform.model.generic import GenericBlock
from powerform.model.readouts import DecisionReadout
from powerform.model.spec import TaskSpec, standardize

_EPS = 1e-8


class PowerForm(nn.Module):
    """K untied blocks with type-masked bus variables and decisions (p_g, vm_set).

    n_cost_feats    per-bus cost features embedded into the initial S: 12 with
                    the economic-dispatch prior, 10 without, 0 for a single-grid
                    model whose costs never vary
    maps            "physics" or "generic"
    generic_hidden  width of the generic block's MLPs
    type_masking    False: constants only initialise the state and are not
                    re-imposed at each block
    merged_readout  True: both decisions read the whole [S, V]
    """

    def __init__(self, d_model: int = 216, K: int = 8, n_cost_feats: int = 0,
                 maps: str = "physics", generic_hidden: int | None = None,
                 type_masking: bool = True, merged_readout: bool = False):
        super().__init__()
        if maps not in ("physics", "generic"):
            raise ValueError(f"maps must be 'physics' or 'generic', got {maps!r}")
        if maps == "physics" and generic_hidden is not None:
            raise ValueError("generic_hidden needs maps='generic'")
        self.d = d_model
        self.K = K
        self.n_cost_feats = n_cost_feats
        self.maps = maps
        self.generic_hidden = generic_hidden
        self.type_masking = bool(type_masking)
        self.merged_readout = bool(merged_readout)
        # Constants: loads of the PQ buses, raw and standardised.
        self.embed_s_const = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        if n_cost_feats:
            # Initial S at the free buses. S_const reaches only the clamped
            # buses, so the decision buses need the cost signal in their own
            # evolving stream.
            self.embed_s_var = nn.Sequential(
                nn.Linear(n_cost_feats + 2, d_model), nn.GELU(),
                nn.Linear(d_model, d_model))
        self.embed_v_const = nn.Sequential(     # slack voltage
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.embed_v_init = nn.Sequential(      # initial V per bus
            nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        # Line constants: [log r, log x, sign x, b_chg, tap, shift, rating, m_edge]
        self.embed_edge = nn.Sequential(
            nn.Linear(8, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        # Attached by ReconfigurationNet. Attaching adds state_dict keys, so a
        # strict load of a reconfiguration checkpoint into a plain trunk fails.
        self.z_stream = None
        if maps == "generic":
            self.gen_blocks = nn.ModuleList(
                GenericBlock(d_model, hidden=generic_hidden) for _ in range(K))
        else:
            self.flow_nets = nn.ModuleList(FlowNet(d_model) for _ in range(K))
            self.bus_nets = nn.ModuleList(BusNet(d_model) for _ in range(K))
        self.readouts = nn.ModuleList(
            DecisionReadout(d_model, merged=self.merged_readout)
            for _ in range(K))
        self.double()

    def forward(self, inst, spec: TaskSpec) -> dict:
        """history: {p_g, vm_set} of every block, the last one also under the
        top-level keys. ell_K: the terminal log-gate, with a track attached."""
        dtype = torch.float64
        dev = inst.p_load.device
        n, e = inst.n_nodes, inst.n_edges

        # A merged instance (data/batch.py) standardises per segment, which is
        # the same computation as standardising each instance on its own.
        segs = getattr(inst, "node_segments", None)
        if segs is None:
            p_hat = standardize(inst.p_load).to(dtype)
            q_hat = standardize(inst.q_load).to(dtype)
        else:
            p_hat = standardize_segments(inst.p_load, segs).to(dtype)
            q_hat = standardize_segments(inst.q_load, segs).to(dtype)

        S_const = self.embed_s_const(torch.stack(
            [inst.p_load, inst.q_load, p_hat, q_hat], dim=-1))
        v_sl = inst.v_slack_scalar
        V_const = self.embed_v_const(torch.stack(
            [v_sl, torch.zeros_like(v_sl)], dim=-1))

        tg = inst.tgrid
        E_emb = self.embed_edge(torch.stack(
            [torch.log(inst.edge_r.clamp_min(_EPS)),
             torch.log(inst.edge_x.clamp_min(_EPS)),
             torch.sign(tg.br_x).to(dtype), tg.br_b.to(dtype),
             tg.tap.to(dtype), tg.shift_rad.to(dtype),
             inst.s_max.clamp_max(20.0), spec.m_edge.to(dtype)], dim=-1))

        V_lat = self.embed_v_init(torch.stack(
            [inst.is_sub.to(dtype), spec.m_node.to(dtype), p_hat, q_hat,
             inst.depth], dim=-1))
        if self.n_cost_feats:
            S_lat = self.embed_s_var(torch.cat(
                [inst.cost_feats.to(dtype), spec.p_min.unsqueeze(-1),
                 spec.p_max.unsqueeze(-1)], dim=-1))
        else:
            S_lat = torch.zeros(n, self.d, dtype=dtype, device=dev)
        Fb = torch.zeros(e, self.d, dtype=dtype, device=dev)
        z = torch.ones(e, dtype=dtype, device=dev)
        ell = torch.zeros(e, dtype=dtype, device=dev)
        if self.z_stream is not None:
            self.z_stream.reset()

        inj_free = spec.m_node.unsqueeze(-1)          # S variable here
        v_const_mask = inst.is_sub.unsqueeze(-1)      # V constant here

        if not self.type_masking:
            S_lat = torch.where(inj_free, S_lat, S_const)
            V_lat = torch.where(v_const_mask, V_const, V_lat)
        history = []
        for k in range(self.K):
            if self.type_masking:
                S_eff = torch.where(inj_free, S_lat, S_const)
                V_eff = torch.where(v_const_mask, V_const, V_lat)
            else:
                S_eff, V_eff = S_lat, V_lat
            # A constant for the whole forward, unless a track makes it a
            # variable the model updates between blocks. The track multiplies
            # z and does not replace it.
            z_eff = z if self.z_stream is None else z * torch.exp(ell)
            if self.maps == "generic":
                block = self.gen_blocks[k]
                V_lat, S_lat, Fb = block(
                    V_lat, S_lat, V_eff, S_eff, Fb, E_emb, inst.edge_uv, n,
                    z_eff)
                if self.z_stream is not None:
                    ell = ell + self.z_stream(k, block.last_inputs)
            else:
                flow = self.flow_nets[k]
                Fb = flow(V_eff, S_eff, Fb, E_emb, inst.edge_uv, z_eff)
                if self.z_stream is not None:
                    ell = ell + self.z_stream(k, flow.last_inputs)
                # The same z_eff FlowNet consumed: the gate acts on both halves
                # of the layer, and block k's decision reaches block k + 1.
                V_lat, S_lat = self.bus_nets[k](
                    V_lat, S_lat, V_eff, Fb, E_emb, inst.edge_uv, n, z_eff)
            p_g, vm_set = self.readouts[k](S_lat, V_lat, spec)
            history.append({"p_g": p_g, "vm_set": vm_set})

        out = {"p_g": p_g, "vm_set": vm_set, "history": history,
               "V_lat": V_lat, "S_lat": S_lat, "Fb": Fb}
        if self.z_stream is not None:
            out["ell_K"] = ell
        return out
