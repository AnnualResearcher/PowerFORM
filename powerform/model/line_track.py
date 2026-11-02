"""The line-status track: the reconfiguration decision as a stream of its own.

Each line gets a latent that the blocks advance and the gate is read from it,
so a decision changes the network the next block routes over:

    zlat_0   = 0                         per line, width z_d
    zlat_k+1 = zlat_k + U_k(inputs_k)    U_k per block, zero-initialised
    ell_k    = w(zlat_k) - w(0)          w shared across blocks
    z_eff    = z * exp(ell_k)            multiplies the next block's routing

U_k starts at zero, so zlat and ell stay 0, exp(0) = 1 exactly, and the first
forward is bit-identical to the trunk without a track. The gate is exp(ell)
rather than a sigmoid because a sigmoid's identity sits at an infinite logit,
where the gradient is dead.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LineStatusTrack(nn.Module):
    """U_k reads what the block reads rather than what it produces: the
    endpoint voltage and injection latents, the incoming line latent, the line
    embedding and the current gate. The row keeps the branch's from/to order;
    that order is corpus data, and relabelling buses preserves it."""

    N_INPUT_BLOCKS = 6          # Vn[u], Vn[v], Sn[u], Sn[v], Fb, E_emb

    def __init__(self, d: int, K: int, z_d: int = 32):
        super().__init__()
        self.z_d = z_d
        in_dim = self.N_INPUT_BLOCKS * d + 1        # + log z
        self.updates = nn.ModuleList(
            [nn.Linear(in_dim, z_d) for _ in range(K)])
        for lin in self.updates:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        # One readout shared across blocks: the latent is a persistent state,
        # so the map from state to gate must not change between blocks.
        self.readout = nn.Linear(z_d, 1)
        self._zlat = None

    def reset(self) -> None:
        """Clear the per-line state; PowerForm.forward does this per scenario."""
        self._zlat = None

    @staticmethod
    def block_inputs(sig: dict) -> torch.Tensor:
        """A block's stashed inputs as one row per line, 6d + 1 wide on both
        map families. The generic block hands over its own line-update input."""
        logz = torch.log(sig["z"].clamp_min(1e-6)).unsqueeze(-1)
        if "e_in" in sig:
            return torch.cat([sig["e_in"], logz], dim=-1)
        uv = sig["uv"]
        u, v = uv[:, 0], uv[:, 1]
        Vn, Sn = sig["Vn"], sig["Sn"]
        return torch.cat([Vn[u], Vn[v], Sn[u], Sn[v],
                          sig["Fb"], sig["E_emb"], logz], dim=-1)

    def forward(self, k: int, sig: dict) -> torch.Tensor:
        """The increment of ell: w(zlat_{k+1}) - w(zlat_k). Only changes of the
        latent reach the gate, so the readout's bias cancels."""
        upd = self.updates[k](self.block_inputs(sig))
        prev = torch.zeros_like(upd) if self._zlat is None else self._zlat
        new = prev + upd
        self._zlat = new
        return (self.readout(new) - self.readout(prev))[:, 0]


class ReconfigurationNet(nn.Module):
    """A PowerForm trunk whose line gate is a stream the trunk itself updates.

    The forward returns the terminal log-gate ell_K; the lines to open are those
    with the lowest values.
    """

    def __init__(self, trunk, z_d: int = 32):
        super().__init__()
        self.trunk = trunk
        trunk.z_stream = LineStatusTrack(trunk.d, trunk.K, z_d)
        blocks = trunk.gen_blocks if trunk.maps == "generic" else trunk.flow_nets
        for block in blocks:
            block.stash_inputs = True
        self.double()

    def forward(self, inst, spec):
        return self.trunk(inst, spec)["ell_K"]
