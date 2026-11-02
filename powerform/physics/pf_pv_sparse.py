"""Sparse PV power flow, for grids where a dense Ybus does not fit.

The same formulation as ``physics/newton_solver.py``, with a sparse Jacobian and
a no-load start.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def build_ybus_sparse(g) -> sp.csr_matrix:
    """Sparse pi-model Ybus, the same formulas as ``build_ybus``.

        t = tau exp(j phi)
        Y_ff = (y + j b/2) / tau^2      Y_ft = -y / conj(t)
        Y_tf = -y / t                   Y_tt =  y + j b/2
    """
    n = int(g.p_load.shape[0])
    f, t = _np(g.f_bus).astype(np.int64), _np(g.t_bus).astype(np.int64)
    y = 1.0 / (_np(g.br_r) + 1j * _np(g.br_x))
    bc2 = 1j * _np(g.br_b) / 2.0
    tau = _np(g.tap)
    tc = tau * np.exp(1j * _np(g.shift_rad))

    y_ff = (y + bc2) / tau ** 2
    y_ft = -y / np.conj(tc)
    y_tf = -y / tc
    y_tt = y + bc2

    rows = np.concatenate([f, f, t, t, np.arange(n)])
    cols = np.concatenate([f, t, f, t, np.arange(n)])
    vals = np.concatenate([y_ff, y_ft, y_tf, y_tt,
                           _np(g.gs) + 1j * _np(g.bs)])
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n),
                         dtype=np.complex128).tocsr()


def _dSbus_dV(Y: sp.csr_matrix, V: np.ndarray):
    """MATPOWER dSbus_dV in polar form, sparse."""
    n = len(V)
    Ibus = Y @ V
    diagV = sp.diags(V)
    diagIbus = sp.diags(Ibus)
    diagVnorm = sp.diags(V / np.abs(V))
    dS_dVa = 1j * diagV @ (diagIbus - Y @ diagV).conj()
    dS_dVm = diagV @ (Y @ diagVnorm).conj() + diagIbus.conj() @ diagVnorm
    assert dS_dVa.shape == (n, n)
    return dS_dVa.tocsr(), dS_dVm.tocsr()


def noload_voltage(Y: sp.csr_matrix, vm_set: np.ndarray, slack: np.ndarray,
                   pq: np.ndarray) -> Optional[np.ndarray]:
    """Zero-injection voltages: solve ``(Y V)_pq = 0`` for the PQ buses.

    PV and slack buses are held at ``vm_set`` with zero angle. Returns ``None``
    if the reduced matrix is singular, which happens when the PQ set contains a
    component with no PV or slack bus, such as an island created by a branch
    removal. The caller then uses a flat start and the convergence check at the
    end decides.
    """
    fixed = ~pq
    V = np.where(fixed, vm_set, 1.0).astype(np.complex128)
    if not pq.any():
        return V
    try:
        A = Y[pq][:, pq].tocsc()
        rhs = -(Y[pq][:, fixed] @ V[fixed])
        sol = spla.spsolve(A, rhs)
    except (RuntimeError, ValueError):
        return None
    if not np.all(np.isfinite(sol)):
        return None
    V[pq] = sol
    return V


def pf_pv_sparse(
    g,
    p_g,
    vm_set,
    pv_mask,
    n_iter: int = 50,
    tol: float = 1e-10,
    V_init: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Polar Newton AC power flow with PV buses; slack angle pinned to 0.

    Arguments mirror ``transmission_pf_newton_pv``. Returns ``V`` (complex, per
    bus), ``q_g`` (reactive dispatch derived at PV and slack), ``p_slack`` and
    ``residual`` (largest absolute mismatch on the solved equations, pu).
    """
    Y = build_ybus_sparse(g)
    n = Y.shape[0]
    slack = _np(g.is_slack).astype(bool)
    pv = _np(pv_mask).astype(bool) & ~slack
    pq = ~slack & ~pv
    ns_idx = np.where(~slack)[0]
    pq_idx = np.where(pq)[0]
    n_ns = len(ns_idx)

    p_load, q_load = _np(g.p_load), _np(g.q_load)
    pg = _np(p_g)
    vms = _np(vm_set)
    # Q is specified only at PQ buses; PV and slack reactive output is free.
    S_spec = -((p_load - pg) + 1j * q_load)

    if V_init is None:
        V_init = noload_voltage(Y, vms, slack, pq)
    if V_init is None:
        Vm = np.where(pv | slack, vms, 1.0)
        Va = np.zeros(n)
    else:
        Vm = np.where(pv | slack, vms, np.abs(V_init))
        Va = np.angle(V_init)
        Va = Va - Va[slack][0]
        Va[slack] = 0.0

    def mismatch(Vm_, Va_):
        V_ = Vm_ * np.exp(1j * Va_)
        m_ = V_ * np.conj(Y @ V_) - S_spec
        return np.concatenate([m_.real[ns_idx], m_.imag[pq_idx]])

    for _ in range(n_iter):
        F = mismatch(Vm, Va)
        f_inf = np.max(np.abs(F)) if F.size else 0.0
        if f_inf < tol:
            break
        V = Vm * np.exp(1j * Va)
        dS_dVa, dS_dVm = _dSbus_dV(Y, V)
        J = sp.bmat([
            [dS_dVa.real[ns_idx][:, ns_idx], dS_dVm.real[ns_idx][:, pq_idx]],
            [dS_dVa.imag[pq_idx][:, ns_idx], dS_dVm.imag[pq_idx][:, pq_idx]],
        ], format="csc")
        dx = spla.spsolve(J, -F)
        if not np.all(np.isfinite(dx)):
            break

        step = 1.0
        accepted = False
        for _ls in range(20):
            Va_t, Vm_t = Va.copy(), Vm.copy()
            Va_t[ns_idx] += step * dx[:n_ns]
            Vm_t[pq_idx] = np.maximum(Vm_t[pq_idx] + step * dx[n_ns:], 0.2)
            if np.max(np.abs(mismatch(Vm_t, Va_t))) < f_inf:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            # On jumper-heavy grids no damped step reduces ||F||_inf from the
            # start, while the plain Newton step does make progress. The
            # residual returned below still reports the outcome.
            Va_t, Vm_t = Va.copy(), Vm.copy()
            Va_t[ns_idx] += dx[:n_ns]
            Vm_t[pq_idx] = np.maximum(Vm_t[pq_idx] + dx[n_ns:], 0.2)
        Va, Vm = Va_t, Vm_t

    V = Vm * np.exp(1j * Va)
    S_calc = V * np.conj(Y @ V)
    resid = np.max(np.abs(np.concatenate([
        (S_calc.real - S_spec.real)[ns_idx],
        (S_calc.imag - S_spec.imag)[pq_idx]]))) if n_ns else 0.0

    q_g = np.where(pv | slack, S_calc.imag + q_load, 0.0)
    p_slack = np.where(slack, S_calc.real + p_load, 0.0)
    return {"V": V, "q_g": q_g, "p_slack": p_slack, "residual": float(resid)}
