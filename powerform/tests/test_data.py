"""Data and physics checks, each with a failure mode: a corrupted value must be caught.

Everything synthetic runs anywhere. Checks on the corpora skip with the fetch command
when the data is absent. ``--deep`` also re-measures the App. B load factors from the
datakit corpus, which reads two columns of the full bus table.

    python powerform/tests/test_data.py [--deep]
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from powerform.data import cost_features as CF  # noqa: E402
from powerform.data import datakit_loader as DL  # noqa: E402
from powerform.data import opfdata as O  # noqa: E402
from powerform.data.allocation import allocate_bus_p_exact  # noqa: E402
from powerform.data.binding import reconfiguration_binding  # noqa: E402
from powerform.data.corpus import datakit_export as X  # noqa: E402
from powerform.data.corpus import fetch_opfdata as F  # noqa: E402
from powerform.data.corpus import fetch_test_split as T  # noqa: E402
from powerform.data.corpus import load_factors as LF  # noqa: E402
from powerform.data.opfdata_case import allocate_bus_p_to_gens, build_instance  # noqa: E402
from powerform.model.spec import TaskSpec  # noqa: E402
from powerform.paths import DATA_HOME, ENV_VAR, shown  # noqa: E402
from powerform.physics.feasibility import VM_UNCHECKED, open_lines, solve_loss  # noqa: E402
from powerform.physics.newton_solver import (TransmissionGrid, build_ybus,  # noqa: E402
                                              transmission_pf_newton_pv)
from powerform.physics.pf_pv_sparse import build_ybus_sparse, pf_pv_sparse  # noqa: E402

torch.set_num_threads(1)
F64 = torch.float64
CASE30 = "pglib_opf_case30_ieee"
DEEP = False


class Skip(Exception):
    pass


def has_shard(case: str, variant: str, split: str) -> bool:
    return O.shard_path(case, variant, split).exists()


def need_shard(case: str, variant: str = "fulltop", split: str = "test") -> None:
    if not has_shard(case, variant, split):
        raise Skip(f"{case}/{variant}/{split} absent; fetch with `{O.fetch_command(case, variant)}`")


def need_datakit() -> None:
    if not X.RAW_DIR.is_dir() or not X.OUT_DIR.is_dir():
        raise Skip(f"datakit corpus absent under {shown(X.DATAKIT_HOME)}; see "
                   f"powerform/data/corpus/README.md, step 4")


# ---------------------------------------------------------------------------
# Synthetic: no data needed
# ---------------------------------------------------------------------------
def check_data_home() -> None:
    probe = ("import json;"
             "from powerform import paths as P;"
             "from powerform.data import opfdata as O;"
             "from powerform.data.corpus import datakit_export as X;"
             "print(json.dumps({'home': str(P.DATA_HOME), 'shards': str(O.DATA_ROOT),"
             "'files': [str(v) for v in O.TEST_SPLIT_DIRS.values()],"
             "'raw': str(X.RAW_DIR), 'hold': str(X.OUT_DIR)}))")

    def roots(value):
        env = dict(os.environ)
        env.pop(ENV_VAR, None)
        if value is not None:
            env[ENV_VAR] = value
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                           cwd=str(REPO), env=env)
        assert r.returncode == 0, r.stderr[-400:]
        return json.loads(r.stdout.strip().splitlines()[-1])

    def under(got: dict, home: Path) -> bool:
        vals = [got["home"], got["shards"], got["raw"], got["hold"], *got["files"]]
        return all(Path(v) == home or home in Path(v).parents for v in vals)

    default = REPO / "datasets"
    got = roots(None)
    assert Path(got["home"]) == default and under(got, default), got
    assert roots("")["home"] == got["home"], "an empty value must mean unset"
    with tempfile.TemporaryDirectory() as t:
        home = Path(t).resolve()
        moved = roots(str(home))
        assert Path(moved["home"]) == home and under(moved, home), moved
        assert not under(moved, default), "the override moved nothing"
        assert Path(moved["raw"]) == home / "case500_datakit" / "case500_30k" / "case500_goc" / "raw"
    # shown() never writes an absolute path
    assert shown(REPO / "results" / "a.json") == "results/a.json"
    outside = Path(tempfile.gettempdir()).resolve() / "elsewhere" / "b.json"
    assert shown(outside) == "b.json", shown(outside)
    print("POWERFORM_DATA moves every corpus root, unset means <repo>/datasets; shown() "
          "never writes an absolute path")


def check_splits() -> None:
    assert O.SPLIT_BOUNDS == {"train": (0, 13500), "val": (13500, 14250), "test": (14250, 15000)}
    assert (F.TRAIN_END, F.VAL_END, F.N_GROUP) == (O.TRAIN_END, O.VAL_END, O.N_GROUP)
    assert (T.TEST_START, T.TEST_END) == O.SPLIT_BOUNDS["test"]
    idx = list(range(100, 233))
    picked = O.subsample(idx, 17)
    assert picked == O.subsample(idx, 17) and len(picked) == 17 and picked[0] == 100 and picked[-1] == 232
    gaps = np.diff(picked)
    assert gaps.max() - gaps.min() <= 1, "not an even subsample"
    assert O.subsample(idx, 0) == idx and O.subsample(idx, 999) == idx

    def pool(ids, split="val", variant="fulltop"):
        return SimpleNamespace(case="case57", variant=variant, split=split, example_indices=ids,
                               alias="case57")
    O.assert_disjoint([pool(list(range(0, 9)), "train")], [pool([13500, 13501])])
    for train, ev, what in (([13500], [13500], "shared index"),
                            ([0], [14250], "index outside its split")):
        try:
            O.assert_disjoint([pool(train, "train")], [pool(ev)])
        except SystemExit:
            continue
        raise AssertionError(f"{what} not refused")
    # a different variant of the same example is a different scenario
    O.assert_disjoint([pool([13500], "train", "n1")], [pool([13500])])
    print("split bounds agree across the loader and both fetchers; even deterministic "
          "subsample; train/eval overlap refused")


class NetworkTouched(RuntimeError):
    pass


def _no_network(*_a, **_k):
    raise NetworkTouched("urlopen called during an offline check")


def check_fetchers_stay_offline() -> None:
    assert F.source_url(CASE30, "fulltop") == (
        "https://storage.googleapis.com/gridopt-dataset/dataset_release_1/pglib_opf_case30_ieee_0.tar.gz")
    assert F.source_url(CASE30, "n1") == (
        "https://storage.googleapis.com/gridopt-dataset/dataset_release_1_nminusone/pglib_opf_case30_ieee_0.tar.gz")
    assert T.source_url("case2000", "fulltop") == (
        "https://storage.googleapis.com/gridopt-dataset/dataset_release_1/pglib_opf_case2000_goc_0.tar.gz")
    assert T.member_index("x/group_0/example_14250.json") == 14250 and T.member_index("x/y.json") is None
    real = urllib.request.urlopen
    urllib.request.urlopen = _no_network
    try:
        with tempfile.TemporaryDirectory() as t:
            for mod, argv in ((F, ["--case", CASE30, "--variant", "fulltop", "--verify-only", "--out", t]),
                              (F, ["--all", "--variant", "both", "--dry-run", "--out", t]),
                              (T, ["--case", "case2000", "--variant", "fulltop", "--dry-run", "--out", t])):
                old = sys.argv
                sys.argv = [mod.__name__] + argv
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        mod.main()
                except SystemExit:
                    pass
                finally:
                    sys.argv = old
        try:
            urllib.request.urlopen("https://example.invalid")
        except NetworkTouched:
            pass
        else:
            raise AssertionError("the trap does not fire")
    finally:
        urllib.request.urlopen = real
    try:
        T.default_out("case30")
    except SystemExit:
        pass
    else:
        raise AssertionError("a test split nothing reads was not refused")
    assert T.default_out("case6470") == O.TEST_SPLIT_DIRS["case6470"]
    print("fetch URLs exact; --verify-only and --dry-run never open a URL (trap fires when called)")


def _rewrite_line(shard: Path, line_no: int, edit) -> None:
    with gzip.open(shard, "rb") as fh:
        lines = fh.read().split(b"\n")
    lines[line_no] = edit(lines[line_no])
    with gzip.open(shard, "wb") as fh:
        fh.write(b"\n".join(lines))


def check_shard_verification() -> None:
    """A fake test shard written the way fetch() writes one."""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        d = root / CASE30 / "fulltop"
        d.mkdir(parents=True)
        idx = list(range(F.VAL_END, F.N_GROUP))
        shard = d / "test.jsonl.gz"
        body = b"".join(json.dumps({"i": i, "v": i * 0.125}, separators=(",", ":")).encode() + b"\n"
                        for i in idx)
        with gzip.open(shard, "wb") as fh:
            fh.write(body)
        man = {"case": CASE30, "variant": "fulltop", "release": F.RELEASE["fulltop"], "group": 0,
               "source_url": F.source_url(CASE30, "fulltop"), "n_group": F.N_GROUP,
               "split_rule": {"train": [0, F.TRAIN_END], "val": [F.TRAIN_END, F.VAL_END],
                              "test": [F.VAL_END, F.N_GROUP]},
               "splits": {"test": {"count": len(idx), "example_indices": idx,
                                   "uncompressed_bytes": len(body), "file": "test.jsonl.gz"}}}
        mp = d / "manifest.json"
        mp.write_text(json.dumps(man))
        expected = {f"opfdata_hf/{CASE30}/fulltop/test": F.shard_digest(shard)}
        verify = lambda deep: F.verify(CASE30, "fulltop", root, splits=("test",), deep=deep,
                                       expected=expected)
        assert verify(True) == [], verify(True)
        # a manifest index moved out of its split
        moved = json.loads(mp.read_text())
        moved["splits"]["test"]["example_indices"][0] = 0
        mp.write_text(json.dumps(moved))
        assert verify(False), "manifest index outside its split not caught"
        mp.write_text(json.dumps(man))
        # one digit changed: same line and byte counts, only the digest sees it
        _rewrite_line(shard, 3, lambda b: b.replace(b'"i":', b'"j":'))
        assert verify(False) == [] and any("sha256" in p for p in verify(True)), verify(True)
        # one scenario dropped from the shard
        with gzip.open(shard, "rb") as fh:
            data = fh.read().split(b"\n", 1)[1]
        with gzip.open(shard, "wb") as fh:
            fh.write(data)
        assert any("lines" in p for p in verify(True)), verify(True)
        shard.unlink()
        assert any("missing" in p for p in verify(False))
    print("shard verification: moved index, changed digit (digest only) and dropped line are caught")


def check_test_split_verification() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        d = root / "fulltop"
        d.mkdir()
        for i in range(T.TEST_START, T.TEST_END):
            with gzip.open(d / f"example_{i}.json.gz", "wb") as fh:
                fh.write(b"{}")
        assert T.verify("case2000", "fulltop", root, deep=False) == []
        got = T.tree_digest(d)
        good = {"opfdata_case2000/fulltop": {"sha256": got["sha256"]}}
        bad = {"opfdata_case2000/fulltop": {"sha256": "0" * 64}}
        assert T.verify("case2000", "fulltop", root, deep=True, expected=good) == []
        assert T.verify("case2000", "fulltop", root, deep=True, expected=bad), "wrong content not caught"
        (d / "example_14300.json.gz").unlink()
        (d / "example_14301.json.gz.part").write_bytes(b"")
        assert len(T.verify("case2000", "fulltop", root)) == 2, "missing file or partial not caught"
    checks = json.loads(F.CHECKSUMS.read_text())
    assert "case500_datakit/holdout_opfdata" in checks and "opfdata_case2000/fulltop" in checks
    assert not any(k.split("/")[0] not in ("_about", "case500_datakit", "opfdata_case2000",
                                           "opfdata_case6470", "opfdata_hf", "pglib") for k in checks)
    print("test-split verification: wrong content, missing file and partial download are caught")


def _small_grid():
    """Five buses in a ring: slack 0, generator 2, loads elsewhere."""
    f = torch.tensor([0, 1, 2, 3, 4, 0])
    t = torch.tensor([1, 2, 3, 4, 0, 2])
    e = len(f)
    one = lambda v, n: torch.full((n,), v, dtype=F64)
    p_load = torch.tensor([0.0, 0.3, 0.1, 0.4, 0.2], dtype=F64)
    q_load = 0.3 * p_load
    g = TransmissionGrid(f_bus=f, t_bus=t, br_r=one(0.01, e), br_x=one(0.08, e), br_b=one(0.02, e),
                         tap=one(1.0, e), shift_rad=one(0.0, e), gs=one(0.0, 5), bs=one(0.0, 5),
                         p_load=p_load, q_load=q_load,
                         is_slack=torch.tensor([True, False, False, False, False]))
    m_free = torch.tensor([True, False, True, False, False])
    inst = SimpleNamespace(tgrid=g, n_edges=e, n_nodes=5, p_load=p_load, m_free=m_free,
                           edge_uv=torch.stack([f, t], dim=1), edge_r=g.br_r.clone(),
                           edge_x=g.br_x.clone(), s_max=one(5.0, e))
    lab = SimpleNamespace(pg_bus=torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0], dtype=F64),
                          qg_bus=torch.zeros(5, dtype=F64),
                          vm=torch.tensor([1.02, 1.0, 1.01, 1.0, 1.0], dtype=F64))
    return inst, lab


def check_power_flow() -> None:
    inst, lab = _small_grid()
    g = inst.tgrid
    ns = ~g.is_slack
    pv = inst.m_free & ns
    p_bus = torch.where(pv, lab.pg_bus, torch.zeros(5, dtype=F64))
    d = transmission_pf_newton_pv(g, p_bus, lab.vm, pv)
    s = pf_pv_sparse(g, p_bus, lab.vm, pv)
    assert float(d["residual"]) < 1e-9 and s["residual"] < 1e-9, (d["residual"], s["residual"])
    dV = float(np.abs(d["V"].numpy() - s["V"]).max())
    assert dV < 1e-9, dV
    assert np.abs(build_ybus(g).numpy() - build_ybus_sparse(g).toarray()).max() < 1e-12
    # Kirchhoff at the solution: the injected power equals the load balance
    V = d["V"]
    S = V * (build_ybus(g) @ V).conj()
    want = d["p_slack"] + p_bus - g.p_load
    assert float((S.real - want).abs().max()) < 1e-9
    # failure mode: the wrong dispatch does not solve the same voltages
    other = transmission_pf_newton_pv(g, p_bus * 1.5, lab.vm, pv)
    assert float((other["V"] - V).abs().max()) > 1e-4

    # feasibility
    base = solve_loss(inst, lab, VM_UNCHECKED)
    assert base is not None and base > 0, base
    wide = (torch.full((5,), 0.8, dtype=F64), torch.full((5,), 1.2, dtype=F64))
    assert solve_loss(inst, lab, wide) == base
    tight = (torch.full((5,), 1.015, dtype=F64), torch.full((5,), 1.2, dtype=F64))
    assert solve_loss(inst, lab, tight) is None, "a voltage outside the box was accepted"
    try:
        solve_loss(inst, lab, None)
    except ValueError:
        pass
    else:
        raise AssertionError("a forgotten voltage box was accepted")
    # An opened line is gone from the network, charging included.
    opened = open_lines(inst, (5,))
    assert opened.n_edges == 5 and inst.n_edges == 6, "open_lines mutated the instance"
    assert opened.edge_uv.shape == (5, 2) and opened.tgrid.br_b.shape == (5,)
    Y0, Y1 = build_ybus(g), build_ybus(opened.tgrid)
    assert abs(complex(Y1[0, 2])) == 0.0 and abs(complex(Y0[0, 2])) > 0.0
    L = solve_loss(opened, lab, VM_UNCHECKED)
    assert L is not None and L != base
    print(f"dense == sparse to {dV:.1e}; KCL holds at the solution; box enforced, a forgotten box "
          f"refused; an opened line leaves Ybus")


def check_allocation() -> None:
    labels = SimpleNamespace(gen_bus=torch.tensor([0, 1, 1]), c1=torch.tensor([1.0, 2.0, 3.0], dtype=F64),
                             c2=torch.tensor([0.1, 0.2, 0.4], dtype=F64),
                             pmin_gen=torch.zeros(3, dtype=F64),
                             pmax_gen=torch.tensor([2.0, 3.0, 3.0], dtype=F64))
    p_bus = torch.tensor([1.5, 3.0, 0.0], dtype=F64)
    p = allocate_bus_p_exact(p_bus, labels, 3)
    assert abs(float(p[0]) - 1.5) < 1e-9 and abs(float(p[1:].sum()) - 3.0) < 1e-9, p
    # the marginal costs of units sharing a bus meet at the optimum
    m = labels.c1 + 2 * labels.c2 * p
    assert abs(float(m[1] - m[2])) < 1e-6, m
    # one unit per bus: the closed form is the bisection's answer
    single = SimpleNamespace(gen_bus=torch.tensor([0, 1]), c1=labels.c1[:2], c2=labels.c2[:2],
                             pmin_gen=torch.zeros(2, dtype=F64), pmax_gen=labels.pmax_gen[:2])
    p_bus = torch.tensor([1.5, 3.7], dtype=F64)
    closed = allocate_bus_p_exact(p_bus, single, 2)
    assert closed.tolist() == [1.5, 3.0], "the bus total was not clipped to the unit box"
    assert float((closed - allocate_bus_p_to_gens(p_bus, single)).abs().max()) < 1e-9

    a = DL.equal_marginal_alloc(3.3, np.array([1e-3, 2e-3, 0.0]), np.array([1.0, 2.0, 3.0]),
                                np.zeros(3), np.full(3, 2.0))
    assert abs(a.sum() - 3.3) < 1e-9 and (a >= 0).all() and (a <= 2.0).all(), a
    assert a[0] >= a[1] >= a[2], "the cheapest unit did not fill first"
    print("allocation meets the bus total at equal marginal cost; closed form == bisection")


def check_cost_features() -> None:
    n = 6
    eligible = torch.tensor([False, True, True, True, False, True])
    gen_bus = torch.tensor([1, 2, 3, 5, 5])
    c1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 6.0], dtype=F64)
    c2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=F64)
    w = torch.tensor([10.0, 20.0, 30.0, 10.0, 30.0], dtype=F64)
    feats = CF.bus_cost_features(c1, c2, w, gen_bus, n, eligible)
    assert abs(float(feats[eligible].mean(0).abs().max())) < 1e-12
    assert abs(float(feats[eligible].std(0).max()) - 1.0) < 1e-12
    assert torch.equal(feats[~eligible], torch.zeros(2, 2, dtype=F64))
    # currency scale does not move the merit-order feature
    assert float((CF.bus_cost_features(c1 * 7, c2 * 7, w, gen_bus, n, eligible) - feats).abs().max()) < 1e-12
    # capacity-weighted at the shared bus
    c1_bus, _ = CF.bus_cost_means(c1, c2, w, gen_bus, n)
    assert abs(float(c1_bus[5]) - (4.0 * 10 + 6.0 * 30) / 40) < 1e-12

    single = torch.zeros(n, dtype=torch.bool)
    single[2] = True
    assert torch.equal(CF.bus_cost_features(c1, c2, w, gen_bus, n, single), torch.zeros(n, 2, dtype=F64))

    p_min = torch.zeros(n, dtype=F64)
    p_max = torch.tensor([0.0, 3.0, 3.0, 3.0, 0.0, 3.0], dtype=F64)
    c1_b, c2_b = CF.bus_cost_means(c1, c2, w, gen_bus, n)
    ed = CF.economic_dispatch_prior(5.0, c1_b, c2_b, p_min, p_max, eligible)
    assert abs(float(ed[:, 0].sum()) - 5.0) < 1e-9, "the prior does not balance the load"
    assert torch.equal(ed[~eligible], torch.zeros(2, 2, dtype=F64))
    assert bool(((ed[eligible, 1] >= 0) & (ed[eligible, 1] <= 1)).all())
    tight = CF.economic_dispatch_prior(100.0, c1_b, c2_b, p_min, p_max, eligible)
    assert abs(float(tight[:, 0].sum()) - float(p_max.sum())) < 1e-9, "a tight grid must saturate"
    other = CF.economic_dispatch_prior(6.0, c1_b, c2_b, p_min, p_max, eligible)
    assert float((other - ed).abs().max()) > 1e-3, "the prior does not read the load"

    ctx = CF.global_context_features(
        p_load_pu=torch.tensor([1.0, 2.0], dtype=F64), q_load_pu=torch.tensor([0.5, 0.5], dtype=F64),
        gen_pmax_pu=torch.tensor([4.0, 2.0], dtype=F64), gen_qmax_pu=torch.tensor([1.0, 1.0], dtype=F64),
        gen_qmin_pu=torch.tensor([-1.0, -1.0], dtype=F64), vm_min=torch.full((2,), 0.9, dtype=F64),
        vm_max=torch.full((2,), 1.1, dtype=F64), rate_pu=torch.tensor([0.0, 2.0], dtype=F64),
        br_r=torch.tensor([0.0, 0.0], dtype=F64), br_x=torch.tensor([0.1, 0.5], dtype=F64), n_bus=2)
    assert ctx.shape == (2, 8) and torch.equal(ctx[0], ctx[1])
    assert abs(float(ctx[0, 2]) - 0.5) < 1e-12 and abs(float(ctx[0, 3]) - 0.25) < 1e-12
    assert abs(float(ctx[0, 5]) - np.log1p(1.0)) < 1e-12 and abs(float(ctx[0, 7]) - np.log1p(10.0)) < 1e-9
    print("cost columns are a z-score of capacity-weighted means, scale-free; the ED prior balances "
          "the load and saturates on a tight grid; the graph context matches a hand calculation")


def check_binding() -> None:
    n = 4
    inst = SimpleNamespace(tgrid=SimpleNamespace(is_slack=torch.tensor([True, False, False, False])),
                           n_edges=3, p_load=torch.ones(n, dtype=F64))
    lab = SimpleNamespace(pg_bus=torch.tensor([9.0, 0.5, -1e-8, 0.0], dtype=F64),
                          qg_bus=torch.tensor([1.0, 0.1, 0.1, 0.0], dtype=F64),
                          vm=torch.tensor([1.02, 1.01, 1.03, 1.0], dtype=F64))
    spec0 = TaskSpec(m_node=torch.tensor([True, True, True, False]), m_edge=torch.zeros(3, dtype=torch.bool),
                     p_min=torch.zeros(n, dtype=F64), p_max=torch.full((n,), 7.0, dtype=F64),
                     vm_min=torch.full((n,), 0.94, dtype=F64), vm_max=torch.full((n,), 1.06, dtype=F64))
    i2, s = reconfiguration_binding(inst, lab, spec0)
    assert i2 is not inst and i2.p_load is inst.p_load, "the instance must be a shallow copy"
    assert s.p_max.tolist() == [0.0, 0.5, 0.0, 0.0], s.p_max
    assert torch.equal(s.p_min, torch.zeros(n, dtype=F64))
    assert s.vm_min.tolist() == [0.94, 1.01, 1.03, 0.94] and s.vm_max.tolist() == [1.06, 1.01, 1.03, 1.06]
    assert s.m_node.tolist() == [False, True, True, True] and not bool(s.m_edge.any())
    assert spec0.p_max.tolist() == [7.0] * n, "the pool's spec was mutated"
    print("the binding pins the dispatch on p_max and the voltage at generators, without mutating the pool")


def check_table1_split_rules() -> None:
    ids = sorted(set(range(30000)) - {16496, 16498, 16582})
    train, hold = DL.table1_split(ids)
    assert hold == list(range(29500, 30000)) and max(train) < 29500 and len(train) == 29497
    missing = [i for i in ids if i != 29777]
    try:
        DL.table1_split(missing)
    except SystemExit:
        pass
    else:
        raise AssertionError("a hold-out with a missing scenario was accepted")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "train_opfdata").mkdir()
        (root / "holdout_opfdata").mkdir()
        for i in range(29500, 30000):
            (root / "holdout_opfdata" / X.scenario_filename(i)).touch()
        (root / "train_opfdata" / X.scenario_filename(7)).touch()
        tr, ho = DL.pyg_split(root)
        assert len(tr) == 1 and len(ho) == 500
        leak = root / "train_opfdata" / X.scenario_filename(29500)
        leak.touch()
        try:
            DL.pyg_split(root)
        except SystemExit:
            pass
        else:
            raise AssertionError("a hold-out id among the training files was accepted")
    print("hold-out is exactly ids 29,500-29,999; a missing or leaked scenario is refused")


def check_missing_corpus_messages() -> None:
    """A missing Table 1 corpus says what is missing and where to get it."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        raw = DL.raw_dir_of(tmp)
        calls = (("corpus_ids", lambda: DL.corpus_ids(tmp)),
                 ("load_case500", lambda: DL.load_case500(tmp, scenario_ids=[29500])),
                 ("template_gen_df", lambda: DL.template_gen_df(tmp, 0.0)),
                 ("build_static", lambda: DL.build_static(raw)),
                 ("pyg_split", lambda: DL.pyg_split(tmp)))
        for name, call in calls:
            try:
                call()
            except SystemExit as e:
                assert "README" in str(e) and str(tmp) in str(e), (name, str(e))
            else:
                raise AssertionError(f"{name} did not refuse a missing corpus")
        # failure mode: with the table in place the guard must not fire
        raw.mkdir(parents=True)
        pd.DataFrame({"load_scenario_idx": [3.0, 3.0, 7.0]}).to_parquet(raw / "bus_data.parquet")
        assert DL.corpus_ids(tmp) == [3, 7]
    print(f"{len(calls)} datakit entry points refuse a missing corpus, naming the place and the data README")


def check_datakit_config() -> None:
    cfg = yaml.safe_load((Path(X.__file__).parent / "datakit" / "config_case500_30k.yaml").read_text())
    assert cfg["settings"]["data_dir"] == "./datasets/case500_datakit/case500_30k"
    assert cfg["load"]["scenarios"] == 30000 and cfg["settings"]["seed"] == 0
    print("the shipped config writes under the default data home, 30,000 scenarios, seed 0")


def check_datakit_config_matches_record() -> None:
    cfg = yaml.safe_load((Path(X.__file__).parent / "datakit" / "config_case500_30k.yaml").read_text())
    log = X.RAW_DIR / "args.log"
    if not log.exists():
        raise Skip(f"generator record absent under {shown(X.DATAKIT_HOME)}; see "
                   f"powerform/data/corpus/README.md, step 4")
    blocks = re.split(r"^New generation started at .*$", log.read_text(), flags=re.M)
    recorded = yaml.safe_load(blocks[-1])

    def mismatches(c: dict) -> list:
        out = []
        for sec, vals in c.items():
            for k, v in vals.items():
                # where the generator wrote is a local choice
                if (sec, k) != ("settings", "data_dir") and recorded.get(sec, {}).get(k, "<none>") != v:
                    out.append(f"{sec}.{k}")
        return out

    assert not mismatches(cfg), mismatches(cfg)
    wrong = json.loads(json.dumps(cfg))
    wrong["load"]["sigma"] = 0.21
    assert mismatches(wrong) == ["load.sigma"]
    print("every key of the shipped config matches the generator's record; sigma 0.21 is caught")


# ---------------------------------------------------------------------------
# On the corpora
# ---------------------------------------------------------------------------
def check_local_corpora_verify() -> None:
    expected = F.load_checksums()
    n = 0
    for case in F.CASES:
        for v in ("fulltop", "n1"):
            if not (O.DATA_ROOT / case / v).is_dir():
                continue
            small = case in ("pglib_opf_case30_ieee", "pglib_opf_case57_ieee")
            # a fetch with --splits holds only those splits
            have = tuple(s for s in F.SPLITS if O.shard_path(case, v, s).exists())
            if not have:
                continue
            problems = F.verify(case, v, O.DATA_ROOT, splits=have, deep=small or DEEP,
                                expected=expected)
            assert not problems, (case, v, problems)
            n += 1
    for alias in ("case2000", "case6470"):
        if not (O.TEST_SPLIT_DIRS[alias] / "fulltop").is_dir():
            continue
        problems = T.verify(alias, "fulltop", O.TEST_SPLIT_DIRS[alias], deep=DEEP, expected=expected)
        assert not problems, (alias, problems)
        n += 1
    if not n:
        raise Skip(f"no corpus under {shown(DATA_HOME)}; fetch with "
                   f"`python -m powerform.data.corpus.fetch_opfdata --all --variant both`")
    print(f"{n} local corpora pass offline verification")


def check_cost_layouts_on_real_data() -> None:
    need_shard("case57", split="val")
    p12 =O.load_pool("case57", "fulltop", "val", n=4, n_cost_feats=12, verbose=False)
    p10 = O.load_pool("case57", "fulltop", "val", n=4, n_cost_feats=10, verbose=False)
    assert p12.example_indices == p10.example_indices
    for a, b in zip(p12.insts, p10.insts):
        assert a.cost_feats.shape[1] == 12 and b.cost_feats.shape[1] == 10
        assert torch.equal(a.cost_feats[:, :10], b.cost_feats), "columns 0-9 differ"
        assert float(a.cost_feats[a.m_node_ns, 10:12].abs().max()) > 0.0, "ED columns are empty"
    assert p12.provenance()["split_range"] == [13500, 14250]
    try:
        O.load_pool("case57", "fulltop", "val", n=1, n_cost_feats=3, verbose=False)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown cost-feature layout was accepted")
    # the cost convention agrees with the solver's own objective
    check = O.label_cross_check(p12)
    assert check["pass"], check
    for lab in p12.labels:
        lab.cost *= 1.001
    assert not O.label_cross_check(p12)["pass"], "a corrupted cost convention passed"
    print(f"10 columns == columns 0-9 of 12, ED columns non-zero; label cost == objective "
          f"(rel_max {check['rel_max']:.1e}), a 0.1% cost error is caught")


def _ybus_residual(inst, lab) -> float:
    """max |V (Y V)* - S_spec| at the label operating point: algebra on the label
    voltages and dispatch, no solver."""
    g = inst.tgrid
    V = (lab.vm * torch.exp(1j * lab.va)).to(torch.complex128)
    S = V * (build_ybus(g) @ V).conj()
    want = ((lab.pg_bus - g.p_load) + 1j * (lab.qg_bus - g.q_load)).to(torch.complex128)
    return float((S - want).abs().max())


def check_labels_satisfy_the_physics() -> None:
    """Ybus identity and a Newton round trip certify r, x, b, tap, shift, shunts and
    per-unit conventions, on the base grid and under outages, for both solvers."""
    need_shard("case57")
    pools = [p for p in (("case57", "fulltop"), ("case57", "n1"), ("case118", "fulltop"))
             if has_shard(*p, "test")]
    n = 0
    worst_y = worst_v = 0.0
    for grid, variant in pools:
        pool = O.load_pool(grid, variant, "test", n=3, n_cost_feats=12, verbose=False)
        for inst, lab in zip(pool.insts, pool.labels):
            yb = _ybus_residual(inst, lab)
            assert yb < 1e-6, (grid, variant, yb)
            g, pv = inst.tgrid, inst.m_node_ns
            d = transmission_pf_newton_pv(g, lab.pg_bus, lab.vm, pv, n_iter=50)
            s = pf_pv_sparse(g, lab.pg_bus, lab.vm, pv, n_iter=50)
            dv = max(float((d["V"].abs() - lab.vm).abs().max()),
                     float(np.abs(np.abs(s["V"]) - lab.vm.numpy()).max()))
            assert dv < 1e-6, (grid, variant, dv)
            sd = float(np.abs(s["V"] - d["V"].numpy()).max())
            assert sd < 1e-7, (grid, variant, sd)
            worst_y, worst_v = max(worst_y, yb), max(worst_v, dv)
            n += 1
    # failure mode: one branch reactance off by 1%
    inst = pool.insts[0]
    bad = SimpleNamespace(tgrid=copy.copy(inst.tgrid))
    bad.tgrid.br_x = inst.tgrid.br_x.clone()
    bad.tgrid.br_x[0] *= 1.01
    wrong = _ybus_residual(bad, pool.labels[0])
    assert wrong > 1e-4, wrong
    print(f"{n} scenarios of {[f'{g}/{v}' for g, v in pools]}: label residual {worst_y:.1e}, round trip "
          f"{worst_v:.1e}, dense == sparse; reactance +1% on one branch gives {wrong:.1e}")


def _digest(inst, spec, lab) -> str:
    h = hashlib.sha256()
    for t in (inst.p_load, inst.q_load, inst.edge_uv, inst.edge_r, inst.edge_x, inst.s_max,
              inst.cost_feats, inst.depth, spec.p_min, spec.p_max, lab.vm, lab.va, lab.pg_gen):
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _print_pool_digests() -> None:
    """The ``--probe`` mode: pools as this process's data home serves them."""
    out = {}
    for case, n in (("case30", 4), ("case2000", 0)):
        p = O.load_pool(case, "fulltop", "test", n=n, verbose=False)
        out[case] = {str(i): _digest(a, b, c) for i, a, b, c in
                     zip(p.example_indices, p.insts, p.specs, p.labels)}
    print("PROBE " + json.dumps(out))


def check_moved_data_home_reads_the_same() -> None:
    need_shard("case30")
    src2000 = O.TEST_SPLIT_DIRS["case2000"] / "fulltop"
    if not src2000.is_dir():
        raise Skip(f"case2000 absent; fetch with `{O.fetch_command('case2000', 'fulltop')}`")
    sample = (14250, 14600, 14999)

    def make(home: Path) -> Path:
        d = home / "opfdata_hf" / CASE30 / "fulltop"
        d.mkdir(parents=True)
        for name in ("manifest.json", "test.jsonl.gz"):
            shutil.copyfile(O.DATA_ROOT / CASE30 / "fulltop" / name, d / name)
        t = home / "opfdata_case2000" / "fulltop"
        t.mkdir(parents=True)
        for i in sample:
            shutil.copyfile(src2000 / f"example_{i}.json.gz", t / f"example_{i}.json.gz")
        return d / "test.jsonl.gz"

    def probe(home: Path) -> dict:
        env = dict(os.environ, **{ENV_VAR: str(home)})
        r = subprocess.run([sys.executable, __file__, "--probe"], capture_output=True,
                           text=True, cwd=str(REPO), env=env)
        assert r.returncode == 0, r.stderr[-400:]
        line = next(x for x in r.stdout.splitlines() if x.startswith("PROBE "))
        return json.loads(line[len("PROBE "):])

    def here() -> dict:
        out = {"case30": {}, "case2000": {}}
        p = O.load_pool("case30", "fulltop", "test", n=4, verbose=False)
        for i, a, b, c in zip(p.example_indices, p.insts, p.specs, p.labels):
            out["case30"][str(i)] = _digest(a, b, c)
        for i in sample:
            with gzip.open(src2000 / f"example_{i}.json.gz", "rt", encoding="utf-8") as fh:
                a, b, c = build_instance(json.load(fh), torch.device("cpu"), 12)
            out["case2000"][str(i)] = _digest(a, b, c)
        return out

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        make(tmp / "good")
        assert probe(tmp / "good") == here(), "a moved data home read different tensors"
        # failure mode: one load changed in the copy shows up in that scenario only
        shard = make(tmp / "bad")
        man = json.loads((shard.parent / "manifest.json").read_text())
        line_no = man["splits"]["test"]["example_indices"].index(14250)

        def bump(line: bytes) -> bytes:
            obj = json.loads(line)
            obj["grid"]["nodes"]["load"][0][0] += 0.01
            return json.dumps(obj, separators=(",", ":")).encode()

        _rewrite_line(shard, line_no, bump)
        got, ref = probe(tmp / "bad"), here()
        changed = [k for k in ref["case30"] if got["case30"][k] != ref["case30"][k]]
        assert changed == ["14250"], changed
    print("a moved data home reads bit-identical pools (shard and per-file); one changed load "
          "shows up in exactly that scenario")


def check_datakit_conversion() -> None:
    """The converter reproduces the files on disk byte for byte."""
    need_datakit()
    if not X.REFERENCE_PATH.exists():
        raise Skip(f"reference absent; fetch with `{O.fetch_command('case500', 'fulltop')}`")
    ref = X.read_reference(X.REFERENCE_PATH)
    with contextlib.redirect_stdout(io.StringIO()):
        ids = np.array([29500.0, 29777.0, 29999.0])
        sg = X.static_grid_for(ids, X.RAW_DIR, ref)
    n = 0
    for sid, obj in X.iter_scenarios(sg, ids, X.RAW_DIR):
        assert json.dumps(obj) == (X.OUT_DIR / X.scenario_filename(sid)).read_text(), sid
        n += 1
    with contextlib.redirect_stdout(io.StringIO()):
        X.verify_flow_label_order(obj)
    # failure mode: one load moved by one part in 1e12
    sid = ids[:1]
    bus, gen, br = X._read_chunk(sid, X.RAW_DIR)
    bus = bus.sort_values("bus").copy()
    col = bus.columns.get_loc("Pd")
    row = int(np.flatnonzero(bus["Pd"].values != 0)[0])
    bus.iloc[row, col] = bus.iloc[row, col] * (1 + 1e-12)
    bad = json.dumps(X.build_scenario(sg, bus, gen.sort_values("idx"), br, int(sid[0])))
    assert bad != (X.OUT_DIR / X.scenario_filename(sid[0])).read_text(), "corruption not detected"
    # failure mode: swapped flow-label columns
    lab = obj["solution"]["edges"]["ac_line"]["features"]
    obj["solution"]["edges"]["ac_line"]["features"] = [[r[2], r[3], r[0], r[1]] for r in lab]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            X.verify_flow_label_order(obj)
    except SystemExit:
        pass
    else:
        raise AssertionError("swapped flow labels passed")
    assert X.verify_outputs(X.OUT_DIR, "holdout") == []
    with tempfile.TemporaryDirectory() as t:
        assert X.verify_outputs(Path(t), "holdout"), "an empty hold-out passed"
    print(f"{n} hold-out scenarios byte-identical to the files on disk; a Pd change of 1e-12 and "
          f"swapped flow labels are caught")


def check_datakit_loader() -> None:
    need_datakit()
    ids = DL.corpus_ids(X.CORPUS_DIR)
    train, hold = DL.table1_split(ids)
    assert len(ids) == 29987 and len(train) == 29487 and len(hold) == 500
    data = DL.load_case500(X.CORPUS_DIR, scenario_ids=hold[:2])
    assert data.scenario_ids == hold[:2]
    sc = data.scenarios[0]
    inst = DL.TransmissionInstance(data.grid, sc)
    assert inst.n_nodes == 500 and inst.n_edges == data.grid.n_edges == 728
    # the label dispatch prices at the corpus's own cost
    gen_df = DL.template_gen_df(X.CORPUS_DIR)
    ns = gen_df[(gen_df.in_service > 0) & ~gen_df.is_slack_gen.astype(bool)]
    coef = DL.slack_quadratic(gen_df)
    with torch.no_grad():
        m_free = data.m_node | data.grid.is_slack
        pf = transmission_pf_newton_pv(inst.tgrid, sc.pg_bus,
                                       torch.where(m_free, sc.vm_ref, torch.ones(500, dtype=F64)),
                                       data.m_node)
    assert float(pf["residual"]) < 1e-6
    p_slack = float(pf["p_slack"].sum())
    cost = DL.dispatch_cost(sc.pg_bus.numpy(), p_slack, data.gen_bus.numpy(), ns, coef)
    rel = abs(cost - sc.cost_ref) / sc.cost_ref
    assert rel < 1e-3, rel
    wrong = DL.dispatch_cost(sc.pg_bus.numpy() * 1.01, p_slack, data.gen_bus.numpy(), ns, coef)
    assert abs(wrong - sc.cost_ref) / sc.cost_ref > 2 * rel, "a 1% dispatch error priced the same"
    try:
        DL.load_case500(X.CORPUS_DIR, scenario_ids=[16496])
    except SystemExit:
        pass
    else:
        raise AssertionError("a scenario that failed to solve was loaded")
    print(f"29,987 scenarios split 29,487 / 500; the label dispatch prices within {rel:.1e} of "
          f"cost_ref, 1% off is caught; a failed solve is refused")


APP_B = {("profile", "corpus"): (0.711, 1.204), ("profile", "holdout"): (0.840, 1.052),
         ("converted", "holdout"): (0.840, 1.052), ("solved", "holdout"): (0.840, 1.052)}


def _app_b_problems(res: dict) -> list:
    out = []
    for (src, split), (lo, hi) in APP_B.items():
        b = res[src][split]
        if (round(b["min"], 3), round(b["max"], 3)) != (lo, hi):
            out.append(f"{src}/{split} {b['min']:.3f}-{b['max']:.3f}")
    gb = res["generator_band"]
    if (round(gb.get("l", -1), 2), round(gb.get("u", -1), 2)) != (0.72, 1.20):
        out.append(f"band {gb}")
    return out


def check_app_b_load_factors() -> None:
    if not DEEP:
        raise Skip("reads two columns of the full bus table; run with --deep")
    need_datakit()
    need_shard("case500", split="train")
    res = LF.measure()
    problems = _app_b_problems(res)
    assert not problems, problems
    worse = {**res, "profile": {**res["profile"], "holdout": {**res["profile"]["holdout"], "max": 1.0535}}}
    assert _app_b_problems(worse), "a hold-out max off by 0.0015 passed"
    summ = X.raw_summary()
    err = (X.RAW_DIR / "error.log").read_text()
    logged = sorted(int(m) for m in re.findall(r"at scenario (\d+)", err))
    assert logged == summ["unsolved_ids"] and summ["n_solved"] == 29987
    want = F.load_checksums()["case500_datakit/case500_30k/case500_goc/raw"]
    assert want["load_profile_sha256"] == summ["load_profile_sha256"]
    print(f"App. B band re-measured, nominal {res['nominal_mw']:.1f} MW; 13 unsolved ids == error.log; "
          f"load profile digest {summ['load_profile_sha256'][:12]}...")


CHECKS = (check_data_home, check_splits, check_fetchers_stay_offline, check_shard_verification,
          check_test_split_verification, check_power_flow, check_allocation, check_cost_features,
          check_binding, check_table1_split_rules, check_missing_corpus_messages, check_datakit_config,
          check_datakit_config_matches_record, check_local_corpora_verify, check_cost_layouts_on_real_data, check_labels_satisfy_the_physics,
          check_moved_data_home_reads_the_same, check_datakit_conversion, check_datakit_loader,
          check_app_b_load_factors)


def main() -> None:
    global DEEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.probe:
        _print_pool_digests()
        return
    DEEP = args.deep
    ran, failed, skipped = 0, 0, []
    for check in CHECKS:
        name = check.__name__[len("check_"):]
        try:
            check()
            ran += 1
            print(f"[PASS] {name}", flush=True)
        except Skip as e:
            skipped.append(f"{name}: {e}")
            print(f"[SKIP] {name}: {e}", flush=True)
        except (Exception, SystemExit):  # noqa: BLE001 -- reported, and the exit code says so
            ran += 1
            failed += 1
            traceback.print_exc()
            print(f"[FAIL] {name}", flush=True)
    why = f" ({'; '.join(skipped)})" if skipped else ""
    print(f"{ran} checks ran, {failed} failed, {len(skipped)} skipped{why}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
