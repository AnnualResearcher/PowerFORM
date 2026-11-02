"""Model checks, each with a failure mode: a perturbed input must be caught.

Runs on synthetic graphs, so it needs neither data nor checkpoints; the checks on
real instances and shipped checkpoints skip when those are absent.

    python powerform/tests/test_model.py
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from powerform.data.batch import merge_instances, split_edges, split_nodes  # noqa: E402
from powerform.eval import checkpoint  # noqa: E402
from powerform.eval.checkpoint import build_model, config_of, load_model  # noqa: E402
from powerform.model.busnet import BusNet  # noqa: E402
from powerform.model.flownet import FlowNet  # noqa: E402
from powerform.model.generic import GenericBlock  # noqa: E402
from powerform.model.line_track import ReconfigurationNet  # noqa: E402
from powerform.model.powerform import PowerForm  # noqa: E402
from powerform.model.spec import TaskSpec  # noqa: E402
from powerform.paths import CHECKPOINTS  # noqa: E402

torch.set_num_threads(1)
F64 = torch.float64
D, K = 48, 3
HIDDEN = 64
TOL = 1e-12
FETCH_CKPT = "python scripts/download_checkpoints.py"


class Skip(Exception):
    pass


ARMS = {
    "physics": dict(maps="physics"),
    "generic": dict(maps="generic", generic_hidden=HIDDEN),
    "physics F1": dict(maps="physics", type_masking=False),
    "physics F2": dict(maps="physics", merged_readout=True),
    "generic F1": dict(maps="generic", generic_hidden=HIDDEN, type_masking=False),
    "generic F2": dict(maps="generic", generic_hidden=HIDDEN, merged_readout=True),
}


def build(seed=0, d=D, k=K, n_cost_feats=12, **kw):
    torch.manual_seed(seed)
    return PowerForm(d_model=d, K=k, n_cost_feats=n_cost_feats, **kw).eval()


def synthetic(n=12, extra=6, n_cost_feats=12, seed=0):
    """A ring with chords: every bus keeps a line when one is removed."""
    g = torch.Generator().manual_seed(seed)
    ring = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n], dim=1)
    u = torch.randint(0, n, (extra,), generator=g)
    chords = torch.stack([u, (u + torch.randint(2, n - 1, (extra,), generator=g)) % n], dim=1)
    uv = torch.cat([ring, chords])
    e = uv.shape[0]
    rnd = lambda *shape: torch.rand(*shape, generator=g, dtype=F64)
    is_sub = torch.zeros(n, dtype=torch.bool)
    is_sub[0] = True
    gen = rnd(n) < 0.4
    gen[1] = True
    m_free = gen | is_sub
    inst = SimpleNamespace(
        n_nodes=n, n_edges=e, edge_uv=uv, edge_r=0.01 + 0.05 * rnd(e),
        edge_x=0.05 + 0.2 * rnd(e), s_max=1.0 + 5.0 * rnd(e), p_load=rnd(n),
        q_load=0.3 * rnd(n), is_sub=is_sub, v_slack_scalar=1.0 + 0.02 * rnd(n),
        depth=rnd(n), cost_feats=rnd(n, max(n_cost_feats, 2)), m_free=m_free,
        tgrid=SimpleNamespace(br_x=0.05 + 0.2 * rnd(e), br_b=0.1 * rnd(e), tap=1.0 + 0.05 * rnd(e),
                              shift_rad=0.02 * rnd(e), br_r=0.01 + 0.05 * rnd(e)))
    spec = TaskSpec(m_node=m_free, m_edge=torch.zeros(e, dtype=torch.bool),
                    p_min=torch.zeros(n, dtype=F64), p_max=2.0 * rnd(n) * gen,
                    vm_min=torch.full((n,), 0.94, dtype=F64), vm_max=torch.full((n,), 1.06, dtype=F64))
    return inst, spec


def maxdiff(a, b) -> float:
    return float((a - b).abs().max().detach())


def live_track(net):
    """A track off its identity point, as after a first optimizer step."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for lin in net.trunk.z_stream.updates:
            lin.weight.copy_(0.05 * torch.randn(lin.weight.shape, generator=g, dtype=lin.weight.dtype))
    return net


def check_parameter_counts() -> None:
    """The numbers of ARCHITECTURE.md, and every arm within 0.5% of the model."""
    n = lambda m: sum(p.numel() for p in m.parameters())
    for cf, rest, total in ((12, 1_002_904, 14_503_928), (10, 1_002_472, 14_503_496),
                            (0, 952_792, 14_453_816)):
        m = PowerForm(n_cost_feats=cf)
        assert n(m.flow_nets) == 5_636_864, n(m.flow_nets)
        assert n(m.bus_nets) == 7_864_160, n(m.bus_nets)
        assert n(m) - n(m.flow_nets) - n(m.bus_nets) == rest, n(m)
        assert n(m) == total, n(m)
    m = PowerForm(n_cost_feats=12)
    net = ReconfigurationNet(m)
    assert n(net.trunk.z_stream) == 332_321 and n(net) == 14_836_249, n(net)
    ref = 14_453_816
    for name, kw in (("generic", dict(maps="generic", generic_hidden=450)),
                     ("F1", dict(maps="generic", generic_hidden=450, type_masking=False)),
                     ("F2", dict(maps="generic", generic_hidden=450, merged_readout=True))):
        got = n(PowerForm(**kw))
        assert abs(got / ref - 1) < 0.005, (name, got)
    print("FlowNet 5,636,864; BusNet 7,864,160; model 14,503,928/14,503,496/14,453,816; "
          "track 332,321; total 14,836,249; arms within 0.5%")


def check_batched_equals_unbatched() -> None:
    pairs = [synthetic(9, 3, seed=1), synthetic(14, 6, seed=2), synthetic(12, 4, seed=3)]
    merged, mspec = merge_instances([i for i, _ in pairs], [s for _, s in pairs])
    with torch.no_grad():
        for name, kw in ARMS.items():
            m = build(**kw)
            ob = m(merged, mspec)
            for j, (inst, spec) in enumerate(pairs):
                oi = m(inst, spec)
                for key in ("p_g", "vm_set", "V_lat", "S_lat"):
                    assert maxdiff(split_nodes(ob[key], merged)[j], oi[key]) <= TOL, (name, key, j)
                assert maxdiff(split_edges(ob["Fb"], merged)[j], oi["Fb"]) <= TOL, (name, "Fb", j)
                for hb, hi in zip(ob["history"], oi["history"]):
                    assert maxdiff(split_nodes(hb["p_g"], merged)[j], hi["p_g"]) <= TOL, name
                    assert maxdiff(split_nodes(hb["vm_set"], merged)[j], hi["vm_set"]) <= TOL, name
        for name, kw in (("physics", ARMS["physics"]), ("generic", ARMS["generic"])):
            net = live_track(ReconfigurationNet(build(**kw), z_d=8).eval())
            lb = net(merged, mspec)
            for j, (inst, spec) in enumerate(pairs):
                assert maxdiff(split_edges(lb, merged)[j], net(inst, spec)) <= TOL, (name, "ell_K", j)

    # gradients: one batched loss against the sum of per-instance losses
    loss = lambda o: (o["p_g"] ** 2).sum() + (o["vm_set"] ** 2).sum() + sum(
        (h["p_g"] ** 2).sum() for h in o["history"])
    m = build(maps="physics")
    m.zero_grad()
    loss(m(merged, mspec)).backward()
    gb = {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    for inst, spec in pairs:
        loss(m(inst, spec)).backward()
    scale = max(float(g.abs().max()) for g in gb.values())
    worst = max(maxdiff(p.grad, gb[k]) for k, p in m.named_parameters() if k in gb)
    assert worst <= 1e-9 * scale + TOL, (worst, scale)

    # failure modes: a perturbed instance, and a merge without its segments
    m = build(maps="physics")
    with torch.no_grad():
        good = m(merged, mspec)["p_g"]
        inst2 = copy.copy(pairs[1][0])
        inst2.p_load = inst2.p_load.clone()
        inst2.p_load[3] += 0.5
        bad = m(*merge_instances([pairs[0][0], inst2, pairs[2][0]], [s for _, s in pairs]))["p_g"]
        moved = [maxdiff(a, b) for a, b in zip(split_nodes(good, merged), split_nodes(bad, merged))]
        assert moved[0] == 0.0 and moved[2] == 0.0 and moved[1] > 1e-6, moved
        segs = merged.node_segments
        del merged.node_segments
        whole = m(merged, mspec)["p_g"]
        merged.node_segments = segs
    delta = maxdiff(good, whole)
    assert delta > 1e-6, delta
    print(f"{len(ARMS)} arms + tracks agree to {TOL:g}; gradients worst {worst:.1e} of {scale:.1e}; "
          f"perturbed instance moves only itself ({moved[1]:.1e}); dropped segments caught ({delta:.1e})")


def _bus_inputs(n=9, m=14, d=16, seed=0):
    torch.manual_seed(seed)
    uv = torch.randint(0, n, (m, 2))
    same = uv[:, 0] == uv[:, 1]
    uv[same, 1] = (uv[same, 1] + 1) % n
    r = lambda *s: torch.randn(*s, dtype=F64)
    return uv, r(n, d), r(n, d), r(n, d), r(m, d), r(m, d)


def check_busnet_permutation_equivariance() -> None:
    d, n, m = 16, 9, 14
    uv, V, S, Veff, F, E = _bus_inputs(n, m, d)
    z = torch.rand(m, dtype=F64).clamp_min(0.05)
    bn = BusNet(d).double()
    V2, S2 = bn(V, S, Veff, F, E, uv, n, z)
    assert V2.shape == (n, d) and S2.shape == (n, d)
    perm = torch.randperm(n)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    V2p, S2p = bn(V[perm], S[perm], Veff[perm], F, E, inv[uv], n, z)
    assert maxdiff(V2p, V2[perm]) < 1e-12 and maxdiff(S2p, S2[perm]) < 1e-12
    wrong = torch.roll(perm, 1)
    assert maxdiff(V2p, V2[wrong]) > 1e-6, "a wrong permutation matched"
    print("BusNet is permutation equivariant; a wrong permutation is caught")


def check_closed_line_is_deleted_line() -> None:
    d, n, m = 16, 9, 14
    uv, V, S, Veff, F, E = _bus_inputs(n, m, d)
    bn = BusNet(d).double()
    deg = torch.bincount(uv.reshape(-1), minlength=n)
    e = next(i for i in range(m) if deg[uv[i, 0]] > 1 and deg[uv[i, 1]] > 1)
    keep = torch.arange(m) != e
    z1 = torch.ones(m, dtype=F64)
    z0 = z1.clone()
    z0[e] = 0.0
    Vc, Sc = bn(V, S, Veff, F, E, uv, n, z0)
    Vd, Sd = bn(V, S, Veff, F[keep], E[keep], uv[keep], n, z1[keep])
    gap = max(maxdiff(Vc, Vd), maxdiff(Sc, Sd))
    assert gap < 1e-5, gap
    # failure mode: the same line left at z = 1 leaks
    Vl, Sl = bn(V, S, Veff, F, E, uv, n, z1)
    leak = max(maxdiff(Vl, Vd), maxdiff(Sl, Sd))
    assert leak > 1e-3 and leak > 100 * gap, (leak, gap)
    # a bus with every incident line closed receives exactly zero
    b = int(uv[e, 0])
    z_all = z1.clone()
    z_all[(uv[:, 0] == b) | (uv[:, 1] == b)] = 0.0
    owner = torch.cat([uv[:, 0], uv[:, 1]])
    att = bn.att_v(V, torch.cat([Veff[uv[:, 1]], Veff[uv[:, 0]]]), torch.cat([F, -F]),
                   torch.cat([E, E]), owner, n, torch.cat([z_all, z_all]))
    assert torch.equal(att[b], torch.zeros(d, dtype=F64))
    assert not torch.equal(att[(b + 1) % n], torch.zeros(d, dtype=F64))
    # FlowNet and the generic block: a closed line carries exactly zero flow
    fn = FlowNet(d, n_heads=4).double()
    Fo = fn(Veff, S, F, E, uv, z0)
    assert torch.equal(Fo[e], torch.zeros(d, dtype=F64)) and float(Fo.abs().max()) > 0
    gb = GenericBlock(d, H=4, hidden=32).double()
    _, _, Fg = gb(V, S, Veff, S, F, E, uv, n, z0)
    assert torch.equal(Fg[e], torch.zeros(d, dtype=F64)) and float(Fg.abs().max()) > 0
    Vg, Sg, _ = gb(V, S, Veff, S, F, E, uv, n, z0)
    Vgd, Sgd, _ = gb(V, S, Veff, S, F[keep], E[keep], uv[keep], n, z1[keep])
    gen_gap = max(maxdiff(Vg, Vgd), maxdiff(Sg, Sgd))
    assert gen_gap < 1e-4, gen_gap
    print(f"closed line == deleted line: BusNet gap {gap:.1e} (open-line leak {leak:.1e}), "
          f"generic gap {gen_gap:.1e}; fully closed bus gets exactly zero; zero flow on a closed line")


def check_line_status_track() -> None:
    inst, spec = synthetic(seed=4)
    other, other_spec = synthetic(10, 3, seed=5)
    for name, kw in (("physics", ARMS["physics"]), ("generic", ARMS["generic"])):
        plain = build(**kw)
        with_track = build(**kw)
        net = ReconfigurationNet(with_track, z_d=8).eval()
        with torch.no_grad():
            a, b = plain(inst, spec), net.trunk(inst, spec)
            assert torch.equal(b["ell_K"], torch.zeros_like(b["ell_K"])), "ell_K != 0 at init"
            for ha, hb in zip(a["history"], b["history"]):
                assert torch.equal(ha["p_g"], hb["p_g"]) and torch.equal(ha["vm_set"], hb["vm_set"])
            # A nonzero U_0 leaves block 0 alone and moves every later block.
            w = net.trunk.z_stream.updates[0].weight
            w.copy_(0.05 * torch.randn(w.shape, generator=torch.Generator().manual_seed(0), dtype=w.dtype))
            b = net.trunk(inst, spec)
            assert float(b["ell_K"].abs().max()) > 0
            assert torch.equal(a["history"][0]["p_g"], b["history"][0]["p_g"]), "block 0 changed"
            assert maxdiff(a["history"][1]["p_g"], b["history"][1]["p_g"]) > 0, "the gate reached nothing"
            # Per-line state does not survive into the next scenario. Without
            # the reset, a scenario with other lines meets a stale latent.
            first = net(inst, spec)
            net(other, other_spec)
            assert torch.equal(net(inst, spec), first), "evaluation order changed a prediction"
            track = net.trunk.z_stream
            track.reset = lambda: None
            try:
                net(other, other_spec)
            except RuntimeError:
                pass
            else:
                raise AssertionError("a stale latent went unnoticed")
            del track.reset
        # Gradient reaches every U_k from the identity point.
        grad_net = ReconfigurationNet(build(**kw), z_d=8)
        grad_net(inst, spec).sum().backward()
        assert all(float(u.weight.grad.norm()) > 0 for u in grad_net.trunk.z_stream.updates), name
        # A strict load into a plain trunk refuses the extra tensors.
        try:
            build(**kw).load_state_dict(with_track.state_dict())
        except RuntimeError:
            pass
        else:
            raise AssertionError("a plain trunk accepted a checkpoint with a track")
    print("identity at init bit for bit on every block, ell_K = 0; U_0 moves only later blocks; "
          "per-scenario reset (mutant caught); gradient on every U_k")


def check_flags_change_the_function() -> None:
    inst, spec = synthetic(seed=6)
    with torch.no_grad():
        for maps, extra in (("physics", {}), ("generic", {"generic_hidden": HIDDEN})):
            a = build(maps=maps, **extra)
            spelled = build(maps=maps, type_masking=True, merged_readout=False, **extra)
            assert torch.equal(a(inst, spec)["V_lat"], spelled(inst, spec)["V_lat"]), "defaults differ"
            b = build(maps=maps, type_masking=False, **extra)
            b.load_state_dict(a.state_dict())
            delta = maxdiff(a(inst, spec)["V_lat"], b(inst, spec)["V_lat"])
            assert delta > 1e-6, f"type_masking on {maps} is dead ({delta:.1e})"
    merged = build(maps="physics", merged_readout=True)
    assert merged.readouts[0].net_p[0].in_features == 2 * D + 2
    assert merged.readouts[0].net_p[0].out_features == D // 2
    for bad in (dict(maps="other"), dict(maps="physics", generic_hidden=450)):
        try:
            build(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    print("defaults are the full model; type_masking changes the function with the same weights; "
          "merged readout changes the structure; bad maps refused")


def check_inputs_reach_the_decisions() -> None:
    """An arm must not silently lose an input it did not mean to remove."""
    inst, spec = synthetic(seed=7)
    loads = copy.copy(inst)
    # not a uniform scale, which the standardised load inputs would absorb
    scale = torch.linspace(0.7, 1.3, inst.n_nodes, dtype=F64)
    loads.p_load, loads.q_load = inst.p_load * scale, inst.q_load * scale
    costs = copy.copy(inst)
    costs.cost_feats = inst.cost_feats * torch.tensor([2.0] * 2 + [1.0] * 10, dtype=F64)
    context = copy.copy(inst)
    context.cost_feats = inst.cost_feats * torch.tensor([1.0] * 2 + [1.5] * 8 + [1.0] * 2, dtype=F64)
    prior = copy.copy(inst)
    prior.cost_feats = inst.cost_feats * torch.tensor([1.0] * 10 + [1.7] * 2, dtype=F64)

    def response(m):
        with torch.no_grad():
            base = m(inst, spec)["p_g"]
            return [maxdiff(base, m(i, spec)["p_g"]) for i in (loads, costs, context, prior)]

    for name, kw in ARMS.items():
        got = response(build(**kw))
        assert min(got) > 1e-9, (name, got)
    # failure mode: the same check on a model cut off from its inputs
    blind = build(maps="physics")
    with torch.no_grad():
        for p in blind.embed_s_const[0].parameters():
            p.zero_()
        blind.embed_v_init[0].weight[:, 2:4] = 0.0
        blind.embed_s_var[0].weight[:, :12] = 0.0
    got = response(blind)
    assert max(got) < 1e-12, got
    print(f"{len(ARMS)} arms read loads, cost, context and ED columns; a blinded mutant is caught")


def check_cost_feature_layouts() -> None:
    for maps, extra in (("physics", {}), ("generic", {"generic_hidden": HIDDEN})):
        m12, m10 = build(n_cost_feats=12, maps=maps, **extra), build(n_cost_feats=10, maps=maps, **extra)
        diff = [n for (n, p), (_, q) in zip(m12.named_parameters(), m10.named_parameters())
                if p.shape != q.shape]
        assert diff == ["embed_s_var.0.weight"], diff
        n12 = sum(p.numel() for p in m12.parameters())
        assert n12 - sum(p.numel() for p in m10.parameters()) == 2 * D
        inst12, spec12 = synthetic(n_cost_feats=12)
        try:
            with torch.no_grad():
                m10(inst12, spec12)
        except RuntimeError:
            continue
        raise AssertionError("a 12-column instance ran through a 10-feature model")
    assert not hasattr(build(n_cost_feats=0), "embed_s_var")
    print("12 vs 10 cost features differ only in embed_s_var.0.weight (2d params); a mismatch is refused")


def _roundtrip(tmp: Path, name: str, model, state=None) -> Path:
    p = tmp / name
    torch.save({"model": state or model.state_dict(), "config": config_of(model)}, p)
    return p


def check_load_model() -> None:
    inst, spec = synthetic(seed=8)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        # A flag no tensor shows comes back from the config.
        m = build(maps="generic", generic_hidden=HIDDEN, type_masking=False)
        got = load_model(_roundtrip(tmp, "f1.pt", m))
        default = build(maps="generic", generic_hidden=HIDDEN)
        default.load_state_dict(m.state_dict())
        with torch.no_grad():
            assert torch.equal(got(inst, spec)["V_lat"], m(inst, spec)["V_lat"])
            assert maxdiff(default(inst, spec)["V_lat"], m(inst, spec)["V_lat"]) > 1e-6
        assert not got.training and config_of(got) == config_of(m)
        # Reconfiguration: float32 weights on disk, float64 in the model.
        net = live_track(ReconfigurationNet(build(), z_d=8))
        stored = {k: v.float() for k, v in net.state_dict().items()}
        got = load_model(_roundtrip(tmp, "reconfig.pt", net, stored))
        assert all(p.dtype == F64 for p in got.parameters())
        assert isinstance(got, ReconfigurationNet) and config_of(got)["z_d"] == 8
        for k, v in got.state_dict().items():
            assert v.dtype == F64 and torch.equal(v, stored[k].double()), k
        # Refusals.
        base = {"model": m.state_dict(), "config": config_of(m)}
        for what, blob in (
                ("unknown config key", dict(base, config=dict(base["config"], tag="x"))),
                ("missing config key", dict(base, config={k: v for k, v in base["config"].items()
                                                          if k != "type_masking"}))):
            p = tmp / "bad.pt"
            torch.save(blob, p)
            try:
                load_model(p)
            except ValueError:
                continue
            raise AssertionError(f"accepted a checkpoint with a {what}")
        extra = dict(m.state_dict(), extra=torch.zeros(1))
        p = tmp / "extra.pt"
        torch.save({"model": extra, "config": config_of(m)}, p)
        try:
            load_model(p)
        except RuntimeError:
            pass
        else:
            raise AssertionError("accepted an extra tensor")
        # A missing file names itself and the release group that holds it.
        manifest = tmp / "manifest.json"
        manifest.write_text(json.dumps({"files": [{"name": "listed_seed0.pt", "group": "some_group"}]}))
        real = checkpoint.MANIFEST
        checkpoint.MANIFEST = manifest
        try:
            for name, want in (("listed_seed0.pt", "--group some_group"), ("other.pt", "--list")):
                try:
                    load_model(tmp / name)
                except SystemExit as e:
                    assert name in str(e) and want in str(e), str(e)
                else:
                    raise AssertionError(f"no refusal for the missing {name}")
            checkpoint.MANIFEST = tmp / "absent.json"
            try:
                load_model(tmp / "other.pt")
            except SystemExit as e:
                assert "--list" in str(e), str(e)
            else:
                raise AssertionError("no refusal without a manifest")
        finally:
            checkpoint.MANIFEST = real
    assert isinstance(build_model(config_of(net)), ReconfigurationNet)
    print("flags and z_d come from the config; float32 weights widen exactly; unknown key, "
          "missing key and extra tensor refused; a missing file names its download group")


def check_shipped_checkpoints() -> None:
    names = ("opf_generalist_physics_seed0", "opf_generalist_generic_seed0",
             "opf_generalist_physics_ed_seed0", "case500_physics_seed0",
             "case500_generic_no_type_masking_seed0", "reconfig_physics_pretrained_seed0")
    paths = [CHECKPOINTS / f"{n}.pt" for n in names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise Skip(f"{len(missing)} checkpoint(s) absent; fetch with `{FETCH_CKPT}`")
    params = {}
    for name, p in zip(names, paths):
        params[name] = sum(q.numel() for q in load_model(p).parameters())
    assert params["opf_generalist_physics_seed0"] == 14_503_496
    assert params["opf_generalist_physics_ed_seed0"] == 14_503_928
    assert params["case500_physics_seed0"] == 14_453_816
    assert params["reconfig_physics_pretrained_seed0"] == 14_836_249
    assert not load_model(paths[4]).type_masking
    print(f"{len(names)} shipped checkpoints load through load_model with the paper's parameter counts")


def check_real_instances() -> None:
    from powerform.data.opfdata import fetch_command, load_pool, shard_path
    if not shard_path("case57", "fulltop", "val").exists():
        raise Skip(f"case57 absent; fetch with `{fetch_command('case57', 'fulltop')}`")
    pool = load_pool("case57", "fulltop", "val", n=4, n_cost_feats=12, verbose=False)
    pairs = list(zip(pool.insts, pool.specs))
    merged, mspec = merge_instances([i for i, _ in pairs], [s for _, s in pairs])
    plain = build(maps="physics")
    with torch.no_grad():
        ob = plain(merged, mspec)
        for j, (inst, spec) in enumerate(pairs):
            assert maxdiff(split_nodes(ob["p_g"], merged)[j], plain(inst, spec)["p_g"]) <= TOL, j
            assert maxdiff(split_nodes(ob["vm_set"], merged)[j], plain(inst, spec)["vm_set"]) <= TOL, j
        net = ReconfigurationNet(build(maps="physics"), z_d=8).eval()
        inst, spec = pairs[0]
        assert torch.equal(net(inst, spec), torch.zeros(inst.n_edges, dtype=F64))
        assert torch.equal(plain(inst, spec)["p_g"], net.trunk(inst, spec)["p_g"])
        # a decision stays inside its box
        out = plain(inst, spec)
        free = spec.m_node
        assert bool((out["vm_set"][free] >= spec.vm_min[free]).all())
        assert bool((out["vm_set"][free] <= spec.vm_max[free]).all())
        assert bool((out["p_g"][free] <= spec.p_max[free]).all())
        assert torch.equal(out["p_g"][~free], torch.zeros_like(out["p_g"][~free]))
    print("case57: batching exact to 1e-12, identity of the track, decisions in their boxes")


CHECKS = (check_parameter_counts, check_batched_equals_unbatched,
          check_busnet_permutation_equivariance, check_closed_line_is_deleted_line,
          check_line_status_track, check_flags_change_the_function,
          check_inputs_reach_the_decisions, check_cost_feature_layouts, check_load_model,
          check_shipped_checkpoints, check_real_instances)


def main() -> None:
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
