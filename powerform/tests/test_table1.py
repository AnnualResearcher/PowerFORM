"""Table 1 and the App. D in-distribution table against results/table1/.

Each printed row, mean and sd, parameter count, samples and wall-clock against the
records; one re-score per kind of row when its checkpoint and the data are present.
Run the graphkit decision row's check in graphkit's environment.

    python powerform/tests/test_table1.py
    python powerform/tests/test_table1.py --summarize [DIR]   # default outputs/case500/rescore

`--summarize` compares your re-scores with the records: exit 0 means every row and seed
was run and agrees, 1 a disagreement, 2 no disagreement but some not run.
"""
from __future__ import annotations

import atexit
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from powerform.data.corpus.datakit_export import CORPUS_DIR, DATAKIT_HOME  # noqa: E402
from powerform.paths import CHECKPOINTS, OUTPUTS, RESULTS  # noqa: E402

torch.set_num_threads(1)
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="gridsfm_cache_")
    atexit.register(shutil.rmtree, os.environ["XDG_CACHE_HOME"], ignore_errors=True)

TABLE = RESULTS / "table1"
FETCH_CKPT = "python scripts/download_checkpoints.py --group table1"
FETCH_DATA = "see powerform/data/corpus/README.md, the Table 1 corpus"
# Re-scores agree across machines to rounding: float64 rows, float32 rows.
TOL_F64, TOL_F32 = 1e-9, 1e-3

# row: (cost MAPE mean, sd, decimals of the sd, params in M, samples in k), as printed
PRINTED = {
    "physics": (0.137, 0.053, 3, 14.5, 80),
    "generic": (0.123, 0.024, 3, 14.5, 80),
    "generic_no_type_masking": (0.122, 0.009, 3, None, 80),
    "generic_merged_readout": (0.121, 0.010, 3, None, 80),
    "generic_final_layer_only": (0.127, 0.004, 3, None, 80),
    "mlp": (0.234, 0.047, 3, 15.1, 80),
    "gnn": (0.197, 0.049, 3, 15.0, 80),
    "gridsfm_decision": (0.212, 0.019, 3, 15.1, 80),
    "gridsfm_state": (1.050, 0.564, 3, 15.1, 88),
    "graphkit_decision": (1.183, 0.321, 3, 20.1, 80),
    "graphkit_state": (10.075, 1.58, 2, 20.1, 79.6),
}
# row: printed wall-clock range in seconds, over the seeds that recorded one
WALL_CLOCK = {
    "mlp": (315, 384), "gnn": (987, 1108), "graphkit_decision": (2519, 2551),
    "gridsfm_decision": (8306, 9087), "generic": (17964, 21389), "physics": (31208, 31751),
}
SEEDS = (0, 1, 2)
# Every arm is within this fraction of the physics maps' parameter count.
ARM_BUDGET = 0.005
GENERIC_ARMS = ("generic", "generic_no_type_masking", "generic_merged_readout",
                "generic_final_layer_only")
# row: (seed, tolerance, the .pyg.json export needed, packages needed)
SPOT_CHECKS = {
    "physics": (0, TOL_F64, False, ()),
    "generic": (1, TOL_F64, False, ()),
    "mlp": (0, TOL_F32, True, ()),
    "gnn": (2, TOL_F32, True, ()),
    "gridsfm_decision": (0, TOL_F32, True, ("torch_geometric",)),
    "gridsfm_state": (0, TOL_F32, True, ("torch_geometric", "huggingface_hub")),
    "graphkit_decision": (0, TOL_F32, True, ("torch_geometric", "gridfm_graphkit")),
    "graphkit_state": (0, TOL_F32, False, ()),
}
VENDORED_SHA256 = "3f45d67371adf12389bcdb49ef1abdb1d428354f449ce75af81784b5c2044221"
IMPORTS = ("powerform.training.case500", "powerform.training.generalist",
           "powerform.training.reconfiguration", "powerform.data.datakit_loader",
           "powerform.model.baselines.common", "powerform.model.baselines.mlp",
           "powerform.model.baselines.gnn", "powerform.model.baselines.gridsfm",
           "powerform.model.baselines.recipe", "powerform.model.baselines.graphkit")


class Skip(Exception):
    pass


def records(row: str) -> list:
    return [json.loads((TABLE / f"case500_{row}_seed{s}.json").read_text()) for s in SEEDS]


def artifact(row: str, s: int) -> Path:
    name = f"case500_{row}_seed{s}"
    return CHECKPOINTS / (name if row == "graphkit_state" else f"{name}.pt")


# ---------------------------------------------------------------------------
def check_printed_rows() -> None:
    """Each row's mean and sd over 3 seeds round to the printed cell."""
    for row, (mean, sd, digits, _, _) in PRINTED.items():
        costs = [r["cost_mape"] for r in records(row)]
        got = (round(float(np.mean(costs)), 3), round(float(np.std(costs, ddof=1)), digits))
        assert got == (mean, sd), f"{row}: {got} != printed {(mean, sd)}"
    # failure mode: a record of another seed in place of one
    good = records("physics")
    costs = [good[0]["cost_mape"], good[0]["cost_mape"], good[2]["cost_mape"]]
    assert round(float(np.mean(costs)), 3) != PRINTED["physics"][0], "a swapped seed passed"


def check_record_consistency() -> None:
    """The per-scenario list is the cost mean; the seed is the file's; 500 scenarios."""
    for row in PRINTED:
        for s, r in zip(SEEDS, records(row)):
            per = r["per_scenario_cost_mape"]
            assert len(per) == 500, f"{row} seed {s}: {len(per)} scenarios"
            assert float(np.mean(per)) == r["cost_mape"], f"{row} seed {s}: mean != cost_mape"
            assert r["seed"] == s, f"{row} seed {s}: record says {r['seed']}"
            if "n_nonconverged" in r:
                assert r["n_nonconverged"] == 0, f"{row} seed {s}: non-converged"


def check_params_and_samples() -> None:
    base = records("physics")[0]["params"]
    for row, (_, _, _, params_m, samples_k) in PRINTED.items():
        for r in records(row):
            if params_m is not None:
                assert round(r["params"] / 1e6, 1) == params_m, f"{row}: params {r['params']}"
            assert round(r["samples"] / 1e3, 1 if samples_k == 79.6 else 0) == samples_k, (
                f"{row}: samples {r['samples']}")
    for row in GENERIC_ARMS:
        for r in records(row):
            assert abs(r["params"] / base - 1) < ARM_BUDGET, f"{row}: {r['params']} vs {base}"


def check_wall_clock() -> None:
    for row, (lo, hi) in WALL_CLOCK.items():
        recorded = [r["train_s"] for r in records(row) if r["train_s"] is not None]
        got = (round(min(recorded)), round(max(recorded)))
        assert got == (lo, hi), f"{row}: {got} != printed {(lo, hi)}"


def check_imports_leave_the_device_probe() -> None:
    """Importing a trainer or a baseline must not change `torch.cuda.is_available`."""
    probe = ("import importlib, sys, torch\n"
             f"sys.path.insert(0, {str(REPO)!r})\n"
             "before = torch.cuda.is_available\n"
             "if sys.argv[1] == '--control':\n"
             "    torch.cuda.is_available = lambda: False\n"
             "else:\n"
             "    importlib.import_module(sys.argv[1])\n"
             "sys.exit(0 if torch.cuda.is_available is before else 3)\n")
    env = {**os.environ, "FORCE_CPU": "1"}

    def code(mod: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-c", probe, mod], env=env, cwd=str(REPO),
                              capture_output=True, text=True)
    for mod in IMPORTS:
        res = code(mod)
        if "No module named 'torch_geometric'" in res.stderr and "gridsfm" in mod:
            continue
        assert res.returncode == 0, f"{mod}: exit {res.returncode}\n{res.stderr[-400:]}"
    assert code("--control").returncode == 3, "the probe misses a patched probe"


def check_vendored_gridsfm() -> None:
    """15 module files byte-identical to the recorded commit."""
    vendored = REPO / "powerform" / "third_party" / "gridsfm"
    want = {"blocks", "checkpoint", "cycle_basis", "data", "dc_prior", "eval",
            "finetune_opfdata", "hodge_pe", "loss", "model", "pe_features", "schema",
            "signed_incidence", "stress_features", "synthetic"}
    names = sorted(p.stem for p in vendored.glob("*.py") if p.stem != "__init__")
    assert set(names) == want, f"vendored modules: {names}"
    digest = hashlib.sha256()
    for name in names:
        digest.update((vendored / f"{name}.py").read_bytes())
    assert digest.hexdigest() == VENDORED_SHA256, f"vendored digest {digest.hexdigest()}"


def require_for_spot_check(row: str, ck: Path) -> None:
    """Skip, with the reason and the way to get it, unless the row can be re-scored."""
    _, _, needs_pyg, packages = SPOT_CHECKS[row]
    if not ck.exists():
        raise Skip(f"no checkpoints; {FETCH_CKPT}")
    if not (CORPUS_DIR / "case500_goc" / "raw").is_dir():
        raise Skip(f"no Table 1 corpus; {FETCH_DATA}")
    if needs_pyg and not (DATAKIT_HOME / "holdout_opfdata").is_dir():
        raise Skip(f"no .pyg.json export; {FETCH_DATA}")
    for package in packages:
        if importlib.util.find_spec(package) is None:
            raise Skip(f"{package} is not installed; see powerform/model/baselines/README.md")


def rescore_args(row: str, ck: Path, scratch: str) -> object:
    from powerform.training import case500
    argv = ["--eval-only", str(ck)]
    if row == "graphkit_decision":
        from powerform.model.baselines.graphkit import make_splits
        norm = CHECKPOINTS / "case500_graphkit_normalizer.pt"
        if not norm.exists():
            raise Skip(f"no checkpoints; {FETCH_CKPT}")
        make_splits(CORPUS_DIR / "case500_goc" / "raw", scratch)
        argv += ["--graphkit-splits", scratch, "--graphkit-norm-stats", str(norm)]
    return case500.parse_args(argv)


def spot_check(row: str) -> None:
    """A live re-score of 2 hold-out scenarios equals the row's record."""
    s, tol, _, _ = SPOT_CHECKS[row]
    ck = artifact(row, s)
    require_for_spot_check(row, ck)
    from powerform.training import case500
    with tempfile.TemporaryDirectory() as scratch:
        args = rescore_args(row, ck, scratch)
        live = case500.rescore(ck, args, torch.device("cpu"), limit=2)["per_scenario_cost_mape"]
    want = json.loads((TABLE / f"case500_{row}_seed{s}.json").read_text())["per_scenario_cost_mape"][:2]
    worst = max(abs(a - b) / b for a, b in zip(live, want))
    assert worst <= tol, f"{row} seed {s}: relative {worst:.2e} > {tol}"
    print(f"    relative difference {worst:.2e}")
    # failure mode: the record of another seed
    other = json.loads((TABLE / f"case500_{row}_seed{(s + 1) % 3}.json").read_text())
    worst = max(abs(a - b) / b for a, b in zip(live, other["per_scenario_cost_mape"][:2]))
    assert worst > tol, f"{row}: another seed's record passed"


# What a re-score writes; a training record adds the run's settings.
SCORED = ("v_mape", "th_mae", "pg_mae", "qg_mae", "cost_mape", "n_nonconverged",
          "per_scenario_cost_mape")


def compare_record(got: dict, want: dict, tol: float) -> str:
    """'exact', 'within tolerance' or 'disagree'."""
    keys = [k for k in SCORED if k in want]
    if any(k not in got for k in keys):
        return "disagree"
    a = [x for k in keys for x in np.atleast_1d(got[k]).tolist()]
    b = [x for k in keys for x in np.atleast_1d(want[k]).tolist()]
    if len(a) != len(b):
        return "disagree"
    if a == b:
        return "exact"
    return "within tolerance" if all(abs(x - y) <= tol * abs(y) for x, y in zip(a, b)) else "disagree"


def summarize(rescore_dir: Path) -> int:
    """Each row and seed from the `--eval-only` records in `rescore_dir`; the exit code."""
    if not rescore_dir.is_dir():
        print(f"{rescore_dir} does not exist: nothing to compare")
        return 2
    counts = dict.fromkeys(("exact", "within tolerance", "disagree", "not run"), 0)
    for row in PRINTED:
        tol = TOL_F64 if row == "physics" or row in GENERIC_ARMS else TOL_F32
        verdicts = []
        for s in SEEDS:
            name = f"case500_{row}_seed{s}.json"
            got = rescore_dir / name
            verdicts.append("not run" if not got.exists() else compare_record(
                json.loads(got.read_text()), json.loads((TABLE / name).read_text()), tol))
            counts[verdicts[-1]] += 1
        print(f"  {row:28s} seeds {SEEDS}: {', '.join(verdicts)}")
    print(", ".join(f"{v} {k}" for k, v in counts.items()))
    if counts["disagree"]:
        return 1
    return 2 if counts["not run"] else 0


def silent_summary(d: Path) -> int:
    """The exit code alone: this check runs the tool on copies of the records."""
    with contextlib.redirect_stdout(io.StringIO()):
        return summarize(d)


def check_summary_exit_codes() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for row in PRINTED:
            for s in SEEDS:
                shutil.copy(TABLE / f"case500_{row}_seed{s}.json", tmp)
        assert silent_summary(tmp) == 0
        f = tmp / "case500_physics_seed0.json"
        rec = json.loads(f.read_text())
        v = rec["cost_mape"]
        rec["cost_mape"] = v * (1 + 1e-13)
        f.write_text(json.dumps(rec))
        assert silent_summary(tmp) == 0, "a rounding difference is not a disagreement"
        rec["cost_mape"] = v * (1 + 1e-6)
        f.write_text(json.dumps(rec))
        assert silent_summary(tmp) == 1
        f.unlink()
        assert silent_summary(tmp) == 2
        assert silent_summary(tmp / "absent") == 2
    print("--summarize on copies of the records: exit 0 when complete, 1 on a moved value, "
          "2 on a missing record")


CHECKS = [(fn.__name__, fn) for fn in (
    check_printed_rows, check_record_consistency, check_params_and_samples,
    check_wall_clock, check_vendored_gridsfm, check_imports_leave_the_device_probe,
    check_summary_exit_codes)]
CHECKS += [(f"re-score {row}", lambda row=row: spot_check(row)) for row in SPOT_CHECKS]


def main() -> int:
    if "--summarize" in sys.argv:
        rest = sys.argv[sys.argv.index("--summarize") + 1:]
        return summarize(Path(rest[0]) if rest else OUTPUTS / "case500" / "rescore")
    failed = rescored = 0
    skipped = []
    for name, fn in CHECKS:
        try:
            fn()
            rescored += name.startswith("re-score")
            print(f"PASS {name}", flush=True)
        except Skip as e:
            skipped.append(str(e))
            print(f"SKIP {name}: {e}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
    line = (f"{len(CHECKS) - len(skipped)} checks ran, {failed} failed, {rescored} of "
            f"{len(SPOT_CHECKS)} rows re-scored")
    if skipped:
        line += f", {len(skipped)} skipped ({'; '.join(sorted(set(skipped)))})"
    print(f"\n{line}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
