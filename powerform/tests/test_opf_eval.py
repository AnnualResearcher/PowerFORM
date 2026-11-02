"""Table 2 and the App. D OOD table against `results/table2`, and re-scored cells.

The printed numbers are checked from the shipped records. With data and checkpoints
present, cells are also re-scored: floats agree within 1e-10 relative or 1e-12
absolute (under the pinned requirements they are bit-identical), counts exactly.

`--summarize` compares the cells `eval/opf_eval.py` wrote to a directory, by default
outputs/table2, with the records and the printed tables. It exits 0 when every cell
was run and agrees, 1 on any disagreement, 2 when nothing disagrees but some cells
were not run.

    python powerform/tests/test_opf_eval.py
    python powerform/tests/test_opf_eval.py --summarize [DIR]
"""
from __future__ import annotations

import contextlib
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

from powerform.data.opfdata import fetch_command, load_pool, shard_path  # noqa: E402
from powerform.eval.checkpoint import load_model  # noqa: E402
from powerform.eval.opf_eval import eval_grid  # noqa: E402
from powerform.paths import CHECKPOINTS, OUTPUTS, RESULTS  # noqa: E402

TABLE2 = RESULTS / "table2"
FETCH_CKPT = "python scripts/download_checkpoints.py --group table2"
REL_TOL, ABS_FLOOR = 1e-10, 1e-12
CELLS = [("case57", "fulltop"), ("case57", "n1"), ("case118", "fulltop"), ("case118", "n1"),
         ("case500", "fulltop"), ("case500", "n1"), ("case30", "fulltop"),
         ("case2000", "fulltop"), ("case6470", "fulltop")]
# The tables as printed, in the order of CELLS.
PRINTED = {
    "physics": ["0.38", "1.55", "0.05", "0.38", "0.29", "0.76", "6.9", "28.8", "non-conv."],
    "generic": ["0.42", "1.71", "0.04", "0.41", "0.14", "0.59", "19.8", "19.2", "non-conv."],
    "gridsfm": ["0.45", "1.62", "0.07", "0.61", "0.51", "0.75", "21.8", "29.7", "non-conv."],
    "ed": ["7.93", "8.00", "3.80", "3.79", "7.35", "7.37", "7.3", "0.96", "0.95"],
}
SEEDS = {"physics": [0, 1, 2], "generic": [0, 1, 2], "gridsfm": [0], "ed": [None]}


class Skip(Exception):
    pass


def cell_path(row: str, seed, grid: str, variant: str) -> Path:
    stem = row if seed is None else f"{row}_seed{seed}"
    return TABLE2 / f"{stem}_{grid}_{variant}.json"


def rendered(values: list, printed: str) -> str:
    """A table entry from its per-seed cost MAPEs, at the printed precision."""
    if all(v is None for v in values):
        return "non-conv."
    if any(v is None for v in values):
        return "mixed"
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    return f"{np.mean(values):.{decimals}f}"


def table_values(row: str, grid: str, variant: str) -> list:
    out = []
    for s in SEEDS[row]:
        out.append(json.loads(cell_path(row, s, grid, variant).read_text()).get("cost_mape_pct"))
    return out


def diff(a, b, path: str = "", exact: bool = True) -> list[str]:
    """Every path where two cells differ; `exact=False` applies the float tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}: keys {sorted(set(a) ^ set(b))}"] if set(a) != set(b) else []
        for k in set(a) & set(b):
            out += diff(a[k], b[k], f"{path}.{k}", exact)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in diff(x, y, f"{path}[{i}]", exact)]
    if isinstance(a, float) and isinstance(b, float) and not exact:
        gap = abs(a - b)
        return [] if gap <= ABS_FLOOR or gap <= REL_TOL * max(abs(a), abs(b)) else [f"{path}: {a!r} vs {b!r}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def check_printed_tables() -> None:
    n = 0
    for row, printed in PRINTED.items():
        for (grid, variant), want in zip(CELLS, printed):
            got = rendered(table_values(row, grid, variant), want)
            assert got == want, f"{row} {grid}/{variant}: {got} vs printed {want}"
            n += 1
    # a moved value must change what is rendered
    vals = table_values("physics", "case57", "fulltop")
    assert rendered([v * 1.05 for v in vals], "0.38") != "0.38"
    print(f"{n} printed entries of Table 2 and the App. D OOD table")


def check_seed_ranges() -> None:
    """Every seed diverges on case6470, and only case118 N-1 separates the two map
    families."""
    assert all(c["n_scored"] == 0 for c in (
        json.loads(cell_path(m, s, "case6470", "fulltop").read_text())
        for m in ("physics", "generic") for s in (0, 1, 2)))
    assert json.loads(cell_path("gridsfm", 0, "case6470", "fulltop").read_text())["n_scored"] == 0
    separated = []
    for grid, variant in CELLS[:-1]:
        p = table_values("physics", grid, variant)
        g = table_values("generic", grid, variant)
        if max(p) < min(g) or max(g) < min(p):
            separated.append((grid, variant))
    assert separated == [("case118", "n1")], separated
    p, g = table_values("physics", "case118", "n1"), table_values("generic", "case118", "n1")
    assert [f"{min(p):.3f}", f"{max(p):.3f}", f"{min(g):.3f}", f"{max(g):.3f}"] == \
        ["0.372", "0.386", "0.389", "0.432"], (p, g)
    print("case6470 diverges for every seed; only case118 N-1 separates the families")


def strip(record: dict) -> dict:
    return {k: v for k, v in record.items()
            if k not in ("model", "checkpoint", "grid", "variant")}


def check_one_schema() -> None:
    """A scored cell and a failed cell each have one set of fields, whichever row
    wrote them, and no path or machine-specific string."""
    scored, failed = set(), set()
    for row in PRINTED:
        for s in SEEDS[row]:
            for grid, variant in CELLS:
                cell = json.loads(cell_path(row, s, grid, variant).read_text())
                keys = frozenset(cell) | frozenset(f"provenance.{k}" for k in cell.get("provenance", {}))
                (scored if cell["n_scored"] else failed).add(keys)
    assert len(scored) == 1 and len(failed) == 1, (scored, failed)
    print(f"{len(next(iter(scored)))} fields in a scored cell, {len(next(iter(failed)))} in a failed one")


def rescored_rows() -> list[str]:
    """The rows `eval/opf_eval.py` can write; the GridSFM row ships as records only."""
    return [row for row in PRINTED if row != "gridsfm"]


def compare_cell(path: Path, want: dict) -> str:
    """'exact', 'within tolerance', 'disagree', or 'not run' for a missing cell or a
    subset run, which says nothing about the whole split."""
    if not path.exists():
        return "not run"
    got = strip(json.loads(path.read_text()))
    if got.get("n_eval") != want["n_eval"]:
        return "not run"
    if not diff(got, want):
        return "exact"
    return "within tolerance" if not diff(got, want, exact=False) else "disagree"


def summarize(cells_dir: Path) -> int:
    """Each printed entry from the cells in `cells_dir`; returns the exit code."""
    if not cells_dir.is_dir():
        print(f"{cells_dir} does not exist: nothing to compare")
        return 2
    counts = dict.fromkeys(("exact", "within tolerance", "disagree", "not run"), 0)
    bad_entry = False
    for row in rescored_rows():
        for (grid, variant), printed in zip(CELLS, PRINTED[row]):
            verdicts, values = [], []
            for s in SEEDS[row]:
                name = cell_path(row, s, grid, variant).name
                want = strip(json.loads((TABLE2 / name).read_text()))
                verdicts.append(compare_cell(cells_dir / name, want))
                counts[verdicts[-1]] += 1
                if verdicts[-1] != "not run":
                    values.append(json.loads((cells_dir / name).read_text()).get("cost_mape_pct"))
            tally = ", ".join(f"{verdicts.count(v)} {v}" for v in counts if v in verdicts)
            label = f"{row:8s}{grid:9s}{variant:8s}printed {printed:10s}"
            if len(values) == len(verdicts):
                got = rendered(values, printed)
                bad_entry |= got != printed
                print(f"  {label}re-scored {got:10s}{tally}")
            else:
                print(f"  {label}{'not run' if not values else 'incomplete, ' + tally}")
    print(f"{sum(counts.values())} cells: " + ", ".join(f"{n} {k}" for k, n in counts.items())
          + ". The GridSFM row ships as records only.")
    if counts["disagree"] or bad_entry:
        return 1
    return 2 if counts["not run"] else 0


def silent_summary(d: Path) -> int:
    """The exit code alone: these checks run the tool on copies of the records."""
    with contextlib.redirect_stdout(io.StringIO()):
        return summarize(d)


def check_summary_exit_codes() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for row in rescored_rows():
            for s in SEEDS[row]:
                for grid, variant in CELLS:
                    shutil.copy(cell_path(row, s, grid, variant), tmp)
        assert silent_summary(tmp) == 0
        f = tmp / "physics_seed0_case57_fulltop.json"
        cell = json.loads(f.read_text())
        v = cell["cost_mape_pct"]
        cell["cost_mape_pct"] = v * (1 + 1e-13)
        f.write_text(json.dumps(cell))
        assert silent_summary(tmp) == 0, "a rounding difference is not a disagreement"
        cell["cost_mape_pct"] = v * (1 + 1e-6)
        f.write_text(json.dumps(cell))
        assert silent_summary(tmp) == 1
        cell["cost_mape_pct"], cell["n_eval"] = v, 32
        f.write_text(json.dumps(cell))
        assert silent_summary(tmp) == 2, "a subset run counts as not run"
        f.unlink()
        assert silent_summary(tmp) == 2
        assert silent_summary(tmp / "absent") == 2
    print("--summarize on copies of the records: exit 0 when complete, 1 on a moved value, "
          "2 on a missing or subset cell")


def need_case57() -> None:
    if not shard_path("case57", "fulltop", "test").exists():
        raise Skip(f"case57 absent; fetch with `{fetch_command('case57', 'fulltop')}`")


def check_ed_rescores_to_the_record() -> None:
    need_case57()
    pool = load_pool("case57", "fulltop", "test", n_cost_feats=12, verbose=False)
    cell = eval_grid(None, pool, baseline="ed")
    want = strip(json.loads(cell_path("ed", None, "case57", "fulltop").read_text()))
    d = diff(cell, want, exact=False)
    assert not d, d[:5]
    print(f"ED case57: cost MAPE {cell['cost_mape_pct']:.4f}%, record agrees")


def check_physics_rescores_to_the_record() -> None:
    need_case57()
    ckpt = CHECKPOINTS / "opf_generalist_physics_seed0.pt"
    if not ckpt.exists():
        raise Skip(f"{ckpt.name} absent; fetch with `{FETCH_CKPT}`")
    model = load_model(ckpt, "cpu")
    pool = load_pool("case57", "fulltop", "test", n_cost_feats=10, verbose=False)
    cell = eval_grid(model, pool)
    want = strip(json.loads(cell_path("physics", 0, "case57", "fulltop").read_text()))
    d = diff(cell, want, exact=False)
    assert not d, d[:5]
    print(f"physics seed 0 case57: cost MAPE {cell['cost_mape_pct']:.4f}%, record agrees")

    small = load_pool("case57", "fulltop", "test", n=10, n_cost_feats=10, verbose=False)
    base = eval_grid(model, small)
    with torch.no_grad():
        p = next(iter(model.parameters()))
        original = p.detach().clone()
        p.view(-1)[0] += 1e-9
    assert diff(base, eval_grid(model, small)), "a moved weight went unnoticed"
    with torch.no_grad():
        p.copy_(original)
    g = small.insts[0].tgrid
    g.p_load = g.p_load.clone()
    g.p_load[int(torch.argmax(g.p_load.abs()))] += 1e-7
    assert diff(base, eval_grid(model, small)), "a moved load went unnoticed"
    print("a weight + 1e-9 and a load + 1e-7 pu each change the cell")


def check_cli_writes_to_out_dir() -> None:
    need_case57()
    with tempfile.TemporaryDirectory() as td:
        env = {**os.environ, "PYTHONPATH": str(REPO)}
        r = subprocess.run([sys.executable, "-m", "powerform.eval.opf_eval", "--baseline", "ed",
                            "--eval", "case57", "--n", "4", "--out-dir", td, "--device", "cpu"],
                           env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        rec = json.loads((Path(td) / "ed_case57_fulltop_n4.json").read_text())
        assert rec["n_eval"] == 4 and rec["model"] == "ed" and rec["checkpoint"] is None
    print("--baseline ed --n 4 writes a subset record to --out-dir")


CHECKS = (check_printed_tables, check_seed_ranges, check_one_schema, check_summary_exit_codes,
          check_ed_rescores_to_the_record, check_physics_rescores_to_the_record,
          check_cli_writes_to_out_dir)


def main() -> None:
    if "--summarize" in sys.argv:
        rest = sys.argv[sys.argv.index("--summarize") + 1:]
        sys.exit(summarize(Path(rest[0]) if rest else OUTPUTS / "table2"))
    ran, failed, skipped = 0, 0, []
    for check in CHECKS:
        name = check.__name__[len("check_"):]
        try:
            check()
            ran += 1
            print(f"[PASS] {name}", flush=True)
        except Skip as e:
            skipped.append(str(e))
            print(f"[SKIP] {name}: {e}", flush=True)
        except (Exception, SystemExit):  # noqa: BLE001 -- reported, and the exit code says so
            ran += 1
            failed += 1
            traceback.print_exc()
            print(f"[FAIL] {name}", flush=True)
    why = f" ({'; '.join(dict.fromkeys(skipped))})" if skipped else ""
    print(f"{ran} checks ran, {failed} failed, {len(skipped)} skipped{why}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
