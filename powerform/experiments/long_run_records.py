"""Turn finished long runs into the record files Figure 2 reads (App. D).

For one arm and seed, from the run of each budget given: the run's record, copied byte for
byte as `case500_<row>_steps<N>_seed<s>.json`; and, for a run whose milestones were scored
(`powerform.eval.milestones`), the scores merged into `milestones_<arch>_steps<N>_seed<s>.json`
(rows sorted by step, values untouched) and its loss log into `curve_<arch>_steps<N>_seed<s>.json`.
`results/long_run/` was written this way. Give the output folder to
`powerform.experiments.training_curve --long-run` to draw a retrained figure.

    python -m powerform.experiments.long_run_records --arch physics --steps 60000 120000
    python -m powerform.experiments.long_run_records --arch gridsfm --steps 60000 120000
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from powerform.paths import OUTPUTS, shown
from powerform.training.long_run import work_dir

OUT_DIR = OUTPUTS / "long_run" / "records"
ROW = {"physics": "case500_physics", "gridsfm": "case500_gridsfm_decision"}
SCORED_ON = "Table 1 hold-out, 500 scenarios"
CURVE_FIELDS = "loss_mean over the interval, lr after the step, s = wall-clock"


def json_lines(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for p in paths for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def spacing(steps: list[int]) -> int:
    """The one gap between consecutive steps, starting from 0; refuses uneven or missing rows."""
    gaps = {b - a for a, b in zip([0] + steps, steps)}
    if len(gaps) != 1:
        raise SystemExit(f"rows are not evenly spaced: steps {steps[:3]} ... {steps[-3:]}")
    return gaps.pop()


def write(path: Path, doc: dict) -> None:
    path.write_bytes(json.dumps(doc, indent=1).encode("utf-8"))
    print(f"-> {shown(path)}")


def export(arch: str, seed: int, steps: int, work: Path, out: Path) -> None:
    """The record files of one run; its milestones and loss log only if they were scored."""
    stem = f"{ROW[arch]}_seed{seed}"
    record = work / "out" / f"{stem}.json"
    if not record.exists():
        raise SystemExit(f"no record at {shown(record)}; the run has not finished")
    shutil.copyfile(record, out / f"{ROW[arch]}_steps{steps}_seed{seed}.json")
    print(f"-> {shown(out / f'{ROW[arch]}_steps{steps}_seed{seed}.json')}")

    scored = sorted((work / "out").glob(f"milestones_{stem}*.jsonl"))
    if not scored:
        print(f"   no scored milestones for {stem} at {steps} steps: record only")
        return
    rows = [r for r in json_lines(scored) if r["name"].startswith("step")]
    for r in rows:
        r["step"] = int(r.pop("name")[len("step"):])
    rows.sort(key=lambda r: r["step"])
    every = spacing([r["step"] for r in rows])
    if rows[-1]["step"] != steps:
        raise SystemExit(f"{stem}: milestones end at {rows[-1]['step']}, the run at {steps}")
    write(out / f"milestones_{arch}_steps{steps}_seed{seed}.json",
          {"arch": arch, "steps": steps, "seed": seed, "every": every, "scored_on": SCORED_ON,
           "rows": rows})

    curve = sorted(json_lines([work / "curves" / f"{stem}.jsonl"]), key=lambda r: r["step"])
    write(out / f"curve_{arch}_steps{steps}_seed{seed}.json",
          {"arch": arch, "steps": steps, "seed": seed, "log_every": spacing([r["step"] for r in curve]),
           "fields": CURVE_FIELDS, "rows": curve})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--arch", required=True, choices=sorted(ROW))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs="+", required=True, help="the budgets to export")
    ap.add_argument("--work-root", type=Path, default=None,
                    help="folder holding steps<N>/ for each budget; default outputs/long_run")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for steps in a.steps:
        work = (a.work_root / f"steps{steps}") if a.work_root else work_dir(steps)
        export(a.arch, a.seed, steps, work, a.out_dir)


if __name__ == "__main__":
    main()
