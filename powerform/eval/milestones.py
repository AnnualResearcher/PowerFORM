"""Score a long run's milestones on the Table 1 hold-out, through each arm's own scorer.

A milestone is not what a shorter run would reach: the cosine schedule spans the whole
run, so a milestone is taken at a learning rate above the floor. The figure therefore
marks what shorter budgets reach with the ends of their own runs. The last milestone is
scored first and must reproduce the run's recorded cost, otherwise this process stops
before writing it; with `--part k/n` only the process holding the last milestone (k = 0)
makes that check.

For the training curve only; no checkpoint is chosen from these numbers. Scores go to
`<work>/out/milestones_<run>.jsonl` (`.part<k>.jsonl` with `--part`), which
`powerform.experiments.long_run_records` merges into Figure 2's record.

    python -m powerform.eval.milestones --arch physics --steps 120000
    python -m powerform.eval.milestones --arch gridsfm --steps 120000 --part 0/3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch

from powerform.data.datakit_loader import (
    build_static, corpus_ids, load_case500, pyg_split, raw_dir_of, scenario_labels,
    table1_split, template_gen_df,
)
from powerform.eval.checkpoint import load_model
from powerform.model.baselines import recipe
from powerform.model.baselines.common import default_dtype
from powerform.paths import shown
from powerform.training import case500
from powerform.training.long_run import work_dir

STEM = {"physics": "case500_physics_seed{}", "gridsfm": "case500_gridsfm_decision_seed{}"}
KEEP = ("cost_mape", "n_nonconverged", "v_mape", "th_mae", "pg_mae", "qg_mae")
MATCH_TOL = 1e-3   # percentage points of cost MAPE


def physics_scorer(data_dir: str, device: torch.device) -> Callable[[Path], dict]:
    """Scores a physics-maps checkpoint on the Table 1 hold-out (float64)."""
    with default_dtype(torch.float64):
        _, hold_ids = table1_split(corpus_ids(data_dir))
        data = load_case500(data_dir, scenario_ids=hold_ids)
        gen_df = template_gen_df(data_dir)
        spec, m_free = case500.decision_spec(data, device)

    def score(ck: Path) -> dict:
        with default_dtype(torch.float64):
            return case500.evaluate(load_model(ck, device), data, spec, m_free, data.scenarios,
                                    gen_df, device)
    return score


def gridsfm_scorer(data_dir: str, pyg_dir: str, device: torch.device) -> Callable[[Path], dict]:
    """Scores a GridSFM decision checkpoint on the Table 1 hold-out (float32)."""
    st = build_static(raw_dir_of(data_dir))
    _, hold_paths = pyg_split(pyg_dir)
    hold_l = [scenario_labels(p, st) for p in hold_paths]

    def score(ck: Path) -> dict:
        model = recipe.build_gridsfm_decision()
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)["model"])
        model.eval()
        return recipe.score_decisions(recipe.gridsfm_decider(model, hold_paths, st, device),
                                      hold_l, st)
    return score


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--arch", required=True, choices=sorted(STEM))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, required=True, help="the run's budget")
    ap.add_argument("--every", type=int, default=5000, help="the run's milestone spacing")
    ap.add_argument("--work", default=None, help="the run's folder; default outputs/long_run/steps<N>")
    ap.add_argument("--part", default="0/1",
                    help="k/n: score every n-th milestone, so n processes can share a run")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=None, help="as for powerform.training.case500")
    ap.add_argument("--pyg-dir", default=None, help="as for powerform.training.case500")
    a = ap.parse_args(argv)
    if a.steps % a.every:
        ap.error("--steps must be a multiple of --every")
    row = case500.parse_args(["--arch", a.arch, "--seed", str(a.seed)]
                             + (["--data-dir", a.data_dir] if a.data_dir else [])
                             + (["--pyg-dir", a.pyg_dir] if a.pyg_dir else []))
    device = torch.device(a.device)
    stem = STEM[a.arch].format(a.seed)
    work = Path(a.work) if a.work else work_dir(a.steps)
    record = work / "out" / f"{stem}.json"
    if not record.exists():
        raise SystemExit(f"no record at {shown(record)}; the run has not finished")
    recorded = json.loads(record.read_text())["cost_mape"]
    k, n = (int(x) for x in a.part.split("/"))
    out = work / "out" / (f"milestones_{stem}.jsonl" if n == 1 else f"milestones_{stem}.part{k}.jsonl")
    done = {json.loads(line)["name"] for p in (work / "out").glob(f"milestones_{stem}*.jsonl")
            for line in p.read_text().splitlines() if line.strip()}

    # The last milestone first: if it misses the record, nothing after it means anything.
    last = f"step{a.steps:06d}"
    jobs = [(f"step{s:06d}", work / "milestones" / stem / f"step{s:06d}.pt")
            for s in range(a.steps, 0, -a.every)]
    missing = [shown(ck) for _, ck in jobs if not ck.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} milestones missing, e.g. {missing[0]}; "
                         "check --steps and --every against the run's --milestone-every")
    score = (physics_scorer(row.data_dir, device) if a.arch == "physics"
             else gridsfm_scorer(row.data_dir, row.pyg_dir, device))
    for name, ck in jobs[k::n]:
        if name in done:
            continue
        t0 = time.time()
        res = score(ck)
        line = {"name": name, **{key: res[key] for key in KEEP}, "s": round(time.time() - t0)}
        print(json.dumps(line), flush=True)
        if name == last and abs(res["cost_mape"] - recorded) > MATCH_TOL:
            raise SystemExit(f"{last} gives {res['cost_mape']}, the run recorded {recorded}")
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
