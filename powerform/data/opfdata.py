"""Corpus loading: which OPFData scenarios a run sees.

Scenarios are chosen deterministically: split bounds are fixed and subsampling
is even, so the run seed varies the optimisation and not the corpus. The training
grids and case30 are consolidated shards (``corpus/fetch_opfdata.py``); the two
largest zero-shot grids are test splits of one file per scenario
(``corpus/fetch_test_split.py``).
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from powerform.data.opfdata_case import build_instance
from powerform.paths import DATA_HOME

DATA_ROOT = DATA_HOME / "opfdata_hf"

TRAIN_END = 13500
VAL_END = 14250
N_GROUP = 15000
SPLIT_BOUNDS = {"train": (0, TRAIN_END), "val": (TRAIN_END, VAL_END),
                "test": (VAL_END, N_GROUP)}

CASE_ALIAS = {
    "case30": "pglib_opf_case30_ieee",
    "case57": "pglib_opf_case57_ieee",
    "case118": "pglib_opf_case118_ieee",
    "case500": "pglib_opf_case500_goc",
    "case2000": "pglib_opf_case2000_goc",
    "case6470": "pglib_opf_case6470_rte",
}
ALIAS_OF = {v: k for k, v in CASE_ALIAS.items()}

# Grids with a per-file test split instead of a shard.
TEST_SPLIT_DIRS = {
    "case2000": DATA_HOME / "opfdata_case2000",
    "case6470": DATA_HOME / "opfdata_case6470",
}


def resolve_case(name: str) -> str:
    """``case30`` or ``pglib_opf_case30_ieee`` -> the full bucket name."""
    return CASE_ALIAS.get(name, name)


def test_split_dir(case: str, variant: str) -> Path | None:
    """Per-file test-split directory for ``case``/``variant``, if one exists."""
    root = TEST_SPLIT_DIRS.get(ALIAS_OF.get(resolve_case(case), case))
    if root is None:
        return None
    d = root / variant
    return d if d.is_dir() and any(d.glob("example_*.json.gz")) else None


def shard_path(case: str, variant: str, split: str,
               root: Path = DATA_ROOT) -> Path:
    return Path(root) / resolve_case(case) / variant / f"{split}.jsonl.gz"


def fetch_command(case: str, variant: str) -> str:
    """The command that fetches ``case``/``variant``."""
    alias = ALIAS_OF.get(resolve_case(case), case)
    if alias in TEST_SPLIT_DIRS:
        return (f"python -m powerform.data.corpus.fetch_test_split "
                f"--case {alias} --variant {variant}")
    return (f"python -m powerform.data.corpus.fetch_opfdata "
            f"--case {resolve_case(case)} --variant {variant}")


def read_manifest(case: str, variant: str, root: Path = DATA_ROOT) -> dict:
    mp = Path(root) / resolve_case(case) / variant / "manifest.json"
    if not mp.exists():
        raise SystemExit(f"{case}/{variant} is not under {root}; fetch it with "
                         f"{fetch_command(case, variant)}, or point "
                         f"POWERFORM_DATA at the data home")
    return json.loads(mp.read_text())


def iter_raw(case: str, variant: str, split: str, root: Path = DATA_ROOT,
             wanted: set[int] | None = None) -> Iterator[tuple[int, dict]]:
    """Yield ``(example_index, scenario_dict)`` from a consolidated shard.

    The manifest records the example index of every line in on-disk order, so a
    subsampled pool skips the ``json.loads`` of the lines it does not want.
    """
    man = read_manifest(case, variant, root)
    info = man["splits"].get(split)
    if info is None:
        raise SystemExit(f"split {split!r} not in manifest for {case}/{variant}")
    order = info["example_indices"]
    path = shard_path(case, variant, split, root)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            idx = order[line_no]
            if wanted is not None and idx not in wanted:
                continue
            yield idx, json.loads(line)


def _iter_test_files(d: Path, wanted: set[int]) -> Iterator[tuple[int, dict]]:
    files = sorted(d.glob("example_*.json.gz"),
                   key=lambda p: int(p.name.split("_")[1].split(".")[0]))
    for p in files:
        idx = int(p.name.split("_")[1].split(".")[0])
        if idx not in wanted:
            continue
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            yield idx, json.load(fh)


def subsample(indices: list[int], n: int) -> list[int]:
    """Deterministic even subsample: two runs asking for the same ``n`` see the
    same scenarios whatever their seed."""
    if n <= 0 or n >= len(indices):
        return list(indices)
    sel = np.linspace(0, len(indices) - 1, n).round().astype(int)
    return [indices[i] for i in dict.fromkeys(sel.tolist())]


@dataclass
class Pool:
    """Built instances for one (case, variant, split) selection."""

    case: str
    variant: str
    split: str
    n_cost_feats: int
    example_indices: list[int] = field(default_factory=list)
    insts: list = field(default_factory=list)
    specs: list = field(default_factory=list)
    labels: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.insts)

    @property
    def alias(self) -> str:
        return ALIAS_OF.get(resolve_case(self.case), self.case)

    def provenance(self) -> dict:
        return {
            "case": resolve_case(self.case),
            "alias": self.alias,
            "variant": self.variant,
            "split": self.split,
            "n": len(self.insts),
            "split_range": list(SPLIT_BOUNDS[self.split]),
            "example_index_min": min(self.example_indices) if self.example_indices else None,
            "example_index_max": max(self.example_indices) if self.example_indices else None,
            "n_cost_feats": self.n_cost_feats,
        }

    def _add(self, idx: int, obj: dict, device) -> None:
        inst, spec, lab = build_instance(obj, device, self.n_cost_feats)
        self.example_indices.append(idx)
        self.insts.append(inst)
        self.specs.append(spec)
        self.labels.append(lab)


def load_pool(case: str, variant: str, split: str, n: int = 0,
              n_cost_feats: int = 12, device=torch.device("cpu"),
              root: Path = DATA_ROOT, verbose: bool = True) -> Pool:
    """``n`` evenly subsampled scenarios of a split; 0 means the whole split."""
    pool = Pool(case=resolve_case(case), variant=variant, split=split,
                n_cost_feats=n_cost_feats)
    d = test_split_dir(case, variant)
    if d is not None and not shard_path(case, variant, split, root).exists():
        if split != "test":
            raise SystemExit(f"{case}/{variant}: only a test split exists on "
                             f"disk; split={split!r} needs a shard fetch")
        all_idx = sorted(int(p.name.split("_")[1].split(".")[0])
                         for p in d.glob("example_*.json.gz"))
        for idx, obj in _iter_test_files(d, set(subsample(all_idx, n))):
            pool._add(idx, obj, device)
    else:
        all_idx = read_manifest(case, variant, root)["splits"][split]["example_indices"]
        lo, hi = SPLIT_BOUNDS[split]
        off = [i for i in all_idx if not (lo <= i < hi)]
        if off:
            raise SystemExit(
                f"{case}/{variant}/{split}: {len(off)} example indices fall "
                f"outside [{lo},{hi}) (first: {off[:5]}); the shard was written "
                f"with different split arithmetic")
        wanted = set(subsample(sorted(all_idx), n))
        for idx, obj in iter_raw(case, variant, split, root, wanted=wanted):
            pool._add(idx, obj, device)
    if verbose:
        n0 = pool.insts[0].n_nodes if pool.insts else 0
        e0 = pool.insts[0].n_edges if pool.insts else 0
        print(f"  pool {pool.alias}/{variant}/{split}: {len(pool)} scenarios "
              f"(N={n0} E={e0}), examples "
              f"{min(pool.example_indices)}..{max(pool.example_indices)}",
              flush=True)
    return pool


def assert_disjoint(train_pools: list[Pool], eval_pools: list[Pool]) -> None:
    """Refuse to run if any evaluation scenario appears in training."""
    for ep in eval_pools:
        lo, hi = SPLIT_BOUNDS[ep.split]
        bad = [i for i in ep.example_indices if not (lo <= i < hi)]
        if bad:
            raise SystemExit(
                f"eval pool {ep.alias}/{ep.split} leaves its split range: {bad[:5]}")
        for tp in train_pools:
            if resolve_case(tp.case) != resolve_case(ep.case):
                continue
            if tp.variant != ep.variant:
                continue
            overlap = set(tp.example_indices) & set(ep.example_indices)
            if overlap:
                raise SystemExit(
                    f"train/eval leakage on {ep.alias}/{ep.variant}: "
                    f"{len(overlap)} shared example indices "
                    f"(first: {sorted(overlap)[:5]})")


def label_cross_check(pool: Pool, rtol: float = 1e-5) -> dict:
    """Re-sum the label cost and compare against OPFData's own ``objective``: the
    cheapest check that the cost convention (coefficients, basis, generators)
    agrees with the one that produced the dataset."""
    rels = []
    for lab in pool.labels:
        denom = max(abs(lab.objective), 1.0)
        rels.append(abs(lab.cost - lab.objective) / denom)
    rels = np.asarray(rels)
    return {
        "n": int(rels.size),
        "rel_max": float(rels.max()) if rels.size else None,
        "rel_mean": float(rels.mean()) if rels.size else None,
        "rtol": rtol,
        "pass": bool(rels.size and rels.max() < rtol),
    }
