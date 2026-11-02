"""Fetch the OPFData test split of one pglib case, one file per scenario.

The zero-shot grids (``case2000``, ``case6470``) are only ever evaluated, so only
their test split is needed. A whole group costs about 135 GB of JSON per variant
(15k files x ~9 MB), of which the test split is 5%. This script streams the
``.tar.gz`` from the OPFData bucket and writes just the test members, each
gzip-compressed (~0.3-0.9 MB).

PyG's ``OPFDataset(..., num_groups=1, split='test')`` assigns example ``i`` of
``group_0`` to

    train  i <  int(15000 * 1 * 0.90) = 13500
    val    i <  13500 + int(15000 * 1 * 0.05) = 14250
    test   i >= 14250

so the evaluation set is ``example_14250 .. example_14999``, the same 750 graphs
for every model.

The stream is single-pass, because gzip is not seekable: the archive is pulled
over the network until all 750 members have been seen, 4-5 min per variant at
18 MB/s.

The output directory is where ``load_pool`` looks for it. ``--verify-only``
checks an existing copy offline: all 750 files present, no partial downloads,
and (``--deep``) every file decompresses and the per-variant content digest
matches ``checksums.json``.

Usage (from the repository root):
    python -m powerform.data.corpus.fetch_test_split --case case2000 --variant fulltop
    python -m powerform.data.corpus.fetch_test_split --case case6470 --variant fulltop
    python -m powerform.data.corpus.fetch_test_split --case case6470 --verify-only --deep
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from powerform.data.opfdata import ALIAS_OF, TEST_SPLIT_DIRS, resolve_case

BUCKET = "https://storage.googleapis.com/gridopt-dataset"

# PyG OPFDataset split boundaries for num_groups=1 (see module docstring).
TEST_START = 14250
TEST_END = 15000  # exclusive
N_TEST = TEST_END - TEST_START

RELEASE = {"fulltop": "dataset_release_1", "n1": "dataset_release_1_nminusone"}

CHECKSUMS = Path(__file__).resolve().parent / "checksums.json"


def default_out(case: str) -> Path:
    """Where the loaders read this case's per-file test split. A case the loader
    has no directory for is refused: fetching it would produce files nothing
    reads."""
    alias = ALIAS_OF.get(resolve_case(case), case)
    if alias not in TEST_SPLIT_DIRS:
        raise SystemExit(
            f"{case}: only {', '.join(sorted(TEST_SPLIT_DIRS))} are read from per-file "
            "test splits; the other grids come from fetch_opfdata. To fetch it anyway, "
            "pass --out")
    return TEST_SPLIT_DIRS[alias]


def member_index(name: str) -> int | None:
    """``.../group_0/example_14250.json`` -> 14250; ``None`` for anything else."""
    stem = name.rsplit("/", 1)[-1]
    if not stem.startswith("example_") or not stem.endswith(".json"):
        return None
    try:
        return int(stem[len("example_"):-len(".json")])
    except ValueError:
        return None


def source_url(case: str, variant: str) -> str:
    return f"{BUCKET}/{RELEASE[variant]}/{resolve_case(case)}_0.tar.gz"


def fetch_variant(variant: str, out_dir: Path, case: str, timeout: float = 120.0) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    have = {int(p.name[len("example_"):-len(".json.gz")])
            for p in out_dir.glob("example_*.json.gz")}
    want = set(range(TEST_START, TEST_END))
    missing = want - have
    if not missing:
        print(f"[{variant}] all {N_TEST} test members already present, skipping")
        return 0

    url = source_url(case, variant)
    print(f"[{variant}] streaming {url}")
    print(f"[{variant}] want examples {TEST_START}..{TEST_END - 1} "
          f"({len(missing)} still missing)")

    t0 = time.time()
    n_written, n_scanned = 0, 0
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                n_scanned += 1
                idx = member_index(member.name)
                if idx is None or idx not in missing:
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                # A temp name first, so an interrupted run never leaves a
                # truncated file that a later run would treat as present.
                tmp = out_dir / f"example_{idx}.json.gz.part"
                with src, gzip.open(tmp, "wb", compresslevel=6) as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                tmp.replace(out_dir / f"example_{idx}.json.gz")
                missing.discard(idx)
                n_written += 1
                if n_written % 25 == 0 or not missing:
                    el = time.time() - t0
                    print(f"[{variant}] {n_written}/{N_TEST} written, "
                          f"{n_scanned} members scanned, {el / 60:.1f} min",
                          flush=True)
                if not missing:
                    break

    if missing:
        raise SystemExit(
            f"[{variant}] tar ended with {len(missing)} members missing "
            f"(first few: {sorted(missing)[:5]}); archive layout changed?")
    print(f"[{variant}] done: {n_written} written in "
          f"{(time.time() - t0) / 60:.1f} min -> {out_dir}")
    return n_written


# ---------------------------------------------------------------------------
# Verification: offline, against what fetch_variant() writes
# ---------------------------------------------------------------------------

def tree_digest(var_dir: Path, chunk: int = 1 << 20) -> dict:
    """SHA-256 over the 750 test files' decompressed content, in index order.

    Each file contributes ``example_<i>.json <sha256 of its content>\\n`` to
    an outer digest, so the result pins both the set of files and every
    byte of every scenario, and is independent of gzip's timestamped
    container (which differs on every fetch).
    """
    outer = hashlib.sha256()
    n_bytes = 0
    for i in range(TEST_START, TEST_END):
        p = var_dir / f"example_{i}.json.gz"
        h = hashlib.sha256()
        with gzip.open(p, "rb") as fh:
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                h.update(block)
                n_bytes += len(block)
        outer.update(f"example_{i}.json {h.hexdigest()}\n".encode())
    return {"files": N_TEST, "bytes": n_bytes, "sha256": outer.hexdigest()}


def verify(case: str, variant: str, out_root: Path, deep: bool = False,
           expected: dict | None = None) -> list[str]:
    """Problems with the local test split of ``case``/``variant``; empty = OK."""
    d = out_root / variant
    if not d.is_dir():
        return [f"{d} missing"]
    have = {int(p.name[len("example_"):-len(".json.gz")])
            for p in d.glob("example_*.json.gz")}
    want = set(range(TEST_START, TEST_END))
    problems = []
    if have - want:
        problems.append(f"{len(have - want)} files outside the test split "
                        f"(first {sorted(have - want)[:3]})")
    if want - have:
        problems.append(f"{len(want - have)} of {N_TEST} test files missing "
                        f"(first {sorted(want - have)[:3]})")
    partial = sorted(p.name for p in d.glob("*.part"))
    if partial:
        problems.append(f"{len(partial)} partial downloads left: {partial[:3]}")
    empty = [p.name for p in d.glob("example_*.json.gz") if p.stat().st_size == 0]
    if empty:
        problems.append(f"{len(empty)} empty files: {empty[:3]}")
    if deep and not problems:
        try:
            got = tree_digest(d)
        except (OSError, EOFError) as exc:
            return problems + [f"a file does not decompress: {exc}"]
        alias = ALIAS_OF.get(resolve_case(case), case)
        ref = (expected or {}).get(f"opfdata_{alias}/{variant}")
        if ref is not None and ref.get("sha256") != got["sha256"]:
            problems.append(f"content sha256 {got['sha256'][:16]}... != "
                            f"expected {ref['sha256'][:16]}...")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=["fulltop", "n1", "both"],
                    default="fulltop")
    ap.add_argument("--case", required=True,
                    help="pglib case name as it appears in the bucket, or its "
                         "short alias (case2000, case6470, ...)")
    ap.add_argument("--out", default=None,
                    help="output root (default: opfdata_<alias>/ under the "
                         "data home, where load_pool reads it)")
    ap.add_argument("--verify-only", action="store_true",
                    help="check the local copy; never touches the network")
    ap.add_argument("--deep", action="store_true",
                    help="with --verify-only: decompress every file and check "
                         "the content sha256")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be fetched and where, then stop")
    args = ap.parse_args()
    out_root = Path(args.out).resolve() if args.out else default_out(args.case)
    variants = ["fulltop", "n1"] if args.variant == "both" else [args.variant]

    if args.verify_only:
        expected = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
        bad = 0
        for v in variants:
            problems = verify(args.case, v, out_root, deep=args.deep,
                              expected=expected)
            bad += bool(problems)
            print(f"[{'OK' if not problems else 'FAIL'}] {args.case}/{v} "
                  f"({out_root / v})")
            for p in problems:
                print(f"       {p}")
        raise SystemExit(1 if bad else 0)

    for v in variants:
        if args.dry_run:
            print(f"{source_url(args.case, v)} -> {out_root / v}")
            continue
        try:
            fetch_variant(v, out_root / v, case=resolve_case(args.case))
        except urllib.error.HTTPError as e:
            raise SystemExit(f"could not fetch {source_url(args.case, v)}: {e}") from e


if __name__ == "__main__":
    main()
