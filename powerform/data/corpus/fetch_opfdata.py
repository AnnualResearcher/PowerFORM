"""Fetch whole OPFData groups (train+val+test) as consolidated JSONL shards.

OPFData ships each group as one ``.tar.gz`` of 15,000 pretty-printed JSON files.
This script streams the archive once and writes one gzipped JSONL per split,
each line one example re-dumped compact (a case500 example is 790 kB raw against
221 kB compact). The shard is built in the system temp directory and moved into
place only once complete, so the destination never holds a half-written file.

Split arithmetic follows PyG's ``OPFDataset`` with ``num_groups=1``:

    train  i <  int(15000 * 0.90) = 13500
    val    i <  13500 + int(15000 * 0.05) = 14250
    test   i >= 14250

The manifest beside the shards records the example index of every line in
on-disk order (tar member order is not numeric), the split rule and the source
URL.

``--verify-only`` checks an existing copy without touching the network: the
manifest, the layout, the split arithmetic, and (``--deep``) every shard's line
count, byte count and content digest against ``checksums.json``. The digest is
over the decompressed lines, because the gzip container records a timestamp.

Usage (from the repository root):
    python -m powerform.data.corpus.fetch_opfdata --case pglib_opf_case57_ieee --variant both
    python -m powerform.data.corpus.fetch_opfdata --all --variant both
    python -m powerform.data.corpus.fetch_opfdata --all --variant both --verify-only --deep
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from powerform.data.opfdata import DATA_ROOT, resolve_case

BUCKET = "https://storage.googleapis.com/gridopt-dataset"
RELEASE = {"fulltop": "dataset_release_1", "n1": "dataset_release_1_nminusone"}

# PyG OPFDataset split boundaries for num_groups=1 (see module docstring).
N_GROUP = 15000
TRAIN_END = 13500
VAL_END = 14250

DEFAULT_OUT = DATA_ROOT

# case57/118/500 are the training grids (and their N-1 variants the
# unseen-contingency test); case30 is the smallest zero-shot grid. The two larger
# zero-shot grids are fetched as test splits only, by fetch_test_split.py.
CASES = [
    "pglib_opf_case30_ieee",
    "pglib_opf_case57_ieee",
    "pglib_opf_case118_ieee",
    "pglib_opf_case500_goc",
]

SPLITS = ("train", "val", "test")

CHECKSUMS = Path(__file__).resolve().parent / "checksums.json"


def split_of(idx: int) -> str:
    if idx < TRAIN_END:
        return "train"
    if idx < VAL_END:
        return "val"
    return "test"


def member_index(name: str) -> int | None:
    """``.../group_0/example_14250.json`` -> 14250; ``None`` for anything else."""
    stem = name.rsplit("/", 1)[-1]
    if not stem.startswith("example_") or not stem.endswith(".json"):
        return None
    try:
        return int(stem[len("example_"):-len(".json")])
    except ValueError:
        return None


def source_url(case: str, variant: str, group: int = 0) -> str:
    return f"{BUCKET}/{RELEASE[variant]}/{case}_{group}.tar.gz"


def move_into_place(src: Path, dst: Path, retries: int = 6) -> None:
    """Move a finished shard into the destination tree.

    ``os.replace`` is atomic within a volume, but a file-sync client can hold
    an open handle on the destination and raise ``PermissionError``. Back off,
    then unlink and move. A final failure re-raises: a shard that did not land
    must not be reported as fetched.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(1 + attempt)
    dst.unlink(missing_ok=True)
    os.replace(src, dst)


def manifest_path(out_root: Path, case: str, variant: str) -> Path:
    return out_root / case / variant / "manifest.json"


def is_complete(out_root: Path, case: str, variant: str, splits) -> bool:
    mp = manifest_path(out_root, case, variant)
    if not mp.exists():
        return False
    try:
        man = json.loads(mp.read_text())
    except (OSError, ValueError):
        return False
    for s in splits:
        info = man.get("splits", {}).get(s)
        if info is None:
            return False
        if not (out_root / case / variant / f"{s}.jsonl.gz").exists():
            return False
    return True


def fetch(case: str, variant: str, out_root: Path, group: int = 0,
          splits=SPLITS, timeout: float = 120.0, compresslevel: int = 6) -> dict:
    out_dir = out_root / case / variant
    if is_complete(out_root, case, variant, splits):
        print(f"[{case}/{variant}] manifest complete, skipping")
        return json.loads(manifest_path(out_root, case, variant).read_text())

    url = source_url(case, variant, group)
    print(f"[{case}/{variant}] streaming {url}", flush=True)

    tmpdir = Path(tempfile.mkdtemp(prefix=f"opfdata_{case}_{variant}_"))
    writers, tmp_paths, indices, n_bytes = {}, {}, {}, {}
    for s in splits:
        tmp_paths[s] = tmpdir / f"{s}.jsonl.gz"
        writers[s] = gzip.open(tmp_paths[s], "wb", compresslevel=compresslevel)
        indices[s] = []
        n_bytes[s] = 0

    t0 = time.time()
    n_seen = n_kept = 0
    objective_probe = None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            with tarfile.open(fileobj=resp, mode="r|gz") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    idx = member_index(member.name)
                    if idx is None:
                        continue
                    n_seen += 1
                    s = split_of(idx)
                    if s not in writers:
                        continue
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with src:
                        obj = json.load(src)
                    line = json.dumps(obj, separators=(",", ":")).encode()
                    writers[s].write(line)
                    writers[s].write(b"\n")
                    indices[s].append(idx)
                    n_bytes[s] += len(line) + 1
                    n_kept += 1
                    if objective_probe is None:
                        objective_probe = obj["metadata"]["objective"]
                    if n_kept % 1000 == 0:
                        el = time.time() - t0
                        print(f"[{case}/{variant}] {n_kept} kept / {n_seen} seen, "
                              f"{el / 60:.1f} min", flush=True)
    finally:
        for w in writers.values():
            w.close()

    if n_seen != N_GROUP:
        raise SystemExit(
            f"[{case}/{variant}] expected {N_GROUP} examples in group {group}, "
            f"saw {n_seen}; the split arithmetic in this script assumes PyG's "
            f"num_groups=1 layout")

    # Tar member order is not guaranteed numeric; record the on-disk line order
    # so readers can map line -> example index without re-sorting the shard.
    man = {
        "case": case,
        "variant": variant,
        "release": RELEASE[variant],
        "group": group,
        "source_url": url,
        "n_group": n_seen,
        "split_rule": {"train": [0, TRAIN_END], "val": [TRAIN_END, VAL_END],
                       "test": [VAL_END, N_GROUP]},
        "line_order_is_sorted": {s: indices[s] == sorted(indices[s]) for s in splits},
        "splits": {s: {"count": len(indices[s]),
                       "example_indices": indices[s],
                       "uncompressed_bytes": n_bytes[s],
                       "file": f"{s}.jsonl.gz"} for s in splits},
        "objective_probe": objective_probe,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for s in splits:
        move_into_place(tmp_paths[s], out_dir / f"{s}.jsonl.gz")
    tmp_man = tmpdir / "manifest.json"
    tmp_man.write_text(json.dumps(man, indent=1))
    move_into_place(tmp_man, manifest_path(out_root, case, variant))
    try:
        tmpdir.rmdir()
    except OSError:
        pass

    total_mb = sum((out_dir / f"{s}.jsonl.gz").stat().st_size for s in splits) / 1e6
    counts = ", ".join(f"{s}={len(indices[s])}" for s in splits)
    print(f"[{case}/{variant}] done in {(time.time() - t0) / 60:.1f} min: "
          f"{counts}, {total_mb:.0f} MB on disk -> {out_dir}", flush=True)
    return man


# ---------------------------------------------------------------------------
# Verification: offline, against what fetch() writes
# ---------------------------------------------------------------------------

def shard_digest(path: Path, chunk: int = 1 << 20) -> dict:
    """Line count, byte count and SHA-256 of a shard's DECOMPRESSED content.

    Streams; memory is one chunk regardless of shard size.
    """
    h = hashlib.sha256()
    n_bytes = n_lines = 0
    with gzip.open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
            n_bytes += len(block)
            n_lines += block.count(b"\n")
    return {"lines": n_lines, "bytes": n_bytes, "sha256": h.hexdigest()}


def load_checksums(path: Path = CHECKSUMS) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def verify(case: str, variant: str, out_root: Path, splits=SPLITS,
           deep: bool = False, expected: dict | None = None,
           group: int = 0) -> list[str]:
    """Problems with the local copy of ``case``/``variant``; empty means OK.

    Shallow: the manifest exists and parses, names this case/variant/source,
    declares the split rule the loaders assume, and its per-split index
    lists are consistent with that rule (right count, no duplicates, every
    index inside its split's range, the three splits together covering
    0..14999 exactly); every shard file exists and is non-empty.

    Deep (``deep=True``) additionally streams every shard and checks its line
    count and decompressed byte count against the manifest, and its content
    digest against ``expected`` (``checksums.json``) when that has an entry.
    """
    problems: list[str] = []
    out_dir = out_root / case / variant
    mp = manifest_path(out_root, case, variant)
    if not mp.exists():
        return [f"{mp} missing"]
    try:
        man = json.loads(mp.read_text())
    except (OSError, ValueError) as exc:
        return [f"{mp} unreadable: {exc}"]

    want = {"case": case, "variant": variant, "release": RELEASE[variant],
            "group": group, "source_url": source_url(case, variant, group),
            "n_group": N_GROUP}
    for k, v in want.items():
        if man.get(k) != v:
            problems.append(f"manifest {k}={man.get(k)!r}, expected {v!r}")
    rule = {"train": [0, TRAIN_END], "val": [TRAIN_END, VAL_END],
            "test": [VAL_END, N_GROUP]}
    if man.get("split_rule") != rule:
        problems.append(f"manifest split_rule={man.get('split_rule')!r}, "
                        f"expected {rule!r}")

    seen: list[int] = []
    for s in splits:
        info = man.get("splits", {}).get(s)
        if info is None:
            problems.append(f"split {s!r} not in manifest")
            continue
        idx = info.get("example_indices", [])
        lo, hi = rule[s]
        if info.get("count") != len(idx):
            problems.append(f"{s}: count={info.get('count')} but "
                            f"{len(idx)} example indices")
        if len(idx) != hi - lo:
            problems.append(f"{s}: {len(idx)} examples, split rule wants "
                            f"{hi - lo}")
        if len(set(idx)) != len(idx):
            problems.append(f"{s}: duplicate example indices")
        off = [i for i in idx if not (lo <= i < hi)]
        if off:
            problems.append(f"{s}: {len(off)} indices outside [{lo},{hi}) "
                            f"(first {off[:3]})")
        seen.extend(idx)
        shard = out_dir / info.get("file", f"{s}.jsonl.gz")
        if not shard.exists() or shard.stat().st_size == 0:
            problems.append(f"{shard} missing or empty")
            continue
        if deep:
            got = shard_digest(shard)
            if got["lines"] != len(idx):
                problems.append(f"{s}: shard has {got['lines']} lines, "
                                f"manifest lists {len(idx)}")
            if got["bytes"] != info.get("uncompressed_bytes"):
                problems.append(f"{s}: shard decompresses to {got['bytes']} "
                                f"bytes, manifest says "
                                f"{info.get('uncompressed_bytes')}")
            ref = (expected or {}).get(f"opfdata_hf/{case}/{variant}/{s}")
            if ref is not None and ref.get("sha256") != got["sha256"]:
                problems.append(f"{s}: content sha256 {got['sha256'][:16]}... "
                                f"!= expected {ref['sha256'][:16]}...")
    if set(splits) == set(SPLITS) and sorted(seen) != list(range(N_GROUP)):
        problems.append("the three splits do not cover examples 0..14999 "
                        "exactly once")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default=None,
                    help="pglib case name in the bucket, or its short alias (case57, ...)")
    ap.add_argument("--all", action="store_true", help=f"fetch all of {CASES}")
    ap.add_argument("--variant", choices=["fulltop", "n1", "both"], default="fulltop")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output root (default: opfdata_hf/ under the data "
                         "home, i.e. $POWERFORM_DATA or <repo>/datasets)")
    ap.add_argument("--verify-only", action="store_true",
                    help="check the local copy; never touches the network")
    ap.add_argument("--deep", action="store_true",
                    help="with --verify-only: stream every shard and check "
                         "lines, bytes and content sha256")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be fetched and where, then stop")
    args = ap.parse_args()

    out_root = Path(args.out).resolve() if args.out else DEFAULT_OUT
    cases = CASES if args.all else [args.case]
    if cases == [None]:
        raise SystemExit("pass --case <name> or --all")
    cases = [resolve_case(c) for c in cases]
    variants = ["fulltop", "n1"] if args.variant == "both" else [args.variant]
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    for s in splits:
        if s not in SPLITS:
            raise SystemExit(f"unknown split {s!r}; choose from {SPLITS}")

    if args.verify_only:
        expected = load_checksums()
        bad = 0
        for case in cases:
            for v in variants:
                problems = verify(case, v, out_root, splits, deep=args.deep,
                                  expected=expected, group=args.group)
                bad += bool(problems)
                print(f"[{'OK' if not problems else 'FAIL'}] {case}/{v}")
                for p in problems:
                    print(f"       {p}")
        raise SystemExit(1 if bad else 0)

    for case in cases:
        for v in variants:
            if args.dry_run:
                state = ("complete" if is_complete(out_root, case, v, splits)
                         else "to fetch")
                print(f"{source_url(case, v, args.group)} -> "
                      f"{out_root / case / v} ({state})")
                continue
            try:
                fetch(case, v, out_root, group=args.group, splits=splits)
            except urllib.error.HTTPError as e:
                raise SystemExit(f"could not fetch {source_url(case, v, args.group)}: {e}; "
                                 f"this tool fetches {', '.join(CASES)}, and "
                                 "fetch_test_split the zero-shot grids") from e


if __name__ == "__main__":
    main()
