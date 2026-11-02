"""Download the paper's checkpoints from the GitHub release and verify them.

Reads checkpoints.json at the repository root and fetches each requested file
into checkpoints/, checking its SHA-256 before putting it in place. A file that
is already present with the right hash is not downloaded again. Standard
library only.

    python scripts/download_checkpoints.py --list          # show the groups
    python scripts/download_checkpoints.py                 # everything
    python scripts/download_checkpoints.py --group fig1    # one group, repeatable
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "checkpoints.json"


TIMEOUT_S = 60


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, size: int) -> None:
    done = 0
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r    {done / 1e6:7.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    print()


def download(entry: dict, base_url: str, dest_dir: Path) -> str | None:
    """Put one file in place; returns an error message, or None on success."""
    dest = (dest_dir / entry["name"]).resolve()
    if not dest.is_relative_to(dest_dir):
        return f"refusing a destination outside {dest_dir}: {entry['name']}"
    if dest.exists() and sha256(dest) == entry["sha256"]:
        print(f"  ok      {entry['name']}")
        return None
    print(f"  fetch   {entry['name']}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url = base_url + entry["asset"]
    try:
        fetch(url, part, entry["size"])
    except urllib.error.HTTPError as e:
        part.unlink(missing_ok=True)
        hint = "; is the release public, and does it hold this file?" if e.code == 404 else ""
        return f"could not download {url}: {e}{hint}"
    except (OSError, http.client.HTTPException) as e:
        part.unlink(missing_ok=True)
        return f"download of {entry['name']} failed: {e!r}; run the command again"
    got = part.stat().st_size
    if got != entry["size"]:
        part.unlink()
        return (f"download of {entry['name']} stopped at {got} of {entry['size']} bytes; "
                "run the command again")
    if sha256(part) != entry["sha256"]:
        part.unlink()
        return f"HASH MISMATCH {entry['name']}: the download was discarded"
    part.replace(dest)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", action="append", default=[],
                    help="only this group; repeatable (see --list)")
    ap.add_argument("--list", action="store_true", help="list the groups and exit")
    a = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest["files"]
    if a.list:
        groups: dict[str, list] = {}
        for f in files:
            groups.setdefault(f["group"], []).append(f)
        for g, fs in groups.items():
            print(f"{g:8s} {len(fs):3d} files {sum(f['size'] for f in fs) / 1e6:8.0f} MB")
        return
    if a.group:
        unknown = set(a.group) - {f["group"] for f in files}
        if unknown:
            sys.exit(f"unknown group(s): {sorted(unknown)}; see --list")
        files = [f for f in files if f["group"] in a.group]

    dest_dir = (ROOT / manifest["dest"]).resolve()
    errors = [e for f in files if (e := download(f, manifest["base_url"], dest_dir))]
    for e in errors:
        print(e)
    if errors:
        sys.exit(f"{len(errors)} of {len(files)} file(s) failed")
    print(f"{len(files)} file(s) present and verified in {manifest['dest']}/")


if __name__ == "__main__":
    main()
