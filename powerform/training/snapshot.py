"""Periodic snapshots a long training run resumes from."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import torch


def save_snapshot(path: Path, payload: Dict) -> bool:
    """Write ``path`` atomically and report success.

    A failed snapshot is printed and the run continues: it protects the run, so it
    must not end it. A file-sync client can hold the target open for a moment,
    which makes the rename fail, so a refused rename is retried.
    """
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(payload, tmp)
        for attempt in range(5):
            try:
                tmp.replace(path)
                return True
            except PermissionError:
                time.sleep(0.5 * (attempt + 1))
        path.unlink(missing_ok=True)
        tmp.replace(path)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [snapshot at step {payload.get('step')} skipped: "
              f"{type(exc).__name__}: {exc}]", flush=True)
        return False
