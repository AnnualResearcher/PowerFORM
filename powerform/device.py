"""The device a run uses, chosen at run time from `--device`."""
from __future__ import annotations

import os

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """`auto` takes CUDA when present and otherwise CPU, printing a note, and
    `FORCE_CPU=1` makes it choose CPU. An explicit CUDA request on a machine
    without CUDA is refused."""
    if requested == "auto":
        if os.environ.get("FORCE_CPU") == "1":
            print("device: cpu (FORCE_CPU=1)", flush=True)
            return torch.device("cpu")
        if not torch.cuda.is_available():
            print("device: cpu (no CUDA device available)", flush=True)
            return torch.device("cpu")
        return torch.device("cuda")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"--device {requested} requested but CUDA is not "
                         "available")
    return device
