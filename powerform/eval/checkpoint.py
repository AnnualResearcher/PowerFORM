"""The one loader of PowerForm checkpoints.

A checkpoint is ``{"model": state_dict, "config": {...}}``. Type masking changes
what the network computes without changing any tensor shape, so a strict load
succeeds on a wrongly built model. The function therefore comes from the
checkpoint's own config, and a config key this module does not know is refused
rather than ignored.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import torch

from powerform.model.line_track import ReconfigurationNet
from powerform.model.powerform import PowerForm
from powerform.paths import REPO

MANIFEST = REPO / "checkpoints.json"

_MODEL_KEYS = ("d_model", "K", "n_cost_feats", "maps", "generic_hidden",
               "type_masking", "merged_readout")
_TRACK_KEY = "z_d"       # present on reconfiguration checkpoints only


def build_model(config: dict) -> torch.nn.Module:
    """An untrained model of the function ``config`` describes."""
    unknown = sorted(set(config) - set(_MODEL_KEYS) - {_TRACK_KEY})
    if unknown:
        raise ValueError(f"checkpoint config carries keys this loader does not "
                         f"know: {unknown}; refusing to build a model that may "
                         f"ignore them")
    missing = sorted(set(_MODEL_KEYS) - set(config))
    if missing:
        raise ValueError(f"checkpoint config lacks {missing}")
    trunk = PowerForm(**{k: config[k] for k in _MODEL_KEYS})
    if _TRACK_KEY in config:
        return ReconfigurationNet(trunk, z_d=config[_TRACK_KEY])
    return trunk


def config_of(model: torch.nn.Module) -> dict:
    """The config that rebuilds ``model``, read from the model itself."""
    trunk = model.trunk if isinstance(model, ReconfigurationNet) else model
    config = {"d_model": trunk.d, "K": trunk.K,
              "n_cost_feats": trunk.n_cost_feats, "maps": trunk.maps,
              "generic_hidden": trunk.generic_hidden,
              "type_masking": trunk.type_masking,
              "merged_readout": trunk.merged_readout}
    if isinstance(model, ReconfigurationNet):
        config[_TRACK_KEY] = trunk.z_stream.z_d
    return config


def _missing_message(path: Path) -> str:
    """Which file is absent and which release group holds it."""
    group = None
    if MANIFEST.exists():
        files = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
        group = next((f["group"] for f in files if Path(f["name"]).name == path.name), None)
    if group is None:
        return (f"checkpoint {path} not found, and checkpoints.json does not list "
                f"{path.name}; see python scripts/download_checkpoints.py --list")
    return (f"checkpoint {path} not found; fetch it with "
            f"python scripts/download_checkpoints.py --group {group}")


def read_checkpoint(path) -> dict:
    """The file at ``path`` as a checkpoint of this repository, or a one-line refusal."""
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError, KeyError, OSError) as e:
        raise SystemExit(f"{Path(path).name} is not a checkpoint") from e
    if not isinstance(blob, dict) or "config" not in blob:
        raise SystemExit(f"{Path(path).name} has no config: not a checkpoint of this "
                         "repository (the graphkit state row is re-scored from its "
                         "predict output directory)")
    return blob


def load_model(path, device="cpu") -> torch.nn.Module:
    """Rebuild a trained model in eval mode, in float64.

    The load is strict: a missing or extra tensor means the config and the
    weights describe different networks. Reconfiguration checkpoints store
    float32 weights, which the load widens to the model's float64.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(_missing_message(path))
    blob = read_checkpoint(path)
    model = build_model(blob["config"])
    model.load_state_dict(blob["model"])
    return model.to(device).eval()
