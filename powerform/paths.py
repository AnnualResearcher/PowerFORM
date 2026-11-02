"""Where the repository keeps its files.

Tools read from RESULTS and CHECKPOINTS and write to OUTPUTS. DATA_HOME holds
the corpora: ``$POWERFORM_DATA`` when set, otherwise ``<repo>/datasets``. The
variable is read once, at import, so set it in the environment of the process.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "POWERFORM_DATA"

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
CHECKPOINTS = REPO / "checkpoints"
OUTPUTS = REPO / "outputs"


def _data_home() -> Path:
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(value).expanduser().resolve() if value else REPO / "datasets"


DATA_HOME = _data_home()


def shown(path) -> str:
    """``path`` for a log line or a record, never as an absolute path.

    Relative to the repository when it lies inside it, ``$POWERFORM_DATA/...``
    when it lies under an overridden data home, otherwise its file name.
    """
    path = Path(path).absolute()
    for root, prefix in ((REPO, ""), (DATA_HOME, f"${ENV_VAR}/")):
        try:
            return (prefix + path.relative_to(root).as_posix()).removesuffix("/.")
        except ValueError:
            pass
    return path.name
