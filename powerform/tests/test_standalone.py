"""Every module of the package imports with only the repository on sys.path.

Each import runs in its own process, from a directory outside the repository, so one
module's imports cannot mask a missing one in another.

    python powerform/tests/test_standalone.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "powerform"
PROBE = ("import importlib, sys\n"
         "sys.path[:] = [p for p in sys.path if p]\n"
         "importlib.import_module(sys.argv[1])\n")


def modules() -> list[str]:
    """Every importable module under powerform/, tests aside."""
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(REPO).with_suffix("")
        if "tests" in rel.parts:
            continue
        parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
        found.append(".".join(parts))
    return found


def imports(module: str, cwd: str, path: Path = REPO) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONPATH": str(path), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", PROBE, module], capture_output=True,
                       text=True, cwd=cwd, env=env)
    last = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
    return r.returncode == 0, last[-1] if last and r.returncode else ""


def main() -> None:
    names = modules()
    failed = []
    with tempfile.TemporaryDirectory() as cwd:
        # A module reaching for something absent must fail this probe.
        (Path(cwd) / "broken_module.py").write_text("import not_a_module_anywhere\n")
        ok, _ = imports("broken_module", cwd, Path(cwd))
        if ok:
            raise SystemExit("the import probe missed a broken module")
        (Path(cwd) / "broken_module.py").unlink()
        for module in names:
            ok, detail = imports(module, cwd)
            print(f"  [{'PASS' if ok else 'FAIL'}] {module}" + (f"  {detail}" if detail else ""),
                  flush=True)
            if not ok:
                failed.append(module)
    print(f"{len(names)} modules imported, {len(failed)} failed, 0 skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
