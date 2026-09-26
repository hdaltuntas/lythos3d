#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run Lythos 3D without installing it.

    python main.py                         start the interface in a browser
    python main.py run site.json -o run    analyse a site; open run/report.html
    python main.py site-example -o s.json  write an example site
    python main.py info                    versions and the linear solvers available
    python main.py --help                  every command

This file puts its own directory on the import path, so it works from a fresh
clone with NumPy and SciPy installed; pypardiso (fast solver) and gmsh
(meshing sites) are strongly recommended.
"""

from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

#: what to do when no command is named
DEFAULT_COMMAND = "gui"
COMMANDS = ("gui", "info", "demo", "pit", "site-example", "mesh", "editor", "dxf", "run")

REQUIRED = ("numpy", "scipy")
#: needed for real work, but not to start: what each one is for
RECOMMENDED = {
    "gmsh": "meshing sites drawn in plan - without it only box models can be analysed",
    "pypardiso": "the fast parallel solver - without it models of useful size are very slow",
}


def missing(names) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def explain(needed: list[str], recommended: list[str]) -> None:
    fish = os.path.basename(os.environ.get("SHELL", "")) == "fish"
    activate = "source .venv/bin/activate.fish" if fish else "source .venv/bin/activate"
    if needed:
        print("Lythos 3D needs these Python packages, which are not installed:\n", file=sys.stderr)
        for name in needed:
            print(f"    {name}", file=sys.stderr)
    for name in recommended:
        print(f"note: {name} is not installed: {RECOMMENDED[name]}", file=sys.stderr)
    print("\nEverything Lythos 3D uses, in a virtual environment of its own:\n", file=sys.stderr)
    print(f"    python -m venv .venv\n    {activate}\n    pip install -e \".[dev]\"\n", file=sys.stderr)
    print("gmsh on a machine without a display also needs: "
          "sudo apt install libglu1-mesa libxcursor1 libxinerama1 libxft2\n", file=sys.stderr)


def main() -> int:
    if sys.version_info < (3, 10):
        print(f"Lythos 3D needs Python 3.10 or later; this is {sys.version.split()[0]}.", file=sys.stderr)
        return 1
    needed, recommended = missing(REQUIRED), missing(RECOMMENDED)
    if needed or recommended:
        explain(needed, recommended)
        if needed:
            return 1

    from lythos3d.cli import main as run_cli

    # "python main.py" and "python main.py --port 9000" both mean: open the
    # interface.  Only an explicit command name changes that.
    argv = list(sys.argv[1:])
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv.insert(0, DEFAULT_COMMAND)
    return run_cli(argv)


if __name__ == "__main__":
    sys.exit(main())
