# SPDX-License-Identifier: AGPL-3.0-only
"""The command line."""

import subprocess
import sys
from pathlib import Path

from lythos3d.cli import main


def test_no_command_explains_what_to_do_and_succeeds(capsys):
    """Run from an IDE with no parameters, main.py must not stop on an argparse error."""
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Give a command" in out and "run site.json" in out and "PyCharm" in out


def test_main_py_runs_without_arguments():
    root = Path(__file__).resolve().parent.parent
    done = subprocess.run([sys.executable, str(root / "main.py")], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "Lythos 3D" in done.stdout


def test_info():
    assert main(["info"]) == 0
