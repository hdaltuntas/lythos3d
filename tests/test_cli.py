# SPDX-License-Identifier: AGPL-3.0-only
"""The command line."""

import json
import subprocess
import sys
from pathlib import Path

from lythos3d.cli import main


def test_no_command_explains_what_to_do_and_succeeds(capsys):
    """Run from an IDE with no parameters, main.py must not stop on an argparse error."""
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Give a command" in out and "run site.json" in out and "gui" in out


def test_main_py_runs_a_command():
    root = Path(__file__).resolve().parent.parent
    done = subprocess.run([sys.executable, str(root / "main.py"), "info"], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0 and "lythos3d" in done.stdout


def test_main_py_without_arguments_starts_the_interface():
    """As in 2D Lythos: no command means the interface in a browser."""
    import urllib.request

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.Popen([sys.executable, str(root / "main.py"), "--no-browser", "--port", "8791"],
                            stdout=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        assert "running at http://127.0.0.1:" in line
        url = line.split()[-1]
        page = urllib.request.urlopen(url, timeout=10).read().decode()
        assert "Lythos 3D plan" in page
        names = [e["name"] for e in json.loads(urllib.request.urlopen(url + "api/examples", timeout=10).read())]
        assert "walled pit" in names
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_info():
    assert main(["info"]) == 0
