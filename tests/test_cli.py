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


def test_the_interface_runs_an_analysis_shows_its_progress_and_stops_on_request(tmp_path):
    """Run an example through the server, watch the monitor report iterations, then stop it."""
    import time

    import pytest

    pytest.importorskip("gmsh")
    from lythos3d.gui import Session
    from lythos3d.io.site_json import site_to_dict
    from lythos3d.examples import walled_pit

    session = Session(str(tmp_path))
    ok, _ = session.start(site_to_dict(walled_pit(4.0)), 0.03, fineness=1.0)
    assert ok
    assert not session.start({}, False)[0]                       # one analysis at a time
    deadline = time.time() + 300
    while "iteration" not in session.state()["detail"] and time.time() < deadline:
        time.sleep(0.5)
    state = session.state()
    assert "iteration" in state["detail"] and "linear solver" in state["log"][0]
    session.stop()
    while session.state()["status"] == "running" and time.time() < deadline:
        time.sleep(0.5)
    assert session.state()["status"] == "failed" and session.state()["error"] == "stopped"
