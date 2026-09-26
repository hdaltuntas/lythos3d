# SPDX-License-Identifier: AGPL-3.0-only
"""The interface in a browser: the plan editor, served locally, with analyses run behind it.

``python main.py`` (or ``lythos3d gui``) starts a small HTTP server on the
loopback address and opens the plan editor in the browser, as 2D Lythos
does.  Served this way the editor gains what a file cannot do on its own:
the example sites, a button that meshes and analyses the site drawn, the
progress of the analysis stage by stage, and the report when it is done.

Only the standard library is used for the server.  One analysis runs at a
time, in a worker thread, so the page stays responsive through a strength
reduction search.  Each analysis writes its site file, report and ParaView
files to a folder of its own under ``~/lythos3d_runs`` (or ``--out``).
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

EDITOR = os.path.join(os.path.dirname(__file__), "io", "editor.html")

_FAVICON = (b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
            b'<rect width="32" height="32" rx="6" fill="#1565c0"/>'
            b'<path d="M6 12l10-5 10 5v12l-10 5-10-5z" fill="#f2e394"/>'
            b'<path d="M6 12l10 5 10-5M16 17v12" stroke="#1565c0" stroke-width="1.5" fill="none"/></svg>')


def examples() -> dict:
    """Example sites the interface offers, by name: (description, function)."""
    from . import examples as ex

    return {
        "walled pit": ("A pit on dipping ground inside a diaphragm wall with a strut", lambda: ex.walled_pit()),
        "dewatered pit": ("The same below the water table, pumped dry as it is dug", lambda: ex.dewatered_pit()),
        "seepage": ("The same with the flow under the wall into the pit solved",
                    lambda: ex.dewatered_pit(seepage=True)),
        "open pit": ("An unsupported pit on dipping ground", lambda: ex.sloping_site()),
    }


class Cancelled(Exception):
    """Raised inside the solver's monitor when the page asks to stop."""


class Session:
    """The one analysis this server knows about."""

    def __init__(self, out_root: str):
        self.out_root = out_root
        self.lock = threading.Lock()
        self.status = "idle"            # idle | running | done | failed
        self.log: list[str] = []
        self.progress = (0, 0, "")
        self.report = None
        self.folder = None
        self.error = ""
        self.started = 0.0
        self.detail = ""
        self.cancel = False

    def state(self) -> dict:
        with self.lock:
            return {"status": self.status, "log": self.log[-200:], "progress": list(self.progress),
                    "detail": self.detail,
                    "report": self.report is not None, "folder": self.folder, "error": self.error,
                    "seconds": round(time.time() - self.started, 1) if self.started else 0.0}

    def say(self, line: str) -> None:
        with self.lock:
            self.log.append(line)

    def start(self, site_dict: dict, fos, fineness: float = 1.0) -> tuple[bool, str]:
        with self.lock:
            if self.status == "running":
                return False, "an analysis is already running"
            self.status, self.log, self.report, self.error = "running", [], None, ""
            self.progress, self.started, self.folder = (0, 0, "checking the site"), time.time(), None
            self.detail, self.cancel = "", False
        threading.Thread(target=self._run, args=(site_dict, fos, fineness), daemon=True).start()
        return True, ""

    def stop(self) -> None:
        with self.lock:
            self.cancel = True

    def _monitor(self, line: str) -> None:
        with self.lock:
            self.detail = line
            if self.cancel:
                raise Cancelled()

    def _run(self, site_dict: dict, fos, fineness: float = 1.0) -> None:
        """``fos`` is False, or the width to bracket the factor of safety to (True: 0.01)."""
        try:
            import numpy as np

            from .core.solver import Solver
            from .io.site_json import save_site, site_from_dict
            from .io.viewer import write_report
            from .io.vtu import write_plates, write_stage

            from .core.assembly import LinearSolver

            site = site_from_dict(site_dict)
            # a coarser or finer mesh than the site asks for, from the interface
            site.mesh_size *= fineness
            for exc in site.excavations:
                if exc.mesh_size:
                    exc.mesh_size *= fineness
            stages = [s for s in site.stages if fos or s.kind != "ssr"]
            backend = LinearSolver("auto").backend
            if backend == "pardiso":
                self.say("linear solver: PARDISO")
            else:
                self.say("WARNING: pypardiso is not installed, so the slow SuperLU solver is used; "
                         "a site of useful size may take hours.  Install it: pip install pypardiso")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in site.name)[:40] or "site"
            folder = os.path.join(self.out_root, f"{stamp}-{safe}")
            os.makedirs(folder, exist_ok=True)
            save_site(site, os.path.join(folder, "site.json"))
            with self.lock:
                self.folder = folder
            for message in site.drawdown_warnings():
                self.say("WARNING: " + message)
            self.say(f"meshing {site.name}")
            self.progress = (0, len(stages), "meshing")
            problem = site.build()
            self.say(f"{problem.mesh.n_elements} elements, {problem.n_dof} equations "
                     f"(mesh size {site.mesh_size:g} m)")
            solver = Solver(problem)
            solver.monitor = self._monitor
            if fos and fos is not True:
                solver.ssr_bracket = float(fos)
            results = []
            for k, stage in enumerate(stages):
                with self.lock:
                    self.progress = (k, len(stages), stage.name)
                problem.check_stage_groups(stage)
                r = solver.run_stage(stage)
                results.append(r)
                write_stage(os.path.join(folder, f"stage{k}.vtu"), problem, r)
                write_plates(os.path.join(folder, f"plates{k}.vtu"), problem, r)
                line = f"{stage.name}: {'ok' if r.converged else 'FAILED'}"
                if r.srf is not None:
                    line += f", factor of safety {r.srf:.2f}"
                elif k:
                    line += f", up to {1000 * r.max_displacement:.1f} mm"
                line += f" ({r.seconds:.0f} s)"
                for name, (_, M, _) in r.plate_forces.items():
                    line += f"; {name} {float(np.abs(M).max()):.1f} kNm/m"
                self.say(line)
                if not r.converged:
                    self.say(f"  {r.message}")
                    break
            report = write_report(os.path.join(folder, "report.html"), problem, results, title=site.name)
            with self.lock:
                self.report = report
                self.progress = (len(stages), len(stages), "done")
                self.status = "done" if all(r.converged for r in results) else "failed"
                if self.status == "failed":
                    self.error = "a stage did not converge; the report shows how far it got"
        except Cancelled:
            with self.lock:
                self.status = "failed"
                self.error = "stopped"
                self.log.append("stopped at your request")
        except Exception as err:                                   # shown in the page
            with self.lock:
                self.status = "failed"
                self.error = f"{type(err).__name__}: {err}"
                self.log.append(traceback.format_exc(limit=3))


def profile_grid(site_dict: dict, n: int = 41) -> dict:
    """The top of every soil on a plan grid over the site, from its boreholes, for the 3D view.

    The same interpolation meshing uses, so what the view shows is what gets
    analysed.
    """
    import numpy as np

    from .core.site import Borehole, Soil, SoilProfile

    soils = [Soil(str(s["name"]), None) for s in site_dict.get("soils", [])]
    holes = [Borehole(str(b["name"]), float(b["x"]), float(b["y"]), [(str(a), float(z)) for a, z in b["tops"]])
             for b in site_dict.get("boreholes", [])]
    profile = SoilProfile(soils, holes, float(site_dict["bottom"]))
    (x0, x1), (y0, y1) = site_dict["extent"]["x"], site_dict["extent"]["y"]
    ny = max(2, int(round(n * (y1 - y0) / max(x1 - x0, y1 - y0))))
    nx = max(2, int(round(n * (x1 - x0) / max(x1 - x0, y1 - y0))))
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    tops = profile.tops(np.column_stack([X.ravel(), Y.ravel()]))
    return {"xs": xs.tolist(), "ys": ys.tolist(), "soils": [s.name for s in soils],
            "tops": tops.T.reshape(len(soils), ny, nx).tolist(), "bottom": profile.bottom}


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):                                # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code: int = 200) -> None:
            self._send(code, json.dumps(data).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                with open(EDITOR, "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            elif url.path == "/favicon.ico" or url.path == "/favicon.svg":
                self._send(200, _FAVICON, "image/svg+xml")
            elif url.path == "/api/status":
                self._json(session.state())
            elif url.path == "/api/solver":
                from .core.assembly import LinearSolver

                self._json({"backend": LinearSolver("auto").backend})
            elif url.path == "/api/examples":
                self._json([{"name": k, "description": v[0]} for k, v in examples().items()])
            elif url.path == "/api/example":
                from .io.site_json import site_to_dict

                name = parse_qs(url.query).get("name", [""])[0]
                found = examples().get(name)
                if found is None:
                    self._json({"error": f"no example {name!r}"}, 404)
                else:
                    self._json(site_to_dict(found[1]()))
            elif url.path.startswith("/vendor/") and "/" not in url.path[len("/vendor/"):]:
                path = os.path.join(os.path.dirname(__file__), "io", "vendor", url.path[len("/vendor/"):])
                if os.path.exists(path):
                    with open(path, "rb") as fh:
                        self._send(200, fh.read(), "text/javascript; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain")
            elif url.path == "/report":
                if session.report and os.path.exists(session.report):
                    with open(session.report, "rb") as fh:
                        self._send(200, fh.read(), "text/html; charset=utf-8")
                else:
                    self._send(404, b"no report yet", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            url = urlparse(self.path)
            if url.path == "/api/profile":
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    return self._json(profile_grid(body.get("site", {})))
                except Exception as err:                           # the page falls back on its own
                    return self._json({"error": f"{type(err).__name__}: {err}"}, 400)
            if url.path == "/api/stop":
                session.stop()
                return self._json({"ok": True})
            if url.path != "/api/run":
                return self._send(404, b"not found", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except ValueError:
                return self._json({"ok": False, "error": "the request was not JSON"}, 400)
            fos = body.get("fos", False)
            fos = float(fos) if isinstance(fos, (int, float)) and not isinstance(fos, bool) and fos > 0 else bool(fos)
            ok, why = session.start(body.get("site", {}), fos,
                                    float(body.get("fineness", 1.0)))
            self._json({"ok": ok, "error": why}, 200 if ok else 409)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8778, open_browser: bool = True, out: str | None = None) -> int:
    out = out or os.path.join(os.path.expanduser("~"), "lythos3d_runs")
    session = Session(out)
    for attempt in range(20):                  # the next free port if this one is taken
        try:
            server = ThreadingHTTPServer((host, port + attempt), make_handler(session))
            break
        except OSError:
            continue
    else:
        print(f"no free port from {port} to {port + 19}")
        return 1
    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{server.server_address[1]}/"
    print(f"Lythos 3D is running at {url}", flush=True)
    print(f"analyses are written under {out}", flush=True)
    print("press Ctrl+C to stop", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0
