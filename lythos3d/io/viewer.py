"""A 3D viewer for results in the browser, and an HTML report around it.

:func:`write_viewer` writes one self-contained HTML file: the mesh and every
stage's results are embedded in it (as base64 float32), and three.js is
loaded from a CDN.  It shows the ground as it stands at each stage,
deformed by a chosen factor and coloured by a chosen field, with the
plates on top, and a cut plane that clips the ground and shows the field on
the section.  :func:`write_report` puts the same viewer in a report with the
stage table and the charts: factor of safety search, consolidation, wall
moments and pile forces.

Only corner nodes are sent: the section and the surface are drawn with the
linear sub-tetrahedra, which is what a contour plot resolves anyway.
"""

from __future__ import annotations

import base64
import html
import json
import os

import numpy as np



def _b64(a, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _nodal(problem, values: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Gauss point values (n_points, k) smoothed onto the nodes over the active elements."""
    ce, mesh = problem.continuum, problem.mesh
    ngp = ce.n_gauss
    values = values.reshape(problem.n_points, -1)
    out = ce.nodal_average(np.where(np.repeat(active, ngp)[:, None], values, 0.0), mesh.n_nodes)
    count = np.zeros(mesh.n_nodes)
    np.add.at(count, mesh.elements[active].ravel(), 1.0)
    total = np.zeros(mesh.n_nodes)
    np.add.at(total, mesh.elements.ravel(), 1.0)
    return out * (total / np.maximum(count, 1.0))[:, None]


def stage_fields(problem, result) -> dict[str, np.ndarray]:
    """Nodal fields worth looking at, by name with units."""
    active = np.asarray(result.active, bool)
    u = result.displacement
    s = _nodal(problem, result.state.stress, active)
    mean = -(s[:, 0] + s[:, 1] + s[:, 2]) / 3.0
    dev = s.copy()
    dev[:, :3] += mean[:, None]
    q = np.sqrt(1.5 * ((dev[:, :3] ** 2).sum(axis=1) + 2.0 * (dev[:, 3:] ** 2).sum(axis=1)))
    fields = {
        "displacement |u| (mm)": 1000.0 * np.linalg.norm(u, axis=1),
        "vertical displacement (mm)": 1000.0 * u[:, 2],
        "horizontal displacement x (mm)": 1000.0 * u[:, 0],
        "horizontal displacement y (mm)": 1000.0 * u[:, 1],
        "vertical effective stress (kPa, compression +)": -s[:, 2],
        "mean effective stress p' (kPa)": mean,
        "deviatoric stress q (kPa)": q,
        "plastic strain": _nodal(problem, result.state.eps_p_eq, active)[:, 0],
    }
    if result.pore_pressure is not None and np.any(result.pore_pressure):
        fields["pore pressure (kPa)"] = _nodal(problem, result.pore_pressure, active)[:, 0]
    if result.excess_pore_pressure is not None and np.any(result.excess_pore_pressure):
        fields["excess pore pressure (kPa)"] = _nodal(problem, result.excess_pore_pressure, active)[:, 0]
    if result.head is not None:
        fields["total head (m)"] = np.nan_to_num(result.head)
    return fields


def viewer_data(problem, results, title: str = "Lythos 3D") -> dict:
    """Everything the viewer draws, as a JSON-able dict."""
    mesh = problem.mesh
    corners = np.unique(mesh.elements[:, :4])
    index = np.full(mesh.n_nodes, -1, dtype=np.int64)
    index[corners] = np.arange(len(corners))
    tets = index[mesh.elements[:, :4]]
    # soil nodes split along a wall are one point: faces either side of it
    # are not the ground's surface
    rep = np.arange(mesh.n_nodes)
    for orig, back in getattr(problem, "split_nodes", {}).values():
        rep[back] = orig
    twin = index[rep[corners]]
    twin[twin < 0] = np.arange(len(corners))[twin < 0]
    plates = []
    for k, plate in enumerate(problem.plates):
        faces = np.asarray(plate.faces)[:, :3]
        nodes, tris = np.unique(faces, return_inverse=True)
        plates.append({"name": plate.name, "nodes": nodes, "tris": tris.reshape(-1, 3)})
    stages = []
    for r in results:
        fields = stage_fields(problem, r)
        plate_data = {}
        for k, pl in enumerate(plates):
            if pl["name"] not in r.plate_forces:
                continue
            el = problem.plate_elements[k]
            _, M, _ = r.plate_forces[pl["name"]]
            Mm = M.mean(axis=1)
            vertical_is_x = np.abs(el.R[:, 0, 2]) >= np.abs(el.R[:, 1, 2])
            Mv = np.where(vertical_is_x, Mm[:, 0], Mm[:, 1])
            plate_data[pl["name"]] = {"disp": _b64(r.displacement[pl["nodes"]], np.float32),
                                      "moment": _b64(Mv, np.float32)}
        stages.append({
            "name": r.name, "kind": r.kind, "converged": bool(r.converged),
            "active": _b64(np.asarray(r.active, bool), np.uint8),
            "disp": _b64(r.displacement[corners], np.float32),
            "fields": {name: _b64(v[corners], np.float32) for name, v in fields.items()},
            "plates": plate_data,
            "fos": r.srf,
        })
    return {
        "title": title,
        "nodes": _b64(mesh.nodes[corners], np.float32),
        "tets": _b64(tets, np.uint32),
        "twin": _b64(twin, np.uint32),
        "n_nodes": int(len(corners)),
        "n_tets": int(mesh.n_elements),
        "plates": [{"name": p["name"], "coords": _b64(mesh.nodes[p["nodes"]], np.float32),
                    "tris": _b64(p["tris"], np.uint32), "n": int(len(p["tris"]))} for p in plates],
        "stages": stages,
    }


_VIEWER_CSS = """
.l3v { position: relative; height: 72vh; min-height: 420px; border: 1px solid var(--line);
       border-radius: 8px; overflow: hidden; background: var(--canvas); }
.l3v canvas { display: block; width: 100%; height: 100%; }
.l3v-panel { position: absolute; top: 10px; left: 10px; background: var(--panel); color: var(--ink);
             border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; font: 13px system-ui, sans-serif;
             display: grid; grid-template-columns: auto 1fr; gap: 6px 10px; align-items: center;
             max-width: min(360px, calc(100% - 20px)); }
.l3v-panel select, .l3v-panel input[type=range] { width: 100%; min-width: 0; }
.l3v-panel label { color: var(--muted); }
.l3v-legend { position: absolute; bottom: 12px; left: 12px; right: 12px; max-width: 420px; background: var(--panel);
              border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font: 12px system-ui, sans-serif; color: var(--ink); }
.l3v-bar { height: 12px; border-radius: 3px; margin: 4px 0; background: linear-gradient(90deg,#30123b,#4662d7,#36aaf9,#1ae4b6,#72fe5e,#c8ef34,#faba39,#f66b19,#ca2a04,#7a0403); }
.l3v-ends { display: flex; justify-content: space-between; font-variant-numeric: tabular-nums; }
"""

_VIEWER_JS = r"""
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const TURBO = [[48,18,59],[70,98,215],[54,170,249],[26,228,182],[114,254,94],[200,239,52],[250,186,57],[246,107,25],[202,42,4],[122,4,3]];
function turbo(t) {
  t = Math.min(1, Math.max(0, t)) * (TURBO.length - 1);
  const i = Math.min(TURBO.length - 2, Math.floor(t)), f = t - i, a = TURBO[i], b = TURBO[i + 1];
  return [(a[0] + f * (b[0] - a[0])) / 255, (a[1] + f * (b[1] - a[1])) / 255, (a[2] + f * (b[2] - a[2])) / 255];
}
function decode(b64, Type) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}
const FACES = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]];

export function mount(root, data) {
  const X = decode(data.nodes, Float32Array), T = decode(data.tets, Uint32Array), W = decode(data.twin, Uint32Array);
  const nn = data.n_nodes, nt = data.n_tets;
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nn; i++) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], X[3*i+k]); hi[k] = Math.max(hi[k], X[3*i+k]); }
  const size = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const centre = new THREE.Vector3((lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2);
  const stages = data.stages.map(s => ({...s, act: decode(s.active, Uint8Array), U: decode(s.disp, Float32Array),
    F: Object.fromEntries(Object.entries(s.fields).map(([k, v]) => [k, decode(v, Float32Array)]))}));
  const plates = data.plates.map(p => ({...p, X: decode(p.coords, Float32Array), T: decode(p.tris, Uint32Array)}));

  const renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setClearColor(new THREE.Color(getComputedStyle(root).backgroundColor || '#eef1f5'));
  renderer.localClippingEnabled = true;
  renderer.setPixelRatio(window.devicePixelRatio);
  root.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, size / 100, size * 50);
  camera.up.set(0, 0, 1);
  camera.position.copy(centre).add(new THREE.Vector3(-1.2 * size, -1.6 * size, 1.0 * size));
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(centre);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(-1, -2, 3); scene.add(sun);

  const panel = document.createElement('div'); panel.className = 'l3v-panel'; root.appendChild(panel);
  const legend = document.createElement('div'); legend.className = 'l3v-legend'; root.appendChild(legend);
  function control(label, el) { const l = document.createElement('label'); l.textContent = label; panel.append(l, el); return el; }
  function select(options) { const s = document.createElement('select'); options.forEach((o, i) => s.add(new Option(o, i))); return s; }
  const stageSel = control('Stage', select(stages.map(s => s.name + (s.fos ? ` (FoS ${s.fos.toFixed(3)})` : ''))));
  stageSel.value = stages.length - 1;
  const fieldSel = control('Field', select([]));
  const scale = control('Deformation', Object.assign(document.createElement('input'), {type: 'range', min: 0, max: 1, step: 0.01, value: 0}));
  const axisSel = control('Cut', select(['none', 'x', 'y', 'z']));
  const cutPos = control('Cut at', Object.assign(document.createElement('input'), {type: 'range', min: 0, max: 1, step: 0.005, value: 0.5}));
  const edges = control('Edges', Object.assign(document.createElement('input'), {type: 'checkbox', checked: true}));

  let meshObj = null, lineObj = null, cutObj = null, plateObjs = [];
  const clip = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);
  const material = new THREE.MeshLambertMaterial({vertexColors: true, side: THREE.DoubleSide, clippingPlanes: []});
  const lineMat = new THREE.LineBasicMaterial({color: 0x1b1f24, transparent: true, opacity: 0.25, clippingPlanes: []});
  const cutMat = new THREE.MeshBasicMaterial({vertexColors: true, side: THREE.DoubleSide});
  const plateMat = new THREE.MeshLambertMaterial({vertexColors: true, side: THREE.DoubleSide, clippingPlanes: []});

  function fieldNames(st) { return Object.keys(st.F); }
  function refreshFields() {
    const st = stages[stageSel.value], keep = fieldSel.selectedOptions[0]?.text;
    fieldSel.innerHTML = ''; fieldNames(st).forEach((n, i) => fieldSel.add(new Option(n, n)));
    if (keep && fieldNames(st).includes(keep)) fieldSel.value = keep;
  }
  function dispScale(st) {
    let m = 0; for (let i = 0; i < nn; i++) m = Math.max(m, Math.hypot(st.U[3*i], st.U[3*i+1], st.U[3*i+2]));
    return m > 0 ? 0.08 * size / m : 0;
  }
  function draw() {
    const st = stages[stageSel.value], F = st.F[fieldSel.value], k = scale.value * dispScale(st);
    const used = new Uint8Array(nn);
    for (let e = 0; e < nt; e++) if (st.act[e]) for (let a = 0; a < 4; a++) used[T[4*e+a]] = 1;
    let fmin = Infinity, fmax = -Infinity;
    for (let i = 0; i < nn; i++) if (used[i]) { fmin = Math.min(fmin, F[i]); fmax = Math.max(fmax, F[i]); }
    if (!(fmax > fmin)) { fmax = fmin + 1; }
    const col = v => turbo((v - fmin) / (fmax - fmin));
    const P = i => [X[3*i] + k * st.U[3*i], X[3*i+1] + k * st.U[3*i+1], X[3*i+2] + k * st.U[3*i+2]];
    // free faces of the active tetrahedra
    const count = new Map();
    for (let e = 0; e < nt; e++) if (st.act[e]) for (const f of FACES) {
      const v = [T[4*e+f[0]], T[4*e+f[1]], T[4*e+f[2]]], key = v.map(i => W[i]).sort((a, b) => a - b).join(',');
      const c = count.get(key); if (c) c.n++; else count.set(key, {n: 1, v});
    }
    const pos = [], colors = [], lines = [];
    for (const {n, v} of count.values()) if (n === 1) {
      for (const i of v) { pos.push(...P(i)); colors.push(...col(F[i])); }
      for (let a = 0; a < 3; a++) lines.push(...P(v[a]), ...P(v[(a + 1) % 3]));
    }
    for (const o of [meshObj, lineObj, cutObj, ...plateObjs]) if (o) { scene.remove(o); o.geometry.dispose(); }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    g.computeVertexNormals();
    meshObj = new THREE.Mesh(g, material); scene.add(meshObj);
    const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
    lineObj = new THREE.LineSegments(lg, lineMat); lineObj.visible = edges.checked; scene.add(lineObj);
    // plates, coloured by their vertical bending moment
    plateObjs = [];
    for (const pl of plates) {
      const pd = st.plates[pl.name]; if (!pd) continue;
      const U = decode(pd.disp, Float32Array), M = decode(pd.moment, Float32Array);
      let mm = 1e-9; for (const m of M) mm = Math.max(mm, Math.abs(m));
      const pp = [], pc = [];
      for (let t = 0; t < pl.n; t++) {
        const c = turbo(0.5 + 0.5 * M[t] / mm);
        for (let a = 0; a < 3; a++) { const i = pl.T[3*t+a]; pp.push(pl.X[3*i] + k*U[3*i], pl.X[3*i+1] + k*U[3*i+1], pl.X[3*i+2] + k*U[3*i+2]); pc.push(...c); }
      }
      const pg = new THREE.BufferGeometry();
      pg.setAttribute('position', new THREE.Float32BufferAttribute(pp, 3));
      pg.setAttribute('color', new THREE.Float32BufferAttribute(pc, 3));
      pg.computeVertexNormals();
      const o = new THREE.Mesh(pg, plateMat); scene.add(o); plateObjs.push(o);
    }
    // the cut: clip the surface, and draw the field on the section
    const ax = axisSel.value - 1;
    const planes = ax >= 0 ? [clip] : [];
    material.clippingPlanes = lineMat.clippingPlanes = plateMat.clippingPlanes = planes;
    cutPos.disabled = ax < 0;
    if (ax >= 0) {
      const at = lo[ax] + cutPos.value * (hi[ax] - lo[ax]);
      // keep the half away from the camera, so the section faces it
      const eye = camera.position.getComponent(ax) < centre.getComponent(ax) ? 1 : -1;
      clip.normal.set(ax === 0 ? eye : 0, ax === 1 ? eye : 0, ax === 2 ? eye : 0); clip.constant = -eye * at;
      const cp = [], cc = [];
      for (let e = 0; e < nt; e++) if (st.act[e]) {
        const vs = [T[4*e], T[4*e+1], T[4*e+2], T[4*e+3]], d = vs.map(i => X[3*i+ax] - at);
        const hits = [];
        for (let a = 0; a < 4; a++) for (let b = a + 1; b < 4; b++) if ((d[a] < 0) !== (d[b] < 0)) {
          const s = d[a] / (d[a] - d[b]), pa = P(vs[a]), pb = P(vs[b]);
          hits.push({p: [pa[0] + s*(pb[0]-pa[0]), pa[1] + s*(pb[1]-pa[1]), pa[2] + s*(pb[2]-pa[2])], v: F[vs[a]] + s*(F[vs[b]] - F[vs[a]])});
        }
        if (hits.length < 3) continue;
        const tris = hits.length === 3 ? [[0, 1, 2]] : [[0, 1, 2], [1, 2, 3]];
        if (hits.length === 4) { // order the quadrilateral
          const [h0, h1, h2, h3] = hits; const c = [0, 1, 2].map(q => (h0.p[q]+h1.p[q]+h2.p[q]+h3.p[q]) / 4);
          const u = (ax + 1) % 3, w = (ax + 2) % 3;
          hits.sort((A, B) => Math.atan2(A.p[w]-c[w], A.p[u]-c[u]) - Math.atan2(B.p[w]-c[w], B.p[u]-c[u]));
          tris[1] = [0, 2, 3];
        }
        for (const t of tris) for (const q of t) { cp.push(...hits[q].p); cc.push(...col(hits[q].v)); }
      }
      const cg = new THREE.BufferGeometry();
      cg.setAttribute('position', new THREE.Float32BufferAttribute(cp, 3));
      cg.setAttribute('color', new THREE.Float32BufferAttribute(cc, 3));
      cutObj = new THREE.Mesh(cg, cutMat); scene.add(cutObj);
    } else cutObj = null;
    const fmt = v => Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(2) : v.toFixed(2);
    legend.innerHTML = `<div>${fieldSel.value}${st.converged ? '' : ' - <b>did not converge</b>'}</div><div class="l3v-bar"></div>` +
      `<div class="l3v-ends"><span>${fmt(fmin)}</span><span>${fmt((fmin + fmax) / 2)}</span><span>${fmt(fmax)}</span></div>` +
      (plateObjs.length ? '<div style="margin-top:4px;color:var(--muted)">plates: vertical bending moment, blue to red about zero</div>' : '');
  }
  stageSel.onchange = () => { refreshFields(); draw(); };
  [fieldSel, axisSel, edges].forEach(el => el.onchange = draw);
  controls.addEventListener('end', () => { if (axisSel.value > 0) draw(); });
  [scale, cutPos].forEach(el => el.oninput = draw);
  refreshFields(); draw();
  function resize() { const w = root.clientWidth, h = root.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
  new ResizeObserver(resize).observe(root); resize();
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
}
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg: #f6f7f9; --ink: #1b1f24; --muted: #5b6470; --line: #d5dae1; --panel: rgba(255,255,255,0.92);
         --canvas: #eef1f5; --accent: #2563eb; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #111418; --ink: #e6e9ee; --muted: #9aa4b1; --line: #2b323b; --panel: rgba(24,28,34,0.92);
           --canvas: #1a1f26; --accent: #60a5fa; }}
}}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }}
main {{ max-width: 1200px; margin: 0 auto; padding: 16px; }}
h1 {{ font-size: 1.5rem; margin: 0.5rem 0 0.25rem; }}
h2 {{ font-size: 1.15rem; margin: 1.8rem 0 0.6rem; }}
.muted {{ color: var(--muted); }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ color: var(--muted); font-weight: 600; }}
.scroll {{ overflow-x: auto; }}
.charts {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr)); gap: 16px; }}
figure svg {{ max-height: 300px; }}
figure {{ margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
figcaption {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
svg text {{ fill: var(--muted); font: 11px system-ui, sans-serif; }}
.bad {{ color: #dc2626; font-weight: 600; }}
{css}
</style>
<script type="importmap">{importmap}</script>
</head>
<body>
<main>
{body}
</main>
<script type="application/json" id="l3v-data">{data}</script>
<script type="module">
{js}
mount(document.getElementById('l3v'), JSON.parse(document.getElementById('l3v-data').textContent));
</script>
</body>
</html>
"""


_VENDOR = os.path.join(os.path.dirname(__file__), "vendor")
_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/"


def _importmap(offline: bool = True) -> str:
    """three.js from the copy shipped with Lythos 3D, inlined, so a report opens anywhere offline;
    or from the CDN, for a smaller file."""
    three = os.path.join(_VENDOR, "three.module.min.js")
    orbit = os.path.join(_VENDOR, "OrbitControls.js")
    if offline and os.path.exists(three) and os.path.exists(orbit):
        def url(path):
            with open(path, "rb") as fh:
                return "data:text/javascript;base64," + base64.b64encode(fh.read()).decode("ascii")
        imports = {"three": url(three), "three/addons/controls/OrbitControls.js": url(orbit)}
    else:
        imports = {"three": _CDN + "build/three.module.js", "three/addons/": _CDN + "examples/jsm/"}
    return json.dumps({"imports": imports})


def _page(title: str, body: str, data: dict, offline: bool = True) -> str:
    js = _VIEWER_JS.replace("export function mount", "function mount")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return _PAGE.format(title=html.escape(title), css=_VIEWER_CSS, body=body, data=payload, js=js,
                        importmap=_importmap(offline))


def write_viewer(path: str | os.PathLike, problem, results, title: str = "Lythos 3D", offline: bool = True) -> str:
    """A self-contained HTML page showing every stage in 3D (three.js inlined unless ``offline=False``)."""
    body = (f"<h1>{html.escape(title)}</h1><p class='muted'>Drag to orbit, scroll to zoom, right-drag to pan.</p>"
            "<div class='l3v' id='l3v'></div>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_page(title, body, viewer_data(problem, results, title), offline))
    return str(path)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _svg_chart(series, xlabel: str, ylabel: str, width=440, height=260, invert_y=False, markers=False) -> str:
    """A small line chart: ``series`` is a list of (label, xs, ys)."""
    xs = np.concatenate([np.asarray(s[1], float) for s in series]) if series else np.zeros(1)
    ys = np.concatenate([np.asarray(s[2], float) for s in series]) if series else np.zeros(1)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    if x1 == x0:
        x1 = x0 + 1.0
    if y1 == y0:
        y1 = y0 + 1.0
    L, R, T, B = 56, 12, 12, 40
    W, H = width - L - R, height - T - B

    def px(x):
        return L + (x - x0) / (x1 - x0) * W

    def py(y):
        f = (y - y0) / (y1 - y0)
        return T + (f if invert_y else 1 - f) * H

    colours = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">',
             f'<rect x="{L}" y="{T}" width="{W}" height="{H}" fill="none" stroke="currentColor" stroke-opacity="0.25"/>']
    for k in range(5):
        xv = x0 + k * (x1 - x0) / 4
        yv = y0 + k * (y1 - y0) / 4
        parts.append(f'<text x="{px(xv):.1f}" y="{T + H + 14}" text-anchor="middle">{xv:.3g}</text>')
        parts.append(f'<text x="{L - 6}" y="{py(yv) + 4:.1f}" text-anchor="end">{yv:.3g}</text>')
    parts.append(f'<text x="{L + W / 2}" y="{height - 6}" text-anchor="middle">{html.escape(xlabel)}</text>')
    parts.append(f'<text x="12" y="{T + H / 2}" text-anchor="middle" transform="rotate(-90 12 {T + H / 2})">'
                 f'{html.escape(ylabel)}</text>')
    for i, (label, sx, sy) in enumerate(series):
        c = colours[i % len(colours)]
        pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(sx, sy))
        if markers:
            parts += [f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="2.5" fill="{c}"/>' for a, b in zip(sx, sy)]
        else:
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"/>')
        parts.append(f'<text x="{L + 8}" y="{T + 14 + 13 * i}" style="fill:{c}">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _fmt(v, digits=3):
    if v is None:
        return "-"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"
    return html.escape(str(v))


def write_report(path: str | os.PathLike, problem, results, title: str = "Lythos 3D analysis",
                 materials_table: bool = True, offline: bool = True) -> str:
    """An HTML report: model, stages, charts and the 3D viewer, in one file."""
    mesh = problem.mesh
    rows = []
    for r in results:
        forces = [f"{k}: {v:.0f} kN" for k, v in r.bar_forces.items()]
        moments = [f"{k}: {np.abs(M).max():.1f} kNm/m" for k, (_, M, _) in r.plate_forces.items()]
        extra = []
        if r.flows:
            extra.append(f"inflow {r.flows.get('in', 0):.3g}, pumped {r.flows.get('pumped', 0):.3g}")
        if r.consolidation:
            extra.append(f"t = {r.time:.3g}, excess left {r.consolidation[-1][1]:.1f} kPa")
        status = "ok" if r.converged else f"<span class='bad'>failed</span> {html.escape(r.message)}"
        rows.append(f"<tr><td>{html.escape(r.name)}</td><td>{html.escape(r.kind)}</td><td>{status}</td>"
                    f"<td>{1000 * r.max_displacement:.2f}</td><td>{_fmt(r.srf)}</td>"
                    f"<td>{100 * r.plastic_fraction:.1f}%</td><td>{'<br>'.join(moments + forces + extra) or '-'}</td>"
                    f"<td>{r.seconds:.0f}</td></tr>")
    body = [f"<h1>{html.escape(title)}</h1>",
            f"<p class='muted'>{mesh.n_elements} quadratic tetrahedra, {mesh.n_nodes} nodes, "
            f"{problem.n_dof} equations, {len(results)} stages.</p>"]
    if materials_table:
        mrows = []
        for r, m in problem.materials.items():
            params = {k: getattr(m, k) for k in ("E", "nu", "gamma", "gamma_sat", "c", "phi", "psi", "k", "drainage")
                      if hasattr(m, k) and getattr(m, k) is not None}
            mrows.append(f"<tr><td>{html.escape(m.name)}</td><td>{type(m).__name__}</td><td>"
                         + ", ".join(f"{k} = {_fmt(v)}" for k, v in params.items()) + "</td></tr>")
        body.append("<h2>Materials</h2><div class='scroll'><table><tr><th>Name</th><th>Model</th><th>Parameters</th></tr>"
                    + "".join(mrows) + "</table></div>")
    body.append("<h2>Stages</h2><div class='scroll'><table><tr><th>Stage</th><th>Kind</th><th>Result</th>"
                "<th>Max displacement (mm)</th><th>Factor of safety</th><th>Plastic points</th>"
                "<th>Structures and water</th><th>s</th></tr>" + "".join(rows) + "</table></div>")

    charts = []
    for r in results:
        if r.srf_curve:
            f, d = zip(*r.srf_curve)
            charts.append((f"Strength reduction, {r.name}: factor of safety {_fmt(r.srf)}",
                           _svg_chart([("trials", f, 1000 * np.asarray(d))], "reduction factor",
                                      "max displacement (mm)", markers=True)))
    cons = [r for r in results if r.consolidation]
    if cons:
        series = []
        t = np.concatenate([[c[0] for c in r.consolidation] for r in cons])
        p = np.concatenate([[c[1] for c in r.consolidation] for r in cons])
        series.append(("largest excess pore pressure (kPa)", t, p))
        charts.append(("Consolidation", _svg_chart(series, "time", "kPa")))
        d = np.concatenate([[1000 * c[2] for c in r.consolidation] for r in cons])
        charts.append(("Consolidation: displacement", _svg_chart([("largest displacement (mm)", t, d)],
                                                                 "time", "mm")))
    last = results[-1] if results else None
    for r in reversed(results):
        if r.plate_forces and r.kind != "ssr":
            last = r
            break
    if last is not None:
        for k, plate in enumerate(problem.plates):
            if plate.name not in last.plate_forces:
                continue
            el = problem.plate_elements[k]
            _, M, _ = last.plate_forces[plate.name]
            Mm = M.mean(axis=1)
            vertical_is_x = np.abs(el.R[:, 0, 2]) >= np.abs(el.R[:, 1, 2])
            Mv = np.where(vertical_is_x, Mm[:, 0], Mm[:, 1])
            z = mesh.nodes[np.asarray(plate.faces)[:, :3]].mean(axis=1)[:, 2]
            # the envelope over the wall's length, level by level
            edges = np.linspace(z.min(), z.max(), 25)
            band = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
            mids, lows, highs = [], [], []
            for b in range(len(edges) - 1):
                if np.any(band == b):
                    mids.append(0.5 * (edges[b] + edges[b + 1]))
                    lows.append(Mv[band == b].min())
                    highs.append(Mv[band == b].max())
            charts.append((f"{plate.name}: vertical bending moment against level (envelope along the wall), "
                           f"{last.name}",
                           _svg_chart([("least", lows, mids), ("greatest", highs, mids)],
                                      "moment (kNm/m)", "level (m)")))
        for name, pf in last.pile_forces.items():
            N = -pf["resultants"][:, 0]
            charts.append((f"{name}: axial force along the pile, {last.name}",
                           _svg_chart([("compression (kN)", N, pf["s"])], "kN", "distance from head (m)",
                                      invert_y=True)))
    if charts:
        body.append("<h2>Charts</h2><div class='charts'>" + "".join(
            f"<figure>{svg}<figcaption>{html.escape(cap)}</figcaption></figure>" for cap, svg in charts) + "</div>")
    body.append("<h2>3D results</h2><p class='muted'>Choose a stage and a field; drag to orbit, scroll to zoom, "
                "right-drag to pan. The cut clips the ground and shows the field on the section.</p>"
                "<div class='l3v' id='l3v'></div>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_page(title, "\n".join(body), viewer_data(problem, results, title), offline))
    return str(path)
