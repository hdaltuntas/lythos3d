"""Plans from DXF drawings.

Only what a site plan needs is read: polylines (``LWPOLYLINE``, and the
older ``POLYLINE`` with its ``VERTEX`` entities), ``LINE`` and ``POINT``
entities of ASCII DXF files, by layer.  Blocks, arcs, splines and text are
ignored; draw pits, walls and loads as polylines.  Coordinates are taken as
metres in the model's plan axes.

A site file refers to a drawing wherever it wants a plan outline::

    "polygon": {"dxf": "plan.dxf", "layer": "PIT"}
    "path": {"dxf": "plan.dxf", "layer": "WALL", "index": 1}

``index`` picks one of several polylines on the layer (in the order drawn,
the first by default).  The path is relative to the site file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Polyline:
    layer: str
    points: list[tuple[float, float]]
    closed: bool = False

    def outline(self) -> list[tuple[float, float]]:
        """The points, with a closing point that repeats the first dropped."""
        pts = list(self.points)
        if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9:
            pts = pts[:-1]
        return pts


@dataclass
class Plan:
    polylines: list[Polyline] = field(default_factory=list)
    points: list[tuple[str, tuple[float, float, float]]] = field(default_factory=list)

    @property
    def layers(self) -> list[str]:
        seen = []
        for name in [p.layer for p in self.polylines] + [layer for layer, _ in self.points]:
            if name not in seen:
                seen.append(name)
        return seen

    def on(self, layer: str) -> list[Polyline]:
        return [p for p in self.polylines if p.layer.upper() == layer.upper()]

    def polyline(self, layer: str, index: int = 0) -> Polyline:
        found = self.on(layer)
        if not found:
            raise ValueError(f"the drawing has no polyline on layer {layer!r}; it has {self.layers}")
        if not 0 <= index < len(found):
            raise ValueError(f"layer {layer!r} has {len(found)} polyline(s), not an index {index}")
        return found[index]


def _pairs(text: str):
    lines = text.splitlines()
    for i in range(0, len(lines) - 1, 2):
        yield int(lines[i].strip()), lines[i + 1].strip()


def parse(text: str) -> Plan:
    """Read the polylines, lines and points of an ASCII DXF text."""
    plan = Plan()
    pairs = list(_pairs(text))
    in_entities = False
    i = 0
    n = len(pairs)

    def entity(start):
        """Group codes of the entity starting at ``start``, up to the next 0 code."""
        j = start + 1
        codes = []
        while j < n and pairs[j][0] != 0:
            codes.append(pairs[j])
            j += 1
        return codes, j

    while i < n:
        code, value = pairs[i]
        if code == 0 and value == "SECTION" and i + 1 < n and pairs[i + 1] == (2, "ENTITIES"):
            in_entities = True
            i += 2
            continue
        if code == 0 and value == "ENDSEC":
            in_entities = False
        if not in_entities or code != 0:
            i += 1
            continue
        codes, j = entity(i)
        layer = next((v for c, v in codes if c == 8), "0")
        if value == "LWPOLYLINE":
            xs = [float(v) for c, v in codes if c == 10]
            ys = [float(v) for c, v in codes if c == 20]
            flag = int(next((v for c, v in codes if c == 70), "0"))
            plan.polylines.append(Polyline(layer, list(zip(xs, ys)), bool(flag & 1)))
            i = j
        elif value == "POLYLINE":
            flag = int(next((v for c, v in codes if c == 70), "0"))
            pts = []
            i = j
            while i < n and pairs[i] == (0, "VERTEX"):
                vcodes, i = entity(i)
                pts.append((float(next(v for c, v in vcodes if c == 10)),
                            float(next(v for c, v in vcodes if c == 20))))
            if i < n and pairs[i] == (0, "SEQEND"):
                i = entity(i)[1]
            plan.polylines.append(Polyline(layer, pts, bool(flag & 1)))
        elif value == "LINE":
            get = dict(codes)
            plan.polylines.append(Polyline(layer, [(float(get[10]), float(get[20])),
                                                   (float(get[11]), float(get[21]))]))
            i = j
        elif value == "POINT":
            get = dict(codes)
            plan.points.append((layer, (float(get[10]), float(get[20]), float(get.get(30, 0.0)))))
            i = j
        else:
            i = j
    return plan


def read_plan(path: str | os.PathLike) -> Plan:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse(fh.read())


def resolve(d, base: str | os.PathLike = ".", cache: dict | None = None, key: str | None = None):
    """Replace every ``{"dxf": file, "layer": name[, "index": k]}`` in a site description by its points.

    Under a ``"path"`` key (a wall) a closed polyline comes back closed,
    its first point repeated at the end; anywhere else (a polygon) an
    outline never repeats its first point.
    """
    cache = {} if cache is None else cache
    if isinstance(d, dict):
        if "dxf" in d and "layer" in d:
            path = os.path.join(base, d["dxf"])
            if path not in cache:
                cache[path] = read_plan(path)
            line = cache[path].polyline(d["layer"], int(d.get("index", 0)))
            pts = line.outline()
            if key == "path" and (line.closed or len(pts) < len(line.points)):
                pts = pts + pts[:1]
            return [list(p) for p in pts]
        return {k: resolve(v, base, cache, k) for k, v in d.items()}
    if isinstance(d, list):
        return [resolve(v, base, cache, key) for v in d]
    return d


def write_plan(path: str | os.PathLike, polylines: list[Polyline]) -> str:
    """A minimal ASCII DXF with the given polylines, as ``LWPOLYLINE`` entities."""
    out = ["0", "SECTION", "2", "ENTITIES"]
    for p in polylines:
        out += ["0", "LWPOLYLINE", "8", p.layer, "90", str(len(p.points)), "70", "1" if p.closed else "0"]
        for x, y in p.points:
            out += ["10", repr(float(x)), "20", repr(float(y))]
    out += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return str(path)
