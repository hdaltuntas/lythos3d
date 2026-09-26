# SPDX-License-Identifier: AGPL-3.0-only
"""Sites as JSON files.

```json
{
  "name": "pit on sloping ground",
  "extent": {"x": [0, 30], "y": [0, 20]},
  "bottom": -15,
  "mesh_size": 2.5,
  "soils": [
    {"name": "fill", "model": "mohr-coulomb", "E": 2e4, "nu": 0.3, "gamma": 18,
     "c": 12, "phi": 30, "psi": 0},
    {"name": "sand", "model": "linear-elastic", "E": 6e4, "nu": 0.3, "gamma": 20}
  ],
  "boreholes": [
    {"name": "BH1", "x": 0, "y": 0, "tops": [["fill", 0.0], ["sand", -4.0]]}
  ],
  "excavations": [
    {"name": "pit", "polygon": [[4, 4], [12, 4], [12, 10], [4, 10]],
     "levels": [-1.5, -3.0], "mesh_size": 1.0, "dewatered": true}
  ],
  "water": {"level": -2.0},
  "stages": [
    {"name": "initial", "kind": "initial", "initial_stress": "k0"},
    {"name": "dig 1", "excavate": ["pit 1"]},
    {"name": "dig 2", "excavate": ["pit 2"]},
    {"name": "factor of safety", "kind": "ssr"}
  ]
}
```

``stages`` may be left out: the sequence is then K0 initial stresses, every
lift in order, and the factor of safety.  Units are kN, m and kPa.

``water`` is the water table: ``{"level": z}`` or ``{"wells": [[x, y, z], ...]}``,
with optional ``"drawdowns": [{"polygon": [...], "level": z}]`` and
``"gamma_w"``; left out, the ground is dry.  Add ``"seepage": {}`` (or
``{"closed": ["xmin"], "psi_k": 0.7}``) to solve the steady flow instead of
taking the water as hydrostatic.  A stage may set its own ``"water"`` (the
same form, or ``"dry"``).  Soils take ``gamma_sat`` for their weight below
the water, and ``k`` (and ``k_v``) for their permeability.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..core.materials import LinearElastic, MohrCoulomb
from ..core.model import Site
from ..core.problem import Stage
from ..core.site import Borehole, Excavation, SiteAnchor, SiteFill, SiteWall, Soil, SoilProfile
from ..core.beams import BeamSection, EmbeddedPile
from ..core.interfaces import InterfaceSpec
from ..core.structures import PlateSection
from ..core.water import Drawdown, Seepage, WaterTable

MODELS = {"mohr-coulomb": MohrCoulomb, "linear-elastic": LinearElastic}
_STAGE_KEYS = {"name", "kind", "increments", "excavate", "construct", "install", "loads", "reset_displacements",
               "initial_stress", "srf_min", "srf_max", "water", "drained", "time", "drained_sides"}
_WATER_KEYS = {"level", "wells", "drawdowns", "gamma_w", "seepage"}
_SEEPAGE_KEYS = {"closed", "psi_k", "k_min"}


def load_from_dict(d: dict):
    """``{"area": [[x, y], ...], "q": kPa[, "horizontal": [qx, qy], "name"]}`` on the ground,
    or ``{"pile": name, "force": [fx, fy, fz][, "moment": [...]]}`` on a pile head."""
    from ..core.analysis import AreaLoad
    from ..core.beams import PileLoad

    if "area" in d:
        return AreaLoad(d["area"], float(d["q"]), tuple(d.get("horizontal", (0.0, 0.0))),
                        d.get("name", "area load"))
    if "pile" in d:
        return PileLoad(d["pile"], tuple(map(float, d.get("force", (0, 0, 0)))),
                        tuple(map(float, d.get("moment", (0, 0, 0)))))
    raise ValueError(f"a stage load needs 'area' or 'pile': {sorted(d)}")


def load_to_dict(load) -> dict | None:
    from ..core.analysis import AreaLoad
    from ..core.beams import PileLoad

    if isinstance(load, AreaLoad):
        return {"name": load.name, "area": [list(p) for p in load.polygon], "q": load.q,
                "horizontal": list(load.horizontal)}
    if isinstance(load, PileLoad):
        return {"pile": load.pile, "force": list(map(float, load.force)), "moment": list(map(float, load.moment))}
    return None


def water_from_dict(d):
    """``"dry"``, or ``{"level" | "wells", "drawdowns", "gamma_w"}``; with ``"seepage": {...}``
    (``closed``, ``psi_k``, ``k_min``, all optional) the flow is solved rather than assumed hydrostatic."""
    if d is None:
        return None
    if d == "dry":
        return WaterTable.dry()
    unknown = set(d) - _WATER_KEYS
    if unknown:
        raise ValueError(f"water: unknown key(s) {sorted(unknown)}")
    kw = {"gamma_w": float(d["gamma_w"])} if "gamma_w" in d else {}
    table = WaterTable(level=None if d.get("level") is None else float(d["level"]),
                       wells=[tuple(map(float, w)) for w in d.get("wells", [])],
                       drawdowns=[Drawdown([tuple(map(float, p)) for p in dd["polygon"]], float(dd["level"]))
                                  for dd in d.get("drawdowns", [])], **kw)
    flow = d.get("seepage")
    if flow is None or flow is False:
        return table
    flow = {} if flow is True else dict(flow)
    unknown = set(flow) - _SEEPAGE_KEYS
    if unknown:
        raise ValueError(f"seepage: unknown key(s) {sorted(unknown)}")
    return Seepage(table, **flow)


def water_to_dict(w):
    if w is None:
        return None
    seepage = None
    if isinstance(w, Seepage):
        seepage = {"closed": list(w.closed), "psi_k": w.psi_k, "k_min": w.k_min}
        w = w.table
    if w.is_dry and not w.drawdowns:
        return "dry"
    out = {"level": w.level} if w.level is not None else {"wells": [list(x) for x in w.wells]}
    if w.drawdowns:
        out["drawdowns"] = [{"polygon": [list(p) for p in d.polygon], "level": d.level} for d in w.drawdowns]
    out["gamma_w"] = w.gamma_w
    if seepage is not None:
        out["seepage"] = seepage
    return out


def section_from_dict(d: dict, owner: str) -> PlateSection:
    """``{"E", "nu", "t"}`` or ``{"EA", "EI"}`` per metre, with optional ``nu`` and ``weight``."""
    if "EA" in d or "EI" in d:
        if "EA" not in d or "EI" not in d:
            raise ValueError(f"wall {owner!r}: give both EA and EI")
        return PlateSection.from_stiffness(float(d["EA"]), float(d["EI"]), float(d.get("nu", 0.2)),
                                           float(d.get("weight", 0.0)))
    try:
        return PlateSection(E=float(d["E"]), nu=float(d.get("nu", 0.2)), t=float(d["t"]),
                            weight=float(d.get("weight", 0.0)))
    except KeyError as err:
        raise ValueError(f"wall {owner!r}: needs E and t, or EA and EI") from err


def pile_from_dict(d: dict) -> EmbeddedPile:
    """``{"name", "head", "tip", "E", "D"[, "nu", "weight", "hollow"], "skin": [top, tip] | null,
    "base": kN | null[, "element_size", "isf_skin", "isf_lateral", "isf_base"]}``."""
    try:
        section = BeamSection.circular(float(d["E"]), float(d["D"]), float(d.get("nu", 0.2)),
                                       float(d.get("weight", 0.0)), float(d.get("hollow", 0.0)))
        return EmbeddedPile(d["name"], tuple(map(float, d["head"])), tuple(map(float, d["tip"])), section,
                            skin=None if d.get("skin") is None else tuple(map(float, d["skin"])),
                            base=None if d.get("base") is None else float(d["base"]),
                            element_size=d.get("element_size"), isf_skin=d.get("isf_skin"),
                            isf_lateral=d.get("isf_lateral"), isf_base=d.get("isf_base"))
    except KeyError as err:
        raise ValueError(f"pile {d.get('name')!r} is missing {err}") from None


def _pile_to_dict(p: EmbeddedPile) -> dict:
    s = p.section
    E = s.EA / (np.pi / 4 * (s.diameter ** 2 - s.hollow ** 2))
    nu = E / (2.0 * s.GJ / (np.pi / 32 * (s.diameter ** 4 - s.hollow ** 4))) - 1.0
    return {"name": p.name, "head": list(p.head), "tip": list(p.tip), "E": E, "nu": nu, "D": s.diameter,
            "hollow": s.hollow, "weight": s.weight, "skin": None if p.skin is None else list(p.skin), "base": p.base,
            "element_size": p.element_size, "isf_skin": p.isf_skin, "isf_lateral": p.isf_lateral,
            "isf_base": p.isf_base}


def material_from_dict(d: dict):
    d = dict(d)
    model = d.pop("model", "mohr-coulomb")
    if model not in MODELS:
        raise ValueError(f"soil {d.get('name')!r}: unknown model {model!r}; use one of {sorted(MODELS)}")
    cls = MODELS[model]
    allowed = set(cls.__dataclass_fields__)
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"soil {d.get('name')!r}: unknown parameter(s) {sorted(unknown)} for {model}")
    return cls(**d)


def material_to_dict(m) -> dict:
    model = next(k for k, v in MODELS.items() if type(m) is v)
    return {"model": model, **{k: getattr(m, k) for k in m.__dataclass_fields__}}


def site_from_dict(d: dict) -> Site:
    try:
        soils = [Soil(s["name"], material_from_dict(s)) for s in d["soils"]]
        boreholes = [Borehole(b["name"], float(b["x"]), float(b["y"]),
                              [(str(n), float(z)) for n, z in b["tops"]]) for b in d["boreholes"]]
        profile = SoilProfile(soils, boreholes, float(d["bottom"]))
        excavations = [Excavation(e["name"], [tuple(map(float, p)) for p in e["polygon"]],
                                  [float(z) for z in e["levels"]], e.get("mesh_size"),
                                  bool(e.get("dewatered", False)))
                       for e in d.get("excavations", [])]
        walls = [SiteWall(w["name"], [tuple(map(float, p)) for p in w["path"]], float(w["toe"]),
                          section_from_dict(w, w["name"]), None if w.get("top") is None else float(w["top"]),
                          InterfaceSpec(**w["interface"]) if w.get("interface") else None)
                 for w in d.get("walls", [])]
        anchors = [SiteAnchor(a["name"], tuple(map(float, a["a"])), tuple(map(float, a["b"])), float(a["EA"]),
                              float(a.get("prestress", 0.0)), bool(a.get("fixed_end", False)))
                   for a in d.get("anchors", [])]
        piles = [pile_from_dict(p) for p in d.get("piles", [])]
        fills = [SiteFill(f["name"], [tuple(map(float, p)) for p in f["polygon"]],
                          [float(z) for z in f["levels"]], material_from_dict(f["material"]), f.get("mesh_size"))
                 for f in d.get("fills", [])]
        stages = []
        for s in d.get("stages", []):
            unknown = set(s) - _STAGE_KEYS
            if unknown:
                raise ValueError(f"stage {s.get('name')!r}: unknown key(s) {sorted(unknown)}")
            s = dict(s)
            s["water"] = water_from_dict(s.get("water"))
            s["loads"] = tuple(load_from_dict(x) for x in s.get("loads", ()))
            stages.append(Stage(**s))
        extent = d["extent"]
        return Site(d.get("name", "site"), profile, tuple(extent["x"]), tuple(extent["y"]),
                    excavations, stages, float(d.get("mesh_size", 2.0)), walls, anchors, piles,
                    water_from_dict(d.get("water")), fills,
                    [load_from_dict(x) for x in d.get("loads", [])])
    except KeyError as err:
        raise ValueError(f"the site description is missing {err}") from None


def site_to_dict(site: Site) -> dict:
    profile = site.profile
    return {
        "name": site.name,
        "extent": {"x": list(site.x), "y": list(site.y)},
        "bottom": profile.bottom,
        "mesh_size": site.mesh_size,
        "soils": [{"name": s.name, **{k: v for k, v in material_to_dict(s.material).items() if k != "name"}}
                  for s in profile.soils],
        "boreholes": [{"name": b.name, "x": b.x, "y": b.y, "tops": [list(t) for t in b.tops]}
                      for b in profile.boreholes],
        "excavations": [{"name": e.name, "polygon": [list(p) for p in e.polygon], "levels": list(e.levels),
                         **({"mesh_size": e.mesh_size} if e.mesh_size else {}),
                         **({"dewatered": True} if e.dewatered else {})} for e in site.excavations],
        "water": water_to_dict(site.water),
        "walls": [{"name": w.name, "path": [list(p) for p in w.path], "toe": w.toe, "top": w.top,
                   "E": w.section.E, "nu": w.section.nu, "t": w.section.t, "weight": w.section.weight,
                   "interface": None if w.interface is None else {
                       "R": w.interface.R, "virtual_thickness": w.interface.virtual_thickness,
                       "tensile": w.interface.tensile, "c": w.interface.c, "phi": w.interface.phi}}
                  for w in site.walls],
        "anchors": [{"name": a.name, "a": list(a.a), "b": list(a.b), "EA": a.EA, "prestress": a.prestress,
                     "fixed_end": a.fixed_end} for a in site.anchors],
        "piles": [_pile_to_dict(p) for p in site.piles],
        "loads": [load_to_dict(x) for x in site.loads],
        "fills": [{"name": f.name, "polygon": [list(p) for p in f.polygon], "levels": list(f.levels),
                   "material": material_to_dict(f.material),
                   **({"mesh_size": f.mesh_size} if f.mesh_size else {})} for f in site.fills],
        "stages": [{"name": s.name, "kind": s.kind, "increments": s.increments,
                    "excavate": list(s.excavate), "construct": list(s.construct), "install": list(s.install),
                    "reset_displacements": s.reset_displacements,
                    "initial_stress": s.initial_stress, "srf_min": s.srf_min, "srf_max": s.srf_max,
                    "water": water_to_dict(s.water), "drained": s.drained,
                    "time": s.time, "drained_sides": list(s.drained_sides),
                    "loads": [load_to_dict(x) for x in s.loads]}
                   for s in site.stages if all(load_to_dict(x) is not None for x in s.loads)],
    }


def load_site(path: str | os.PathLike) -> Site:
    """A site from its JSON file; plan outlines may come from DXF drawings (see :mod:`lythos3d.io.dxf`)."""
    from .dxf import resolve

    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return site_from_dict(resolve(d, os.path.dirname(os.path.abspath(path))))


def save_site(site: Site, path: str | os.PathLike) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(site_to_dict(site), fh, indent=2)
    return str(path)
