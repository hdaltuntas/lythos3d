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
     "levels": [-1.5, -3.0], "mesh_size": 1.0}
  ],
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
"""

from __future__ import annotations

import json
import os

from ..core.materials import LinearElastic, MohrCoulomb
from ..core.model import Site
from ..core.problem import Stage
from ..core.site import Borehole, Excavation, SiteAnchor, SiteWall, Soil, SoilProfile
from ..core.interfaces import InterfaceSpec
from ..core.structures import PlateSection

MODELS = {"mohr-coulomb": MohrCoulomb, "linear-elastic": LinearElastic}
_STAGE_KEYS = {"name", "kind", "increments", "excavate", "install", "reset_displacements",
               "initial_stress", "srf_min", "srf_max"}


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
                                  [float(z) for z in e["levels"]], e.get("mesh_size"))
                       for e in d.get("excavations", [])]
        walls = [SiteWall(w["name"], [tuple(map(float, p)) for p in w["path"]], float(w["toe"]),
                          section_from_dict(w, w["name"]), None if w.get("top") is None else float(w["top"]),
                          InterfaceSpec(**w["interface"]) if w.get("interface") else None)
                 for w in d.get("walls", [])]
        anchors = [SiteAnchor(a["name"], tuple(map(float, a["a"])), tuple(map(float, a["b"])), float(a["EA"]),
                              float(a.get("prestress", 0.0)), bool(a.get("fixed_end", False)))
                   for a in d.get("anchors", [])]
        stages = []
        for s in d.get("stages", []):
            unknown = set(s) - _STAGE_KEYS
            if unknown:
                raise ValueError(f"stage {s.get('name')!r}: unknown key(s) {sorted(unknown)}")
            stages.append(Stage(**s))
        extent = d["extent"]
        return Site(d.get("name", "site"), profile, tuple(extent["x"]), tuple(extent["y"]),
                    excavations, stages, float(d.get("mesh_size", 2.0)), walls, anchors)
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
                         **({"mesh_size": e.mesh_size} if e.mesh_size else {})} for e in site.excavations],
        "walls": [{"name": w.name, "path": [list(p) for p in w.path], "toe": w.toe, "top": w.top,
                   "E": w.section.E, "nu": w.section.nu, "t": w.section.t, "weight": w.section.weight,
                   "interface": None if w.interface is None else {
                       "R": w.interface.R, "virtual_thickness": w.interface.virtual_thickness,
                       "tensile": w.interface.tensile, "c": w.interface.c, "phi": w.interface.phi}}
                  for w in site.walls],
        "anchors": [{"name": a.name, "a": list(a.a), "b": list(a.b), "EA": a.EA, "prestress": a.prestress,
                     "fixed_end": a.fixed_end} for a in site.anchors],
        "stages": [{"name": s.name, "kind": s.kind, "increments": s.increments,
                    "excavate": list(s.excavate), "install": list(s.install),
                    "reset_displacements": s.reset_displacements,
                    "initial_stress": s.initial_stress, "srf_min": s.srf_min, "srf_max": s.srf_max}
                   for s in site.stages if not s.loads],
    }


def load_site(path: str | os.PathLike) -> Site:
    with open(path, encoding="utf-8") as fh:
        return site_from_dict(json.load(fh))


def save_site(site: Site, path: str | os.PathLike) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(site_to_dict(site), fh, indent=2)
    return str(path)
