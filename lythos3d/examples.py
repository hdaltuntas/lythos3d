# SPDX-License-Identifier: AGPL-3.0-only
"""Worked example models."""

from __future__ import annotations

from .core.materials import MohrCoulomb
from .core.model import Model, Stratum, Volume
from .core.problem import Stage


def excavation_pit(half_width: float = 4.0, depth: float = 3.0, lifts: int = 2,
                   mesh_size: float = 1.5, trench: bool = False) -> Model:
    """An unsupported square pit dug in lifts, then its factor of safety.

    A quarter of the pit is modelled, the symmetry planes x = 0 and y = 0
    being on rollers like the far sides.  With ``trench`` the same section is
    a slice one element thick, held in plane strain: a long trench of the
    same width and depth.  The difference between the two factors of safety
    is the support the corners and the ends of a pit's walls give each other,
    which a plane-strain analysis cannot see.
    """
    upper = MohrCoulomb("sandy clay", E=2.5e4, nu=0.3, gamma=19.0, c=12.0, phi=26.0)
    lower = MohrCoulomb("stiff clay", E=6.0e4, nu=0.3, gamma=20.0, c=25.0, phi=24.0)
    extent = half_width + 3.0 * depth
    y_extent = mesh_size if trench else extent
    y_pit = y_extent if trench else half_width
    step = depth / lifts
    volumes = [Volume(f"lift {k + 1}", (0.0, 0.0, -(k + 1) * step), (half_width, y_pit, -k * step))
               for k in range(lifts)]
    stages = [Stage("initial stresses", kind="initial", initial_stress="k0")]
    stages += [Stage(f"excavate to {-(k + 1) * step:.1f} m", excavate=(v.name,))
               for k, v in enumerate(volumes)]
    stages.append(Stage("factor of safety", kind="ssr", srf_min=0.8, srf_max=3.0))
    return Model(
        name=("trench" if trench else "square pit") + f" {2 * half_width:g} m wide, {depth:g} m deep",
        x=(0.0, extent), y=(0.0, y_extent), bottom=-(depth + 2.0 * depth + 1.0),
        strata=[Stratum("sandy clay", upper, 0.0), Stratum("stiff clay", lower, -(depth + 1.0))],
        volumes=volumes, stages=stages, mesh_size=mesh_size,
    )


def sloping_site(mesh_size: float = 2.5):
    """A pit on ground whose layers dip and thin between three boreholes.

    The clay met in BH1 and BH3 is missing from BH2, so it pinches out
    across the site; the ground falls 2 m from one side to the other.  The
    pit is 8 m by 6 m, dug to 3 m in two lifts.
    """
    from .core.model import Site
    from .core.site import Borehole, Excavation, Soil, SoilProfile

    fill = MohrCoulomb("fill", E=2.0e4, nu=0.3, gamma=18.0, c=12.0, phi=30.0)
    clay = MohrCoulomb("clay", E=3.0e4, nu=0.3, gamma=19.0, c=15.0, phi=24.0)
    sand = MohrCoulomb("sand", E=6.0e4, nu=0.3, gamma=20.0, c=1.0, phi=34.0)
    profile = SoilProfile(
        [Soil("fill", fill), Soil("clay", clay), Soil("sand", sand)],
        [Borehole("BH1", 0.0, 0.0, [("fill", 0.0), ("clay", -2.0), ("sand", -6.0)]),
         Borehole("BH2", 30.0, 0.0, [("fill", 1.0), ("sand", -4.0)]),
         Borehole("BH3", 0.0, 20.0, [("fill", -1.0), ("clay", -3.0), ("sand", -8.0)])],
        bottom=-15.0,
    )
    pit = Excavation("pit", [(4.0, 4.0), (12.0, 4.0), (12.0, 10.0), (4.0, 10.0)], [-1.5, -3.0])
    return Site("pit on sloping ground", profile, (0.0, 30.0), (0.0, 20.0), [pit], mesh_size=mesh_size)


def walled_pit(mesh_size: float = 2.5):
    """The pit of :func:`sloping_site`, now inside a diaphragm wall with a strut.

    A 0.6 m wall runs round the pit down to 9 m, installed before digging.
    A strut crosses the pit at 1 m below the original ground at BH1, and is
    fitted after the first lift has exposed it.
    """
    from .core.site import SiteAnchor, SiteWall
    from .core.structures import PlateSection

    site = sloping_site(mesh_size)
    wall = SiteWall("diaphragm wall", [(4.0, 4.0), (12.0, 4.0), (12.0, 10.0), (4.0, 10.0), (4.0, 4.0)],
                    toe=-9.0, section=PlateSection(E=3.0e7, nu=0.2, t=0.6, weight=0.0))
    # a strut is preloaded in compression: a negative lock-off load
    strut = SiteAnchor("strut", (4.0, 7.0, -1.0), (12.0, 7.0, -1.0), EA=2.0e6, prestress=-100.0)
    site.name = "walled pit on sloping ground"
    site.walls, site.anchors = [wall], [strut]
    site.stages = site.default_stages()
    return site


def dewatered_pit(mesh_size: float = 2.5, seepage: bool = False):
    """The walled pit of :func:`walled_pit` below the water table, pumped dry as it is dug.

    The water stands 1.5 m below the original ground at BH1 (level -1.5).
    Inside the wall it is drawn down to each formation level as it is
    reached; outside it stays where it was, so the wall carries the
    difference.  The wall is modelled with interfaces, which carry the
    water pressure across to it on both sides.

    With ``seepage`` the flow is solved: the wall is impermeable, water
    comes in from the sides of the model and under the toe, and is pumped
    out of the pit floor.  The permeabilities are typical of the soils.
    """
    from dataclasses import replace

    from .core.interfaces import InterfaceSpec
    from .core.water import WaterTable

    site = walled_pit(mesh_size)
    for soil, gamma_sat, k in zip(site.profile.soils, (20.0, 20.0, 21.0), (1e-5, 1e-8, 1e-4)):
        soil.material = replace(soil.material, gamma_sat=gamma_sat, k=k)
    site.name = "dewatered walled pit on sloping ground"
    site.walls[0].interface = InterfaceSpec(R=0.67)
    site.excavations[0].dewatered = True
    site.water = WaterTable(level=-1.5)
    if seepage:
        from .core.water import Seepage

        site.name += ", with seepage"
        site.water = Seepage(site.water)
    site.stages = site.default_stages()
    return site
