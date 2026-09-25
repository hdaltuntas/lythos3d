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
