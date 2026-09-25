# Lythos 3D

Three-dimensional finite element analysis for geotechnical engineering: the 3D
counterpart of [Lythos](https://github.com/hdaltuntas/lythos).

Plane strain is the right idealisation for a long slope or a long wall, and the
wrong one for the corner of an excavation pit, a pile group, a raft, or a slope
whose failure is bounded at its ends. Lythos 3D is for those.

> **Status: early development.** Mohr-Coulomb plasticity, staged excavation
> and strength reduction work on layered ground in a box, verified against
> closed-form solutions and against 2D Lythos. General geometry and structures
> come next; see the [roadmap](#roadmap).

## What works now

- **10-node quadratic tetrahedra**, fully vectorised: the 3D counterpart of
  the 6-node triangles Lythos uses, and chosen for the same reason. Linear
  tetrahedra lock badly under the constant-volume plastic flow that
  determines a factor of safety.
- **Layered ground in a box**: a structured mesh whose grid lines fall on
  layer boundaries, with one material per region.
- **Loads**: self weight, and uniform tractions on any part of a boundary
  plane, such as a footing, a strip or a surcharge.
- **Box restraints**: a fixed base, and sides on rollers.
- **PARDISO**: Intel's parallel sparse direct solver, through `pypardiso`.
  SciPy's SuperLU cannot factorise a 3D model of useful
  size in reasonable time or memory.
- **Mohr-Coulomb** with a tension cut-off and non-associated flow: the 2D
  exact return mapping in principal stresses, carried into six stress
  components with its consistent tangent.
- **Staged construction**: K0 or gravity initial stresses, excavation by
  removing volumes of ground in lifts, surface loads per stage.
- **Factor of safety** by strength reduction.
- **ParaView output** (`.vtu`): displacements, smoothed stresses and plastic
  strain on quadratic cells, stage by stage, with excavated ground left out.

## Installing and trying it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

lythos3d info                    # versions, and which linear solvers are available
lythos3d demo -o out             # a footing on sand over clay -> out/footing.vtu
lythos3d pit -o pit              # a square pit dug in two lifts, then its factor of safety
lythos3d pit --trench -o trench  # the same section as a long trench, in plane strain
pytest
```

From a clone without installing, `python main.py info` and
`python main.py demo` do the same. You will need NumPy, SciPy and, on x86-64,
`pypardiso`.

```
30720 elements, 44649 nodes, 125400 equations
assembled in 8.48 s, solved by pardiso in 8.36 s
settlement under the centre of the footing: 14.8 mm
```

## From a script

A square pit, a quarter of it by symmetry, dug in two lifts:

```python
from lythos3d.core.materials import MohrCoulomb
from lythos3d.core.model import Model, Stratum, Volume
from lythos3d.core.problem import Stage
from lythos3d.io.vtu import write_stage

clay = MohrCoulomb("sandy clay", E=2.5e4, nu=0.3, gamma=19.0, c=12.0, phi=26.0)
stiff = MohrCoulomb("stiff clay", E=6.0e4, nu=0.3, gamma=20.0, c=25.0, phi=24.0)

model = Model(
    name="pit", x=(0, 13), y=(0, 13), bottom=-10,
    strata=[Stratum("sandy clay", clay, 0.0), Stratum("stiff clay", stiff, -4.0)],
    volumes=[Volume("lift 1", (0, 0, -1.5), (4, 4, 0)),
             Volume("lift 2", (0, 0, -3.0), (4, 4, -1.5))],
    stages=[Stage("initial", kind="initial", initial_stress="k0"),
            Stage("dig to -1.5", excavate=("lift 1",)),
            Stage("dig to -3.0", excavate=("lift 2",)),
            Stage("factor of safety", kind="ssr")],
    mesh_size=1.0,
)
problem, results = model.run(verbose=True)
print(results[-1].factor_of_safety)
for k, r in enumerate(results):
    write_stage(f"pit_{k}.vtu", problem, r)
```

The x = 0 and y = 0 planes are on rollers, as are the far sides, so they act
as planes of symmetry. For a single load case on elastic ground,
`lythos3d.core.analysis.linear_static` does without stages.

Units are kN, m and kPa, as in Lythos. `z` points up, and stresses are
tension positive, stored in the order `[xx, yy, zz, xy, yz, zx]`.

## Linear solver

| | PARDISO (`pypardiso`) | SuperLU (SciPy) |
| --- | --- | --- |
| 25k equations | 0.8 s | 20 s |
| 195k equations | 16–20 s (Cholesky), 40 s (LU) | not finished after 10 min, 13 GB in use |

PARDISO is installed automatically on x86-64 Linux and Windows. It is not
built for ARM, so on Apple silicon Lythos 3D falls back to SuperLU and warns;
keep models small there. Elastic stiffness matrices are factorised by
Cholesky, which takes half the time and memory. The elasto-plastic tangent is
unsymmetric under non-associated flow, so it will use LU.

On Debian and Ubuntu, a system-wide `pip` puts MKL under `/usr/local/lib`,
where `pypardiso` does not look. Lythos 3D finds it there itself. If MKL is
somewhere else entirely, set `PYPARDISO_MKL_RT` to the path of `libmkl_rt`.

## Verification

Every row is a test in `tests/`:

| Check | Reference | Lythos 3D |
| --- | --- | --- |
| Patch test, linear field, distorted mesh | constant strain | exact to 1e-12 |
| Rigid body modes of one element | 6 | 6 |
| Geostatic stress under self weight | `σv = γz`, `σh = K0 σv` | exact |
| Settlement profile under self weight | `w = γ/M (H z − z²/2)` | exact |
| Uniform surcharge | `qH/M` | exact |
| Two-layer column | sum of layer compressions | exact |
| Cantilever tip deflection (2 elements through the depth) | `PL³/3EI + PL/κGA` | within 1% |
| Footing symmetry | x ↔ y swap | exact |
| Singular (unrestrained) model | must be refused | refused |
| Mohr-Coulomb triaxial strength | `σ1 = σ3 Kp + 2c√Kp` | within 1e-6 |
| Consistent tangent | finite-difference derivative | within 1e-9 E |
| K0 procedure, layered ground | `σh = K0 σv`, no imbalance | exact, zero iterations |
| Excavating a layer | heave `γ h H / M` | exact |
| 2:1 slope as a plane-strain slice (`pytest -m slow`) | 2D Lythos: 1.430 | 1.438 |

At the same element sizes the plane-strain slice follows 2D Lythos to within
0.5%, and falls with refinement the same way:

| Element size | 2D Lythos | Lythos 3D slice |
| --- | --- | --- |
| 2.5 m | 1.430 | 1.438 |
| 2.0 m | 1.416 | 1.423 |

A strength reduction factor is found to within its bisection bracket
(0.007). Near failure, whether a single trial converges depends on
round-off, and PARDISO's parallel factorisation does not fix the order in
which it sums. So a repeated run can land one bracket lower, for example
1.416 instead of 1.423.

## Roadmap

1. ~~**Core**: quadratic tetrahedra, elastic solution, PARDISO, ParaView output.~~
2. ~~**Plasticity and staging**: Mohr-Coulomb in six stress components, K0
   and gravity initial stresses, excavation in lifts, strength reduction,
   checked against 2D Lythos in plane strain.~~
   Still to come here: groundwater and pore pressure, and constructing
   volumes (fill) as well as removing them.
3. **Geometry**: soil layers from boreholes, excavation pits and structures
   drawn in plan with depths, meshed by gmsh.
4. **Structures**: plates for diaphragm and pile walls, embedded beams for
   piles, anchors, interfaces. In 3D a pile row no longer has to be smeared
   into a plate.
5. **Interface**: a three.js viewer for contours and cut planes, a plan
   editor, DXF plan import and an HTML report.

## Licence

MIT.
