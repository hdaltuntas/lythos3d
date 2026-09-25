# Lythos 3D

Three-dimensional finite element analysis for geotechnical engineering: the 3D
counterpart of [Lythos](https://github.com/hdaltuntas/lythos).

Plane strain is the right idealisation for a long slope or a long wall, and the
wrong one for the corner of an excavation pit, a pile group, a raft, or a slope
whose failure is bounded at its ends. Lythos 3D is for those.

> **Status: early development.** What exists today is the linear elastic core,
> verified against closed-form solutions. Staged construction, Mohr-Coulomb
> plasticity and strength reduction come next; see the [roadmap](#roadmap).

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
- **ParaView output** (`.vtu`): displacements and smoothed stresses on
  quadratic cells.

## Installing and trying it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

lythos3d info                    # versions, and which linear solvers are available
lythos3d demo -o out             # a footing on sand over clay -> out/footing.vtu
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

```python
import numpy as np
from lythos3d.core.analysis import SurfaceLoad, linear_static
from lythos3d.core.materials import LinearElastic
from lythos3d.core.mesh import box_mesh, graded
from lythos3d.io.vtu import write_result

sand = LinearElastic("sand", E=4.0e4, nu=0.3, gamma=18.0)
clay = LinearElastic("clay", E=1.2e4, nu=0.35, gamma=19.0)

mesh = box_mesh(graded(-15, 15, 1.0), graded(-15, 15, 1.0),
                graded(-20, 0, 1.0, breaks=[-4.0]))        # a grid line at the layer boundary
mesh.assign_regions(lambda c: np.where(c[:, 2] > -4.0, 0, 1))

footing = SurfaceLoad("z", 0.0, (0, 0, -150.0),
                      where=lambda c: (abs(c[:, 0]) < 1.5) & (abs(c[:, 1]) < 1.5))
result = linear_static(mesh, [sand, clay], gravity=False, loads=[footing])
write_result("footing.vtu", result)
```

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

## Roadmap

1. ~~**Core**: quadratic tetrahedra, elastic solution, PARDISO, ParaView output.~~
2. **Plasticity and staging**: Mohr-Coulomb in six stress components (the
   2D return mapping already works in principal stresses), K0 and gravity
   initial stresses, excavation by element removal, strength reduction. The
   check: a 3D slice held in plane strain must give Lythos's 2:1 slope factor
   of safety of 1.381.
3. **Geometry**: soil layers from boreholes, excavation pits and structures
   drawn in plan with depths, meshed by gmsh.
4. **Structures**: plates for diaphragm and pile walls, embedded beams for
   piles, anchors, interfaces. In 3D a pile row no longer has to be smeared
   into a plate.
5. **Interface**: a three.js viewer for contours and cut planes, a plan
   editor, DXF plan import and an HTML report.

## Licence

MIT.
