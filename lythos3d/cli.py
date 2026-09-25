"""Command line: ``lythos3d demo`` and ``lythos3d info``."""

from __future__ import annotations

import argparse
import os
import sys


def _info(args) -> int:
    import numpy
    import scipy

    from . import __version__
    from .core import assembly

    print(f"lythos3d {__version__}")
    print(f"numpy {numpy.__version__}, scipy {scipy.__version__}")
    backends = assembly.available_backends()
    print(f"linear solvers: {', '.join(backends)} (default {assembly.default_backend()})")
    if "pardiso" not in backends:
        print("  pypardiso is not available; install it for models of any size:"
              " pip install pypardiso")
    return 0


def _demo(args) -> int:
    import numpy as np

    from .core.analysis import SurfaceLoad, linear_static
    from .core.materials import LinearElastic
    from .core.mesh import box_mesh, graded
    from .io.vtu import write_result

    # a 3 x 3 m footing on 4 m of sand over clay, a quarter of it by symmetry
    B, q, size = 3.0, 150.0, args.size
    sand = LinearElastic("sand", E=4.0e4, nu=0.3, gamma=18.0)
    clay = LinearElastic("clay", E=1.2e4, nu=0.35, gamma=19.0)
    mesh = box_mesh(graded(0, 15, size, breaks=[B / 2]),
                    graded(0, 15, size, breaks=[B / 2]),
                    graded(-20, 0, size, breaks=[-4.0]))
    mesh.assign_regions(lambda c: np.where(c[:, 2] > -4.0, 0, 1))
    footing = SurfaceLoad("z", 0.0, (0.0, 0.0, -q),
                          where=lambda c: (c[:, 0] < B / 2) & (c[:, 1] < B / 2))
    result = linear_static(mesh, [sand, clay], gravity=False, loads=[footing],
                           backend=args.solver)

    os.makedirs(args.out, exist_ok=True)
    path = write_result(os.path.join(args.out, "footing.vtu"), result)
    i = result.info
    centre = np.nonzero(np.all(np.isclose(mesh.nodes, 0.0), axis=1))[0][0]
    print(f"{i['n_elements']} elements, {i['n_nodes']} nodes, {i['n_free']} equations")
    print(f"assembled in {i['assemble_seconds']:.2f} s, "
          f"solved by {i['backend']} in {i['solve_seconds']:.2f} s")
    print(f"settlement under the centre of the footing: "
          f"{-1000 * result.displacement[centre, 2]:.1f} mm")
    print(f"written {path} - open it in ParaView")
    return 0


def _pit(args) -> int:
    import time

    from .core.solver import Solver
    from .examples import excavation_pit
    from .io.vtu import write_stage

    model = excavation_pit(half_width=args.width / 2, depth=args.depth, lifts=args.lifts,
                           mesh_size=args.size, trench=args.trench)
    problem = model.build()
    print(f"{model.name}: {problem.mesh.n_elements} elements, {problem.n_dof} equations")
    os.makedirs(args.out, exist_ok=True)
    solver = Solver(problem, tolerance=2e-3, backend=args.solver, verbose=args.verbose)
    started = time.perf_counter()
    for k, stage in enumerate(model.stages):
        problem.check_stage_groups(stage)
        result = solver.run_stage(stage)
        path = write_stage(os.path.join(args.out, f"stage{k}.vtu"), problem, result)
        line = f"  {stage.name}: {'ok' if result.converged else 'FAILED'}"
        if result.srf is not None:
            line += f", factor of safety {result.srf:.2f}"
        elif k:
            line += f", max displacement {1000 * result.max_displacement:.1f} mm"
        print(f"{line}  ({result.seconds:.0f} s) -> {path}", flush=True)
        if not result.converged:
            print(f"    {result.message}")
            break
    print(f"total {time.perf_counter() - started:.0f} s.  A coarse mesh overestimates the "
          "factor of safety; refine with --size until it stops changing.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lythos3d", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="versions and the linear solvers available")
    p.set_defaults(func=_info)

    p = sub.add_parser("demo", help="analyse a footing on layered ground and write it for ParaView")
    p.add_argument("-o", "--out", default="lythos3d_out", help="output directory")
    p.add_argument("--size", type=float, default=1.0, help="plan element size in m")
    p.add_argument("--solver", default="auto", choices=("auto", "pardiso", "superlu"))
    p.set_defaults(func=_demo)

    p = sub.add_parser("pit", help="dig an unsupported square pit in lifts and find its factor of safety")
    p.add_argument("-o", "--out", default="lythos3d_pit", help="output directory")
    p.add_argument("--width", type=float, default=8.0, help="width of the square pit in m")
    p.add_argument("--depth", type=float, default=3.0, help="depth in m")
    p.add_argument("--lifts", type=int, default=2, help="number of excavation lifts")
    p.add_argument("--size", type=float, default=1.5, help="element size in m")
    p.add_argument("--trench", action="store_true",
                   help="the same section as a long trench, in plane strain, for comparison")
    p.add_argument("--solver", default="auto", choices=("auto", "pardiso", "superlu"))
    p.add_argument("-v", "--verbose", action="store_true", help="report every strength reduction trial")
    p.set_defaults(func=_pit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
