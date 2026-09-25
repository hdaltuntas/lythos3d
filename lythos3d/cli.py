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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
