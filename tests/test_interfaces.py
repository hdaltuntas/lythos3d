# SPDX-License-Identifier: AGPL-3.0-only
"""The interface element on its own."""

import numpy as np
import pytest

from lythos3d.core.interfaces import N_GAUSS, STATE_WIDTH, InterfaceElements

TRI6 = np.array([[0, 0, 0], [2, 0, 0], [0, 1.5, 0], [1, 0, 0], [1, 0.75, 0], [0, 0.75, 0]], float)


def _element(tan_phi=np.tan(np.radians(25.0)), c=5.0, tilt=True):
    frame = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0] if tilt else np.eye(3)
    x = TRI6 @ frame.T
    nodes = np.vstack([x, x])                              # soil 0..5, wall 6..11, same places
    normal = frame[:, 2]
    el = InterfaceElements(nodes, [np.arange(6)], [np.arange(6, 12)], [normal],
                           kn=1e6, ks=4e5, c=c, tan_phi=tan_phi)
    return el, frame


def _move(frame, dn, ds1, ds2=0.0):
    """Element dofs for the soil face moved by (dn, ds1, ds2) in the element's axes, the wall held."""
    el_axes = frame @ np.array([ds1, ds2, dn])          # frame columns: s1, s2, normal
    u = np.zeros(36)
    for a in range(6):
        u[3 * a:3 * a + 3] = el_axes
    return u[None]


def _zero_state(n=1):
    return np.zeros((n, N_GAUSS, STATE_WIDTH))


def test_elastic_contact_and_pressure():
    el, frame = _element()
    Fe, Ke, trial, _ = el.respond(_move(frame, -1e-5, 2e-6), _zero_state(), np.array([False]))
    t = trial[0, :, 3:]
    # the local tangent axes are arbitrary in the plane: compare invariants
    assert np.allclose(t[:, 0], 1e6 * -1e-5)
    assert np.allclose(np.linalg.norm(t[:, 1:], axis=1), 4e5 * 2e-6)
    # the forces on the two faces balance, and sum to traction times area
    assert np.allclose(Fe[0, :18].reshape(6, 3).sum(0), -Fe[0, 18:].reshape(6, 3).sum(0))
    assert np.allclose(np.linalg.norm(Fe[0, :18].reshape(6, 3).sum(0)),
                       np.linalg.norm([-10.0, 0.8]) * el.area[0], rtol=1e-9)


def test_sliding_at_the_mohr_coulomb_limit():
    el, frame = _element()
    tn = 1e6 * -1e-4                                        # 100 kPa of contact pressure
    _, _, trial, _ = el.respond(_move(frame, -1e-4, 1e-3), _zero_state(), np.array([False]))
    limit = 5.0 - tn * np.tan(np.radians(25.0))
    assert np.allclose(np.linalg.norm(trial[0, :, 4:], axis=1), limit)
    assert np.allclose(trial[0, :, 3], tn)


def test_the_gap_opens_under_tension():
    el, frame = _element()
    _, Ke, trial, _ = el.respond(_move(frame, 1e-4, 1e-4), _zero_state(), np.array([False]))
    assert np.allclose(trial[0, :, 3:], 0.0)
    assert np.abs(Ke).max() < 1e-2 * 1e6 * el.area[0]


def test_slip_direction_follows_the_trial_shear():
    el, frame = _element()
    _, _, trial, _ = el.respond(_move(frame, -1e-4, 1e-3, 1e-3), _zero_state(), np.array([False]))
    t = trial[0, 0, 4:]
    d = trial[0, 0, 1:3]
    assert np.allclose(t / np.linalg.norm(t), d / np.linalg.norm(d))


@pytest.mark.parametrize("dn, ds1, ds2", [(-1e-5, 2e-6, 0.0),        # sticking
                                          (-1e-4, 1e-3, 4e-4),         # sliding
                                          (2e-5, 1e-5, 0.0)])          # open
def test_consistent_tangent(dn, ds1, ds2):
    """Exact in every state once the deliberate residual stiffness is taken out."""
    el, frame = _element()
    el.residual_stiffness = 0.0
    u0 = _move(frame, dn, ds1, ds2)
    rigid = np.array([False])
    _, K, _, _ = el.respond(u0, _zero_state(), rigid)
    h = 1e-9
    num = np.empty((36, 36))
    for j in range(36):
        e = np.zeros((1, 36))
        e[0, j] = h
        num[:, j] = (el.respond(u0 + e, _zero_state(), rigid)[0] - el.respond(u0 - e, _zero_state(), rigid)[0])[0] / (2 * h)
    assert np.abs(num - K[0]).max() < 1e-5 * max(np.abs(K[0]).max(), 1e6 * el.area[0])


def test_unloading_after_slip_is_elastic():
    el, frame = _element()
    rigid = np.array([False])
    _, _, slid, _ = el.respond(_move(frame, -1e-4, 1e-3), _zero_state(), rigid)
    _, _, back, _ = el.respond(_move(frame, -1e-4, 1e-3 - 1e-6), slid, rigid)
    drop = np.linalg.norm(slid[0, :, 4:], axis=1) - np.linalg.norm(back[0, :, 4:], axis=1)
    assert np.allclose(drop, 4e5 * 1e-6, rtol=1e-6)


def test_rigid_tie_before_the_wall_is_installed():
    el, frame = _element(c=0.0)
    _, K, trial, _ = el.respond(_move(frame, 1e-4, 1e-3), _zero_state(), np.array([True]))
    # no slip and no gap: the tie carries the whole relative displacement
    assert np.allclose(trial[0, :, 3], el.tie[0] * 1e-4)
    assert np.allclose(np.linalg.norm(trial[0, :, 4:], axis=1), el.tie[0] * 1e-3)


def test_rigid_body_motion_of_both_faces_is_free():
    el, frame = _element()
    for motion in np.eye(3):
        u = np.tile(motion, 12)[None]
        Fe, K, _, _ = el.respond(u, _zero_state(), np.array([False]))
        assert np.abs(Fe).max() < 1e-9 and np.abs(K[0] @ u[0]).max() < 1e-6


# --------------------------------------------------------------------------- in the ground
def _sliding_block(backend="auto"):
    """A 2 x 2 x 0.25 m block on a plate, pressed down by q and pushed sideways.

    The lower ground and the plate are held; the block rests on the
    interface alone.  It is thin so that the push does not tip it and open a
    gap at the heel, which would lower the capacity below the exact one.
    """
    from lythos3d.core.interfaces import InterfaceSpec
    from lythos3d.core.materials import LinearElastic
    from lythos3d.core.mesh import box_mesh, graded
    from lythos3d.core.problem import Problem
    from lythos3d.core.solver import Solver
    from lythos3d.core.structures import Plate, PlateSection

    mesh = box_mesh(graded(0, 2, 0.5), graded(0, 2, 0.5), graded(-1.25, 0, 0.25))
    faces = mesh.faces_in_box((0, 0, -0.25), (2, 2, -0.25))
    plate = Plate("base", faces, PlateSection(E=3e7, nu=0.2, t=0.3),
                  interface=InterfaceSpec(c=5.0, phi=20.0), side=lambda p: np.sign(p[:, 2] + 0.25))
    problem = Problem(mesh, [LinearElastic("block", 5e4, 0.25, 0.0)], fixed=np.zeros(0, dtype=np.int64),
                      plates=[plate])
    upper = set(problem.interface_elements.soil_faces[:len(faces)].ravel().tolist())
    held = [n for n in np.nonzero(problem.mesh.nodes[:, 2] < -0.25 + 1e-9)[0] if n not in upper]
    problem.fixed = (3 * np.array(held)[:, None] + np.arange(3)).ravel()
    return problem, Solver(problem, tolerance=1e-6, backend=backend)


@pytest.mark.slow
def test_a_block_slides_at_the_interface_strength():
    """Capacity c A + N tan(phi), exactly: equilibrium at 98% of it, none at 102%.

    The block is elastic, so only the sliding contact makes the tangent
    unsymmetric; this is the case that once had PARDISO factorise it by
    Cholesky and return nonsense.
    """
    from lythos3d.core.analysis import SurfaceLoad
    from lythos3d.core.problem import Stage

    q = 100.0
    limit = 5.0 + q * np.tan(np.radians(20.0))
    problem, solver = _sliding_block()
    assert solver.run_stage(Stage("seat", install=("base",), loads=(SurfaceLoad("z", 0.0, (0, 0, -q)),),
                                  increments=2)).converged
    below = solver.run_stage(Stage("0.98", loads=(SurfaceLoad("z", 0.0, (0.98 * limit, 0, -q)),),
                                   increments=8))
    assert below.converged
    # the block's side of the plate comes first; its faces are all the same
    # size, so the mean shear traction on them must equal the applied shear
    tractions = below.interface_tractions["base"]
    upper = slice(0, len(problem.plates[0].faces))
    # the shear traction points the way the soil moves relative to the plate
    assert tractions["shear"][upper, 0].mean() == pytest.approx(0.98 * limit, rel=1e-6)
    assert np.abs(tractions["shear"][upper, 1].mean()) < 1e-6 * limit
    assert tractions["tn"][upper].mean() == pytest.approx(-q, rel=1e-6)
    above = solver.run_stage(Stage("1.02", loads=(SurfaceLoad("z", 0.0, (1.02 * limit, 0, -q)),),
                                   increments=8))
    assert not above.converged


def _wall_slice(interface, install=True, h=1.0):
    from lythos3d.core.materials import MohrCoulomb
    from lythos3d.core.model import Model, Stratum, Volume, Wall
    from lythos3d.core.problem import Stage
    from lythos3d.core.structures import PlateSection

    soil = MohrCoulomb("sand", E=3e4, nu=0.3, gamma=18.0, c=5.0, phi=30.0)
    stages = [Stage("initial", kind="initial")]
    if install:
        stages.append(Stage("wall", install=("wall",)))
    stages.append(Stage("dig", excavate=("dig",)))
    return Model("slice", (0, 20), (0, h), -12.0, [Stratum("sand", soil, 0.0)],
                 volumes=[Volume("dig", (0, 0, -3), (5, h, 0))],
                 walls=[Wall("wall", (5, 0, -8), (5, h, 0), PlateSection(E=3e7, nu=0.2, t=0.5),
                             interface=interface)],
                 stages=stages, mesh_size=h)


def _top(problem, result):
    nodes = np.unique(problem.plates[0].faces)
    top = nodes[np.all(np.isclose(problem.mesh.nodes[nodes], [5, 0, 0]), axis=1)][0]
    return result.displacement[top, 0]


def test_splitting_leaves_the_toe_tied_and_everything_else_free():
    from lythos3d.core.interfaces import InterfaceSpec

    plain = _wall_slice(None).build()
    split = _wall_slice(InterfaceSpec()).build()
    wall = np.unique(plain.plates[0].faces)
    toe = np.isclose(plain.mesh.nodes[wall, 2], -8.0)
    # every wall node but those on the toe line gets two copies: the wall's and the far soil's
    assert split.mesh.n_nodes - plain.mesh.n_nodes == 2 * np.count_nonzero(~toe)
    assert np.count_nonzero(toe) == 3                          # the toe line of a one-element slice
    assert split.interface_elements.n_elements == 2 * len(plain.plates[0].faces)


def test_before_it_is_installed_the_wall_line_is_continuous_ground():
    """An uninstalled wall's interfaces tie the soil across the line it will occupy.

    Elastic ground under a load straddling the wall line, with the wall
    not yet built: the same as ground with no wall at all, but for the
    penalty compliance of the tie.  (A linear case on purpose: near a
    collapse the answer depends on round-off and cannot be compared.)
    """
    from lythos3d.core.analysis import SurfaceLoad
    from lythos3d.core.interfaces import InterfaceSpec
    from lythos3d.core.materials import LinearElastic
    from lythos3d.core.model import Model, Stratum, Wall
    from lythos3d.core.problem import Stage
    from lythos3d.core.structures import PlateSection

    soil = LinearElastic("clay", E=2e4, nu=0.3, gamma=18.0)
    load = SurfaceLoad("z", 0.0, (0, 0, -100.0), where=lambda c: np.abs(c[:, 0] - 5.0) < 2.0)

    def run(walls):
        model = Model("x", (0, 12), (0, 1), -8.0, [Stratum("clay", soil, 0.0)], walls=walls,
                      stages=[Stage("initial", kind="initial"), Stage("load", loads=(load,))], mesh_size=0.5)
        return model.run(tolerance=1e-8)

    wall = Wall("wall", (5, 0, -5), (5, 1, 0), PlateSection(E=3e7, nu=0.2, t=0.5), interface=InterfaceSpec())
    problem, results = run([wall])
    p0, r0 = run([])
    n = p0.mesh.n_nodes                                     # the original nodes come first in both
    assert np.allclose(problem.mesh.nodes[:n], p0.mesh.nodes)
    u, u0 = results[-1].displacement[:n], r0[-1].displacement
    assert np.abs(u - u0).max() < 1e-3 * np.abs(u0).max()      # 1e-4 measured


def test_less_wall_friction_lets_the_wall_move_more():
    """Bonded, then a full-strength interface, then R = 0.67: each lets the wall move more.

    A smoother wall carries less of the soil's weight in friction and is
    pushed harder; the interface's own elastic give adds to that.
    """
    from lythos3d.core.interfaces import InterfaceSpec

    moves = []
    for spec in (None, InterfaceSpec(R=1.0), InterfaceSpec(R=0.67)):
        problem, results = _wall_slice(spec).run(tolerance=1e-4)
        assert all(r.converged for r in results)
        moves.append(-_top(problem, results[-1]))
    assert 0 < moves[0] < moves[1] < moves[2], moves
    # and the interface on the dug side above the formation has gone with the soil
    tractions = results[-1].interface_tractions["wall"]
    assert len(tractions["tn"]) < problem.interface_elements.n_elements
    assert np.all(tractions["mobilised"] <= 1.0 + 1e-6)


def test_interface_round_trips_through_json(tmp_path):
    import json

    from lythos3d.core.interfaces import InterfaceSpec
    from lythos3d.examples import walled_pit
    from lythos3d.io.site_json import load_site, save_site, site_to_dict

    site = walled_pit()
    site.walls[0].interface = InterfaceSpec(R=0.5, phi=None, c=None)
    path = save_site(site, tmp_path / "site.json")
    again = load_site(path)
    assert again.walls[0].interface.R == 0.5
    assert site_to_dict(again) == json.load(open(path))


def test_next_to_undrained_soil_friction_acts_on_the_effective_normal_stress():
    """100 kPa of contact pressure of which 40 kPa is excess pore water: the limit is c + 60 tan(phi)."""
    el, frame = _element()
    u = _move(frame, -1e-4, 1e-3)
    _, _, total, _ = el.respond(u, _zero_state(), np.array([False]))
    _, _, eff, _ = el.respond(u, _zero_state(), np.array([False]), pore=np.array([40.0]))
    tan_phi = np.tan(np.radians(25.0))
    assert np.allclose(np.linalg.norm(total[0, :, 4:], axis=1), 5.0 + 100.0 * tan_phi)
    assert np.allclose(np.linalg.norm(eff[0, :, 4:], axis=1), 5.0 + 60.0 * tan_phi)
    # more excess than contact pressure leaves the cohesion alone
    _, _, none, _ = el.respond(u, _zero_state(), np.array([False]), pore=np.array([150.0]))
    assert np.allclose(np.linalg.norm(none[0, :, 4:], axis=1), 5.0)
