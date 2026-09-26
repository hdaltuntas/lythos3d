# SPDX-License-Identifier: AGPL-3.0-only
"""Mohr-Coulomb with a stiffness that depends on stress and on unloading."""

import numpy as np
import pytest

from lythos3d.core.analysis import SurfaceLoad
from lythos3d.core.materials import MaterialState, MohrCoulomb, StressDependentMohrCoulomb
from lythos3d.core.mesh import box_mesh, graded
from lythos3d.core.model import Model, Stratum, Volume
from lythos3d.core.problem import Problem, Stage
from lythos3d.core.solver import Solver

#: strong enough to stay elastic, so the stiffness alone is tested
STIFF = dict(E=2.0e4, nu=0.3, c=500.0, phi=30.0, tension_cutoff=None)


def _oedometric(E, nu):
    return E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))


def test_the_modulus_follows_the_minor_principal_stress():
    soil = StressDependentMohrCoulomb("clay", E=1e4, nu=0.3, c=0.0, phi=30.0, m=0.5, p_ref=100.0)
    stress = np.array([[-50.0, -50.0, -200.0, 0, 0, 0], [-400.0, -400.0, -400.0, 0, 0, 0],
                       [0.0, 0.0, -10.0, 0, 0, 0]])
    assert np.allclose(soil.factor(stress), [np.sqrt(50 / 100), np.sqrt(4.0), np.sqrt(10 / 100)])
    # with cohesion the reference shifts by c cot(phi)
    cohesive = StressDependentMohrCoulomb("clay", E=1e4, nu=0.3, c=10.0, phi=30.0, m=1.0)
    cc = 10.0 / np.tan(np.radians(30.0))
    assert cohesive.factor(stress[:1]) == pytest.approx((50 + cc) / (100 + cc))
    assert StressDependentMohrCoulomb("s", E=1e4, nu=0.3).unloading_modulus == 3e4


def test_loading_is_on_e_and_unloading_on_e_ur():
    """A confined column under a surcharge settles q H / M(E), and rebounds q H / M(E_ur) when it comes off."""
    H, q = 4.0, 80.0
    soil = StressDependentMohrCoulomb("clay", m=0.0, E_ur=7.0e4, **STIFF)
    mesh = box_mesh(graded(0, 1, 1.0), graded(0, 1, 1.0), graded(-H, 0, 1.0))
    load = SurfaceLoad("z", 0.0, (0, 0, -q))
    solver = Solver(Problem(mesh, [soil]), tolerance=1e-10)
    loaded = solver.run_stage(Stage("load", loads=(load,), increments=1))
    unloaded = solver.run_stage(Stage("unload", increments=1))
    top = np.isclose(mesh.nodes[:, 2], 0.0)
    assert np.allclose(loaded.displacement[top, 2], -q * H / _oedometric(2.0e4, 0.3), rtol=1e-9)
    assert np.allclose(unloaded.displacement[top, 2], q * H / _oedometric(7.0e4, 0.3), rtol=1e-9)
    assert loaded.state.q_max.max() > 0


def test_an_excavation_floor_heaves_on_the_unloading_modulus():
    """Dig the top 2 m off the whole site: the floor rises gamma h (H - h) / M(E_ur), a third of Mohr-Coulomb's."""
    H, h, gamma = 8.0, 2.0, 18.0

    def heave(soil):
        model = Model("dig", (0, 1), (0, 1), -H, [Stratum("clay", soil, 0.0)],
                      volumes=[Volume("top", (0, 0, -h), (1, 1, 0))],
                      stages=[Stage("initial", kind="initial"), Stage("dig", excavate=("top",), increments=1)],
                      mesh_size=1.0)
        problem, results = model.run(tolerance=1e-10)
        floor = np.isclose(problem.mesh.nodes[:, 2], -h)
        return results[-1].displacement[floor, 2]

    plain = heave(MohrCoulomb("clay", gamma=gamma, K0=0.5, **STIFF))
    sd = heave(StressDependentMohrCoulomb("clay", gamma=gamma, K0=0.5, m=0.0, **STIFF))
    assert np.allclose(plain, gamma * h * (H - h) / _oedometric(2.0e4, 0.3), rtol=1e-9)
    assert np.allclose(sd, gamma * h * (H - h) / _oedometric(6.0e4, 0.3), rtol=1e-9)


def test_the_model_round_trips_through_json():
    from lythos3d.io.site_json import material_from_dict, material_to_dict

    soil = StressDependentMohrCoulomb("sand", E=3e4, nu=0.25, c=1.0, phi=34.0, E_ur=9e4, m=0.5)
    back = material_from_dict(material_to_dict(soil))
    assert back == soil and material_to_dict(soil)["model"] == "stress-dependent-mohr-coulomb"


def test_the_state_carries_the_largest_deviatoric_stress():
    soil = StressDependentMohrCoulomb("clay", m=0.0, **STIFF)
    state = MaterialState.zeros(1)
    _, _, loaded = soil.update(state, np.array([[0, 0, -1e-3, 0, 0, 0]]))
    _, _, back = soil.update(loaded, np.array([[0, 0, 5e-4, 0, 0, 0]]))
    assert back.q_max[0] == pytest.approx(loaded.q_max[0]) and loaded.q_max[0] > 0
