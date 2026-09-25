"""Mohr-Coulomb in six stress components."""

import math

import numpy as np
import pytest

from lythos3d.core.materials import (
    LinearElastic, MaterialState, MohrCoulomb, from_principal, principal_stresses,
    principal_values, rotation_to_global,
)

rng = np.random.default_rng(0)


def _stresses(n=300):
    return rng.normal(size=(n, 6)) * 60.0 - np.array([100, 110, 90, 0, 0, 0.0])


def test_principal_values_in_closed_form_match_the_eigensolver():
    s = _stresses()
    s[:5, 3:] = 0.0                                    # already principal
    s[5] = [-50, -50, -50, 0, 0, 0]                    # hydrostatic: all three equal
    s[6] = [-50, -50, -80, 0, 0, 0]                    # two equal
    v, V = principal_stresses(s)
    assert np.allclose(principal_values(s), v, atol=1e-9)
    assert np.all(np.diff(v, axis=1) <= 1e-12)
    assert np.allclose(from_principal(v, V), s, atol=1e-10)


def test_rotation_maps_the_elastic_principal_tangent_onto_the_isotropic_one():
    mat = MohrCoulomb("s", E=1e5, nu=0.3)
    _, V = principal_stresses(_stresses(50))
    Dp = np.zeros((50, 6, 6))
    Dp[:, :3, :3] = mat._elastic_principal()
    Dp[:, 3:, 3:] = mat.shear_modulus * np.eye(3)
    T = rotation_to_global(V)
    assert np.allclose(T @ Dp @ T.transpose(0, 2, 1), mat.elastic(), atol=1e-9 * mat.E)


def test_returned_stresses_lie_on_the_surface():
    mat = MohrCoulomb("s", E=1e5, nu=0.3, c=10.0, phi=30.0, psi=5.0)
    state = MaterialState.zeros(400)
    state.stress[:] = [-100, -120, -80, 0, 0, 0]
    stress, _, new = mat.update(state, rng.normal(size=(400, 6)) * 2e-3)
    f = mat.yield_function(stress)
    assert new.yielding.sum() > 100
    assert np.all(f <= 1e-8 * mat.E)
    on_shear = np.isclose(f[new.yielding], 0.0, atol=1e-8 * mat.E)
    on_cutoff = principal_values(stress[new.yielding])[:, 0] >= -1e-6
    assert np.all(on_shear | on_cutoff)
    assert np.all(principal_values(stress)[:, 0] <= 1e-6)             # no tension (kPa)


@pytest.mark.parametrize("cutoff", [None, 0.0])
def test_consistent_tangent_matches_a_numerical_derivative(cutoff):
    """The tangent is exact on faces and edges.

    ``residual_stiffness`` is set to zero here: it deliberately keeps a
    thousandth of the shear stiffness on an edge return, where the exact
    value is zero, so that the global matrix stays invertible.
    """
    mat = MohrCoulomb("s", E=1e5, nu=0.3, c=10.0, phi=30.0, psi=5.0,
                      tension_cutoff=cutoff, residual_stiffness=0.0)
    state = MaterialState.zeros(300)
    state.stress[:] = [-100, -120, -80, 0, 0, 0]
    de = rng.normal(size=(300, 6)) * 2e-3
    _, D, new = mat.update(state, de)
    h = 1e-7
    errors = []
    for i in np.nonzero(new.yielding)[0]:
        sub = state.take([i])
        Dn = np.empty((6, 6))
        for k in range(6):
            e = np.zeros(6)
            e[k] = h
            Dn[:, k] = (mat.update(sub, (de[i] + e)[None])[0] - mat.update(sub, (de[i] - e)[None])[0])[0] / (2 * h)
        errors.append(np.abs(Dn - D[i]).max() / mat.E)
    errors = np.array(errors)
    assert len(errors) > 100
    # a point whose finite difference straddles a corner of the surface can
    # disagree; everywhere else the tangent is exact to round-off
    assert np.quantile(errors, 0.98) < 1e-7


def test_elastic_step_returns_the_elastic_matrix():
    mat = MohrCoulomb("s", E=5e4, nu=0.25, c=50.0, phi=35.0)
    state = MaterialState.zeros(10)
    state.stress[:] = [-100, -100, -100, 0, 0, 0]
    stress, D, new = mat.update(state, 1e-5 * rng.normal(size=(10, 6)))
    assert not new.yielding.any()
    assert np.allclose(D, mat.elastic())


def _triaxial(mat, confining, steps=60, axial_strain=-0.02):
    """Strain-driven triaxial compression: axial strain imposed, lateral stress held.

    The lateral strains are found by Newton iteration on the material's own
    tangent, which also exercises it.
    """
    state = MaterialState.zeros(1)
    state.stress[:] = [-confining, -confining, -confining, 0, 0, 0]
    d_axial = axial_strain / steps
    lateral = [0, 1]
    for _ in range(steps):
        de = np.zeros(6)
        de[2] = d_axial
        for _ in range(30):
            s, D, new = mat.update(state, de[None])
            r = s[0, lateral] + confining
            if np.abs(r).max() < 1e-9 * confining:
                break
            de[lateral] -= np.linalg.solve(D[0][np.ix_(lateral, lateral)], r)
        state = new
    return state.stress[0]


def test_triaxial_strength_is_the_mohr_coulomb_value():
    c, phi, s3 = 10.0, 30.0, 100.0
    mat = MohrCoulomb("s", E=3e4, nu=0.3, c=c, phi=phi, psi=0.0)
    s = _triaxial(mat, s3)
    kp = (1 + math.sin(math.radians(phi))) / (1 - math.sin(math.radians(phi)))
    assert -s[2] == pytest.approx(s3 * kp + 2 * c * math.sqrt(kp), rel=1e-6)
    assert s[0] == pytest.approx(-s3, rel=1e-9) and s[1] == pytest.approx(-s3, rel=1e-9)


def test_strength_reduction_divides_c_and_tan_phi():
    mat = MohrCoulomb("s", E=1e5, nu=0.3, c=12.0, phi=32.0, psi=8.0)
    r = mat.reduced(1.6)
    assert r.c == pytest.approx(12.0 / 1.6)
    assert math.tan(math.radians(r.phi)) == pytest.approx(math.tan(math.radians(32.0)) / 1.6)
    assert r.psi <= r.phi and r.E == mat.E
    elastic = LinearElastic("e", 1e4, 0.3)
    assert elastic.reduced(2.0) is elastic


def test_k0_defaults():
    assert MohrCoulomb("s", E=1e4, nu=0.3, phi=30.0).k0 == pytest.approx(0.5)
    assert MohrCoulomb("s", E=1e4, nu=0.3, phi=30.0, K0=0.8).k0 == 0.8
    assert LinearElastic("e", 1e4, 0.25).k0 == pytest.approx(1 / 3)


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError):
        MohrCoulomb("s", E=1e4, nu=0.3, phi=20.0, psi=25.0)
    with pytest.raises(ValueError):
        MohrCoulomb("s", E=1e4, nu=0.5)
