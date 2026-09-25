"""Linear analyses against closed-form solutions."""

import numpy as np
import pytest

from lythos3d.core.analysis import SurfaceLoad, box_fixities, linear_static
from lythos3d.core.materials import LinearElastic
from lythos3d.core.mesh import box_mesh, graded

SOIL = LinearElastic("clay", E=2.0e4, nu=0.3, gamma=19.0)


def _column(H=10.0, breaks=()):
    return box_mesh(graded(0, 3, 1.5), graded(0, 2, 1), graded(-H, 0, 1.0, breaks))


def test_geostatic_stress_and_settlement_under_self_weight():
    """Zero lateral strain: sigma_v = gamma z, sigma_h = K0 sigma_v, settlement gamma H^2 / 2M."""
    H = 10.0
    mesh = _column(H)
    r = linear_static(mesh, [SOIL])
    z = r.mesh.nodes[:, 2]
    depth = -z
    s = r.nodal_stress
    assert np.allclose(s[:, 2], -SOIL.gamma * depth, atol=1e-8)
    assert np.allclose(s[:, 0], SOIL.k0 * s[:, 2], atol=1e-8)
    assert np.allclose(s[:, 1], SOIL.k0 * s[:, 2], atol=1e-8)
    assert np.abs(s[:, 3:]).max() < 1e-8

    # w(z) = -gamma / M (H z_b - z_b^2 / 2), z_b measured up from the base
    zb = z + H
    expected = -SOIL.gamma / SOIL.oedometer_modulus * (H * zb - zb ** 2 / 2)
    assert np.allclose(r.displacement[:, 2], expected, atol=1e-12)
    assert np.abs(r.displacement[:, :2]).max() < 1e-12


def test_uniform_surcharge_gives_one_dimensional_compression():
    q, H = 50.0, 8.0
    mesh = _column(H)
    r = linear_static(mesh, [SOIL], gravity=False,
                      loads=[SurfaceLoad("z", 0.0, (0.0, 0.0, -q))])
    top = np.isclose(mesh.nodes[:, 2], 0.0)
    assert np.allclose(r.displacement[top, 2], -q * H / SOIL.oedometer_modulus, rtol=1e-10)
    assert np.allclose(r.stress[:, 2], -q, atol=1e-9)


def test_layered_column_settles_by_the_sum_of_its_layers():
    top_layer = LinearElastic("sand", E=5.0e4, nu=0.25, gamma=18.0)
    bottom_layer = LinearElastic("clay", E=1.0e4, nu=0.35, gamma=20.0)
    h1, h2 = 3.0, 5.0
    mesh = _column(h1 + h2, breaks=[-h1])
    mesh.assign_regions(lambda c: np.where(c[:, 2] > -h1, 0, 1))
    r = linear_static(mesh, [top_layer, bottom_layer])

    top = np.isclose(mesh.nodes[:, 2], 0.0)
    g1, g2 = top_layer.gamma, bottom_layer.gamma
    M1, M2 = top_layer.oedometer_modulus, bottom_layer.oedometer_modulus
    expected = -(g1 * h1 ** 2 / 2 / M1 + (g1 * h1 * h2 + g2 * h2 ** 2 / 2) / M2)
    assert np.allclose(r.displacement[top, 2], expected, rtol=1e-10)
    base = np.isclose(mesh.nodes[:, 2], -(h1 + h2))
    assert np.allclose(r.nodal_stress[base, 2], -(g1 * h1 + g2 * h2), rtol=1e-10)


def test_missing_material_is_reported():
    mesh = _column(4.0)
    mesh.region[:] = 3
    with pytest.raises(ValueError, match=r"region\(s\) \[3\]"):
        linear_static(mesh, [SOIL])


def test_cantilever_tip_deflection():
    """A tip-loaded cantilever against Timoshenko beam theory, within 1%.

    Unlike the column tests the exact solution is not in the element's
    polynomial space, so this exercises bending on a coarse mesh: two
    quadratic tetrahedra across the depth.
    """
    L, b, h, P = 10.0, 1.0, 1.0, 1.0
    mat = LinearElastic("steel-ish", E=1.0e6, nu=0.0)
    mesh = box_mesh(graded(0, L, 0.5), graded(0, b, 0.5), graded(0, h, 0.5))
    x = mesh.nodes
    clamped = np.nonzero(np.isclose(x[:, 0], 0.0))[0]
    fixed = (3 * clamped[:, None] + np.arange(3)).ravel()
    r = linear_static(mesh, [mat], gravity=False, fixed=fixed,
                      loads=[SurfaceLoad("x", L, (0.0, 0.0, -P / (b * h)))])
    tip = np.isclose(x[:, 0], L)
    deflection = -r.displacement[tip, 2].mean()
    inertia, G = b * h ** 3 / 12, mat.E / 2
    expected = P * L ** 3 / (3 * mat.E * inertia) + P * L / (5 / 6 * G * b * h)
    assert deflection == pytest.approx(expected, rel=0.01)


def test_footing_load_is_carried_symmetrically():
    B, q = 2.0, 100.0
    mesh = box_mesh(graded(-6, 6, 1), graded(-6, 6, 1), graded(-8, 0, 1))
    footing = SurfaceLoad("z", 0.0, (0.0, 0.0, -q),
                          where=lambda c: (np.abs(c[:, 0]) < B / 2) & (np.abs(c[:, 1]) < B / 2))
    r = linear_static(mesh, [SOIL], gravity=False, loads=[footing])
    x, w = mesh.nodes, r.displacement[:, 2]
    centre = np.nonzero(np.all(np.isclose(x, 0.0), axis=1))[0][0]
    assert w[centre] == pytest.approx(w.min())
    idx = {tuple(np.round(p, 9)): i for i, p in enumerate(x)}

    def image(transform):
        return np.array([idx[tuple(np.round(transform(p), 9))] for p in x])

    # every cell is cut into the six tetrahedra of all axis orderings, so the
    # mesh - and the answer - is exactly symmetric under swapping x and y
    assert np.allclose(w, w[image(lambda p: p[[1, 0, 2]])], atol=1e-12 * abs(w.min()))
    # but the cells are all cut along the same diagonal, so mirror symmetry
    # holds only to discretisation accuracy; the worst of it, under 3% of the
    # peak settlement on this 1 m mesh, is at the corners of the footing
    # where the traction jumps from q to zero inside an element
    for mirror in (np.array([-1, 1, 1]), np.array([1, -1, 1])):
        assert np.abs(w - w[image(lambda p: p * mirror)]).max() < 0.05 * abs(w.min())
    # the base reactions carry the load
    assert r.stress.reshape(-1, 4, 6)[:, :, 2].min() < 0


def test_box_fixities_restrain_sides_normal_only():
    mesh = _column(3.0)
    fixed = set(box_fixities(mesh).tolist())
    lo, hi = mesh.bounds
    for n, p in enumerate(mesh.nodes):
        on_x = np.isclose(p[0], lo[0]) or np.isclose(p[0], hi[0])
        on_y = np.isclose(p[1], lo[1]) or np.isclose(p[1], hi[1])
        on_base = np.isclose(p[2], lo[2])
        assert (3 * n in fixed) == (on_x or on_base)
        assert (3 * n + 1 in fixed) == (on_y or on_base)
        assert (3 * n + 2 in fixed) == on_base
