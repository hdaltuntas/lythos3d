import pytest
import math
from src.materials.soil import MohrCoulomb
from src.materials.pile import Pile

def test_mohr_coulomb_reduction():
    soil = MohrCoulomb("Test Soil", density=2000, youngs_modulus=50e6, poissons_ratio=0.3,
                       cohesion=10000, friction_angle=30, dilation_angle=0)

    reduced_soil = soil.get_reduced_strength(srf=2.0)

    assert reduced_soil.cohesion == 5000

    # tan(phi_reduced) = tan(30) / 2
    expected_phi_rad = math.atan(math.tan(math.radians(30)) / 2)
    expected_phi_deg = math.degrees(expected_phi_rad)

    assert math.isclose(reduced_soil.friction_angle, expected_phi_deg, rel_tol=1e-5)

def test_pile_properties():
    # Pile with 1m diameter, 30 MPa concrete
    pile = Pile("Test Pile", density=2500, diameter=1.0, concrete_strength_fc=30)

    # Area = pi * (0.5)^2
    expected_area = math.pi * 0.25
    assert math.isclose(pile.area, expected_area, rel_tol=1e-5)

    # E = 4700 * sqrt(30) * 1e6
    expected_e = 4700 * math.sqrt(30) * 1e6
    assert math.isclose(pile.youngs_modulus, expected_e, rel_tol=1e-5)
