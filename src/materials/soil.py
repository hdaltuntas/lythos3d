class Material:
    """Base class for all materials."""
    def __init__(self, name, density):
        self.name = name
        self.density = density

class Soil(Material):
    """Base class for soil materials."""
    def __init__(self, name, density, youngs_modulus, poissons_ratio):
        super().__init__(name, density)
        self.youngs_modulus = youngs_modulus
        self.poissons_ratio = poissons_ratio

class LinearElastic(Soil):
    """Linear elastic soil model."""
    def __init__(self, name, density, youngs_modulus, poissons_ratio):
        super().__init__(name, density, youngs_modulus, poissons_ratio)

class MohrCoulomb(Soil):
    """Mohr-Coulomb soil model with failure criteria."""
    def __init__(self, name, density, youngs_modulus, poissons_ratio, cohesion, friction_angle, dilation_angle):
        super().__init__(name, density, youngs_modulus, poissons_ratio)
        self.cohesion = cohesion
        self.friction_angle = friction_angle
        self.dilation_angle = dilation_angle

    def get_reduced_strength(self, srf):
        """Returns a new MohrCoulomb model with strength reduced by the Strength Reduction Factor (SRF)."""
        import numpy as np
        reduced_c = self.cohesion / srf
        # tan(phi_reduced) = tan(phi) / srf
        phi_rad = np.radians(self.friction_angle)
        reduced_phi_rad = np.arctan(np.tan(phi_rad) / srf)
        reduced_phi = np.degrees(reduced_phi_rad)

        return MohrCoulomb(
            name=f"{self.name}_SRF_{srf:.2f}",
            density=self.density,
            youngs_modulus=self.youngs_modulus,
            poissons_ratio=self.poissons_ratio,
            cohesion=reduced_c,
            friction_angle=reduced_phi,
            dilation_angle=self.dilation_angle # Typically dilation is not reduced in the same way, or handled specially
        )
