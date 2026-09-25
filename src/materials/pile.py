from .soil import Material
import math

class Pile(Material):
    """Represents a pile structural element."""
    def __init__(self, name, density, diameter, concrete_strength_fc):
        super().__init__(name, density)
        self.diameter = diameter
        self.concrete_strength_fc = concrete_strength_fc # f'c in MPa

        # Estimate Young's modulus from f'c (ACI empirical formula: E = 4700 * sqrt(f'c))
        self.youngs_modulus = 4700 * math.sqrt(self.concrete_strength_fc) * 1e6 # Pa

        # Calculate cross-sectional area
        self.area = math.pi * (self.diameter / 2) ** 2

        # Calculate moment of inertia
        self.moment_of_inertia = (math.pi * self.diameter ** 4) / 64
