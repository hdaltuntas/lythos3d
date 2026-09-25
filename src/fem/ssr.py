from .solver import FEMSolver
from ..materials.soil import MohrCoulomb

class SSRAnalysis:
    """Shear Strength Reduction (SSR) analysis for slope stability."""
    def __init__(self, mesh, initial_srf=1.0, srf_increment=0.1, max_iterations=50):
        self.mesh = mesh
        self.current_srf = initial_srf
        self.srf_increment = srf_increment
        self.max_iterations = max_iterations
        self.fos = None # Factor of Safety

    def update_mesh_materials(self, srf):
        """Updates the materials in the mesh with reduced strength parameters."""
        print(f"Updating materials for SRF = {srf:.2f}")
        for element_id, element in self.mesh.elements.items():
            if isinstance(element.material, MohrCoulomb):
                # We need a way to store the original material if we want to reset,
                # but for now we just create a reduced one on the fly.
                # In a full implementation, the mesh would store original properties.
                # For this template, we assume element.material is the original.
                reduced_material = element.material.get_reduced_strength(srf)
                # In a real solver, we'd pass this reduced material to the element for K assembly
                pass

    def check_convergence(self, solver):
        """Dummy convergence check. In reality, checks if the non-linear solver converged."""
        # For this template, we simulate non-convergence randomly or after a few steps
        # We will just say it fails if SRF > 1.5 for demonstration
        if self.current_srf > 1.5:
            return False
        return True

    def run(self):
        """Executes the SSR loop."""
        print("Starting SSR Analysis...")
        for i in range(self.max_iterations):
            print(f"\n--- SSR Iteration {i+1} ---")

            # 1. Update materials with current SRF
            self.update_mesh_materials(self.current_srf)

            # 2. Setup solver
            solver = FEMSolver(self.mesh)

            # 3. Assemble and solve (would be non-linear Newton-Raphson in reality)
            solver.assemble_stiffness()
            solver.apply_boundary_conditions()
            try:
                solver.solve()
                converged = self.check_convergence(solver)
            except Exception as e:
                print(f"Solver failed: {e}")
                converged = False

            # 4. Check results
            if converged:
                print(f"Model converged for SRF = {self.current_srf:.2f}")
                self.current_srf += self.srf_increment
            else:
                print(f"Model failed to converge at SRF = {self.current_srf:.2f}")
                self.fos = self.current_srf - self.srf_increment
                break

        if self.fos is not None:
            print(f"\nSSR Analysis complete. Factor of Safety (FoS) ~ {self.fos:.2f}")
        else:
            print("\nSSR Analysis reached max iterations without failure.")

        return self.fos
