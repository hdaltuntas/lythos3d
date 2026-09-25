import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class FEMSolver:
    """Core FEM solver handling matrix assembly and solving."""
    def __init__(self, mesh):
        self.mesh = mesh
        self.num_dofs = len(mesh.nodes) * 3 # Assuming 3 DOFs per node
        self.K_global = None
        self.F_global = None
        self.U_global = None

    def assemble_stiffness(self):
        """Assembles the global stiffness matrix (dummy implementation)."""
        print("Assembling global stiffness matrix...")
        # In a real implementation, loop over elements, get local K, and add to global K
        # Here we just create a dummy sparse identity matrix for structural layout

        # We need a sparse matrix, typically Lil or Coo for assembly, CSR for solving
        self.K_global = sp.lil_matrix((self.num_dofs, self.num_dofs))
        self.F_global = np.zeros(self.num_dofs)

        # Dummy assembly: just setting diagonal to 1 to make it non-singular
        self.K_global.setdiag(1.0)

        # Convert to CSR for efficient solving
        self.K_global = self.K_global.tocsr()
        print("Assembly complete.")

    def apply_boundary_conditions(self):
        """Applies boundary conditions to K and F (dummy implementation)."""
        print("Applying boundary conditions...")
        pass

    def solve(self):
        """Solves the linear system KU = F."""
        if self.K_global is None:
            raise ValueError("Stiffness matrix not assembled.")

        print("Solving linear system...")
        self.U_global = spla.spsolve(self.K_global, self.F_global)
        print("Solution found.")
        return self.U_global
