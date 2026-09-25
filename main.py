import sys

# Core modules
from src.core.mesh import Mesh, Node, Element3D
from src.materials.soil import MohrCoulomb
from src.materials.pile import Pile
from src.fem.ssr import SSRAnalysis
from src.postprocess.visualizer import Visualizer3D
from src.gui.main_window import MainWindow, QApplication

def build_dummy_model():
    """Builds a minimal test model for integration check."""
    mesh = Mesh()
    n1 = Node(1, 0, 0, 0)
    n2 = Node(2, 1, 0, 0)
    n3 = Node(3, 1, 1, 0)
    n4 = Node(4, 0, 1, 0)

    mesh.add_node(n1)
    mesh.add_node(n2)
    mesh.add_node(n3)
    mesh.add_node(n4)

    soil = MohrCoulomb("Clay", density=2000, youngs_modulus=50e6, poissons_ratio=0.3,
                       cohesion=10000, friction_angle=25, dilation_angle=0)

    el = Element3D(1, [n1, n2, n3, n4], material=soil)
    mesh.add_element(el)

    return mesh

def main():
    print("Initializing Lythos3D...")

    # Test headless backend
    mesh = build_dummy_model()
    ssr = SSRAnalysis(mesh, max_iterations=2)
    # Just a quick dry run to ensure imports and logic flow
    print("Dry run SSR...")
    ssr.run()

    viz = Visualizer3D(mesh)
    viz.plot_mesh()

    # Launch GUI
    print("Launching GUI...")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
