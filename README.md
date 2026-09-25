# lythos3d

Lythos3d is a 3D Finite Element Method (FEM) analysis program tailored for geotechnical engineering. It provides capabilities similar to Plaxis 3D or RS3, specifically focusing on slopes, embankments, and deep excavations.

## Features
- **3D Core Engine**: Handles nodes and finite elements.
- **Material Models**: Support for standard soil models (e.g. Linear Elastic, Mohr-Coulomb) and structural elements like piles.
- **FEM Solver**: Framework for assembling stiffness matrices and solving systems using SciPy.
- **Shear Strength Reduction (SSR)**: Calculates Factor of Safety for geotechnical stability.
- **3D Visualization**: Integration with PyVista for 3D visualization.
- **User Interface**: PyQt-based GUI to enter soil/pile parameters and trigger analysis.

## Project Structure
- `src/core/`: Mesh geometries and elements.
- `src/materials/`: Soil and structural material models.
- `src/fem/`: FEM solvers and the SSR framework.
- `src/gui/`: Application interface.
- `src/postprocess/`: PyVista rendering functions.
- `tests/`: Unit testing suite.
