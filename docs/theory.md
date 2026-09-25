# Formulation

Units are kN, m and kPa, as in Lythos. `z` points up and gravity acts in
`-z`. Stresses are **tension positive** and stored as the six components
`[σxx, σyy, σzz, τxy, τyz, τzx]`. Strains use the same order, with
engineering shear strains `γxy = ∂u/∂y + ∂v/∂x`, so that `σ = D ε` with the
usual isotropic `D`. The 2D program carried `σzz` explicitly for plane
strain; in 3D it is simply one of the six.

## The element

Continuum elements are **10-node quadratic tetrahedra**. The reason is the
one given for 6-node triangles in Lythos: linear tetrahedra lock under
isochoric plastic flow, and the collapse load, and with it the factor of
safety, would come out too high.

Nodes are ordered as in VTK's quadratic tetrahedron: corners 0 to 3, then
the mid-edge nodes of edges (0,1), (1,2), (0,2), (0,3), (1,3) and (2,3). With
barycentric coordinates `L0..L3` the shape functions are

    corner i:     Ni = Li (2 Li − 1)
    edge (i, j):  N  = 4 Li Lj

Integration uses the 4-point Gauss rule, which is exact to degree 2. On a
straight-sided element `B` is linear, so `BᵀDB` is quadratic and the
stiffness is integrated exactly. So are the consistent body forces, where the
shape functions are quadratic and the load is constant.

The element stiffness is formed as a single product per element. `B` at the
four Gauss points is scaled by `√(det J · w)` and stacked into one 24 × 30
operator, so `K = B̃ᵀ D̃ B̃`. That takes one (30 × 24)(24 × 30) product where
the direct sum takes four (30 × 6)(6 × 30) products. Batched matrix products
in NumPy cost per product, not per flop, so this is about twenty times faster.

## Assembly

A 3D model assembled through COO triplets means sorting and merging tens of
millions of entries every time the matrix is formed. In a Newton iteration
that is every iteration. The matrix structure is therefore computed once, by
`SparsityPattern`:

1. Node pairs sharing an element give a node-level pattern. That is nine
   times fewer pairs than the dof-level pattern.
2. Each node pair is expanded into a 3 × 3 block. Dof row `3a + i` starts at
   `9·start[a] + 3i·deg(a)`, where `start` is the node-level row pointer and
   `deg(a)` the number of neighbours of node `a`.
3. The position in that structure of every entry of every element matrix is
   stored once.

Forming the matrix is then a single `bincount` over the element matrices.
Leaving out elements, as excavation will, only drops their rows from the
`bincount`.

## Restraints and the linear solver

Restrained degrees of freedom are not sliced out of the matrix. Their rows
and columns are zeroed in place, and a diagonal entry scaled to the matrix is
set. This keeps the matrix symmetric and avoids the two full copies that
boolean slicing of a sparse matrix makes.

Systems are solved by PARDISO (`pypardiso`) when it is available. A
symmetric positive definite (elastic) matrix is factorised by Cholesky from
its upper triangle (`mtype = 2`), and anything else by LU (`mtype = 11`).
SuperLU is the fallback.

A singular stiffness, from a model that is not restrained against rigid body
motion, is almost never reported as singular. Round-off makes it factorisable
and the solution is a rigid motion of 10¹⁰ m. Such a system cannot balance a
general load, though, so every solution is checked against its residual
(`‖Ku − f‖ ≤ 10⁻⁶ ‖f‖`) and refused if it fails.

## Stress recovery

Gauss point stresses are extrapolated to the corners through the linear
field they define. This is exact for the linear stress field of a
straight-sided quadratic element. Mid-edge nodes take the mean of their two
corners, and each node then averages over the elements that share it.

## The structured mesh

`box_mesh` cuts each hexahedral grid cell into the six tetrahedra of the
Kuhn subdivision, one for each ordering of the axes, all around the main
diagonal. Every cell is cut the same way, so the diagonals on shared faces
agree and the mesh is conforming. The subdivision is symmetric under any
permutation of the axes. It is not symmetric under reflection, though,
because every cell's diagonal runs the same way. A symmetric problem therefore
gives an exactly symmetric answer under swapping `x` and `y`, but a mirror
image differs by discretisation error.
