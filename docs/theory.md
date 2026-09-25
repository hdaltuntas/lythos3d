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

## Mohr-Coulomb in three dimensions

The constitutive update is the one 2D Lythos uses: an exact return mapping
in principal stress space (Clausen, Damkilde & Andersen, 2006). With the
principal stresses sorted `s1 ≥ s2 ≥ s3`, the Mohr-Coulomb pyramid and the
tension cut-off are planes. For any set of active planes the return is a small
linear system, and its tangent follows in closed form. The active set is
found by trying the candidates in turn: the main face, an edge, the cut-off,
their intersections, the apex. None of this depends on the number of
dimensions, and the code is carried over unchanged.

What 3D changes is the way into and out of principal space.

**Principal values.** The 2D code rotated in the plane. In 3D the stress is
a symmetric 3 × 3 tensor with an eigenproblem at every Gauss point, and the
eigenvectors are needed only where the point yields. Every point is
therefore screened first with principal values in closed form (the
trigonometric solution of the characteristic cubic), and `eigh` is called
only for points at or beyond the surface.

**Tangent.** The principal-space tangent `Dp` (3 × 3) is rotated into global
axes as `D = T D̃ Tᵀ`. Here `T` is the Voigt transformation built from the
principal directions, and `D̃` is the 6 × 6 principal-frame stiffness: `Dp` in
the normal block, and for each pair of principal axes `(a, b)` the shear
stiffness

    G_ab = μ (s_a − s_b) / (s_a^trial − s_b^trial)

which comes from the rotation of the principal axes. For an elastic step it
is the shear modulus `μ`. 2D has one such term; 3D has three. Leaving them
out costs Newton its quadratic convergence wherever the soil yields.

The tangent is verified against a finite-difference derivative of the
stress update. On the faces and edges of the surface it agrees to round-off
(`10⁻⁹ E`). `residual_stiffness` deliberately keeps a thousandth of `μ` where
the exact value is zero (on an edge, `s_a = s_b` after the return), so that
the global matrix stays invertible.

With non-associated flow (`ψ < φ`) the tangent is unsymmetric, and it is used
as it is. PARDISO factorises it by LU with `mtype = 1`, structurally
symmetric: every finite element matrix has entry `(i, j)` exactly when it
has `(j, i)`, whatever the values, which spares the weighted matching
needed for a general unsymmetric matrix. While no point has yielded the
tangent is symmetric positive definite, and Cholesky is used.

## Staged construction and strength reduction

The solver is the 2D one without the structural elements:

- **Newton-Raphson** on the consistent tangent. Each load increment is
  interpolated between the internal force at the start of the stage and the
  external load, so an excavation's unloading is applied gradually rather
  than all at once.
- **Adaptive sub-stepping.** An increment that will not converge is halved,
  and once a step size has failed the step never grows back to it.
- **Backtracking line search.** The full step is evaluated together with its
  tangent. It is accepted in most iterations, and then that evaluation is
  exactly what the next iteration needs, so it is handed on rather than
  repeated.
- **Symbolic factorisation reused.** The sparsity pattern never changes
  during an analysis. PARDISO's reordering (phase 11) therefore runs once,
  and every later iteration only factorises and solves (phase 23). That is
  2–3× faster per solve.
- **Excavation** switches elements off. Nodes left with no active element
  around them are held in place, so the system stays regular.
- **K0 procedure.** `σv` comes from the stratum profile (`Model.overburden`),
  not from integrating through the mesh. This is where borehole-defined
  strata will plug in, and it stays exact however the mesh is graded.
- **Strength reduction.** `c` and `tan φ` are divided by a trial factor, and
  the factor is marched up until equilibrium is lost, then bisected. Each
  trial starts from the last converged one. A trial that has taken eight
  times the iterations of the hardest successful one is judged to have
  failed. Tightening that budget, or the iteration limit per increment,
  makes the search faster but declares trials failed that would have
  converged: 20 iterations per increment instead of 40 moved the benchmark
  slope from 1.423 to 1.402. The limits are therefore kept where the answer
  no longer depends on them.
