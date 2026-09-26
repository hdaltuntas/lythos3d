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

## Ground from boreholes

A borehole lists the soils it meets and the level at which each starts. The
top of every soil is interpolated between boreholes:

- **One borehole**: every surface is level.
- **Collinear boreholes**: linear along the line, constant across it.
- **Three or more**: linear over the Delaunay triangulation of the borehole
  positions. Outside their convex hull a surface takes its value at the
  nearest point of the hull. Plain nearest-borehole extrapolation would put
  a step in the surface at the hull edge; this keeps it continuous.

A soil a borehole does not meet has zero thickness there: its top is set to
the top of the next soil down. Between boreholes the layer thins out to that
point, as a lens does. Surfaces are then forced not to cross
(`top_k = min(top_k, top_{k-1})`) and not to go below the base.

The same profile gives the overburden for the K0 procedure:
`σv = Σ γ_i · (thickness of soil i above the point)`. The initial stresses
and the meshed geometry therefore come from one description. On dipping
strata a K0 field is not quite in equilibrium, and the first stage's Newton
iterations remove the difference.

## Meshing a site

gmsh's OpenCASCADE kernel builds the solid the way the ground is described:

1. A box over the plan extent, from the base to above the highest ground.
2. The ground surface, extruded upwards, is cut away from it.
3. The soil interfaces fragment what remains.
4. Each lift of each excavation is a prism, its plan polygon between two
   levels. It is clipped to the ground and fragmented in.

Every surface is a loft through splines sampled from the soil profile. A
plane is reproduced exactly; an interpolated surface is reproduced to within
the smoothing of its kinks, which amounts to 0.02% of the layer volumes on
the test site.

**Labelling.** After meshing, every gmsh volume takes the soil and the lift
that most of its elements' centroids fall in. Its own centroid would not do:
a ring of soil round a pit has its centroid inside the pit.

**Sizing.** The global size applies everywhere, with finer boxes round each
excavation. Two things matter for quality:

- **Short edges.** A dipping layer that passes close to a pit corner, or a
  layer pinching out against the side of the model, leaves edges a few
  millimetres long. The mesh has to resolve them, and without grading the
  elements beside them are slivers (radius ratio 10⁻⁴ on the test site).
  Every edge shorter than a quarter of the element size therefore gets a
  size field that starts at its own length and grows to the global size
  over three element lengths.
- **The 3D algorithm.** HXT, not gmsh's default Delaunay. On a half-metre
  slab of soil between a lift floor and a layer boundary, the default left
  a worst radius ratio of 0.01 and HXT 0.30. It runs on one thread, so a
  site always gives the same mesh.

What remains is geometric. A layer dipping at 5° that crosses a horizontal
pit floor leaves a 5° wedge of ground, and a wedge of angle α cannot be
filled with elements much better than about α. Those elements are harmless
to a direct solver. `Site.build` warns only about degenerate elements.

## Modified Newton

A factorisation of a 100 000-equation tangent costs seconds; a
back-substitution with it costs a small fraction of that. In a construction
stage the tangent changes little from one iteration to the next, since most
of the ground stays elastic. So the solver keeps the last factorisation while
each iteration still cuts the imbalance by at least a factor of three. When
that stops, or when the old factorisation no longer gives a descent
direction, it factorises the current tangent. Convergence is always judged
on the true residual, so the answer is unchanged. On the test site the
initial stage took 8 s instead of 44 s, and the first lift 10 s instead of
41 s.

Strength reduction trials still run full Newton. Near collapse, whether a
trial converges depends on how its iterations are spent: modified Newton
converged trials that full Newton gave up on, and moved the benchmark slope
from 1.438 to 1.459 with no change in the physics. The factor of safety is
kept on the path it was verified on.

## Plates

Walls and rafts are 6-node flat shells on faces of the 10-node tetrahedra,
so they share nodes with the soil. Each node keeps its three translations
and gains three rotations about the global axes, numbered after all the
translations. In the element's own frame:

- **Membrane:** plane stress, `N = D_m ε_m`.
- **Bending:** Reissner–Mindlin, `u(z) = u + zθ_y`, `v(z) = v − zθ_x`,
  `κ = [∂θ_y/∂x, −∂θ_x/∂y, ∂θ_y/∂y − ∂θ_x/∂x]`.
- **Transverse shear:** `γ = [∂w/∂x + θ_y, ∂w/∂y − θ_x]`, with κ = 5/6.
- **Drilling:** the rotation about the normal is tied to the membrane's
  in-plane rotation `(∂v/∂x − ∂u/∂y)/2` by a penalty of `10⁻³ G t`
  (Hughes–Brezzi). Tying it rather than springing it to zero leaves a rigid
  rotation free of energy.

**Integration and spurious modes.** Membrane and bending are integrated by
the 3-point rule, which is exact on a flat element. Shear and drilling are
quartic and need the 6-point rule. Under the 3-point rule the drilling term
loses rank: the 18 in-plane dofs, less 3 rigid motions, need 15 independent
modes, while 3 points give the membrane at most 9 and drilling at most 3. The
element then had nine zero-energy modes instead of six. With exact
integration it has exactly six, which is tested.

**Shear locking.** The shear stiffness is scaled by `t²/(t² + a h²)`
(Lyly, Stenberg & Vihinen), with `h` the longest edge. `a = 0.03` is
calibrated on a simply supported square plate with `t/a = 0.005`:

| `a` | 4 × 4 mesh | 12 × 12 mesh |
| --- | --- | --- |
| 0 | −19% (locked) | −2% |
| 0.03 | −3% | +0.5% |
| 0.1 | +10% | +3% |

Cantilever strips from `L/t = 10` to `1000` are within 0.3% of beam theory.

**In the ground.** A plate installed at a stage stores the displacement at
that moment, and its forces follow `K (u − u_installed)`: it is built
stress-free into ground that has already moved. Until then its rotations
are held. The system matrix is the union of the soil pattern and the plates'
and bars' dof pairs, found once (`CombinedPattern`). Assembly stays a single
`bincount`, and PARDISO's symbolic factorisation is still reused.

**Against 2D.** A 3D slice held in plane strain with a cantilever wall
retaining a 3 m cut, compared with 2D Lythos given the plane-strain plate
stiffness `E t³/12(1 − ν²)`:

| Element size | Max moment 2D / 3D (kNm/m) | Top displacement 2D / 3D (mm) |
| --- | --- | --- |
| 1.0 m | 9.04 / 7.92 | 1.15 / 1.09 |
| 0.5 m | 9.83 / 9.65 | 1.22 / 1.21 |
| 0.25 m | 10.62 / 10.40 | 1.32 / 1.28 |

The coarse 3D moment is low because it is read at Gauss points inside the
elements and misses the peak. From 0.5 m the two agree within 2% in moment
and 3% in displacement, and refine together.

## Anchors and struts

A bar joins two mesh nodes, or a node and a fixed point. In the stage that
installs it, it is a pair of jack forces `P₀` pulling its ends together. At
the end of that stage it is locked off, and from then on it carries
`N = P₀ + EA/L · (extension − extension at lock-off)`, with stiffness
`EA/L e eᵀ`. A strut is a bar with `P₀ = 0`.

## Walls and anchors on a site

A wall drawn in plan is a polyline with a toe level and, optionally, a top
level (by default the ground). Each segment becomes a vertical rectangle,
clipped to the ground by intersecting it with the soil solid, and is
fragmented into the geometry with the soil interfaces and the lifts. A wall
along the edge of a pit coincides with the lift's side, and the fragment
merges the two surfaces. After meshing, the wall's triangles are taken from
the surfaces the fragment made of it, and their mid-edge nodes are looked up
on the tetrahedra's edges. The plate is therefore conforming by
construction, which is tested.

Two things need care:

- **A wall that stops short of the base** does not divide the soil. It is
  embedded in a volume rather than bounding one, so the clean-up that
  removes leftover surfaces must spare it.
- **Anchor ends** are embedded as points. A point on a wall is embedded in
  the wall surface reliably. A point free in the soil should be embedded in
  its volume by the fragment, but OpenCASCADE's inside test can fail near a
  lofted soil surface: it did so 0.36 m below one, where the layer itself
  was reproduced to 2 mm. The point is then silently left out, and a
  missing node index of −1 picked the last node of the mesh. Now every such
  point's volume is found from a first mesh, by a barycentric test on its
  tetrahedra, and the point is embedded explicitly before meshing again. A
  point that still has no node is an error, never a silent substitute.

With no stages given, walls are installed after the initial stresses, and
each anchor is stressed right after the lift that exposes its head.

## Interfaces

A wall with an interface gets its own nodes. The tetrahedra touching it
are sorted onto its two sides by the wall's side function: for a wall in
the box model, the side of its plane; for a wall drawn in plan, the side of
the nearest segment, so a closed wall has the pit on one side. The soil on
the + side keeps the original nodes, and the wall and the − side soil get
new ones at the same places. An interface element joins each side to the
wall.

**Edges.** Where the soil runs on past an edge of the wall (below the toe
of a wall that stops short of the base, or past its end), splitting would
open a crack along the wall's plane. A node is therefore left shared wherever
splitting it would divide a face that is not the wall's own. In a
plane-strain slice this leaves exactly the toe line tied, which is tested.
At the ground surface there is no face beyond the wall, and nothing is
tied.

**The element.** The element has 12 nodes (6 on the soil face, 6 on the
wall), and the relative displacement is `u_soil − u_wall`, resolved along
the normal (pointing into the soil, so positive opens a gap) and in the
plane:

- elastic: `kn = E_oed / t_v`, `ks = G / t_v`, where `t_v` is a tenth of the
  local element size, as in 2D Lythos;
- sliding: `|τ| > c_i − σn tan φ_i`, where the shear traction is returned
  radially onto the limit (keeping its direction in the plane), with no
  dilation;
- open: `σn` above the tensile capacity (zero by default), with no traction.

The update is incremental, so unloading after slip is elastic. The tangent is
the exact one of the radial return, including the coupling
`∂τ/∂δn = −tan φ_i kn m`. A residual stiffness of 10⁻³ of the elastic one
is kept where the exact tangent has none, which is tested with it removed.
Integration uses the 6-point rule, exact for the quartic integrand; the
3-point rule gives rank 9 against 18 relative dofs.

**Before and at installation.** Until the wall is installed its
interfaces tie the two sides by a penalty a hundred times the contact
stiffness, so the ground is continuous; against ground with no wall the
difference is 10⁻⁴. At installation the tie's tractions are carried into
the contact, so the wall is built into ground that is already in
equilibrium. (2D Lythos clears them, and the soil stress on the wall line
is then briefly unbalanced.)

**Strength.** `c_i = R c'` and `tan φ_i = R tan φ'` from the soil the
element sits against, or a `c` and `φ` of the interface's own. Strength
reduction divides the interface strength along with the soil's. 2D
reduces only the soil, which leaves the wall friction at full strength in
the factor of safety.

**Convergence.** Contact points close to the limit switch between
sticking and sliding from one iteration to the next, and the iteration
stalls. As in 2D, once it slows (after four iterations, less than halving
the imbalance) every contact is held in its current state for the rest of
the increment. The next increment starts free again. On the plane-strain
wall this took the dig from no convergence in 380 iterations to 69.

**A bug the sliding block found.** Whether the tangent is symmetric, and
so whether PARDISO may use Cholesky, was decided only by whether any soil
had yielded. On elastic ground with a sliding interface the unsymmetric
contact tangent was factorised as if symmetric, and the solutions had
relative residuals of order one. The tangent now counts as unsymmetric
whenever a contact slides.

## Beams and embedded piles

**The beam.** A pile is a 3-node Timoshenko beam with six dofs per node.
In its own axes, the strains are axial `du1/dx`, torsion `dθ1/dx`,
curvatures `dθ2/dx` and `dθ3/dx`, and shears `du2/dx − θ3` and
`du3/dx + θ2`, with `D = diag(EA, GA2, GA3, GJ, EI2, EI3)`. Everything is
integrated by the 2-point rule. That is exact for the axial, torsion and
bending terms of a straight quadratic element, and a reduced integration of
shear that stops a slender beam locking. It gives twelve strain modes for
eighteen dofs less six rigid motions, and exactly six zero-energy modes,
which is tested. Cantilever deflection `PL³/3EI + PL/GA` is reproduced
exactly in any orientation.

**Embedding.** The beam's nodes are not mesh nodes. Every point where the
pile meets the soil is located in the mesh: its tetrahedron is found from
a k-d tree of element centroids and confirmed by a barycentric test. The
soil's displacement there is interpolated with the tetrahedron's shape
functions.

**Where it is tied: perimeter, not axis.** The first version tied the
pile to the soil along its axis, as PLAXIS's embedded beam does. Against the
same pile modelled with solid concrete elements in elastic ground, it
settled more than twice as much. Making the springs nearly rigid did not
help, and refining the mesh made it worse:

| Mesh | Solid pile | Axis-tied, rigid springs | Perimeter-tied |
| --- | --- | --- | --- |
| 2.0 m | 1.17 mm | 1.73 mm | 1.24 mm |
| 1.0 m | 1.22 mm | 1.79 mm | 1.28 mm |
| 0.7 m | 1.24 mm | 2.47 mm | 1.31 mm |

A pile tied on its axis puts its load into the continuum along a line. The
displacement under a line load is logarithmically singular, so the finer
the mesh the softer the pile, without limit. The pile is therefore tied at
its perimeter instead. At each of three stations per beam element, eight
points on the pile's surface move with the cross-section as a rigid disc,
`u + θ × r`. The tip is tied at seven points over its base. The load then
enters the ground over a cylinder of the pile's own size, and the result
converges: within 6% of the solid pile in settlement and 7–11% laterally,
at every mesh. The remainder is the springs' own give and the solid pile's
square section. The perimeter points also resist the pile's twist, which
an axis alone cannot. With the axis coupling, torsion was a zero-energy mode
and PARDISO reported a zero pivot.

**Springs.** They stand for a layer of the surrounding soil a tenth of the
pile's radius thick: `2πG R / 0.1R = 20πG` per metre along and across the
pile, and `G A / 0.1R` under the tip. Along the pile, the skin friction is
elastic up to its capacity `T_max` (kN per metre, varying linearly from
head to tip) and then slides. Across it, it is elastic. At the tip, it is
compression only, up to the end bearing capacity `F_max`, shared between
the tip's points. Tractions are incremental from the committed state, so
unloading is elastic. The axial capacity is therefore exactly
`∫T_max ds + F_max`, whatever the stiffness or the mesh, which is tested.

**Installation.** When a stage installs a pile, its nodes take the soil's
displacement where they are, and its springs start from there free of
force. Until then its dofs are held. A spring whose tetrahedron has been
excavated switches off with it.

**Strength reduction** leaves a pile's capacities as they are. They are
the engineer's numbers for the pile, not soil strengths.
