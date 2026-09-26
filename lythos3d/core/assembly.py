# SPDX-License-Identifier: AGPL-3.0-only
"""Global system assembly and the sparse linear solver.

A 3D model runs to hundreds of thousands of equations, which is out of reach
of SciPy's SuperLU in both time and memory.  Intel's PARDISO, through
:mod:`pypardiso`, factorises the same systems in parallel and is used whenever
it is installed.  It is only built for x86-64, so on other machines (Apple
silicon, ARM Linux) the solver falls back to SuperLU and says so once.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _import_pypardiso():
    """Import pypardiso, helping it find the MKL runtime if it has to.

    pypardiso looks for ``libmkl_rt`` under ``sys.prefix``.  A system-wide pip
    on Debian and Ubuntu installs it under ``/usr/local`` while ``sys.prefix``
    is ``/usr``, and the import then fails although MKL is present.
    """
    import glob
    import os
    import sys

    try:
        import pypardiso as module
        return module
    except (ImportError, OSError):      # OSError: an MKL library that will not load
        pass
    if "PYPARDISO_MKL_RT" not in os.environ:
        for root in (os.path.join(sys.prefix, "local"), sys.base_prefix,
                     os.path.join(sys.base_prefix, "local")):
            found = sorted(glob.glob(os.path.join(root, "lib*", "libmkl_rt.so*")), key=len)
            if found:
                os.environ["PYPARDISO_MKL_RT"] = found[0]
                for name in [m for m in sys.modules if m.startswith("pypardiso")]:
                    del sys.modules[name]
                try:
                    import pypardiso as module
                    return module
                except (ImportError, OSError):
                    del os.environ["PYPARDISO_MKL_RT"]
    return None


pypardiso = _import_pypardiso()

#: solver names accepted by :func:`solve`
BACKENDS = ("auto", "pardiso", "superlu")

_warned_fallback = False


def available_backends() -> list[str]:
    return (["pardiso"] if pypardiso is not None else []) + ["superlu"]


def default_backend() -> str:
    return "pardiso" if pypardiso is not None else "superlu"


def _resolve_backend(backend: str) -> str:
    global _warned_fallback
    if backend not in BACKENDS:
        raise ValueError(f"unknown solver backend {backend!r}; choose from {BACKENDS}")
    if backend == "auto":
        if pypardiso is None and not _warned_fallback:
            warnings.warn("pypardiso is not installed; using SciPy's SuperLU, which is much "
                          "slower on 3D models", RuntimeWarning, stacklevel=3)
            _warned_fallback = True
        return default_backend()
    if backend == "pardiso" and pypardiso is None:
        raise RuntimeError("the pardiso backend needs pypardiso: pip install pypardiso")
    return backend


class _PardisoHandle:
    """One PARDISO instance for a sequence of matrices sharing a sparsity structure.

    PARDISO's symbolic analysis - the fill-reducing ordering - depends only on
    where the non-zeros are, and in a Newton iteration they never move.  So
    the analysis (phase 11) is run when the structure changes and every later
    matrix is only factorised and solved (phase 23): two to three times faster
    than analysing afresh each time, as ``pypardiso.spsolve`` does.
    """

    def __init__(self, mtype: int):
        self.solver = pypardiso.PyPardisoSolver(mtype=mtype)
        self._indptr = None
        self._indices = None
        self._matrix = None

    def solve(self, A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
        s = self.solver
        s._check_A(A)
        b = s._check_b(A, b)
        same = (self._indptr is not None and len(self._indptr) == len(A.indptr)
                and len(self._indices) == len(A.indices)
                and np.array_equal(self._indptr, A.indptr) and np.array_equal(self._indices, A.indices))
        if not same:
            if self._indptr is not None:
                s.free_memory(everything=True)
            s.set_phase(11)
            s._call_pardiso(A, b)
            self._indptr, self._indices = A.indptr.copy(), A.indices.copy()
        s.set_phase(23)
        self._matrix = A            # iterative refinement in a later re-solve needs the values
        return s._call_pardiso(A, b)

    def resolve(self, b: np.ndarray) -> np.ndarray:
        """Solve with the factorisation already held (phase 33)."""
        s = self.solver
        s.set_phase(33)
        return s._call_pardiso(self._matrix, s._check_b(self._matrix, b))

    def release(self) -> None:
        if self._indptr is not None:
            self.solver.free_memory(everything=True)
            self._indptr = self._indices = self._matrix = None


class LinearSolver:
    """Solves a sequence of sparse systems, reusing what can be reused.

    ``symmetric`` declares the matrix symmetric positive definite, as an
    elastic stiffness is: PARDISO then factorises its upper triangle by
    Cholesky, in about half the time and memory.  Otherwise
    ``structurally_symmetric`` - true of every finite element matrix, whose
    entry (i, j) exists exactly when (j, i) does, whatever the values - lets
    PARDISO skip the matching step it needs for a general unsymmetric matrix.
    """

    def __init__(self, backend: str = "auto"):
        self.backend = _resolve_backend(backend)
        self._handles: dict[int, _PardisoHandle] = {}
        self._last = None               # what factorised the last matrix: a handle or a SuperLU object

    def solve(self, A: sp.spmatrix, b: np.ndarray, symmetric: bool = False,
              structurally_symmetric: bool = False) -> np.ndarray:
        if self.backend == "superlu":
            self._last = spla.splu(sp.csc_matrix(A))
            return self._last.solve(np.asarray(b, dtype=float))
        mtype = 2 if symmetric else (1 if structurally_symmetric else 11)
        A = sp.triu(A, format="csr") if symmetric else sp.csr_matrix(A)
        if mtype not in self._handles:
            self._handles[mtype] = _PardisoHandle(mtype)
        self._last = self._handles[mtype]
        return self._last.solve(A, np.ascontiguousarray(b, dtype=float))

    @property
    def has_factorisation(self) -> bool:
        return self._last is not None

    def resolve(self, b: np.ndarray) -> np.ndarray:
        """Solve with the matrix factorised last, for a new right-hand side.

        A back-substitution only - a small fraction of the cost of a
        factorisation - which is what makes modified Newton pay.
        """
        if self._last is None:
            raise RuntimeError("nothing has been factorised yet")
        b = np.ascontiguousarray(b, dtype=float)
        if self.backend == "superlu":
            return self._last.solve(b)
        return self._last.resolve(b)

    def forget(self) -> None:
        """Mark the last factorisation as no longer describing the problem."""
        self._last = None

    def release(self) -> None:
        """Free the factorisations PARDISO holds outside Python's memory."""
        for handle in self._handles.values():
            handle.release()
        self._handles.clear()
        self._last = None

    def __del__(self):
        try:
            self.release()
        except Exception:                                  # pragma: no cover - interpreter shutdown
            pass


def solve(A: sp.spmatrix, b: np.ndarray, backend: str = "auto",
          symmetric: bool = False) -> np.ndarray:
    """Solve one sparse system ``A x = b`` (see :class:`LinearSolver`)."""
    solver = LinearSolver(backend)
    try:
        return solver.solve(A, b, symmetric)
    finally:
        solver.release()


class SparsityPattern:
    """The sparsity of a finite element matrix, built once and reused.

    Assembling through COO triplets sorts and merges tens of millions of
    entries every time the matrix is formed, which in a Newton iteration is
    every iteration.  Instead the compressed-row structure is found once,
    from the node connectivity, together with the position in it of every
    entry of every element matrix; forming the matrix is then a single
    ``bincount``.

    The pattern is built at node level (an order of magnitude fewer pairs
    than dof level) and expanded to ``dof_per_node`` x ``dof_per_node``
    blocks, dof ``d * node + k``.
    """

    def __init__(self, elements: np.ndarray, n_nodes: int, dof_per_node: int = 3):
        elements = np.asarray(elements, dtype=np.int64)
        ne, nen = elements.shape
        m = dof_per_node
        self.n_dof = m * n_nodes
        self.dof_per_node = m

        a = np.repeat(elements, nen, axis=1).ravel()           # row node of each node pair
        b = np.tile(elements, (1, nen)).ravel()                # column node
        keys, inverse = np.unique(a * n_nodes + b, return_inverse=True)
        inverse = inverse.ravel()
        row_node = keys // n_nodes
        col_node = keys % n_nodes
        degree = np.bincount(row_node, minlength=n_nodes)
        start = np.concatenate([[0], np.cumsum(degree)])       # node-level indptr

        # dof-level CSR: row m*a + i holds, for each neighbour b of a, the m
        # columns m*b + j; it starts at m*m*start[a] + i*m*degree[a]
        row_len = np.repeat(m * degree, m)
        self.indptr = np.concatenate([[0], np.cumsum(row_len)]).astype(np.int64)
        self.nnz = int(self.indptr[-1])
        index_type = np.int32 if self.nnz < 2**31 else np.int64
        # every dof row of node a lists the same columns: m for each neighbour
        blocks = (m * col_node[:, None] + np.arange(m)).reshape(-1)
        node_of_block = np.repeat(np.arange(n_nodes), m * degree)
        within = np.arange(len(blocks)) - m * start[node_of_block]
        base = m * m * start[node_of_block] + within
        indices = np.empty(self.nnz, dtype=np.int64)
        for i in range(m):
            indices[base + i * m * degree[node_of_block]] = blocks
        self.indices = indices.astype(index_type)
        self.indptr = self.indptr.astype(index_type)

        # position in data of every entry (e, m*p + i, m*q + j) of the element matrices
        rank = np.arange(len(keys)) - start[row_node]           # position of each pair in its node row
        pa = row_node[inverse].reshape(ne, nen, nen)
        pr = rank[inverse].reshape(ne, nen, nen)
        deg = degree[pa]
        i = np.arange(m)[None, None, :, None, None]
        j = np.arange(m)[None, None, None, None, :]
        pos = (m * m * start[pa][:, :, None, :, None] + i * m * deg[:, :, None, :, None]
               + m * pr[:, :, None, :, None] + j)
        self.scatter = pos.reshape(ne, nen * m, nen * m).astype(index_type)

    def assemble(self, Ke: np.ndarray, active: np.ndarray | None = None,
                 group_matrices=(), group_active=()) -> sp.csr_matrix:
        """Global matrix from element matrices ``Ke`` (ne, 30, 30).

        (The group arguments exist for :class:`CombinedPattern`'s interface;
        a plain pattern has no groups.)

        ``active`` optionally masks elements out, as excavation will.
        """
        if len(group_matrices):
            raise ValueError("this pattern has no structural groups")
        scatter = self.scatter if active is None else self.scatter[active]
        Ke = Ke if active is None else Ke[active]
        data = np.bincount(scatter.ravel(), weights=Ke.ravel(), minlength=self.nnz)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n_dof, self.n_dof))


class CombinedPattern:
    """A continuum pattern extended by structural elements with their own dofs.

    ``groups`` are dof maps (n_e, m) of further element sets - plates,
    bars - over a system of ``n_dof`` equations, of which the continuum's
    own come first.  The combined structure is the union of both, found
    once; the continuum's scatter indices are remapped into it, so forming
    the matrix remains a single ``bincount`` and its structure never changes,
    which is what lets PARDISO reuse its symbolic factorisation.
    """

    def __init__(self, continuum: SparsityPattern, groups: list[np.ndarray], n_dof: int):
        self.n_dof = n_dof
        c_rows = np.repeat(np.arange(continuum.n_dof, dtype=np.int64), np.diff(continuum.indptr))
        c_keys = c_rows * n_dof + continuum.indices.astype(np.int64)
        extra = [(np.repeat(g, g.shape[1], axis=1) * n_dof + np.tile(g, (1, g.shape[1]))).astype(np.int64)
                 for g in groups]
        keys = np.union1d(c_keys, np.concatenate([e.ravel() for e in extra])) if extra else c_keys
        self.nnz = len(keys)
        index_type = np.int32 if self.nnz < 2**31 else np.int64
        rows = keys // n_dof
        self.indices = (keys % n_dof).astype(index_type)
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n_dof))]).astype(index_type)
        remap = np.searchsorted(keys, c_keys)
        self.scatter = remap[continuum.scatter].astype(index_type)
        self.group_scatter = [np.searchsorted(keys, e).reshape(len(e), g.shape[1], g.shape[1]).astype(index_type)
                              for e, g in zip(extra, groups)]

    def assemble(self, Ke: np.ndarray, active: np.ndarray | None = None,
                 group_matrices=(), group_active=()) -> sp.csr_matrix:
        """Global matrix from continuum matrices ``Ke`` and, per group, its element
        matrices and an optional mask of the elements taking part."""
        parts_i = [self.scatter if active is None else self.scatter[active]]
        parts_w = [Ke if active is None else Ke[active]]
        for k, M in enumerate(group_matrices):
            mask = group_active[k] if k < len(group_active) else None
            parts_i.append(self.group_scatter[k] if mask is None else self.group_scatter[k][mask])
            parts_w.append(M if mask is None else M[mask])
        data = np.bincount(np.concatenate([p.ravel() for p in parts_i]),
                           weights=np.concatenate([w.ravel() for w in parts_w]), minlength=self.nnz)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n_dof, self.n_dof))


def assemble_vector(n_dof: int, dofs: np.ndarray, Fe: np.ndarray) -> np.ndarray:
    """Global vector from element vectors ``Fe`` (n, m) at ``dofs`` (n, m)."""
    return np.bincount(dofs.ravel(), weights=Fe.ravel(), minlength=n_dof)


def solve_constrained(K: sp.csr_matrix, f: np.ndarray, fixed: np.ndarray,
                      prescribed: np.ndarray | None = None,
                      backend: str = "auto", symmetric: bool = False,
                      solver: LinearSolver | None = None) -> np.ndarray:
    """Solve ``K u = f`` with ``fixed`` dofs held at ``prescribed`` values.

    ``K`` is a finite element matrix, structurally symmetric.  Pass a
    ``solver`` to reuse its factorisation data across calls, as a Newton
    iteration does.

    The restrained rows and columns are zeroed in place and given a unit
    diagonal (scaled to the matrix), rather than sliced out: slicing a sparse
    matrix by a boolean mask copies it twice, which on a large 3D model costs
    as much as the assembly.  Symmetry is preserved.
    """
    n = K.shape[0]
    free = np.ones(n, dtype=bool)
    free[fixed] = False
    u = np.zeros(n)
    if prescribed is not None and len(fixed):
        u[fixed] = prescribed
        f = f - K @ u
    if not free.any():
        return u
    K = sp.csr_matrix(K, copy=True)
    K.sum_duplicates()
    rows = np.repeat(np.arange(n), np.diff(K.indptr))
    cols = K.indices
    diagonal = rows == cols
    if np.count_nonzero(diagonal) < n:
        # make room for a diagonal entry in every row (explicit zeros survive)
        coo = K.tocoo()
        K = sp.coo_matrix((np.concatenate([coo.data, np.zeros(n)]),
                           (np.concatenate([coo.row, np.arange(n)]),
                            np.concatenate([coo.col, np.arange(n)]))), shape=K.shape).tocsr()
        rows = np.repeat(np.arange(n), np.diff(K.indptr))
        cols = K.indices
        diagonal = rows == cols
    d = K.data[diagonal & free[rows]]
    scale = float(np.mean(np.abs(d[d != 0]))) if np.any(d != 0) else 1.0
    K.data[~free[rows] | ~free[cols]] = 0.0
    K.data[diagonal & ~free[rows]] = scale
    rhs = np.array(f, dtype=float)
    rhs[~free] = scale * u[~free]
    if solver is None:
        u = solve(K, rhs, backend, symmetric)
    else:
        u = solver.solve(K, rhs, symmetric, structurally_symmetric=True)
    # A singular stiffness is rarely reported as such: round-off makes it
    # factorisable and the "solution" is a rigid body motion of 1e10 metres.
    # The residual gives it away, because such a system cannot balance the
    # load, and costs one product with K.
    residual = np.linalg.norm(K @ u - rhs) if np.all(np.isfinite(u)) else np.inf
    if residual > 1e-6 * max(np.linalg.norm(rhs), 1e-300):
        raise np.linalg.LinAlgError("the stiffness matrix is singular: the model is not "
                                    "restrained against rigid body motion")
    return u
