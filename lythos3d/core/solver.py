"""Non-linear solver: staged construction, plastic analysis and strength reduction.

This is the 2D Lythos solver carried into three dimensions, without the
structural elements for now: Newton-Raphson on the consistent tangent, load
increments that halve when one will not converge, a backtracking line search,
and strength reduction by marching the reduction factor up until equilibrium
is lost and then bisecting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .assembly import LinearSolver, assemble_vector, solve_constrained
from .materials import MaterialState
from .problem import INITIAL, SSR, Problem, Stage


@dataclass
class IterationLog:
    increment: int
    iteration: int
    residual: float


@dataclass
class StageResult:
    """Everything the post-processor needs from one construction stage."""

    name: str
    kind: str
    converged: bool
    #: displacement since the last reset, (n, 3)
    displacement: np.ndarray
    state: MaterialState
    active: np.ndarray
    srf: float | None = None
    #: (reduction factor, maximum displacement) for every strength reduction trial
    srf_curve: list[tuple[float, float]] = field(default_factory=list)
    iterations: list[IterationLog] = field(default_factory=list)
    message: str = ""
    seconds: float = 0.0
    plastic_fraction: float = 0.0

    @property
    def max_displacement(self) -> float:
        return float(np.linalg.norm(self.displacement, axis=1).max()) if len(self.displacement) else 0.0

    @property
    def factor_of_safety(self) -> float | None:
        return self.srf


class Solver:
    """Drives a :class:`Problem` through its construction stages."""

    def __init__(self, problem: Problem, tolerance: float = 1.0e-3, max_iterations: int = 40,
                 backend: str = "auto", verbose: bool = False):
        self.p = problem
        self.tol = tolerance
        self.max_iterations = max_iterations
        self.backend = backend
        self.linear = LinearSolver(backend)
        #: reuse a factorisation across iterations (and increments) while it
        #: keeps reducing the imbalance by at least ``1 / reuse_rate`` per step
        self.modified_newton = True
        self.reuse_rate = 0.3
        self.verbose = verbose
        self.materials = dict(problem.materials)
        #: a strength reduction trial that has taken this many times the
        #: iterations of the hardest successful one is taken to have failed
        self.ssr_budget_factor = 8
        self.ssr_min_budget = 200
        self.results: list[StageResult] = []
        self._u = np.zeros(problem.n_dof)
        self._u_offset = np.zeros(problem.n_dof)
        self._state = MaterialState.zeros(problem.n_points)
        self._active = np.ones(problem.mesh.n_elements, bool)
        for name in problem.absent:
            self._active &= ~problem.groups[name]
        lo, hi = problem.mesh.bounds
        self._size = float(max(hi - lo))

    # ------------------------------------------------------------------ public
    def run(self, stages, progress=None) -> list[StageResult]:
        stages = list(stages)
        for stage in stages:
            self.p.check_stage_groups(stage)
        for i, stage in enumerate(stages):
            if progress:
                progress(i, len(stages), stage.name)
            self.results.append(self.run_stage(stage))
        return self.results

    def run_stage(self, stage: Stage) -> StageResult:
        t0 = time.perf_counter()
        for name in stage.excavate:
            self._active &= ~self.p.groups[name]
        for name in stage.construct:
            self._active |= self.p.groups[name]
        active = self._active.copy()

        if stage.kind == INITIAL:
            result = self._initial_stage(stage, active)
        elif stage.kind == SSR:
            result = self._ssr_stage(stage, active)
        else:
            if stage.reset_displacements:
                self._u_offset = self._u.copy()
            result = self._newton(stage, active, stage.increments, stage.name)
        result.seconds = time.perf_counter() - t0
        gp_active = np.repeat(active, self.p.continuum.n_gauss)
        result.plastic_fraction = float(np.mean(result.state.yielding[gp_active])) if active.any() else 0.0
        if self.verbose:
            print(f"  {stage.name}: {'converged' if result.converged else 'NOT converged'}"
                  f" in {result.seconds:.1f} s {result.message}", flush=True)
        return result

    # ----------------------------------------------------------------- stages
    def _initial_stage(self, stage: Stage, active: np.ndarray) -> StageResult:
        """Initial stresses, by the K0 procedure or by gravity loading.

        Either way the stage leaves no displacement behind: the ground is
        where it is, and what later stages report is movement from here.
        """
        self._u[:] = 0.0
        self._state = MaterialState.zeros(self.p.n_points)
        if stage.initial_stress == "k0":
            self._state.stress[:] = self.p.k0_stress(active)
            # a K0 field is in equilibrium only under level ground and level
            # strata; let the solver remove whatever imbalance is left
            increments = max(1, stage.increments // 2)
        else:
            increments = stage.increments
        res = self._newton(stage, active, increments, stage.name)
        self._u_offset = self._u.copy()
        res.displacement = np.zeros((self.p.mesh.n_nodes, 3))
        return res

    def _ssr_stage(self, stage: Stage, active: np.ndarray) -> StageResult:
        """Shear strength reduction: the largest factor that still equilibrates.

        Cohesion and tan(phi) are divided by a trial factor and the whole
        non-linear problem is solved again.  Past the factor of safety the
        deforming zone links up into a mechanism, displacements run away and
        no equilibrium exists.  Each trial starts from the last converged one,
        so the displacement-versus-factor curve follows one loading path.
        """
        base_materials = dict(self.materials)
        reference = self._u.copy()
        anchor_u, anchor_state = self._u.copy(), self._state.copy()
        curve: list[tuple[float, float]] = []
        successful: list[int] = []

        def attempt(srf: float) -> bool:
            self.materials = {r: m.reduced(srf) for r, m in base_materials.items()}
            self._u, self._state = anchor_u.copy(), anchor_state.copy()
            budget = max(self.ssr_min_budget, self.ssr_budget_factor * max(successful, default=25))
            started = time.perf_counter()
            res = self._newton(stage, active, max(stage.increments, 6), f"SRF {srf:.3f}", budget)
            if res.converged:
                successful.append(len(res.iterations))
            dmax = float(np.linalg.norm((self._u - reference).reshape(-1, 3), axis=1).max())
            curve.append((srf, dmax))
            if self.verbose:
                print(f"    SRF {srf:.3f}: {'equilibrium' if res.converged else 'no equilibrium'},"
                      f" {1000 * dmax:.1f} mm, {len(res.iterations)} iterations,"
                      f" {time.perf_counter() - started:.1f} s", flush=True)
            return res.converged

        # 1. march upwards until equilibrium is lost
        srf, step, last_ok, hi = max(stage.srf_min, 0.5), 0.2, None, stage.srf_max
        for _ in range(30):
            if attempt(srf):
                last_ok = srf
                anchor_u, anchor_state = self._u.copy(), self._state.copy()
                if srf >= stage.srf_max:
                    hi = srf
                    break
                step = min(1.5 * step, 0.5)
                srf = min(srf + step, stage.srf_max)
            else:
                hi = srf
                break

        # 2. bisect between the last stable and the first unstable factor
        if last_ok is None:
            fos = None
            message = "no equilibrium even at the lowest trial factor: unstable as modelled"
        else:
            lo = last_ok
            for _ in range(10):
                if hi - lo <= 0.01:
                    break
                mid = 0.5 * (lo + hi)
                if attempt(mid):
                    lo = mid
                    anchor_u, anchor_state = self._u.copy(), self._state.copy()
                else:
                    hi = mid
            fos = lo
            message = (f"factor of safety {fos:.3f}, bracketed within {hi - lo:.3f}"
                       if hi > lo else f"stable up to the largest factor tried, {fos:.2f}")

        self._u, self._state = anchor_u, anchor_state
        self.materials = base_materials
        return StageResult(name=stage.name, kind=SSR, converged=last_ok is not None,
                           displacement=(self._u - reference).reshape(-1, 3), state=self._state,
                           active=active, srf=fos, srf_curve=sorted(curve), message=message)

    # ----------------------------------------------------------------- newton
    def _newton(self, stage: Stage, active: np.ndarray, increments: int, label: str,
                budget: int | None = None) -> StageResult:
        p = self.p
        fixed = np.union1d(p.fixed, self._orphan_dofs(active))
        f_ext = p.gravity(active) + p.surface_loads(stage.loads)
        logs: list[IterationLog] = []

        u_committed = self._u.copy()
        state_committed = self._state.copy()
        f_int0 = self._internal(u_committed, u_committed, state_committed, active, False)[0]
        # a new stage has new restraints and elements; never start it on the
        # factorisation of the last one
        self.linear.forget()
        # Strength reduction trials run full Newton.  Near collapse a trial's
        # verdict depends on how the iterations are spent, and modified
        # Newton - more iterations, each cheaper - converges trials that full
        # Newton gives up on: it moved the benchmark slope from 1.438 to 1.459
        # with no change in the physics.  The factor of safety is kept on the
        # path it was verified on.
        modified = self.modified_newton and stage.kind != SSR

        # Adaptive stepping: an increment that will not converge is retried at
        # half the size, and once a size has failed the step never grows back
        # to it - walking into the same wall again is what makes a doomed
        # strength reduction trial cost ten times a successful one.
        lam, message, cuts = 0.0, "", 0
        dlam = ceiling = 1.0 / max(increments, 1)
        min_dlam = dlam / 64.0
        scale = max(np.linalg.norm(f_ext), np.linalg.norm(f_int0), 1e-8)

        while lam < 1.0 - 1e-10:
            if budget is not None and len(logs) > budget:
                message = f"gave up after {len(logs)} iterations at {lam * 100:.0f}% of the stage load"
                break
            trial = min(1.0, lam + dlam)
            target = f_int0 + trial * (f_ext - f_int0)
            u_try = u_committed.copy()
            ok, stalled = False, 0
            reused, previous = False, None
            evaluated = self._internal(u_try, u_committed, state_committed, active, True)
            for it in range(self.max_iterations):
                f_int, tangent, symmetric = evaluated
                r = target - f_int
                r[fixed] = 0.0
                rn = float(np.linalg.norm(r))
                logs.append(IterationLog(len(logs), it, rn / scale))
                if rn / scale < self.tol:
                    ok = True
                    break
                # Modified Newton: keep the last factorisation while it still
                # cuts the imbalance by a factor of three an iteration, and
                # factorise the current tangent when it stops doing so.
                refactor = (not modified or not self.linear.has_factorisation
                            or (reused and rn > self.reuse_rate * previous))
                try:
                    if refactor:
                        du = solve_constrained(tangent(), r, fixed, symmetric=symmetric,
                                               solver=self.linear)
                    else:
                        du = self.linear.resolve(r)
                        du[fixed] = 0.0
                except np.linalg.LinAlgError:
                    message = "singular stiffness matrix"
                    break
                reused, previous = not refactor, rn
                du = self._limit_step(du)
                u_try, improved, evaluated = self._line_search(u_try, du, u_committed, state_committed,
                                                               target, active, fixed, rn)
                if evaluated is None:
                    evaluated = self._internal(u_try, u_committed, state_committed, active, True)
                if not improved and reused:
                    # an old factorisation that no longer points downhill is
                    # not a failure of Newton: factorise afresh and go on
                    previous = 0.0
                    continue
                stalled = 0 if improved else stalled + 1
                if stalled >= 3:
                    message = "the Newton step stopped reducing the imbalance"
                    break
                if not np.all(np.isfinite(u_try)) or np.abs(u_try).max() > 1.0e3 * self._size:
                    message = "displacements ran away"
                    break

            if ok:
                state_committed = self._internal(u_try, u_committed, state_committed, active,
                                                 False, return_state=True)[2]
                u_committed = u_try
                lam = trial
                if it < 6 and dlam < ceiling:
                    dlam = min(2.0 * dlam, ceiling)
            else:
                ceiling = min(ceiling, 0.5 * dlam)
                dlam *= 0.5
                cuts += 1
                if dlam < min_dlam or cuts > 20:
                    message = message or f"no equilibrium beyond {lam * 100:.0f}% of the stage load"
                    break

        self._u, self._state = u_committed, state_committed
        return StageResult(name=label, kind=stage.kind, converged=lam >= 1.0 - 1e-10,
                           displacement=(self._u - self._u_offset).reshape(-1, 3),
                           state=self._state, active=active, iterations=logs, message=message)

    def _limit_step(self, du: np.ndarray) -> np.ndarray:
        """Cap a Newton step so one bad tangent cannot throw the solution away."""
        cap = 0.05 * self._size
        peak = float(np.abs(du).max()) if du.size else 0.0
        return du * (cap / peak) if peak > cap > 0.0 else du

    def _line_search(self, u, du, u_committed, state_committed, target, active, fixed, r0):
        """Backtracking line search.

        The full step is tried first and kept whenever it reduces the
        imbalance, so the extra stress updates are paid for only where the
        step misbehaves.  The full step is evaluated together with its tangent: it is accepted
        in most iterations, and then that evaluation is exactly what the next
        iteration needs, so it is handed back rather than repeated.  Returns
        ``(u, improved, evaluation at u or None)``.
        """
        def residual(f_int):
            r = target - f_int
            r[fixed] = 0.0
            return float(np.linalg.norm(r))

        evaluated = self._internal(u + du, u_committed, state_committed, active, True)
        full = residual(evaluated[0])
        if full <= r0 or r0 == 0.0:
            return u + du, True, evaluated
        best_alpha, best = 1.0, full
        for alpha in (0.5, 0.25, 0.1, 0.03):
            value = residual(self._internal(u + alpha * du, u_committed, state_committed,
                                            active, False)[0])
            if value < best:
                best_alpha, best = alpha, value
            if value < r0:
                return u + alpha * du, True, None
        return u + best_alpha * du, best < 1.5 * r0, (evaluated if best_alpha == 1.0 else None)

    # ------------------------------------------------------------- assembly
    def _internal(self, u, u_committed, state_committed, active, tangent: bool,
                  return_state: bool = False):
        """``(internal force, tangent or None, symmetric or new state)``.

        With ``tangent`` the second item is a function that forms the tangent
        matrix, called only when an iteration refactorises, and the third
        says whether it is symmetric - true while no point has yielded - so
        that PARDISO can use Cholesky.
        """
        p = self.p
        ce = p.continuum
        dstrain = ce.strains(u - u_committed)
        stress = np.empty((p.n_points, 6))
        tangents = np.empty((p.n_points, 6, 6)) if tangent else None
        new_state = state_committed.copy() if return_state else None
        yielding = False
        for mat, gp in p.material_groups(self.materials):
            s, t, ns = mat.update(state_committed.take(gp), dstrain[gp])
            stress[gp] = s
            if tangent:
                tangents[gp] = t
                yielding |= bool(ns.yielding.any())
            if return_state:
                new_state.put(gp, ns)

        fe = ce.internal_forces(stress)
        fe[~active] = 0.0
        f_int = assemble_vector(p.n_dof, p.dofs, fe)
        if tangent:
            # the matrix itself is formed only if this iteration factorises it
            def matrix():
                return p.pattern.assemble(ce.stiffness(tangents), active)
            return f_int, matrix, not yielding
        return f_int, None, new_state

    def _orphan_dofs(self, active: np.ndarray) -> np.ndarray:
        """Dofs of nodes that belong to no active element: held so the system stays regular."""
        live = np.zeros(self.p.mesh.n_nodes, bool)
        live[self.p.mesh.elements[active].ravel()] = True
        dead = np.nonzero(~live)[0]
        return (3 * dead[:, None] + np.arange(3)).ravel()
