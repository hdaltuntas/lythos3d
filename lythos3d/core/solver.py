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
    #: stress resultants of every installed plate: name -> (N, M, Q) as
    #: :meth:`PlateElements.resultants` returns them
    plate_forces: dict = field(default_factory=dict)
    #: axial force of every installed bar, kN, tension positive
    bar_forces: dict = field(default_factory=dict)
    #: per plate with an interface: element means of the normal traction
    #: ``tn`` (kPa, tension positive), the shear traction's size ``tau`` and
    #: vector ``shear`` (global axes, pointing the way the soil moves
    #: relative to the wall), and the fraction of the shear strength
    #: mobilised, for the elements in contact with soil
    interface_tractions: dict = field(default_factory=dict)

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
        self._state = self._fresh_state()
        self._srf = 1.0
        #: interface contact modes: those of the last evaluation, and those
        #: held fixed while an increment converges on them
        self._contact_modes = None
        self._contact_frozen = None
        self._active = np.ones(problem.mesh.n_elements, bool)
        for name in problem.absent:
            self._active &= ~problem.groups[name]
        lo, hi = problem.mesh.bounds
        self._size = float(max(hi - lo))
        self._n3 = problem.n_translation
        #: installed plates: index -> displacement when installed
        self._plate_reference: dict[int, np.ndarray] = {}
        #: installed bars: index -> None while being stressed, else extension at lock-off
        self._bar_reference: dict[int, float | None] = {}

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
        plate_index = {pl.name: i for i, pl in enumerate(self.p.plates)}
        bar_index = {b.name: i for i, b in enumerate(self.p.bars)}
        for name in stage.install:
            if name in plate_index:
                self._plate_reference.setdefault(plate_index[name], self._u.copy())
            elif bar_index[name] not in self._bar_reference:
                self._bar_reference[bar_index[name]] = None

        if stage.kind == INITIAL:
            result = self._initial_stage(stage, active)
        elif stage.kind == SSR:
            result = self._ssr_stage(stage, active)
        else:
            if stage.reset_displacements:
                self._u_offset = self._u.copy()
            result = self._newton(stage, active, stage.increments, stage.name)
        # bars stressed in this stage are locked off at the load they carry
        for i, ref in list(self._bar_reference.items()):
            if ref is None:
                self._bar_reference[i] = self._bar_extension(i, self._u)
        result.plate_forces, result.bar_forces = self._structure_forces(self._u)
        result.interface_tractions = self._interface_tractions(active)
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
        self._state = self._fresh_state()
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
            self._srf = srf                    # interfaces are weakened with the soil
            self._u, self._state = anchor_u.copy(), anchor_state.copy()
            budget = max(self.ssr_min_budget, self.ssr_budget_factor * max(successful, default=25))
            started = time.perf_counter()
            res = self._newton(stage, active, max(stage.increments, 6), f"SRF {srf:.3f}", budget)
            if res.converged:
                successful.append(len(res.iterations))
            dmax = float(np.linalg.norm(self._nodal(self._u - reference), axis=1).max())
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
        self._srf = 1.0
        return StageResult(name=stage.name, kind=SSR, converged=last_ok is not None,
                           displacement=self._nodal(self._u - reference), state=self._state,
                           active=active, srf=fos, srf_curve=sorted(curve), message=message)

    # ----------------------------------------------------------------- newton
    def _newton(self, stage: Stage, active: np.ndarray, increments: int, label: str,
                budget: int | None = None) -> StageResult:
        p = self.p
        fixed = np.union1d(p.fixed, self._orphan_dofs(active))
        f_ext = p.gravity(active) + p.surface_loads(stage.loads) + self._structure_loads()
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
            self._contact_frozen = None
            last_rn = None
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
                # Contact points near their limit can switch between sticking
                # and sliding from one iteration to the next, leaving an
                # imbalance no step removes.  Once the iteration slows, hold
                # every contact in the state it is in; the next increment
                # starts free again (as in 2D Lythos).
                if (p.interface_elements is not None and self._contact_frozen is None and it >= 4
                        and last_rn is not None and rn > 0.5 * last_rn):
                    self._contact_frozen = self._contact_modes.copy()
                    evaluated = self._internal(u_try, u_committed, state_committed, active, True)
                    f_int, tangent, symmetric = evaluated
                    r = target - f_int
                    r[fixed] = 0.0
                    rn = float(np.linalg.norm(r))
                    self.linear.forget()
                last_rn = rn
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

        self._contact_frozen = None
        self._u, self._state = u_committed, state_committed
        return StageResult(name=label, kind=stage.kind, converged=lam >= 1.0 - 1e-10,
                           displacement=self._nodal(self._u - self._u_offset),
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

        # structures: linear elastic, measured from when they were installed
        group_matrices, group_active = [], []
        for i, (el, dofs) in enumerate(zip(p.plate_elements, p.plate_dofs)):
            installed = i in self._plate_reference
            if installed:
                du = (u - self._plate_reference[i])[dofs]
                f_int += assemble_vector(p.n_dof, dofs, np.einsum("fij,fj->fi", el.K, du))
            group_matrices.append(el.K)
            group_active.append(np.full(el.n_elements, installed))
        ie = p.interface_elements
        if ie is not None:
            installed = np.array([i in self._plate_reference for i in range(len(p.plates))])
            live = active[p.interface_support]
            Fe, Ke_i, trial, modes = ie.respond(u[p.interface_dofs], state_committed.interface,
                                                ~installed[p.interface_plate], self._srf,
                                                modes=self._contact_frozen)
            self._contact_modes = modes
            # a sliding contact's tangent is unsymmetric (the coupling of the
            # shear traction to the normal one): no Cholesky then, whatever
            # the soil is doing
            contact = live & installed[p.interface_plate]
            yielding |= bool(np.any(modes[contact] == ie.SLIDE))
            Fe[~live] = 0.0
            f_int += assemble_vector(p.n_dof, p.interface_dofs, Fe)
            if return_state:
                new_state.interface = np.where(live[:, None, None], trial, state_committed.interface)
        for i, (axis, length, dofs) in enumerate(p.bar_data):
            locked = self._bar_reference.get(i) is not None
            k = p.bars[i].EA / length
            if locked:
                force = p.bars[i].prestress + k * (self._bar_extension(i, u) - self._bar_reference[i])
                f_int += assemble_vector(p.n_dof, dofs[None, :], self._bar_vector(axis, len(dofs))[None, :] * force)
            b = self._bar_vector(axis, len(dofs))
            group_matrices.append((k * np.outer(b, b))[None])
            group_active.append(np.array([locked]))

        if ie is not None:
            group_matrices.append(Ke_i)
            group_active.append(live)
        if tangent:
            # the matrix itself is formed only if this iteration factorises it
            def matrix():
                return p.system_pattern.assemble(ce.stiffness(tangents), active, group_matrices, group_active)
            return f_int, matrix, not yielding
        return f_int, None, new_state

    # ------------------------------------------------------------- structures
    @staticmethod
    def _bar_vector(axis: np.ndarray, n: int) -> np.ndarray:
        """d(extension)/d(dofs) of a bar: -axis at a, +axis at b."""
        return np.concatenate([-axis, axis]) if n == 6 else -axis

    def _bar_extension(self, i: int, u: np.ndarray) -> float:
        axis, _, dofs = self.p.bar_data[i]
        return float(self._bar_vector(axis, len(dofs)) @ u[dofs])

    def _structure_loads(self) -> np.ndarray:
        """Self weight of installed plates, and the jack forces of bars being stressed."""
        p = self.p
        f = np.zeros(p.n_dof)
        for i in self._plate_reference:
            el = p.plate_elements[i]
            if el.section.weight:
                f += assemble_vector(p.n_dof, p.plate_dofs[i], el.self_weight())
        for i, ref in self._bar_reference.items():
            if ref is None and p.bars[i].prestress:
                axis, _, dofs = p.bar_data[i]
                # the jack pulls the ends together: minus the bar's internal force
                f -= assemble_vector(p.n_dof, dofs[None, :],
                                     self._bar_vector(axis, len(dofs))[None, :] * p.bars[i].prestress)
        return f

    def _structure_forces(self, u: np.ndarray):
        p = self.p
        plates = {}
        for i, ref in self._plate_reference.items():
            plates[p.plates[i].name] = p.plate_elements[i].resultants((u - ref)[p.plate_dofs[i]])
        bars = {}
        for i, ref in self._bar_reference.items():
            k = p.bars[i].EA / p.bar_data[i][1]
            extension = self._bar_extension(i, u)
            bars[p.bars[i].name] = p.bars[i].prestress + (0.0 if ref is None else k * (extension - ref))
        return plates, bars

    def _fresh_state(self) -> MaterialState:
        state = MaterialState.zeros(self.p.n_points)
        if self.p.interface_elements is not None:
            from .interfaces import N_GAUSS, STATE_WIDTH
            state.interface = np.zeros((self.p.interface_elements.n_elements, N_GAUSS, STATE_WIDTH))
        return state

    def _interface_tractions(self, active: np.ndarray) -> dict:
        p = self.p
        ie = p.interface_elements
        if ie is None:
            return {}
        # element means weighted as the element integrates: the 6-point
        # rule's weights are not equal
        w = ie.w / ie.w.sum(axis=1, keepdims=True)
        t = np.einsum("eg,egi->ei", w, self._state.interface[..., 3:])
        tn, tau = t[:, 0], np.linalg.norm(t[:, 1:], axis=1)
        # the shear traction as a vector in global axes (the element's own
        # in-plane axes are arbitrary)
        shear = np.einsum("eij,ei->ej", ie.R[:, 1:], t[:, 1:])
        with np.errstate(divide="ignore", invalid="ignore"):
            strength = ie.c / self._srf - np.minimum(tn, 0.0) * ie.tan_phi / self._srf
            mobilised = np.where(np.isfinite(strength) & (strength > 0), tau / strength, 0.0)
        out = {}
        live = active[p.interface_support]
        for i, plate in enumerate(p.plates):
            mask = (p.interface_plate == i) & live
            if mask.any() and i in self._plate_reference:
                out[plate.name] = {"tn": tn[mask], "tau": tau[mask], "shear": shear[mask],
                                   "mobilised": mobilised[mask]}
        return out

    def _nodal(self, u: np.ndarray) -> np.ndarray:
        """The translations of a system vector, (n_nodes, 3)."""
        return u[:self._n3].reshape(-1, 3)

    def _orphan_dofs(self, active: np.ndarray) -> np.ndarray:
        """Dofs of nodes that belong to no active element: held so the system stays regular."""
        p = self.p
        live = np.zeros(p.mesh.n_nodes, bool)
        live[p.mesh.elements[active].ravel()] = True
        # a plate's own nodes, split off the soil, are held only by its
        # interfaces (and, once installed, by the plate itself)
        if p.interface_elements is not None:
            on = active[p.interface_support]
            live[p.interface_elements.wall_faces[on].ravel()] = True
        for i in self._plate_reference:
            live[p.plates[i].faces.ravel()] = True
        dead = np.nonzero(~live)[0]
        held = [(3 * dead[:, None] + np.arange(3)).ravel()]
        # rotations belong to plates; until one is installed they are held
        rotating = np.zeros(p.n_dof, bool)
        for i in self._plate_reference:
            rotating[p.plate_dofs[i][:, 3::6].ravel()] = True
            rotating[p.plate_dofs[i][:, 4::6].ravel()] = True
            rotating[p.plate_dofs[i][:, 5::6].ravel()] = True
        held.append(np.nonzero(~rotating[self._n3:])[0] + self._n3)
        return np.concatenate(held)
