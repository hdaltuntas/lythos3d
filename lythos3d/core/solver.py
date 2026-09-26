# SPDX-License-Identifier: AGPL-3.0-only
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
from .problem import CONSOLIDATION, INITIAL, SSR, Problem, Stage


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
    #: per installed embedded pile: distance ``s`` from the head and the beam
    #: resultants ``[N, Q2, Q3, T, M2, M3]`` there (N tension positive), the
    #: shaft friction ``skin`` (kN/m, positive where the soil drags the pile
    #: towards its tip) at ``s_skin``, and the tip force ``base`` (kN,
    #: compression positive)
    pile_forces: dict = field(default_factory=dict)
    #: pore pressure at the Gauss points, kPa, compression positive; the
    #: soil's ``state.stress`` is effective, total stress is ``stress - p m``
    pore_pressure: np.ndarray | None = None
    #: with seepage: total head at the nodes (NaN off the active ground),
    #: Darcy velocity at the Gauss points (n_points, 3), and the flows
    #: (``in``, ``out``, ``pumped``, ``seepage_face``) in units of k m2
    #: excess pore pressure at the Gauss points from undrained loading
    #: (included in ``pore_pressure``)
    excess_pore_pressure: np.ndarray | None = None
    #: consolidation: (time since the start of the analysis, largest excess
    #: pore pressure, largest displacement since the reset) after each step
    consolidation: list = field(default_factory=list)
    #: time at the end of the stage (consolidation stages advance it)
    time: float = 0.0
    head: np.ndarray | None = None
    velocity: np.ndarray | None = None
    flows: dict = field(default_factory=dict)

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
        #: Strength reduction needs a tighter tolerance than construction
        #: stages.  Near collapse, an out-of-balance force of a fraction of a
        #: percent of the whole model's load can hold a mechanism that has no
        #: true equilibrium, and the more so the larger the model is next to
        #: the mechanism.  At 2e-3 the benchmark slope came out 1.438 by full
        #: Newton and 1.459 by modified Newton, and the example pit 1.32; at
        #: 2e-4 they are 1.430 (2D Lythos: 1.430) and 1.18, and halving it
        #: again changes neither.  A vertical cut with a tension cut-off was
        #: still moving at 2e-4 (0.96, 0.92, 0.90 at 5e-4, 2e-4, 1e-4):
        #: for a brittle mechanism, check the answer at a tighter tolerance.
        self.ssr_tolerance = 2e-4
        #: a strength reduction trial that has taken this many times the
        #: iterations of the hardest successful one is taken to have failed
        self.ssr_budget_factor = 8
        self.ssr_min_budget = 200
        #: load steps a strength reduction trial takes from the last one, and
        #: whether it starts on the last trial's factorisation (the structure
        #: is the same, and modified Newton refactorises when it stops
        #: helping).  Against 6 steps and a fresh factorisation, the example
        #: pit's search took 289 s instead of 405 s, and gave 1.177, the
        #: value it converges to at a tighter tolerance, rather than 1.191;
        #: the benchmark slope is 1.430 either way.  Cutting the iteration
        #: budget instead moved the pit to 1.198, so the budget stays.
        self.ssr_increments = 3
        self.ssr_keep_factorisation = True
        #: the factor of safety is bracketed to within this
        self.ssr_bracket = 0.01
        #: smallest step, as a fraction of the first, before a trial gives up
        self.ssr_min_step = 1.0 / 64.0
        #: called with a line of text as the analysis goes - every Newton
        #: iteration, every strength reduction trial - by an interface that
        #: shows progress; it may raise to stop the analysis
        self.monitor = None
        self.results: list[StageResult] = []
        self._u = np.zeros(problem.n_dof)
        self._u_offset = np.zeros(problem.n_dof)
        self._state = self._fresh_state()
        self._srf = 1.0
        #: interface contact modes: those of the last evaluation, and those
        #: held fixed while an increment converges on them
        self._contact_modes = None
        self._contact_frozen = None
        #: the water table in force
        self._water = problem.water
        #: time elapsed in consolidation stages
        self._time = 0.0
        #: the pore pressure field that goes with it and the ground as it is
        self._field = None
        self._undrained = True
        self._gauss_z = problem.continuum.gauss_xyz[:, :, 2].ravel()
        problem.reset_regions()
        self._active = np.ones(problem.mesh.n_elements, bool)
        if problem.inactive is not None:
            self._active &= ~problem.inactive
        for name in problem.absent:
            self._active &= ~problem.groups[name]
        lo, hi = problem.mesh.bounds
        self._size = float(max(hi - lo))
        self._n3 = problem.n_translation
        #: installed plates: index -> displacement when installed
        self._plate_reference: dict[int, np.ndarray] = {}
        #: installed bars: index -> None while being stressed, else extension at lock-off
        self._bar_reference: dict[int, float | None] = {}
        #: installed piles: index -> displacement when installed
        self._pile_reference: dict[int, np.ndarray] = {}

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
        ngp = self.p.continuum.n_gauss
        for name in stage.construct:
            new = self.p.groups[name] & ~self._active
            self._active |= self.p.groups[name]
            if name in self.p.construct_region:
                self.p.set_region(self.p.groups[name], self.p.construct_region[name])
            # new ground starts free of stress, whatever its nodes did before
            gp = (np.nonzero(new)[0][:, None] * ngp + np.arange(ngp)).ravel()
            self._state.put(gp, MaterialState.zeros(len(gp)))
        active = self._active.copy()
        if stage.water is not None:
            self._water = stage.water
        plate_index = {pl.name: i for i, pl in enumerate(self.p.plates)}
        bar_index = {b.name: i for i, b in enumerate(self.p.bars)}
        pile_index = {pl.name: i for i, pl in enumerate(self.p.piles)}
        for name in stage.install:
            if name in plate_index:
                self._plate_reference.setdefault(plate_index[name], self._u.copy())
            elif name in pile_index:
                if pile_index[name] not in self._pile_reference:
                    self._install_pile(pile_index[name])
            elif bar_index[name] not in self._bar_reference:
                self._bar_reference[bar_index[name]] = None

        # undrained soil is undrained except in a stage marked drained (and
        # the initial stresses); a drained stage lets the excess pore
        # pressure go, and the soil consolidates under the load it carried
        self._undrained = not (stage.drained or stage.kind in (INITIAL, CONSOLIDATION))
        if (stage.drained or stage.kind == INITIAL) and self._state.excess is not None:
            self._state.excess[:] = 0.0
        if stage.kind != SSR or self._field is None:
            # strength reduction changes neither the ground nor the water
            self._field = self.p.pore_field(self._water, active, installed=tuple(self._plate_reference))
        if stage.kind == INITIAL:
            result = self._initial_stage(stage, active)
        elif stage.kind == SSR:
            result = self._ssr_stage(stage, active)
        elif stage.kind == CONSOLIDATION:
            if stage.reset_displacements:
                self._u_offset = self._u.copy()
            result = self._consolidation_stage(stage, active)
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
        result.pile_forces = self._pile_forces()
        result.time = self._time
        result.excess_pore_pressure = self._state.excess.copy()
        result.pore_pressure = self._field.gauss + result.excess_pore_pressure
        result.head, result.velocity = self._field.head, self._field.velocity
        result.flows = dict(self._field.flows)
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
            self._state.stress[:] = self.p.k0_stress(active, self._water, self._field)
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
            res = self._newton(stage, active, self.ssr_increments, f"SRF {srf:.3f}", budget)
            if res.converged:
                successful.append(len(res.iterations))
            dmax = float(np.linalg.norm(self._nodal(self._u - reference), axis=1).max())
            curve.append((srf, dmax))
            if self.monitor is not None:
                self.monitor(f"strength reduction: factor {srf:.3f} "
                             f"{'holds' if res.converged else 'fails'} ({len(curve)} trials so far)")
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
                if hi - lo <= self.ssr_bracket:
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
        f_ext = (p.gravity(active, self._field) + p.water_loads(self._field, active)
                 + p.surface_loads(stage.loads, active) + self._structure_loads())
        logs: list[IterationLog] = []

        u_committed = self._u.copy()
        state_committed = self._state.copy()
        f_int0 = self._internal(u_committed, u_committed, state_committed, active, False)[0]
        # a new stage has new restraints and elements; never start it on the
        # factorisation of the last one
        if not (stage.kind == SSR and self.ssr_keep_factorisation):
            self.linear.forget()
        modified = self.modified_newton
        tol = min(self.tol, self.ssr_tolerance) if stage.kind == SSR else self.tol

        # Adaptive stepping: an increment that will not converge is retried at
        # half the size, and once a size has failed the step never grows back
        # to it - walking into the same wall again is what makes a doomed
        # strength reduction trial cost ten times a successful one.
        lam, message, cuts = 0.0, "", 0
        dlam = ceiling = 1.0 / max(increments, 1)
        min_dlam = dlam * (self.ssr_min_step if stage.kind == SSR else 1.0 / 64.0)
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
                if self.monitor is not None:
                    self.monitor(f"{label}: {100 * trial:.0f}% of the load, iteration {it + 1}, "
                                 f"imbalance {rn / scale:.1e} (target {tol:.0e})")
                if rn / scale < tol:
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

    def _consolidation_stage(self, stage: Stage, active: np.ndarray) -> StageResult:
        """Time passing: the excess pore pressure flows away and the soil consolidates (Biot).

        See :mod:`lythos3d.core.consolidation`.  The stage's loads and
        construction are applied in proportion to the time elapsed.  A time
        step that will not converge is split in two.
        """
        import scipy.sparse as sp

        from .consolidation import PressureSpace, time_steps
        from .water import GAMMA_WATER

        p = self.p
        ce = p.continuum
        ngp = ce.n_gauss
        gamma_w = getattr(self._water, "gamma_w", GAMMA_WATER)
        space = PressureSpace(p, active, installed=tuple(self._plate_reference),
                              drained_sides=stage.drained_sides, gamma_w=gamma_w)
        n_u, n_p = p.n_dof, space.n
        pres = space.from_gauss(self._state.excess, ngp)
        undrained = pres.copy()
        # the drained boundary drains at once: its pressure goes to zero at
        # the start, and the imbalance that leaves is no part of the load
        # to be spread over the stage
        pres[space.drained] = 0.0
        self._state.excess[:] = 0.0

        Q = sp.coo_matrix((space.Qe.ravel(),
                           (np.repeat(space.udofs[:, :, None], 4, axis=2).ravel(),
                            np.repeat(space.pdofs[:, None, :], 30, axis=1).ravel())),
                          shape=(n_u, n_p)).tocsr()
        pr_rows = np.repeat(space.pdofs, 4, axis=1).ravel()
        pr_cols = np.tile(space.pdofs, (1, 4)).ravel()
        S = sp.coo_matrix((space.Se.ravel(), (pr_rows, pr_cols)), shape=(n_p, n_p)).tocsr()
        H = sp.coo_matrix((space.He.ravel(), (pr_rows, pr_cols)), shape=(n_p, n_p)).tocsr()

        fixed_u = np.union1d(p.fixed, self._orphan_dofs(active))
        fixed = np.concatenate([fixed_u, n_u + space.drained])
        f_ext = (p.gravity(active, self._field) + p.water_loads(self._field, active)
                 + p.surface_loads(stage.loads, active) + self._structure_loads())
        u_n, p_n, state_n = self._u.copy(), pres.copy(), self._state.copy()
        f_int0 = self._internal(u_n, u_n, state_n, active, False)[0] - Q @ undrained
        scale = max(np.linalg.norm(f_ext), np.linalg.norm(f_int0), 1e-8)
        linear = type(self.linear)(self.backend)
        steps = list(time_steps(stage.time, max(stage.increments, 1)))
        logs, history = [], []
        elapsed, splits, message, i = 0.0, 0, "", 0
        while i < len(steps):
            dt = steps[i]
            target = f_int0 + (elapsed + dt) / stage.time * (f_ext - f_int0)
            C = (S + dt * H).tocsr()
            u, pr = u_n.copy(), p_n.copy()
            ok = False
            for it in range(self.max_iterations):
                f_int, tangent, _ = self._internal(u, u_n, state_n, active, True)
                Ru = target - f_int + Q @ pr
                Ru[fixed_u] = 0.0
                flow_terms = (Q.T @ (u - u_n), S @ (pr - p_n), dt * (H @ pr))
                Rp = flow_terms[0] + flow_terms[1] + flow_terms[2]
                Rp[space.drained] = 0.0
                p_scale = max(max(np.linalg.norm(t) for t in flow_terms), 1e-12 * scale)
                ru, rp = np.linalg.norm(Ru) / scale, np.linalg.norm(Rp) / p_scale
                logs.append(IterationLog(len(logs), it, max(ru, rp)))
                if ru < self.tol and rp < self.tol:
                    ok = True
                    break
                A = sp.bmat([[tangent(), -Q], [-Q.T, -C]], format="csr")
                try:
                    delta = solve_constrained(A, np.concatenate([Ru, Rp]), fixed, np.zeros(len(fixed)),
                                              solver=linear)
                except np.linalg.LinAlgError:
                    break
                linear.forget()
                u = u + delta[:n_u]
                pr = pr + delta[n_u:]
                if not np.all(np.isfinite(u)):
                    break
            if ok:
                state_n = self._internal(u, u_n, state_n, active, False, return_state=True)[2]
                u_n, p_n = u, pr
                elapsed += dt
                history.append((self._time + elapsed, float(np.abs(p_n).max(initial=0.0)),
                                float(np.linalg.norm(self._nodal(u_n - self._u_offset), axis=1).max())))
                i += 1
            else:
                splits += 1
                if splits > 20:
                    message = f"no equilibrium after {elapsed:g} of {stage.time:g}"
                    break
                steps[i:i + 1] = [0.5 * dt, 0.5 * dt]
        linear.release()
        self._time += elapsed
        self._u, self._state = u_n, state_n
        self._state.excess[:] = space.to_gauss(p_n, ce.n_elements, ngp)
        res = StageResult(name=stage.name, kind=stage.kind, converged=i == len(steps),
                          displacement=self._nodal(self._u - self._u_offset), state=self._state,
                          active=active, iterations=logs, message=message)
        res.consolidation = history
        return res

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
        m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        for mat, gp in p.material_groups(self.materials):
            committed = state_committed.take(gp)
            s, t, ns = mat.update(committed, dstrain[gp], self._gauss_z[gp])
            ns.excess = committed.excess
            if self._undrained and getattr(mat, "undrained", False):
                # the pore water resists the change of volume: excess pore
                # pressure, and the water's stiffness in the tangent
                kw = mat.water_bulk_modulus
                ns.excess = committed.excess - kw * dstrain[gp, :3].sum(axis=1)
                if tangent:
                    t = t + kw * np.outer(m, m)
            stress[gp] = s if ns.excess is None else s - ns.excess[:, None] * m
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
        if p.piles:
            installed = np.array([i in self._pile_reference for i in range(len(p.piles))])
            for i, beam in enumerate(p.pile_beams):
                if installed[i]:
                    d = p.beam_dofs[p.beam_pile == i]
                    du = (u - self._pile_reference[i])[d]
                    f_int += assemble_vector(p.n_dof, d, np.einsum("eij,ej->ei", beam.K, du))
            K_beam = np.concatenate([b.K for b in p.pile_beams])
            live_skin = installed[p.skin_pile] & active[p.skin_tet]
            Fs, Ks, trial_s = p.embedded.skin(u[p.skin_dofs], state_committed.embedded)
            Fs[~live_skin] = 0.0
            f_int += assemble_vector(p.n_dof, p.skin_dofs, Fs)
            live_tip = installed[p.tip_pile] & active[p.tip_tet]
            Ft, Kt, trial_t = p.embedded.tips(u[p.tip_dofs], state_committed.tips)
            Ft[~live_tip] = 0.0
            f_int += assemble_vector(p.n_dof, p.tip_dofs, Ft)
            if return_state:
                new_state.embedded = np.where(live_skin[:, None], trial_s, state_committed.embedded)
                new_state.tips = np.where(live_tip[:, None], trial_t, state_committed.tips)
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
        if p.piles:
            group_matrices += [K_beam, Ks, Kt]
            group_active += [installed[p.beam_pile], live_skin, live_tip]
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
        for i in self._pile_reference:
            beam = p.pile_beams[i]
            if beam.section.weight:
                f += assemble_vector(p.n_dof, p.beam_dofs[p.beam_pile == i], beam.self_weight())
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
        if self.p.piles:
            state.embedded = np.zeros((len(self.p.skin_dofs), 6))
            state.tips = np.zeros((len(self.p.tip_dofs), 2))
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

    def _install_pile(self, i: int) -> None:
        """Put pile ``i`` into the ground as it now is.

        Its nodes take the soil's displacement where they are, so the pile
        starts where the ground has got to; its springs start from there
        free of force.
        """
        p = self.p
        tets, N = p.pile_node_tet[i], p.pile_node_N[i]
        soil = np.einsum("nk,nkj->nj", N, self._u[p.continuum.dofs()[tets]].reshape(len(tets), 10, 3))
        dofs = p.pile_node_dofs[i]
        self._u[dofs[:, :3]] = soil
        self._u[dofs[:, 3:]] = 0.0
        self._pile_reference[i] = self._u.copy()
        mine = p.skin_pile == i
        d = np.einsum("nij,nj->ni", p.embedded.B_skin[mine], self._u[p.skin_dofs[mine]])
        self._state.embedded[mine] = np.concatenate([d, np.zeros_like(d)], axis=1)
        at_tip = p.tip_pile == i
        dt = np.einsum("mij,mj->mi", p.embedded.B_tip[at_tip], self._u[p.tip_dofs[at_tip]])[:, 0]
        self._state.tips[at_tip] = np.column_stack([dt, np.zeros_like(dt)])

    def _pile_forces(self) -> dict:
        p = self.p
        out = {}
        for i in self._pile_reference:
            beam, pile = p.pile_beams[i], p.piles[i]
            d = p.beam_dofs[p.beam_pile == i]
            res = beam.resultants((self._u - self._pile_reference[i])[d]).reshape(-1, 6)
            s = np.linalg.norm(beam.gauss_points().reshape(-1, 3) - np.asarray(pile.head, float), axis=1)
            mine = p.skin_pile == i
            out[pile.name] = {"s": s, "resultants": res, "s_skin": p.skin_s[mine],
                              "skin": self._state.embedded[mine, 3],
                              "base": -float(self._state.tips[p.tip_pile == i, 1].sum())}
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
        # piles' own dofs, likewise, until the pile is installed
        for i in self._pile_reference:
            rotating[self.p.pile_node_dofs[i].ravel()] = True
        held.append(np.nonzero(~rotating[self._n3:])[0] + self._n3)
        return np.concatenate(held)
