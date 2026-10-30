"""Feasible bundles, local fairness costs, and subsidy optimization."""

from collections import defaultdict
import contextlib
from itertools import combinations
import io
from time import perf_counter
import numpy as np

from config import env_params, RL_params
from common.optimization import (
    Optimizer,
    _COPT_STATUS_NAMES,
    _GUROBI_STATUS_NAMES,
    _status_name,
)


# Guarded backends (same convention as common/optimization.py).
try:
    import gurobipy as gp
except ImportError:  # pragma: no cover - Gurobi backend unavailable without gurobipy
    gp = None

try:
    import coptpy as cp
except ImportError:  # pragma: no cover - COPT backend unavailable without coptpy
    cp = None


def fairness_modes(mode):
    """Return (fair, weighted) flags for "fair"/"weighted"/"efficiency"."""
    return mode == "fair", mode == "weighted"


class Bundling:
    """Bundle enumeration, fairness costs, the joint BMIP, and the subsidy LP for one decision step."""
    def __init__(self, env):
        """Declare candidate/incidence tables and request mappings."""
        self.environment = env  # BatterySwapEnv: joint demand and station state
        self.max_size = env_params["max_bundle_size"]  # effective bundle limit
        self.per_station_cap = int(env_params["max_bundles"])  # per-station enumeration cap

        self.reset()

    def reset(self):
        """Clear per-step candidate state."""
        self.bundles = []  # candidate request-ID tuples
        self.bundle_to_index, self.schedules = {}, {}  # tuple->row map | cached reservations

        self.BS_pairs = []  # list[tuple[int, int]]: station-bundle index pairs
        self.generation_seconds = 0.0  # seconds spent generating the current candidate set

        self.index = {r.id: i for i, r in enumerate(self.environment.new_demand)}  # ID -> row
        self.requests = {r.id: r for r in self.environment.new_demand}  # dict[int, Request]

    def start_bundling(self, max_size=None):
        """Generate bundles and incidence maps, timing the pass."""
        self.reset()
        start = perf_counter()
        self.generate_bundles(max_size=max_size)
        self.BS_pairs.sort()  # preserve bundle-index, then station-ID candidate order
        self.generation_seconds = perf_counter() - start

    def generate_bundles(self, max_size=None):
        """Enumerate feasible request subsets per station (deduplicated, capped)."""
        # Policy-bound the subset size, then keep feasible unique tuples.
        limit = self.max_size if max_size is None else min(self.max_size, max_size)
        for agent in self.environment.Agents:
            demand = self.environment.get_agent_demand(agent)
            size_limit = min(limit, len(demand))
            station_count = 0
            stopped = False
            for size in range(1, size_limit + 1):
                for requests in combinations(demand, size):
                    bundle = tuple(sorted(r.id for r in requests))
                    if self.bundle_feasibility(agent, bundle) is None:
                        continue

                    # Share each unique tuple across its feasible stations.
                    if bundle not in self.bundle_to_index:
                        self.bundle_to_index[bundle] = len(self.bundles)
                        self.bundles.append(bundle)
                    self.BS_pairs.append((self.bundle_to_index[bundle], agent.id))
                    station_count += 1
                    if station_count >= self.per_station_cap:
                        stopped = True
                        break
                if stopped:
                    break

        # One size per bundle + one count per step for service_records.
        self.environment.bundle_size_samples.extend(len(bundle) for bundle in self.bundles)
        self.environment.step_samples["bundle_number"].append(len(self.bundles))

    def bundle_feasibility(self, agent, bundle):
        """Return the cached reservation {id: (arrival, begin, departure, wait)} or None."""
        key = (agent.id, tuple(sorted(bundle)))
        if key not in self.schedules:
            self.schedules[key] = agent.reserve(
                self.environment.current_time, [self.requests[rid] for rid in key[1]]
            )
        return self.schedules[key]

    def build_options(self, enumerated=False, max_size=None, target_fixed=False):
        """Return candidates k=(station, bundle, charging q, value);
        enumerated=True uses the base_stocks 2x2 grid, else the single
        completion-time target."""
        if target_fixed and enumerated:
            raise ValueError(
                "target_fixed applies only to the non-enumerated charging path "
                "(enumerated=False)"
            )
        # Candidate charging targets per learner type.
        limit = self.max_size if max_size is None else min(self.max_size, max_size)
        # One O(|BS_pairs|) grouping pass (BS_pairs sorted, so lists are deterministic).
        bundles_by_station = {}
        for bundle_index, station in self.BS_pairs:
            bundle = self.bundles[bundle_index]
            if len(bundle) <= limit:
                bundles_by_station.setdefault(station, []).append(bundle)

        options = []
        for agent in self.environment.Agents:
            # Enumerated grid or one target per station (fixed/dynamic peak-window).
            levels = (
                agent.enum_base_levels()
                if enumerated
                else (
                    agent.set_base_level(
                        self.environment.current_time, target_fixed=target_fixed
                    ),
                )
            )

            # Fold duplicate charging vectors.
            quantities = set()
            for target in levels:
                q = agent.nested_charging(target)
                quantities.add(tuple(q))

            # Empty bundle () stays as the all-reject outlet; one option per
            # (bundle, q) pair, so filtering here keeps direct calls correct.
            bundle_ids = bundles_by_station.get(agent.id, []) + [()]
            # Precompute one cost per charging vector (depends only on station/q/time).
            quantities = sorted(quantities)  # deterministic candidate order
            q_costs = {
                q: agent.calculate_charging(self.environment.current_time, q)
                for q in quantities
            }
            for bundle in bundle_ids:
                # value = service income (arrival-SOC fees) - discounted charging cost;
                # the empty bundle has no income, so its value is -cost(q).
                income = sum(self.requests[rid].fee_at(agent.id) for rid in bundle)
                for q, cost in q_costs.items():
                    options.append(
                        dict(
                            station=agent.id,
                            bundle=bundle,
                            charging=q,
                            value=income - cost,      # objective score
                        )
                    )

        return options

    def bundle_optimize(
            self,
            options,  # candidate list
            mode,  # "fair" | "weighted" | "efficiency": explicit allocation semantics
            randomize=False,  # randomize=True requires rng
            rng=None,
            solver_seed=None,  # int or None
            policy=None,
            envy_weight=None,  # float or None: weighted-mode objective coefficient
            quick_solve=False,  # True: short time budget returning the best incumbent (exploration)
            target_fixed=False,  # True: all-reject fallback uses the fixed fixed_level charging target
    ):
        """Assemble and solve the joint BMIP; return the unified action dict."""
        start = perf_counter()
        requests = self.environment.new_demand
        fair, weighted = fairness_modes(mode)
        # Use the configured solver backend directly, without license probing.
        backend = RL_params["solver"]

        # Blocks: efficiency x/y, weighted +e, fair +p.
        n, nx = len(requests), len(options)
        y0 = nx
        if fair:
            size = nx + 2 * n
            p0, e0 = nx + n, None
        elif weighted:
            size = nx + 2 * n
            p0, e0 = None, nx + n
        else:
            size = nx + n
            p0 = e0 = None

        # -v_k on x, +1 on p, weight on e; an empty bundle (value -cost(q))
        # enters as +cost(q) and still commits its charging vector.
        objective = np.zeros(size)
        objective[:nx] = -np.array([option["value"] for option in options])
        if fair:
            objective[p0 : p0 + n] = 1.0
        if weighted:
            weight = float(RL_params["envy_weight"] if envy_weight is None else envy_weight)
            objective[e0 : e0 + n] = weight
        if randomize:
            # Exploration: multiply each candidate value by U(-0.5, 2) and
            # randomize y/subsidy coefficients as well.
            values = np.array([option["value"] for option in options])
            objective[:nx] = -(values * rng.uniform(-0.5, 2, nx))
            cr = float(env_params.get("rejection_time_cost", 50))
            objective[y0 : y0 + n] = cr * rng.uniform(0, 4, n)
            if fair:
                objective[p0 : p0 + n] = rng.uniform(0, 4, n)

        # Bounds include arrival-SOC fees; travel/wait bounds alone are insufficient.
        own_upper, cost_gaps = {}, {}
        if fair or weighted:
            time_cost = env_params["VOT"] * (
                env_params["max_detour_minutes"] + env_params["max_wait_time"]
                + 2 * env_params["time_tolerance"]
            )
            for request in requests:
                fees = [request.fee_at(j) for j in request.feasible_stations]
                own_upper[request.id] = max(
                    [request.reject_cost] + [fee + time_cost for fee in fees]
                )
                lower_cost = min([request.reject_cost] + fees)
                cost_gaps[request.id] = max(0.0, own_upper[request.id] - lower_cost)
        cost_spread = max(cost_gaps.values(), default=0.0)
        # P_bar_i = (group size - 1) x S (longest simple chain).
        pay_bounds = None
        if fair:
            group_sizes = {}
            for request in requests:
                key = (request.cluster, request.requested_class)
                group_sizes[key] = group_sizes.get(key, 0) + 1
            pay_bounds = np.array(
                [(group_sizes[r.cluster, r.requested_class] - 1) * cost_spread for r in requests]
            )

        # Bounds: x/y in [0,1]; p in [0,P_bar]; e in [0,S].
        lower = np.zeros(size)
        upper = np.ones(size)
        if fair:
            upper[p0 : p0 + n] = pay_bounds
        if weighted:
            upper[e0 : e0 + n] = [cost_gaps[r.id] for r in requests]
        integrality = np.zeros(size, dtype=int)
        integrality[: nx + n] = 1

        optimizer = Optimizer()
        # Reverse indices: candidates per station and per request.
        by_station = [[] for _ in self.environment.Agents]
        by_request = [[] for _ in requests]
        for k, option in enumerate(options):
            by_station[option["station"]].append(k)
            for rid in option["bundle"]:
                by_request[self.index[rid]].append(k)

        # (a) exactly one candidate per station; the empty bundle is the
        # "serve nothing" outlet that keeps the all-reject solution representable.
        for alternatives in by_station:
            optimizer.add_constraint({k: 1 for k in alternatives}, 1, 1)

        # (b) serve or reject: empty bundles never touch y, so y=1 (all reject) stays valid.
        for i in range(n):
            optimizer.add_constraint({**{k: 1 for k in by_request[i]}, y0 + i: 1}, 1, 1)

        if fair or weighted:
            # big-M relaxes rows whose replacement is infeasible (skip the
            # comparison); it must exceed any achievable row value.
            vot_span = env_params["VOT"] * (
                env_params["max_detour_minutes"] + env_params["max_wait_time"]
                + 2 * float(env_params["time_tolerance"])
            )
            if requests:
                fee_bound = max(
                    (
                        env_params["ready_soc"]
                        * env_params["battery_kwh"][r.requested_class - 1]
                        - env_params["reserve_soc"]
                        * env_params["battery_kwh"][r.installed_class - 1]
                    ) * r.energy_rate + r.upgrade_charge
                    for r in requests
                )
                max_reject = max(r.reject_cost for r in requests)
                big_m = max_reject + fee_bound + vot_span + (n + 2) * cost_spread
            else:
                big_m = 0.0
            for rid, other in self.peer_pairs():
                i, m = self.index[rid], self.index[other]
                request = requests[i]
                coefficients = {
                    y0 + i: request.reject_cost,
                    y0 + m: -request.reject_cost,
                }
                if fair:
                    coefficients[p0 + i], coefficients[p0 + m] = -1, 1
                if weighted:
                    coefficients[e0 + i] = -1
                for k in by_request[i]:
                    option = options[k]
                    cost = self.service_cost(rid, option["bundle"], option["station"])
                    coefficients[k] = coefficients.get(k, 0) + cost
                for k in by_request[m]:
                    option = options[k]
                    cost = self.replace_cost(rid, other, option["bundle"], option["station"])
                    if np.isfinite(cost):
                        coefficients[k] = coefficients.get(k, 0) - cost
                    else:
                        # Impossible replacement: relax the row whenever this candidate serves `other`.
                        coefficients[k] = coefficients.get(k, 0) - big_m
                row_upper = (
                    float(env_params["fairness_tolerance"]) if fair else 0.0
                )
                # Empty bundles add no terms; with every station empty and p=0
                # each row reduces to 0 <= row_upper (always satisfied).
                optimizer.add_constraint(coefficients, -np.inf, row_upper)

        # Assemble the COO -> CSC constraint matrix.
        matrix = optimizer.matrix(size)

        build_seconds = perf_counter() - start
        t0 = perf_counter()
        if backend == "gurobi":
            result = optimizer.gurobi_solve(
                objective, integrality, lower, upper, matrix, optimizer.lows, optimizer.highs,
                solver_seed, policy, quick_solve,
            )
        elif backend == "copt":
            result = optimizer.copt_solve(
                objective, integrality, lower, upper, matrix, optimizer.lows, optimizer.highs,
                policy, quick_solve,
            )
        solve_seconds = perf_counter() - t0

        # Decode the incumbent; a missing/unusable result falls back to all-reject.
        assignment, subsidy = np.full(n, -1, dtype=int), np.zeros(n)
        charging = np.zeros((len(self.environment.Agents), 2), dtype=int)

        feasible = result.x is not None and np.isfinite(result.x).all()
        if feasible:
            status = getattr(result, "status_name", "OPTIMAL")
            selected = [k for k in range(nx) if result.x[k] > 0.5]
            for k in selected:
                option = options[k]
                # Empty bundle: untouched requests stay -1 (reject); q still commits.
                for rid in option["bundle"]:
                    assignment[self.index[rid]] = option["station"]
                charging[option["station"]] = option["charging"]
            if fair:
                subsidy = np.maximum(result.x[p0 : p0 + n], 0.0)
            if status != "OPTIMAL":
                # Timeout: commit the best incumbent; only a solve with no incumbent rejects all.
                rejected = int(np.count_nonzero(assignment == -1))
                rejection_rate = rejected / n if n else 0.0
                print(
                    f"############ BMIP for {policy} at step={self.environment.time_step}: "
                    f"{status} with incumbent ############ \n",
                    f"requests={n}, options={nx}, variables={size}, "
                    f"constraints={len(optimizer.lows)}, current-batch rejections={rejected}/{n} "
                    f"({rejection_rate:.2%}).",
                    flush=True,
                )

        if not feasible:
            # No usable incumbent: reject all requests, keep baseline charging targets
            # (always feasible; passes the fairness audit via booking rejection costs).
            assignment[:] = -1
            subsidy[:] = 0.0
            charging = np.array(
                [
                    agent.nested_charging(
                        agent.set_base_level(
                            self.environment.current_time, target_fixed=target_fixed
                        )
                    )
                    for agent in self.environment.Agents
                ],
                dtype=int,
            )
            status = getattr(result, "status_name", None) or getattr(
                result, "message", "unknown"
            )
            print(
                f"############ BMIP for {policy} solved: ({status}) ############ \n "
                f"All-reject fallback at episode={self.environment.episode}, "
                f"step={self.environment.time_step}; \n"
                f"requests={n}, options={nx}, variables={size}, constraints={len(optimizer.lows)}.",
                flush=True,
            )

        self.environment.last_diagnostics.update(
            dict(
                bundle_seconds=self.generation_seconds,
                model_build_seconds=build_seconds,
                model_solve_seconds=solve_seconds,
                solver_gap=(
                    float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None
                ),
                solver_failed=int(not feasible),
                fallback=int(not feasible),
                exploration=int(randomize),
                solver_quick=int(quick_solve),
                bmip_requests=n,
                bmip_options=nx,
                bmip_variables=size,
                bmip_constraints=len(optimizer.lows),
            )
        )
        return {"assignment": assignment, "charging": charging, "subsidy": subsidy}

    def service_cost(self, rid, bundle, station):
        """Return served disutility, rejection cost, or infinity for an infeasible bundle."""
        # wait is arrival-to-completion minutes; fee uses the candidate station's arrival SOC.
        request = self.requests[rid]
        if station < 0:
            return request.reject_cost

        # Fresh reservation: BMIP rows, subsidy LP, and audit must share identical costs.
        schedule = self.environment.Agents[station].reserve(
            self.environment.current_time,
            [self.requests[r] for r in tuple(sorted(bundle))],
        )
        if schedule is None:
            return float("inf")
        detour = max(request.travel_times[station] - request.nearest_time, 0.0)
        return request.fee_at(station) + env_params["VOT"] * (
            detour + schedule[rid][3]
        )

    def replace_cost(self, rid, other_id, bundle, station):
        """Return the cost of swapping rid in for other_id in a station bundle."""
        if station < 0:
            return self.requests[rid].reject_cost
        replacement = bundle if rid in bundle else tuple(sorted((set(bundle) - {other_id}) | {rid}))
        return self.service_cost(rid, replacement, station)

    def peer_pairs(self):
        """Return directed peer pairs within (cluster, class) groups."""
        groups = defaultdict(list)
        for request in self.environment.new_demand:
            groups[request.cluster, request.requested_class].append(request.id)
        return [(i, m) for group in groups.values() for i in group for m in group if i != m]

    def compare_costs(self, assignment, bundle_assign):
        """Return own costs {id: c_i} and counterfactuals {(i,m): g_{i,m}}."""
        own, counterfactual = {}, {}
        for i, request in enumerate(self.environment.new_demand):
            j = int(assignment[i])
            own[request.id] = self.service_cost(request.id, bundle_assign.get(j, ()), j)
        for rid, other in self.peer_pairs():
            j = int(assignment[self.index[other]])
            counterfactual[rid, other] = self.replace_cost(rid, other, bundle_assign.get(j, ()), j)
        return own, counterfactual

    def fairness_residual(self, assignment, subsidy):
        """Return the largest net envy after independently rebuilding the action's timetables."""
        # Rebuild from station physics, independent of cached BMIP coefficients.
        schedules = []
        for agent in self.environment.Agents:
            accepted = [r for i, r in enumerate(self.environment.new_demand)
                        if assignment[i] == agent.id]
            schedule = agent.reserve(self.environment.current_time, accepted)
            if schedule is None:
                return float("inf")
            schedules.append(schedule)
        subsidy = np.asarray(subsidy)
        if subsidy.shape != (len(assignment),) or not np.isfinite(subsidy).all():
            return float("inf")
        envy = self.environment.realized_envy(assignment, schedules, subsidy)
        return float(envy.max()) if envy.size else 0.0

    def repair_greedy_subsidies(self, assignment, bundles):
        """Break envy-difference negative cycles with local rejections."""
        groups = defaultdict(list)
        for request in self.environment.new_demand:
            groups[(request.cluster, request.requested_class)].append(request.id)

        tolerance = float(env_params["fairness_tolerance"])
        rejected = set()
        for ids in groups.values():
            if len(ids) < 2:
                continue
            active = list(ids)
            while True:
                # Weights recomputed per pass: rejections change own/counterfactual costs.
                own, counterfactual = self.compare_costs(assignment, bundles)
                index = {rid: k for k, rid in enumerate(active)}
                # Edge i -> m has weight g_{i,m} - c_i + eps (the subsidy LP row).
                weights = np.full((len(active), len(active)), np.inf)
                for rid in active:
                    for oid in active:
                        if rid == oid:
                            continue
                        cost = counterfactual.get((rid, oid))
                        if cost is not None and np.isfinite(cost):
                            weights[index[rid], index[oid]] = (
                                cost - own[rid] + tolerance
                            )
                # Floyd-Warshall: negative diagonal = node on a negative cycle.
                sub = weights.copy()
                for k in range(len(active)):
                    sub[k, k] = 0.0
                for k in range(len(active)):
                    sub = np.minimum(sub, sub[:, k, None] + sub[None, k, :])
                negative = [k for k in range(len(active))
                            if sub[k, k] < -env_params["solver_tolerance"]
                            and assignment[self.index[active[k]]] >= 0]
                if not negative:
                    break
                victim = min(negative, key=lambda k: sub[k, k])
                victim_rid = active[victim]
                rejected.add(victim_rid)
                # Rejected node stays in the group: its outcome is still covered.
                assignment[self.index[victim_rid]] = -1
                for station_ids in bundles.values():
                    station_ids[:] = [
                        rid for rid in station_ids if rid != victim_rid
                    ]
        return len(rejected)

    def set_subsidies(self, assignment, bundle_assign, solver_seed=None):
        """Solve and audit minimum subsidies for the selected feasible comparisons."""
        # Accepted and rejected requests remain in the same comparison group.
        n = len(assignment)
        if not n:
            return np.zeros(0)
        own, counterfactual = self.compare_costs(assignment, bundle_assign)
        rows, rhs = [], []
        for (rid, other), cost in counterfactual.items():
            if np.isfinite(cost):
                row = np.zeros(n)
                row[self.index[rid]], row[self.index[other]] = -1, 1
                rows.append(row)
                rhs.append(cost - own[rid] + env_params["fairness_tolerance"])
        if not rows:
            return np.zeros(n)

        # min 1'p s.t. A p <= b, p >= 0.
        objective, matrix, upper = np.ones(n), np.array(rows), np.array(rhs)
        if RL_params["solver"] == "gurobi":
            subsidy, status_name = self.solve_subsidies_gp(
                objective, matrix, upper, solver_seed
            )
        elif RL_params["solver"] == "copt":
            subsidy, status_name = self.solve_subsidies_copt(objective, matrix, upper)
        else:
            subsidy, status_name = None, "unknown solver backend"
        if subsidy is not None:
            residual = self.fairness_residual(assignment, subsidy)
            self.environment.last_diagnostics["subsidy_residual"] = residual
            if residual > env_params["fairness_tolerance"] + env_params["solver_tolerance"]:
                self.environment.last_diagnostics["subsidy_residual_failed"] = 1
                return None
        return subsidy

    @staticmethod
    def solve_subsidies_gp(objective, matrix, upper, solver_seed=None):
        """Gurobi subsidy LP: return (p* (R,) or None, status name)."""
        if gp is None:
            raise ImportError(
                "gurobipy is required for solver='gurobi'; install it or select solver='copt'"
            )
        model = gp.Model("BS_subsidy_LP")
        model.Params.OutputFlag = 0
        model.Params.Threads = Optimizer._threads()
        if solver_seed is not None:
            model.Params.Seed = int(solver_seed)
        model.Params.FeasibilityTol = float(env_params["solver_tolerance"])
        model.Params.OptimalityTol = float(env_params["solver_tolerance"])

        variables = model.addMVar(len(objective), lb=0.0, vtype=gp.GRB.CONTINUOUS)
        model.setObjective(objective @ variables, gp.GRB.MINIMIZE)
        for row, value in zip(matrix, upper):
            model.addConstr(row @ variables <= value)
        model.optimize()
        status = int(model.Status)
        status_name = _status_name(lambda: gp.GRB.Status, _GUROBI_STATUS_NAMES, status)
        if status == gp.GRB.OPTIMAL:
            result = np.maximum(variables.X, 0)
        else:
            result = None
        model.dispose()
        return result, status_name

    @staticmethod
    def solve_subsidies_copt(objective, matrix, upper):
        """COPT subsidy LP: same (subsidy, status name) contract (no seed parameter)."""
        if cp is None:
            raise ImportError(
                "coptpy is required for solver='copt'; install it or select solver='gurobi'"
            )
        # Suppress COPT's per-Env license banner.
        with contextlib.redirect_stdout(io.StringIO()):
            env = cp.Envr()
        model = env.createModel("BS_subsidy_LP")
        model.setParam(cp.COPT.Param.Logging, 0)
        model.setParam(cp.COPT.Param.Threads, Optimizer._threads())
        model.setParam(cp.COPT.Param.FeasTol, float(env_params["solver_tolerance"]))
        model.setParam(cp.COPT.Param.DualTol, float(env_params["solver_tolerance"]))
        variables = [
            model.addVar(lb=0.0, vtype=cp.COPT.CONTINUOUS) for _ in range(len(objective))
        ]
        model.setObjective(
            cp.quicksum(c * v for c, v in zip(objective, variables)), cp.COPT.MINIMIZE
        )
        for row, value in zip(matrix, upper):
            expression = cp.quicksum(
                c * variables[j] for j, c in enumerate(row) if c
            )
            model.addConstr(expression <= value)
        model.solve()
        status = int(model.status)
        status_name = _status_name(lambda: cp.COPT, _COPT_STATUS_NAMES, status)
        if status == cp.COPT.OPTIMAL:
            result = np.maximum(np.array([v.x for v in variables]), 0.0)
        else:
            result = None
        env.close()
        return result, status_name

