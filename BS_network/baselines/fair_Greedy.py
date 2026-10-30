"""Nearest-travel-time greedy allocation with full-group fairness repair.
``target_fixed`` selects the fixed ``fixed_level`` charging target."""

from collections import defaultdict
from time import perf_counter

import numpy as np
from config import RL_params
from common.optimization import Optimizer


class GreedyPolicy:
    """Time-first heuristic with minimum subsidies for its final repaired allocation."""
    def __init__(self, env, target_fixed=False):
        """Set the charging-target switch, identity, and solver seed streams."""
        self.environment = env  # BatterySwapEnv: joint demand and station state
        self.type = "Fair"  # str: reward convention
        # Charging-target and result-file naming switch.
        self.target_fixed = bool(target_fixed)  # bool: fixed fixed_level charging targets
        self.name = "fairGreedy_fixed" if self.target_fixed else "fairGreedy"

        self.seed = int(RL_params["seed"])  # int: root learning seed
        self.seeds = Optimizer.learning_seeds(self.seed)  # dict[str, int] of the four seed streams

    def greedy_assign(self, bundling):
        """Assign each request to the feasible station with the shortest travel time."""
        # assignment: int (R,), -1 for rejection; bundles: station ID -> request ID list.
        assignment = np.full(len(self.environment.new_demand), -1, dtype=int)
        bundles = defaultdict(list)  # station ID -> request IDs accepted this step

        # Revealed order; shortest-travel-time feasible station with bundle capacity.
        for i, request in enumerate(self.environment.new_demand):
            for j in sorted(
                request.feasible_stations, key=lambda j: (request.travel_times[j], j)
            ):
                if len(bundles[j]) >= bundling.max_size:
                    continue
                new_bundle = bundles[j] + [request.id]
                if bundling.bundle_feasibility(self.environment.Agents[j], new_bundle) is not None:
                    assignment[i] = j
                    bundles[j].append(request.id)
                    break
        return assignment, bundles

    def greedy_charging(self):
        """Return int (J,2) charging starts at the selected baseline target."""
        return np.array(
            [a.nested_charging(
                a.set_base_level(
                    self.environment.current_time, target_fixed=self.target_fixed
                )
            )
             for a in self.environment.Agents]
        )

    def greedy_schedule(self, bundling):
        """Build the greedy action; time each stage for the timing diagnostics."""
        self.environment.last_diagnostics = {}
        bundling.reset()

        # Timed stages: bundle construction, charging targets, subsidy LP.
        start = perf_counter()
        assignment, bundles = self.greedy_assign(bundling)
        bundle_seconds = perf_counter() - start

        start = perf_counter()
        charging = self.greedy_charging()
        candidate_seconds = perf_counter() - start

        start = perf_counter()
        subsidy = bundling.set_subsidies(assignment, bundles, self.seeds["solver"])
        solve_seconds = perf_counter() - start
        fairness_rejected = 0
        repair_loops = 0
        if subsidy is None and RL_params["greedy_repair"]:
            # Reject a served participant of a negative envy cycle and retry the LP.
            while subsidy is None:
                repair_loops += 1
                rejected_now = bundling.repair_greedy_subsidies(assignment, bundles)
                fairness_rejected += rejected_now
                start = perf_counter()
                subsidy = bundling.set_subsidies(
                    assignment, bundles, self.seeds["solver"]
                )
                solve_seconds += perf_counter() - start
                if rejected_now == 0:
                    break
        fallback = subsidy is None

        # Failed subsidy repair executes an envy-free all-reject action.
        if fallback:
            assignment[:] = -1
            subsidy = np.zeros(len(assignment))
            rejected = int((assignment == -1).sum())
            print(
                f"{self.name}: episode={self.environment.episode}, "
                f"step={self.environment.time_step}, "
                f"requests={len(self.environment.new_demand)}, "
                f"rejected={rejected}, "
                f"fairness_rejected={fairness_rejected}, "
                f"repair_loops={repair_loops}. ",
                flush=True,
            )
        action = {"assignment": assignment, "charging": charging, "subsidy": subsidy}

        # Greedy builds no BMIP: model_build=0, LP lands in model_solve_seconds.
        self.environment.last_diagnostics.update(
            bundle_seconds=bundle_seconds,
            candidate_seconds=candidate_seconds,
            model_build_seconds=0.0,
            model_solve_seconds=solve_seconds,
            fallback=int(fallback),
            subsidy_repaired=int(fairness_rejected > 0),
            fairness_rejected=fairness_rejected,
            solver_gap=0.0,
            bmip_requests=len(self.environment.new_demand),
            bmip_options=0,
            bmip_variables=0,
            bmip_constraints=0,
        )
        return action

    def scheduling(self, bundling, train_flag=False, episode=None, name=None):
        """Play one episode; train_flag exists only for the uniform call signature."""
        episode = self.environment.episode if episode is None else episode
        start = perf_counter()
        self.environment.reset()
        done = False

        while not done:
            action_start = perf_counter()
            action = self.greedy_schedule(bundling)
            self.environment.last_diagnostics["action_seconds"] = perf_counter() - action_start
            _, _, done = self.environment.step(action, self.type)

        result = self.environment.service_records(episode)
        result.update(run_time=(perf_counter() - start) / 60)
        self.environment.request_records(
            name or self.name, episode, evaluation=not train_flag,
            zero_envy=self.type == "Fair",
        )
        return result
