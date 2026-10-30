"""Demand revelation, joint actions, station transitions, and episode records."""

from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
import hashlib
import math
import numpy as np
import pandas as pd
from config import (
    MODEL_SIZE_COLUMNS,
    STEP_SAMPLE_COLUMNS,
    TIMING_COLUMNS,
    data_params,
    env_params,
    RL_params,
)
from common.geo import haversine_km
from common.outputs import atomic_to_csv, expansion_station_file, scenario_size
from network.BS_networks import BSNetwork
from common.instances import Instance
from simulator.station_agents import StationAgent


def _is_peak_step(step):
    """True for decision steps inside a demand-peak window (8-10, 12-17, 18-20)."""
    minutes = 7 * 60 + int(step) * int(env_params["delta_time"])
    hour = minutes / 60.0
    return any(start <= hour < end for start, end in ((8, 10), (12, 17), (18, 20)))


class BatterySwapEnv:
    """Battery-swap environment: demand streams, station physics, rewards, and records."""
    def __init__(self, seed=None):
        """Initialize configuration, network inputs, demand metadata, and runtime state."""
        self.seed = int(data_params["seed"] if seed is None else seed)
        self.start_time, self.end_time = data_params["start_time"], data_params["end_time"]
        self.decision_end = data_params["decision_end_time"]  # datetime: last order-acceptance cutoff
        self.delta_time, self.gamma = env_params["delta_time"], env_params["gamma"]
        self.total_steps = int(
            (self.end_time - self.start_time).total_seconds() / (60 * self.delta_time)
        )

        self.Clusters, self.Agents = [], []  # list[Cluster | StationAgent]
        self.order_data = pd.DataFrame()  # (N, 11) mapped trajectory table incl. upgrade_days
        self.all_demand, self.new_demand = [], []  # list[Request]: whole scenario | this epoch
        self.neighbor_mask = None  # (J, J) 0-1 static adjacency, filled by get_network
        self.detour_weights = None  # (J, J) static normalized detour edge weights
        self.station_speeds = None  # (J,) per-station road speed km/h, filled by get_network
        self.scenario = None  # int or None: expansion actually loaded, independent of task arguments
        self.source_fingerprints = {}  # input-file SHA256s, retained in resumable checkpoints

        self.episode = 0  # int: current episode index
        self.episode_seed = self.seed  # int: per-episode demand seed
        self.demand_by_step = []  # list[list[Request]]: one revealed bucket per decision interval
        self._reset_runtime()

        # Test-mode per-step sampling; set once and not cleared by _reset_runtime.
        self.capture_action_steps = False
        # Request-table switch: TestRunner enables it only for the pair; survives _reset_runtime.
        self.write_request_records = True

    def _reset_runtime(self):
        """Initialize the decision clock, episode totals, and diagnostic accumulators."""
        self.time_step = 0  # int: current decision index within the episode
        self.current_time = self.start_time + timedelta(minutes=self.delta_time)  # datetime: current decision clock
        self.hour = self.current_time.hour  # int: hour of the current decision
        self.total_rewards = 0.0  # float CNY: discounted episode rewards
        self.total_revenues = 0.0  # float CNY: service revenue
        self.total_subsidies = 0.0  # float CNY: cash subsidies paid
        self.total_envies = 0.0  # float CNY: realized envy
        self.total_charging_pv = 0.0  # float CNY: charging costs, discounted to the episode start
        self.total_charging_cash = 0.0  # float CNY: full committed undiscounted charging costs
        self.total_charged = 0  # int: charging starts
        self.active_charging_at_close = 0  # int: batteries charging at 23:00, before fulfillment
        self.completed_at_close = 0  # int: swaps completed by end_time, before fulfillment
        self.settlement_time = self.end_time  # datetime: final committed event time
        self.last_diagnostics = {}  # dict: per-step solver/timing diagnostics
        # Unified per-step samples (timings/BMIP sizes/bundle count); the dict/list
        # attributes below are views kept for compatibility.
        self.step_samples = {column: [] for column in STEP_SAMPLE_COLUMNS}
        self.timing_samples = {
            column: self.step_samples[column] for column in TIMING_COLUMNS
        }
        self.bmip_size_samples = {
            column: self.step_samples[column] for column in MODEL_SIZE_COLUMNS
        }
        self.bundle_size_samples = []  # list[int]: feasible bundle sizes generated per decision step
        self.bundle_number_samples = self.step_samples["bundle_number"]
        self.station_queue_samples = []  # list[int]: queue lengths per station per decision step
        self.station_queue_low_samples = []  # list[int]: low-class queue lengths per station per step
        self.station_queue_high_samples = []  # list[int]: high-class queue lengths per station per step
        self.station_demand_low_samples = []  # list[int]: reachable low-class demand per station per step
        self.station_demand_high_samples = []  # list[int]: reachable high-class demand per station per step
        self.fallback_steps = 0  # failed BMIP or subsidy-repair steps executing fallback
        self.solver_failed_steps = 0  # BMIP steps with no finite incumbent
        self.gap_steps = 0  # int: steps whose reported solver gap is > 0 (not a timeout indicator)
        self.exploration_steps = 0  # int: exploration steps
        self.action_step_records = []  # list[dict]: optional per-step action records
        # Test mode: per-step ready-stock snapshot for request_records.
        self.step_stock_snapshots = {}  # dict[int, dict[int, tuple[int, int]]]
        # Test mode: per-step mean queue-wait snapshot for request_records.
        self.step_wait_snapshots = {}  # dict[int, dict[int, float]]

    def get_episode_seed(self, episode, training=False):
        """Return the deterministic demand seed for the episode and selected stream."""
        # seed = base + 2*episode + int(training); default odd base gives even train seeds.
        return self.seed + 2 * int(episode) + int(training)

    def training_configuration(self):
        """Return the effective config snapshot plus the static station list."""
        excluded = {"check_dir", "data_dir", "result_dir", "MAX_Tests", "MAX_Episodes"}
        settings = {
            name: {key: value for key, value in group.items() if key not in excluded}
            for name, group in (
                ("data", data_params),
                ("environment", env_params),
                ("learning", RL_params),
            )
        }

        # stations: list of J tuples (id, cluster, lat, lon, stock (2,), kW, swap minutes)
        settings["stations"] = [
            (a.id, a.cluster, a.latitude, a.longitude, a.initial_batteries, a.grid_power, a.swap_time)
            for a in self.Agents
        ]
        settings["demand_base_seed"] = self.seed
        settings["source_fingerprints"] = dict(self.source_fingerprints)
        settings["clock"] = dict(
            start=self.start_time, cutoff=self.decision_end, end=self.end_time,
            delta=self.delta_time, steps=self.total_steps, gamma=self.gamma,
        )
        settings["decision_model"] = {
            "lookahead_demand": "episode-next-bucket-v1",
            "weighted_objective": "action-max-envy-arrival-soc-v2",
            "charging": "fixed-reserve-to-ready-v1",
            "fairness": "feasible-replacements-residual-v2",
            "service": "commitment-safe-battery-gap-v2",
            "feature_dtype": "float32-v1",
        }
        return settings

    def get_network(self, task="training", scenario=None):
        """Load tables and build agents; task="test" + scenario selects an expansion."""
        # Require station IDs 0..J-1 for policy array indexing.
        network = BSNetwork(seed=self.seed).load_network(task, scenario)
        self.Clusters = network.Clusters
        self.order_data = network.order_data
        self.scenario = (
            scenario_size(scenario)
        ) if task == "test" and scenario is not None else None
        station_file = (
            data_params["station_data"] if self.scenario is None
            else expansion_station_file(self.scenario)
        )
        self.source_fingerprints = {}
        for source in (station_file, data_params["grid_data"], data_params["cluster_data"],
                       data_params["order_data"]):
            digest = hashlib.sha256()
            with (Path(data_params["data_dir"]) / source).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            self.source_fingerprints[source] = digest.hexdigest()

        types = {c.id: c.type for c in self.Clusters}
        self.Agents = []
        for s in network.Stations:
            self.Agents.append(
                StationAgent(
                    s.id,
                    s.cluster,
                    types[s.cluster],
                    s.longitude,
                    s.latitude,
                    s.swap_time,
                    s.capacity,
                    s.initial_batteries,
                    s.grid_power)
            )

        # Static graph: driving (J,J) travel minutes -> neighbor_mask/detour_weights.
        locations = np.array([[a.latitude, a.longitude] for a in self.Agents])
        lat, lon = locations[:, 0], locations[:, 1]
        distances = (
            haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
            * env_params["road_distance_factor"]
        )  # float (J,J): road-adjusted driving distance, km
        # Focal-station area speed (centre 25 / suburb 30 km/h); divide each row by it.
        self.station_speeds = np.array(
            [float(env_params["vehicle_speed"][agent.type]) for agent in self.Agents]
        )  # float (J,): per-focal-station road speed, km/h
        driving = distances / self.station_speeds[:, None] * 60  # float (J,J): driving minutes
        # neighbor_mask[j, m] = 1: incoming edge m -> j, self-loop included.
        self.neighbor_mask = (driving <= env_params["max_detour_minutes"] +
                              float(env_params["time_tolerance"])).astype(np.int8)
        # Detour feature: minutes/cap clipped to [0,1].
        self.detour_weights = np.clip(
            np.asarray(
                driving / env_params["max_detour_minutes"], dtype=np.float32
            ),
            0.0,
            1.0,
        )
        for j in range(len(self.Agents)):
            # Report each station's neighbors with their driving distance and time.
            neighbor_rows = [
                (int(m), float(distances[j, m]), float(driving[j, m]))
                for m in np.flatnonzero(self.neighbor_mask[j])
                if m != j
            ]
            if neighbor_rows:
                detail = ", ".join(
                    f"{m} ({dist:.1f} km, {minutes:.1f} min)"
                    for m, dist, minutes in neighbor_rows
                )
            else:
                detail = "none"
            print(f"Station {j}: neighbors -> {detail}")
        return self

    def initialize_instance(self, scale=None, seed=None, episode=0):
        """Generate the episode demand and derive per-request reachability."""
        if scale is None:
            scale = data_params["scale"]
        self.episode = episode
        self.episode_seed = (
            self.get_episode_seed(episode) if seed is None else int(seed)
        )
        instance = Instance(self.order_data, scale, episode, self.episode_seed)
        instance.order_generation()
        self.all_demand = instance.requests

        # Road-adjusted travel in one R x J pass.
        locations = np.array([(a.latitude, a.longitude) for a in self.Agents])
        if self.all_demand:
            station_lat, station_lon = locations[:, 0], locations[:, 1]
            request_lat = np.array([r.latitude for r in self.all_demand])
            request_lon = np.array([r.longitude for r in self.all_demand])
            distances = haversine_km(
                request_lat[:, None], request_lon[:, None],
                station_lat[None, :], station_lon[None, :],
            ) * env_params["road_distance_factor"]
            # Column j scaled by station j's area speed.
            times = distances / self.station_speeds[None, :] * 60
            nearest = np.argmin(times, axis=1)  # int (R,): time-nearest station (detour baseline)

            # Feasible: energy reach plus detour cap.
            usable = np.array(
                [
                    (r.SOC - env_params["reserve_soc"])
                    * env_params["battery_kwh"][r.installed_class - 1]
                    for r in self.all_demand
                ]
            )
            detour = times - times[np.arange(len(self.all_demand)), nearest][:, None]
            feasible = (
                (distances * env_params["consumption_kwh_per_km"]
                 <= usable[:, None] + float(env_params["energy_tolerance"]))
                & (detour <= env_params["max_detour_minutes"]
                   + float(env_params["time_tolerance"]))
            )
            for k, request in enumerate(self.all_demand):
                request.nearest_station = self.Agents[int(nearest[k])].id
                request.nearest_time = float(times[k, nearest[k]])
                request.travel_distances = tuple(distances[k])
                request.travel_times = tuple(times[k])
                request.feasible_stations = tuple(
                    self.Agents[j].id for j in np.flatnonzero(feasible[k])
                )

        # Bucket requests by decision interval (right-closed).
        self.demand_by_step = [[] for _ in range(self.total_steps)]
        for request in self.all_demand:
            elapsed = (request.submission - self.start_time).total_seconds() / 60
            # Defensive clamp: pre-start submissions must not wrap to the last bucket.
            slot = max(0, math.ceil(elapsed / self.delta_time) - 1)
            slot = min(slot, self.total_steps - 1)
            self.demand_by_step[slot].append(request)

        empty_feasible = sum(1 for request in self.all_demand if not request.feasible_stations)
        if empty_feasible > 0:
            print(
                f"Episode {self.episode} (seed {self.episode_seed}): "
                f"{empty_feasible}/{len(self.all_demand)} requests have empty feasible stations."
            )
        return instance

    def reset(self):
        """Reset operations/accounting for the generated scenario; returns joint state."""
        for request in self.all_demand:
            request.reset_service()
        for agent in self.Agents:
            agent.reset_state()

        self._reset_runtime()

        self.reveal_requests()
        return self.get_states()

    def reveal_requests(self):
        """Reveal the current bucket and price its orders (fee + upgrade + reject cost)."""
        self.new_demand = (
            list(self.demand_by_step[self.time_step]) if self.time_step < self.total_steps else []
        )

        peak = (
            env_params["peak_electricity"][0]
            <= self.current_time.hour
            < env_params["peak_electricity"][1]
        )
        price = env_params["electric_price_peak" if peak else "electric_price_off"]
        service = env_params["swap_fee_peak" if peak else "swap_fee_off"]

        for request in self.new_demand:
            request.decision_time = self.current_time
            request.decision_step = self.time_step
            request.soc_reserved = request.SOC  # booking-time SOC
            request.energy_rate = price + service  # quoted CNY/kWh rate
            # Booking quote prices the delivered energy: a full requested-class
            # battery minus the returned battery's residual energy at booking.
            energy = (
                env_params["ready_soc"] * env_params["battery_kwh"][request.requested_class - 1]
                - request.soc_reserved * env_params["battery_kwh"][request.installed_class - 1]
            )
            # Upgrade orders carry their own 1-3 day charge period, drawn per source
            # row by load_network, instead of sharing the old fixed upgrade_days.
            upgrade = (
                env_params["upgrade_fee_per_day"] * request.upgrade_days
                if request.requested_class > request.installed_class
                else 0.0
            )
            request.upgrade_charge = upgrade
            # Booking quote prices the rejection cost; an accepted order is later
            # repriced at the assigned station on the arrival SOC.
            request.service_fee = energy * request.energy_rate + upgrade
            request.reject_cost = request.service_fee + env_params["rejection_time_cost"]

        # Reveal-time stock/wait snapshots, joined by decision_step in request_records.
        if self.capture_action_steps:
            self.step_stock_snapshots[self.time_step] = {
                a.id: tuple(int(x) for x in a.full_batteries) for a in self.Agents
            }
            self.step_wait_snapshots[self.time_step] = {
                a.id: (
                    float(np.mean([r.wait_time for r in a.queue]))
                    if a.queue else 0.0
                )
                for a in self.Agents
            }

    def get_agent_demand(self, agent, new_demand=None):
        """Return the requests that can reach the station."""
        demand = self.new_demand if new_demand is None else new_demand
        return [r for r in demand if agent.id in r.feasible_stations]

    def get_states(self):
        """Return {"time_step", "agents", "memberships", "nearest"} joint state."""
        demand = {a.id: self.get_agent_demand(a) for a in self.Agents}

        memberships = {
            j: tuple(tuple(r.id for r in requests if r.requested_class == l) for l in (1, 2))
            for j, requests in demand.items()
        }

        return {
            "time_step": self.time_step,
            "agents": {
                a.id: a.get_observe(self.time_step, self.hour, demand[a.id], self.current_time)
                for a in self.Agents
            },
            "memberships": memberships,
            # nearest: request ID -> time-nearest station, for the GAT edge features.
            "nearest": {r.id: r.nearest_station for r in self.new_demand},
        }

    def step(self, action, agent_type="Fair", envy_weight=None):
        """Validate and commit {"assignment","charging","subsidy"}, then advance
        one step (validation: fail-fast; otherwise infeasible bundles reject
        locally)."""
        assignments, charging, subsidies = (
            np.asarray(action["assignment"]),
            np.asarray(action["charging"]),
            np.asarray(action["subsidy"]),
        )
        assignments = assignments.astype(int)
        charging = np.array(
            [a.validate_charging(charging[j]) for j, a in enumerate(self.Agents)], dtype=int
        )

        # Reserve all bundles up front (no partial commits); fail-fast or local rejection.
        accepted_by_station, schedules = [], []
        for agent in self.Agents:
            accepted = [r for i, r in enumerate(self.new_demand) if assignments[i] == agent.id]
            schedule = agent.reserve(self.current_time, accepted)
            if schedule is None:
                if RL_params["validation"]:
                    reason = agent.last_reserve_reason or "unknown reservation failure"
                    print(
                        f"############ Station {agent.id}: no feasible timetable for "
                        f"the committed bundle at step {self.time_step}; "
                        f"reason: {reason} ############",
                        flush=True,
                    )
                    raise ValueError(
                        f"Station {agent.id}: infeasible committed bundle at step "
                        f"{self.time_step}: {reason}"
                    )
                # Validation disabled: reject this station's newly assigned
                # orders instead of aborting the step.
                index = {r.id: i for i, r in enumerate(self.new_demand)}
                rejected_ids = [r.id for r in accepted]
                assignments[[index[rid] for rid in rejected_ids]] = -1
                accepted = []
                schedule = agent.reserve(self.current_time, [])
                if schedule is None:
                    print(
                        f"Station {agent.id}: even the empty re-reservation failed "
                        f"at step {self.time_step}; keeping prior commitments only."
                    )
                print(
                    f"Station {agent.id}: assigned bundle infeasible at step "
                    f"{self.time_step}; {len(rejected_ids)} order(s) rejected."
                )
            accepted_by_station.append(accepted)
            schedules.append(schedule)

        # Audit before mutation; solver subsidy precision preserved, reward/envy float32.
        weight = float(RL_params["envy_weight"] if envy_weight is None else envy_weight)
        envy = self.realized_envy(
            assignments, schedules, subsidies if agent_type == "Fair" else None
        )
        if agent_type == "Fair":
            residual = float(envy.max()) if envy.size else 0.0
            self.last_diagnostics["fairness_residual"] = residual
            if (
                RL_params["validation"]
                and residual > env_params["fairness_warning_threshold"]
            ):
                print(f"############ The action violates fairness: residual={residual:.4f} ############")

        # Arrival SOC at the assigned station: booking SOC minus driving consumption.
        for i, request in enumerate(self.new_demand):
            j = int(assignments[i])
            if j < 0:
                request.soc_arrival = None
                continue
            # Arrival SOC and the realized fee are both priced at the assigned station.
            request.soc_arrival = request.arrival_soc(j)
            request.service_fee = request.fee_at(j)

        # Reward uses PV; profit counts the entire undiscounted committed batch cost.
        reward = np.zeros(len(self.Agents), dtype=np.float32)
        revenue = np.zeros_like(reward)
        pv_cost = np.array(
            [a.calculate_charging(self.current_time, q)
             for a, q in zip(self.Agents, charging)],
            dtype=np.float32,
        )
        cash_cost = np.array(
            [
                a.calculate_charging(self.current_time, q, False)
                for a, q in zip(self.Agents, charging)
            ],
            dtype=np.float32,
        )

        # Fees/subsidies credit the assigned station; rejected orders credit nearest.
        for i, request in enumerate(self.new_demand):
            request.station = int(assignments[i])
            request.subsidy = float(subsidies[i])
            request.envy = float(envy[i])
            credited = request.station if request.station >= 0 else request.nearest_station
            if request.station >= 0:
                revenue[credited] += request.service_fee
            reward[credited] -= request.subsidy
        reward += revenue - pv_cost

        if agent_type == "Weighted":
            for i, request in enumerate(self.new_demand):
                credited = request.station if request.station >= 0 else request.nearest_station
                reward[credited] -= weight * envy[i]

        # Commit stations, then advance events to the next decision.
        next_time = self.current_time + timedelta(minutes=self.delta_time)
        for j, agent in enumerate(self.Agents):
            accepted = accepted_by_station[j]
            schedule = schedules[j]

            agent.start_charging(self.current_time, charging[j])  # empty_batteries down, charging/active_power up
            if schedule is not None:
                agent.implement_schedule(accepted, schedule)  # queue up, request time fields written, inventory_position down
            agent.advance(next_time)  # settle finished charging/swaps; advance to the next decision

        # Per-step queue/demand samples for the episode records.
        self.station_queue_samples.extend(len(agent.queue) for agent in self.Agents)
        self.station_queue_low_samples.extend(
            sum(r.requested_class == 1 for r in agent.queue) for agent in self.Agents
        )
        self.station_queue_high_samples.extend(
            sum(r.requested_class == 2 for r in agent.queue) for agent in self.Agents
        )

        # Reachable demand counts every request that can reach the station.
        low_demand = np.zeros(len(self.Agents), dtype=int)
        high_demand = np.zeros(len(self.Agents), dtype=int)
        for request in self.new_demand:
            for station in request.feasible_stations:
                if request.requested_class == 1:
                    low_demand[station] += 1
                else:
                    high_demand[station] += 1
        self.station_demand_low_samples.extend(int(x) for x in low_demand)
        self.station_demand_high_samples.extend(int(x) for x in high_demand)

        # Accumulate episode accounts; rewards discounted by decision index.
        self.total_revenues += float(revenue.sum())
        self.total_charging_pv += self.gamma**self.time_step * float(pv_cost.sum())
        self.total_charging_cash += float(cash_cost.sum())
        self.total_subsidies += float(subsidies.sum())
        self.total_envies += sum(r.envy for r in self.new_demand)
        self.total_rewards += self.gamma**self.time_step * float(reward.sum())
        self.total_charged += int(charging.sum())

        # Fold timing/solver diagnostics into the unified per-step samples.
        for column in TIMING_COLUMNS:
            sample = self.last_diagnostics.get(column)
            self.step_samples[column].append(0.0 if sample is None else float(sample))
        # Test mode additionally samples the BMIP sizes of every decision step.
        if self.capture_action_steps:
            for column in MODEL_SIZE_COLUMNS:
                self.step_samples[column].append(
                    int(self.last_diagnostics.get(column, 0))
                )
        self.fallback_steps += int(bool(self.last_diagnostics.get("fallback", 0)))
        self.solver_failed_steps += int(bool(
            self.last_diagnostics.get("solver_failed", 0)
        ))
        self.gap_steps += int((self.last_diagnostics.get("solver_gap") or 0.0) > 0)
        self.exploration_steps += int(bool(self.last_diagnostics.get("exploration", 0)))

        # Test-mode action records; bundle_number comes from the just-executed pool
        # (0 without one, e.g. Greedy); timing samples were appended above.
        if self.capture_action_steps:
            self.action_step_records.append(
                dict(
                    episode=self.episode,
                    step=self.time_step,
                    is_peak=int(_is_peak_step(self.time_step)),
                    bundle_number=int(self.step_samples["bundle_number"][-1])
                    if self.step_samples["bundle_number"]
                    else 0,
                    bundle_seconds=float(self.step_samples["bundle_seconds"][-1]),
                    candidate_seconds=float(self.step_samples["candidate_seconds"][-1]),
                    bmip_model_seconds=(
                        float(self.step_samples["model_build_seconds"][-1])
                        + float(self.step_samples["model_solve_seconds"][-1])
                    ),
                    action_seconds=float(self.step_samples["action_seconds"][-1]),
                    bmip_requests=int(self.last_diagnostics.get("bmip_requests", 0)),
                    bmip_options=int(self.last_diagnostics.get("bmip_options", 0)),
                    bmip_variables=int(self.last_diagnostics.get("bmip_variables", 0)),
                    bmip_constraints=int(self.last_diagnostics.get("bmip_constraints", 0)),
                )
            )

        # Advance the clock and reveal the next bucket.
        self.time_step += 1
        self.current_time, self.hour = next_time, next_time.hour
        done = (self.time_step >= self.total_steps)
        if done:
            # Final decision committed; charging and swaps keep running until finished.
            self.active_charging_at_close = sum(
                int(agent.charging_batteries.sum()) for agent in self.Agents
            )
            self.completed_at_close = sum(r.complete for r in self.all_demand)
            self.settle_commitments()
        self.reveal_requests()
        next_state = self.get_states()
        return next_state, reward, done

    def realized_envy(self, assignments, schedules, subsidies=None):
        """Recompute each request's maximum feasible envy, optionally net of subsidies."""
        # Lenient path may carry None schedule entries (counted as rejected);
        # infeasible replacements contribute no pair.
        requests = self.new_demand
        index = {r.id: i for i, r in enumerate(requests)}
        by_id = {r.id: r for r in requests}
        by_station = {agent.id: [] for agent in self.Agents}
        own = {}
        for i, request in enumerate(requests):
            j = int(assignments[i])
            if j < 0:
                own[request.id] = request.reject_cost
                continue
            if schedules[j] is None:
                # Lenient path (validation off): unreservable station bundle.
                own[request.id] = request.reject_cost
                continue
            by_station[j].append(request)
            wait = schedules[j][request.id][3]
            detour = max(request.travel_times[j] - request.nearest_time, 0.0)
            own[request.id] = request.fee_at(j) + env_params["VOT"] * (detour + wait)

        groups = defaultdict(list)
        for request in requests:
            groups[(request.cluster, request.requested_class)].append(request.id)
        envy = np.zeros(len(requests), dtype=np.float32)
        for ids in groups.values():
            for rid in ids:
                for other in ids:
                    if rid == other:
                        continue
                    j = int(assignments[index[other]])
                    if j < 0:
                        counterfactual = by_id[rid].reject_cost
                    else:
                        bundle = tuple(sorted(r.id for r in by_station[j]))
                        replacement = (
                            bundle
                            if rid in bundle
                            else tuple(sorted((set(bundle) - {other}) | {rid}))
                        )
                        schedule = self.Agents[j].reserve(
                            self.current_time, [by_id[x] for x in replacement]
                        )
                        if schedule is None:
                            continue  # infeasible replacement: no comparison
                        wait = schedule[rid][3]
                        detour = max(
                            by_id[rid].travel_times[j] - by_id[rid].nearest_time, 0.0
                        )
                        counterfactual = (
                            by_id[rid].fee_at(j) + env_params["VOT"] * (detour + wait)
                        )
                    gap = own[rid] - counterfactual
                    if subsidies is not None:
                        gap += float(subsidies[index[other]]) - float(subsidies[index[rid]])
                    if gap > envy[index[rid]]:
                        envy[index[rid]] = gap
        return envy

    def settle_commitments(self):
        """Complete every committed swap and charging job after the horizon."""
        # Clock advances only to settle committed events; nothing charges beyond commitments.
        horizon = self.current_time
        while any(agent.charging or agent.queue or agent.services for agent in self.Agents):
            upcoming = []
            for agent in self.Agents:
                if agent.charging:
                    upcoming.append(min(finish for finish, _, _ in agent.charging))
                if agent.queue:
                    upcoming.append(min(r.begin_time for r in agent.queue))
                if agent.services:
                    upcoming.append(min(r.departure for r in agent.services))
            horizon = max(horizon, min(upcoming))
            for agent in self.Agents:
                agent.advance(horizon)
        self.settlement_time = horizon

    def _step_sample_statistics(self, columns, suffixes=("mean", "p95", "max", "total")):
        """Return {col}_{suffix} stats over the unified per-step samples;
        empty series report 0.0."""
        records = {}
        for column in columns:
            values = np.asarray(self.step_samples[column], dtype=np.float64)
            if "mean" in suffixes:
                records[f"{column}_mean"] = float(values.mean()) if values.size else 0.0
            if "p95" in suffixes:
                records[f"{column}_p95"] = float(np.quantile(values, 0.95)) if values.size else 0.0
            if "max" in suffixes:
                records[f"{column}_max"] = float(values.max()) if values.size else 0.0
            if "total" in suffixes:
                records[f"{column}_total"] = float(values.sum())
        return records

    def timing_records(self):
        """Return stage timings ({stage}_{mean,p95,max,total}, seconds) plus
        the three solver counters."""
        records = self._step_sample_statistics(TIMING_COLUMNS)

        records["fallback_steps"] = int(self.fallback_steps)
        records["solver_failed_steps"] = int(self.solver_failed_steps)
        records["positive_gap_steps"] = int(self.gap_steps)
        return records

    def action_records(self):
        """Return bundle_mean/bundle_max and exploration_steps."""
        served = [r for r in self.all_demand if r.station >= 0]
        bundles = Counter((r.station, r.decision_time) for r in served)
        decisions = len(self.Agents) * self.total_steps
        return dict(
            bundle_mean=len(served) / decisions if decisions else 0.0,
            bundle_max=max(bundles.values(), default=0),
            exploration_steps=int(self.exploration_steps),
        )

    def step_statistics_records(self):
        """Return per-step action-field statistics (population variance ddof=0;
        empty series report 0.0). bmip_model_seconds = build + solve; test mode
        adds BMIP sizes and bundle_number p95/total."""
        build = np.asarray(self.step_samples["model_build_seconds"], dtype=np.float64)
        solve = np.asarray(self.step_samples["model_solve_seconds"], dtype=np.float64)
        records = {}
        for name, values in (
            ("bundle_number", np.asarray(self.step_samples["bundle_number"], dtype=np.float64)),
            ("bmip_model_seconds", build + solve),
        ):
            records[f"{name}_mean"] = float(values.mean()) if values.size else 0.0
            records[f"{name}_var"] = float(values.var()) if values.size else 0.0
            records[f"{name}_max"] = float(values.max()) if values.size else 0.0
        for column in ("bundle_seconds", "candidate_seconds", "action_seconds"):
            values = np.asarray(self.step_samples[column], dtype=np.float64)
            records[f"{column}_var"] = float(values.var()) if values.size else 0.0
        if self.capture_action_steps:
            records.update(self._step_sample_statistics(MODEL_SIZE_COLUMNS))
            records.update(
                self._step_sample_statistics(
                    ("bundle_number",), suffixes=("p95", "total")
                )
            )
        return records

    def service_records(self, episode, loss=None):
        """Return the one-row episode summary (the *_summary.csv schema)."""
        # Identity and demand/service.
        accepted = [r for r in self.all_demand if r.station >= 0]
        completed = sum(r.complete for r in self.all_demand)
        record = dict(
            episode=episode,
            seed=self.episode_seed,
            total_demand=len(self.all_demand),
            served=len(accepted),
            completed=completed,
            outstanding_accepted=len(accepted) - completed,
            charged=self.total_charged,
            rejection=(1 - len(accepted) / len(self.all_demand)) if self.all_demand else 0.0,
            low_demand=sum(r.requested_class == 1 for r in self.all_demand),
            high_demand=sum(r.requested_class == 2 for r in self.all_demand),
            upgrade_demand=sum(r.installed_class < r.requested_class for r in self.all_demand),
            outstanding_upgrades=sum(
                r.station >= 0 and not r.complete and r.installed_class < r.requested_class
                for r in self.all_demand
            ),
        )

        # Economics; normalized_* scale totals by the request count.
        profit = self.total_revenues - self.total_subsidies - self.total_charging_cash
        demand_count = len(self.all_demand)
        record.update(
            total_rewards=self.total_rewards,
            total_revenues=self.total_revenues,
            total_profits=profit,
            normalized_reward=self.total_rewards / demand_count if demand_count else 0.0,
            normalized_profit=profit / demand_count if demand_count else 0.0,
            committed_cost_pv=self.total_charging_pv,
            committed_cost_cash=self.total_charging_cash,
            total_subsidies=self.total_subsidies,
            total_envies=self.total_envies,
        )

        # Global envy/subsidy.
        envies = [r.envy for r in self.all_demand]
        subsidies = [r.subsidy for r in self.all_demand]
        record.update(
            max_envy=max(envies, default=0.0),
            mean_envy=float(np.mean(envies)) if envies else 0.0,
            mean_subsidy=float(np.mean(subsidies)) if subsidies else 0.0,
            max_subsidy=max(subsidies, default=0.0),
        )

        # Per-class envy/subsidy.
        low = [r for r in self.all_demand if r.requested_class == 1]
        high = [r for r in self.all_demand if r.requested_class == 2]
        low_envies = [r.envy for r in low]
        high_envies = [r.envy for r in high]
        low_subsidies = [r.subsidy for r in low]
        high_subsidies = [r.subsidy for r in high]
        record.update(
            mean_envy_low=float(np.mean(low_envies)) if low_envies else 0.0,
            max_envy_low=max(low_envies, default=0.0),
            mean_envy_high=float(np.mean(high_envies)) if high_envies else 0.0,
            max_envy_high=max(high_envies, default=0.0),
            mean_subsidy_low=float(np.mean(low_subsidies)) if low_subsidies else 0.0,
            max_subsidy_low=max(low_subsidies, default=0.0),
            mean_subsidy_high=float(np.mean(high_subsidies)) if high_subsidies else 0.0,
            max_subsidy_high=max(high_subsidies, default=0.0),
        )

        # Per-class service/charging/rejection.
        low_served = [r for r in low if r.station >= 0]
        high_served = [r for r in high if r.station >= 0]
        low_completed = sum(r.complete for r in low)
        high_completed = sum(r.complete for r in high)
        record.update(
            served_low=len(low_served),
            served_high=len(high_served),
            completed_low=int(low_completed),
            completed_high=int(high_completed),
            charged_low=int(sum(a.charged[0] for a in self.Agents)),
            charged_high=int(sum(a.charged[1] for a in self.Agents)),
            rejection_low=(1 - len(low_served) / len(low)) if low else 0.0,
            rejection_high=(1 - len(high_served) / len(high)) if high else 0.0,
        )

        # Per-class waiting/detour, minutes.
        low_waits = [r.wait_time for r in low_served]
        high_waits = [r.wait_time for r in high_served]
        low_detours = [r.detour_time for r in low_served]
        high_detours = [r.detour_time for r in high_served]
        record.update(
            mean_wait_low=float(np.mean(low_waits)) if low_waits else 0.0,
            max_wait_low=max(low_waits, default=0.0),
            mean_wait_high=float(np.mean(high_waits)) if high_waits else 0.0,
            max_wait_high=max(high_waits, default=0.0),
            mean_detour_low=float(np.mean(low_detours)) if low_detours else 0.0,
            max_detour_low=max(low_detours, default=0.0),
            mean_detour_high=float(np.mean(high_detours)) if high_detours else 0.0,
            max_detour_high=max(high_detours, default=0.0),
        )

        # Station aggregates: served = accepted at that station, completed = finished swaps.
        station_served = [
            int(sum(1 for r in self.all_demand if r.station == a.id)) for a in self.Agents
        ]
        station_completed = [int(a.completed.sum()) for a in self.Agents]
        station_charged = [int(a.charged.sum()) for a in self.Agents]
        record.update(
            mean_station_serve=float(np.mean(station_served)) if station_served else 0.0,
            max_station_serve=max(station_served, default=0),
            mean_station_complete=float(np.mean(station_completed)) if station_completed else 0.0,
            max_station_complete=max(station_completed, default=0),
            mean_station_charge=float(np.mean(station_charged)) if station_charged else 0.0,
            max_station_charge=max(station_charged, default=0),
        )

        # Station queue samples: mean queue length over queued samples only.
        queued_samples = [q for q in self.station_queue_samples if q > 0]
        record.update(
            mean_station_que=(
                float(np.mean(queued_samples)) if queued_samples else 0.0
            ),
            max_station_que=max(self.station_queue_samples, default=0),
            max_station_que_low=float(np.max(self.station_queue_low_samples)) if self.station_queue_low_samples else 0.0,
            max_station_que_high=float(np.max(self.station_queue_high_samples)) if self.station_queue_high_samples else 0.0,
        )

        # Station reachable-demand maxima.
        record.update(
            max_station_demand_low=max(self.station_demand_low_samples, default=0),
            max_station_demand_high=max(self.station_demand_high_samples, default=0),
        )

        # Regional aggregates.
        centre = [r for r in self.all_demand if r.area == "centre"]
        suburb = [r for r in self.all_demand if r.area == "suburb"]
        centre_envies = [r.envy for r in centre]
        suburb_envies = [r.envy for r in suburb]
        centre_subsidies = [r.subsidy for r in centre]
        suburb_subsidies = [r.subsidy for r in suburb]
        record.update(
            mean_centre_envy=float(np.mean(centre_envies)) if centre_envies else 0.0,
            max_centre_envy=max(centre_envies, default=0.0),
            mean_centre_subsidy=float(np.mean(centre_subsidies)) if centre_subsidies else 0.0,
            max_centre_subsidy=max(centre_subsidies, default=0.0),
            mean_suburb_envy=float(np.mean(suburb_envies)) if suburb_envies else 0.0,
            max_suburb_envy=max(suburb_envies, default=0.0),
            mean_suburb_subsidy=float(np.mean(suburb_subsidies)) if suburb_subsidies else 0.0,
            max_suburb_subsidy=max(suburb_subsidies, default=0.0),
        )

        # Global waiting/detour, minutes.
        waits = [r.wait_time for r in accepted]
        detours = [r.detour_time for r in accepted]
        record.update(
            mean_wait=float(np.mean(waits)) if waits else 0.0,
            max_wait=max(waits, default=0.0),
            mean_detour=float(np.mean(detours)) if detours else 0.0,
            max_detour=max(detours, default=0.0),
        )

        # Inventory/horizon.
        record.update(
            active_charging_at_close=self.active_charging_at_close,
            completed_at_close=self.completed_at_close,
            settlement_time=self.settlement_time.isoformat(),
            steps=self.total_steps,
        )

        # Bundle statistics; NaN when no bundle pool was generated (Greedy).
        record.update(
            mean_bundle_size=(
                float(np.mean(self.bundle_size_samples))
                if self.bundle_size_samples else float("nan")
            ),
            max_bundle_size=(
                max(self.bundle_size_samples) if self.bundle_size_samples else float("nan")
            ),
            mean_bundle_number=(
                float(np.mean(self.step_samples["bundle_number"]))
                if self.step_samples["bundle_number"] else float("nan")
            ),
            max_bundle_number=(
                max(self.step_samples["bundle_number"])
                if self.step_samples["bundle_number"] else float("nan")
            ),
        )

        # Episode-mean TD loss (0.0 for static baselines).
        record.update(loss=0.0 if loss is None else float(loss))

        # Timing and action diagnostics.
        record.update(self.timing_records())
        record.update(self.step_statistics_records())
        record.update(self.action_records())
        return record

    def request_records(self, stem, episode, evaluation=False, zero_envy=False):
        """Write per-order rows for evaluation streams; zero_envy records 0.0
        for fully compensated fair policies."""
        if not evaluation or not self.write_request_records:
            return None
        directory = Path(data_params["result_dir"])
        directory.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in self.all_demand:
            # Reveal-time ready stock of the requested class (test-mode snapshots only).
            step_stock = self.step_stock_snapshots.get(r.decision_step, {})
            assigned = step_stock.get(r.station)
            station_stock = assigned[r.requested_class - 1] if assigned else None
            nearest = step_stock.get(r.nearest_station)
            nearest_stock = nearest[r.requested_class - 1] if nearest else None
            wait_step = self.step_wait_snapshots.get(r.decision_step, {})
            cluster_waits = [
                wait_step[agent.id]
                for agent in self.Agents
                if agent.cluster == r.cluster and agent.id in wait_step
            ]
            cluster_pre_wait = (
                float(np.mean(cluster_waits)) if cluster_waits else None
            )
            if r.station >= 0:
                station_agent = self.Agents[r.station]
                station_cluster, station_area = station_agent.cluster, station_agent.type
            else:
                station_cluster = station_area = None
            # cost = fee + VOT*(detour + wait); reject_cost when rejected.
            cost = (
                r.service_fee + env_params["VOT"] * (r.detour_time + r.wait_time)
                if r.station >= 0
                else r.reject_cost
            )
            rows.append(
                dict(
                    episode=episode,
                    seed=self.episode_seed,
                    request=r.id,
                    submission=r.submission.isoformat() if r.submission else None,
                    decision_step=r.decision_step,
                    sample_fallback=r.sample_fallback,
                    cluster=r.cluster,
                    area=r.area,
                    installed_class=r.installed_class,
                    requested_class=r.requested_class,
                    nearest_station=r.nearest_station,
                    nearest_time=r.nearest_time,
                    served=int(r.station >= 0),
                    completed=r.complete,
                    station=r.station,
                    station_cluster=station_cluster,
                    station_area=station_area,
                    station_stock=station_stock,
                    nearest_stock=nearest_stock,
                    cluster_pre_wait=cluster_pre_wait,
                    detour_minutes=r.detour_time,
                    waiting_minutes=r.wait_time,
                    soc_reserved=r.soc_reserved,
                    soc_arrival=r.soc_arrival,
                    service_fee=r.service_fee if r.station >= 0 else 0,
                    subsidy=r.subsidy,
                    max_envy=0.0 if zero_envy else r.envy,
                    net_disutility=cost - r.subsidy,
                )
            )

        # Explicit column order keeps zero-demand files readable.
        request_columns = (
            "episode seed request submission decision_step sample_fallback cluster area "
            "installed_class requested_class nearest_station nearest_time served completed "
            "station station_cluster station_area station_stock nearest_stock cluster_pre_wait "
            "detour_minutes waiting_minutes soc_reserved soc_arrival "
            "service_fee subsidy max_envy net_disutility"
        ).split()
        atomic_to_csv(
            pd.DataFrame(rows, columns=request_columns),
            directory / f"{stem}_requests_{episode:03d}.csv",
        )

        return directory
