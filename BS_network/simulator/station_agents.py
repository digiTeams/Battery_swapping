"""Station battery/queue physics: charging, swaps, scheduling, and costs."""

import copy
import math
from collections import Counter
from datetime import timedelta
from itertools import product

import numpy as np

from config import data_params, env_params, RL_params


class StationAgent:
    """Two-class charging and single-bay FCFS swap service with battery-gap bypass."""
    def __init__(
        self,
        sid,
        cid,
        ctype,
        long,
        lat,
        swap_time,
        battery_cap,
        initial_batteries,
        grid_power,
    ):
        """Build static specs and the fixed-depth two-class charging model."""
        self.id = int(sid)  # int: station identifier
        self.cluster = int(cid)  # int: home fine-cluster ID
        self.type = ctype  # str: area type ("centre" or "suburb")
        self.longitude, self.latitude = float(long), float(lat)  # float degrees: station coordinates
        self.swap_time = float(swap_time)  # float minutes: swap service duration
        self.capacity = int(battery_cap)  # int: compartment count
        self.initial_batteries = tuple(int(x) for x in initial_batteries)  # tuple[int,int] (low,high)
        self.grid_power = float(grid_power)  # float kW: grid connection limit

        # Decision clock and discount copied once from config, mirroring the
        # BatterySwapEnv convention, so every timing/cost helper reads the instance
        # value instead of the mutable global.
        self.delta_time = float(env_params["delta_time"])  # float minutes: decision interval
        self.gamma = float(env_params["gamma"])  # float: per-step discount factor

        # Charging model (class 0/1 = low/high); every returned battery recharges
        # from reserve_soc to ready_soc, so batch energy is SOC-independent.
        self.charging_power = np.array(env_params["charging_power_kw"], dtype=float)  # float64 (2,): ceil-sensitive physical power

        self.reset_state()

    def reset_state(self):
        """Restore initial stock and clear commitments."""
        self.full_batteries = np.array(self.initial_batteries, dtype=int)  # int (2,): ready stock
        self.empty_batteries = np.zeros(2, dtype=int)  # int (2,): depleted stock
        self.charging = []  # list[(finish datetime, class index, quantity)]
        self.queue = []  # unstarted swaps waiting for the single bay: list[Request]
        self.services = []  # swaps currently occupying the bay: list[Request]
        self.busy_until = data_params["start_time"]

        self.completed = np.zeros(2, dtype=int)
        self.charged = np.zeros(2, dtype=int)
        # Why the last reserve() returned None; None on success.
        self.last_reserve_reason = None
        self.refresh_state()

    def batch_steps(self, l):
        """Decision intervals to recharge one class-l battery at full power (fixed reserve->ready)."""
        interval_kwh = float(self.charging_power[l]) * self.delta_time / 60.0
        required = (
            (env_params["ready_soc"] - env_params["reserve_soc"])
            * env_params["battery_kwh"][l]
        )
        return int(math.ceil(required / interval_kwh))

    def refresh_state(self):
        """Recompute charging counts, active power, and net inventory."""
        self.charging_batteries = np.array(
            [sum(n for _, k, n in self.charging if k == l) for l in range(2)], dtype=int
        )  # int (2,)
        self.active_power = sum(int(n) * float(p) for n, p in zip(
            self.charging_batteries, self.charging_power
        ))

        reserved = np.array(
            [sum(r.requested_class == l + 1 for r in self.queue) for l in range(2)], dtype=int
        )
        self.inventory_position = self.full_batteries + self.charging_batteries - reserved

    def get_observe(self, time_step, hour, demand=(), current_time=None):
        """Return this station's demand, queue, waiting, and inventory observation."""
        # avg_waiting includes swap time; next_ready_* = minutes until ready
        # (0 now; a fresh batch's duration if none charging); spare_grid =
        # remaining power fraction; bay_free = minutes until the bay is idle.
        now = (
            current_time
            if current_time is not None
            else data_params["start_time"]
            + timedelta(minutes=(time_step + 1) * self.delta_time)
        )
        next_ready = [0.0, 0.0]
        for l in (0, 1):
            if self.full_batteries[l] <= 0:
                finishes = [finish for finish, k, _ in self.charging if k == l]
                if finishes:
                    next_ready[l] = max(0.0, (min(finishes) - now).total_seconds() / 60.0)
                else:
                    next_ready[l] = float(self.batch_steps(l) * self.delta_time)
        spare_grid = max(0.0, min(1.0, (self.grid_power - self.active_power) / self.grid_power))
        bay_free = max(0.0, (self.busy_until - now).total_seconds() / 60.0)
        return {
            "time_step": time_step,
            "hour": hour,
            "new_order": tuple(sum(r.requested_class == l for r in demand) for l in (1, 2)),

            # upgrade_order: tuple[int] (1,) of low-to-high requests.
            "upgrade_order": (
                sum(r.installed_class == 1 and r.requested_class == 2 for r in demand),
            ),
            "que_order": tuple(sum(r.requested_class == l for r in self.queue) for l in (1, 2)),
            "avg_waiting": tuple(
                (
                    float(np.mean([r.wait_time for r in self.queue if r.requested_class == l]))
                    if any(r.requested_class == l for r in self.queue)
                    else 0.0
                )
                for l in (1, 2)
            ),
            "full_batteries": tuple(int(x) for x in self.full_batteries),
            "inventory_position": tuple(int(x) for x in self.inventory_position),
            "empty_batteries_low": int(self.empty_batteries[0]),
            "empty_batteries_high": int(self.empty_batteries[1]),
            "next_ready_low": next_ready[0],
            "next_ready_high": next_ready[1],
            "spare_grid": spare_grid,
            "bay_free": bay_free,

            # capacity: int compartment count (normalized by max_size downstream).
            "capacity": self.capacity,
            # type: one-hot area indicator, [1, 0] = centre, [0, 1] = suburb.
            "type": (1, 0) if self.type == "centre" else (0, 1),
        }

    def enum_base_levels(self):
        """Return the enumerated base_stocks**2 nested (total, high) target grid."""
        cumulative = np.cumsum(self.initial_batteries[::-1])[::-1]  # int (2,): (total, high)

        levels = []
        for fractions in product(RL_params["base_stocks"], repeat=2):
            targets = tuple(
                int(math.ceil(fraction * c)) for fraction, c in zip(fractions, cumulative)
            )
            levels.append((max(targets[0], targets[1]), targets[1]))
        return tuple(levels)

    def set_base_level(self, current_time, target_fixed=False):
        """Return the baseline (total, high) charging targets by anticipated
        completion time; target_fixed pins every interval to fixed_level."""
        cumulative = np.cumsum(self.initial_batteries[::-1])[::-1]

        intervals = (
            (data_params["peak_hour_centre"],)
            if self.type == "centre"
            else data_params["peak_hour_suburb"]
        )

        levels = []
        for l in range(2):
            finish = current_time + timedelta(
                minutes=self.batch_steps(l) * self.delta_time
            )
            hour = finish.hour + finish.minute / 60
            if target_fixed:
                fraction = float(RL_params["fixed_level"])
            else:
                peak = any(a <= hour < b for a, b in intervals)
                fraction = RL_params["base_stocks"][1 if peak else 0]
            levels.append(math.ceil(cumulative[l] * fraction))
        levels[0] = max(levels[0], levels[1])
        return tuple(levels)

    def nested_charging(self, levels):
        """Return charging starts int (2,) meeting nested targets within stock/power."""
        # Apply one total power budget, including tolerance, across both classes.
        position = self.inventory_position.copy()
        remaining = self.grid_power - self.active_power + float(env_params["power_tolerance"])
        quantities = np.zeros(2, dtype=int)

        # High class first, then total, within depleted stock and spare power.
        for l in (1, 0):
            deficit = max(int(levels[l]) - int(position[l:].sum()), 0)
            quantities[l] = min(
                deficit,
                int(self.empty_batteries[l]),
                max(0, math.floor(remaining / self.charging_power[l])),
            )
            position[l] += quantities[l]
            remaining -= quantities[l] * self.charging_power[l]
        return quantities

    def reserve(self, current_time, requests):
        """Return a feasible single-bay timetable preserving earlier commitments;
        on failure last_reserve_reason records why (no state modified)."""
        # Reject duplicate IDs, unreachable stations, and invalid battery classes.
        merged = list(self.queue) + list(requests)
        ids = [r.id for r in merged] + [r.id for r in self.services]
        if len(ids) != len(set(ids)):
            duplicates = sorted(rid for rid, count in Counter(ids).items() if count > 1)
            self.last_reserve_reason = (
                f"duplicate request IDs {duplicates} among queued/services/new requests"
            )
            print(
                f"Station {self.id}: duplicate request IDs {duplicates} in reserve; "
                "reservation rejected"
            )
            return None

        if any(
            self.id not in r.feasible_stations
            or r.installed_class not in (1, 2)
            or r.requested_class not in (1, 2)
            or r.requested_class < r.installed_class
            for r in requests
        ):
            invalid = sorted(
                r.id for r in requests
                if self.id not in r.feasible_stations
                or r.installed_class not in (1, 2)
                or r.requested_class not in (1, 2)
                or r.requested_class < r.installed_class
            )
            self.last_reserve_reason = (
                f"invalid request(s) {invalid}: station {self.id} is unreachable "
                f"for them or their battery classes are invalid"
            )
            return None

        ready = [[current_time] * int(n) for n in self.full_batteries]  # per-class availability
        for finish, l, number in self.charging:
            ready[l].extend([finish] * number)
        for times in ready:
            times.sort()

        # Arrivals: queued orders keep r.arrival; new ones use current_time + travel.
        arrival = {
            r.id: (
                r.arrival
                if r in self.queue
                else current_time + timedelta(minutes=r.travel_times[self.id])
            )
            for r in merged
        }

        schedule, reason = self._reserve_fcfs(current_time, requests, arrival, ready)
        self.last_reserve_reason = reason
        return schedule

    def _reserve_fcfs(self, current_time, requests, arrival, ready):
        """Return (timetable or None, failure reason or None) under FCFS + bypass."""
        pending = (sorted(self.queue, key=lambda r: (r.begin_time, r.id)) +
                   sorted(requests, key=lambda r: (arrival[r.id], r.id)))
        queued = {r.id for r in self.queue}
        previous = max(current_time, self.busy_until)  # earliest bay availability
        swap = timedelta(minutes=self.swap_time)
        max_wait = env_params["max_wait_time"] + float(env_params["time_tolerance"])
        schedule = {}
        while pending:
            head = pending[0]
            l = head.requested_class - 1
            if not ready[l]:
                return None, (
                    f"no ready battery for requested class {head.requested_class} "
                    f"when serving request {head.id}"
                )

            h_battery = ready[l][0]
            if head.id in queued:
                # Already-queued orders keep their committed begin.
                if max(previous, arrival[head.id], h_battery) > head.begin_time + timedelta(
                    minutes=float(env_params["time_tolerance"])
                ):
                    return None, (
                        f"queued commitment for request {head.id} cannot be met: "
                        f"promised begin {head.begin_time.isoformat()}"
                    )
                head_begin = head.begin_time
            else:
                head_begin = max(previous, arrival[head.id], h_battery)

            # Bypass: a later NEW order may fill an idle gap while the head waits.
            if h_battery > max(previous, arrival[head.id]):
                inserted = None
                for k in range(1, len(pending)):
                    j = pending[k]
                    if j.id in queued:
                        continue
                    lj = j.requested_class - 1
                    battery_index = 1 if lj == l else 0
                    if battery_index >= len(ready[lj]):
                        continue
                    begin = max(previous, arrival[j.id], ready[lj][battery_index])
                    departure = begin + swap
                    if departure > head_begin or (
                        departure - arrival[j.id]
                    ).total_seconds() / 60 > max_wait:
                        continue
                    # Trial consumption must leave batteries and bay time for remaining starts.
                    trial = [times.copy() for times in ready]
                    trial[lj].pop(battery_index)
                    if self._preserves_commitments(pending, arrival, trial, departure):
                        inserted = (k, j, begin, departure, trial)
                        break
                if inserted is not None:
                    k, j, begin, departure, ready = inserted
                    wait = (departure - arrival[j.id]).total_seconds() / 60
                    schedule[j.id] = (arrival[j.id], begin, departure, wait)
                    previous = departure
                    del pending[k]
                    continue

            ready[l].pop(0)
            departure = head_begin + swap
            wait = (departure - arrival[head.id]).total_seconds() / 60
            if wait > max_wait:
                return None, (
                    f"request {head.id}: wait {wait:.2f} min exceeds max_wait "
                    f"{env_params['max_wait_time']} min"
                )
            schedule[head.id] = (arrival[head.id], head_begin, departure, wait)
            previous = departure
            del pending[0]
        return schedule, None

    def _preserves_commitments(self, pending, arrival, ready, bay_time):
        """Check that the remaining batteries and bay preserve every queued start."""
        available = [times.copy() for times in ready]
        queued = {r.id for r in self.queue}
        tolerance = timedelta(minutes=float(env_params["time_tolerance"]))
        swap = timedelta(minutes=self.swap_time)
        for request in pending:
            if request.id not in queued:
                continue
            batteries = available[request.requested_class - 1]
            if not batteries or max(bay_time, arrival[request.id], batteries[0]) > request.begin_time + tolerance:
                return False
            batteries.pop(0)
            bay_time = request.begin_time + swap
        return True

    def implement_schedule(self, requests, schedule):
        """Commit the reservation timetable into the queue and refresh inventory."""
        # Validate all entries before mutating (a partial schedule must never commit).
        if RL_params["validation"]:
            needed = [r.id for r in self.queue] + [r.id for r in requests]
            missing = [rid for rid in needed if rid not in schedule]
            if missing:
                raise ValueError(
                    f"Station {self.id}: schedule is missing timetable entries for "
                    f"request IDs {missing}; queue unchanged"
                )
        self.queue.extend(requests)
        for r in self.queue:
            r.arrival, r.begin_time, r.departure, r.wait_time = schedule[r.id]
            r.detour_time = max(r.travel_times[self.id] - r.nearest_time, 0.0)
        # Keep the queue ordered by (promised begin, ID).
        self.queue.sort(key=lambda r: (r.begin_time, r.id))
        self.refresh_state()

    def validate_charging(self, quantities):
        """Return int (2,) charging starts; with validation enabled, malformed
        quantities raise ValueError before any mutation."""
        quantities = np.asarray(quantities)
        if RL_params["validation"]:
            if quantities.shape != (2,):
                raise ValueError(
                    f"Station {self.id}: charging quantities must have shape (2,), "
                    f"got {quantities.shape}"
                )
            if np.any(~np.isfinite(quantities)) or np.any(quantities < 0):
                raise ValueError(
                    f"Station {self.id}: charging quantities must be finite and "
                    f"nonnegative, got {quantities.tolist()}"
                )
        return quantities.astype(int)

    def start_charging(self, current_time, quantities):
        """Move depleted stock into non-preemptive charging jobs."""
        # Validate the full batch before any stock moves.
        quantities = self.validate_charging(quantities)

        # Depleted stock -> jobs with fixed reserve-to-ready duration.
        for l, number in enumerate(quantities):
            if not number:
                continue
            self.empty_batteries[l] -= number
            self.charged[l] += number
            end = current_time + timedelta(
                minutes=self.batch_steps(l) * self.delta_time
            )
            self.charging.append((end, l, int(number)))
        self.refresh_state()

    def advance(self, until_time):
        """Process charging completions and swap starts/completions up to until_time."""
        # Collect charging completions and swap starts due within the window.
        events = [
            (finish, 0, (l, number)) for finish, l, number in self.charging if finish <= until_time
        ]
        events += [(r.begin_time, 1, r) for r in self.queue if r.begin_time <= until_time]

        # Charging completions before swap starts at the same timestamp.
        for _time, _case, _event in sorted(events, key=lambda x: (x[0], x[1])):
            if _case == 0:
                l, number = _event
                self.full_batteries[l] += number
            else:
                r = _event
                self.full_batteries[r.requested_class - 1] -= 1
                # Arrival SOC only selects the ready/depleted split; recharge is fixed reserve->ready.
                returned_soc = (
                    r.soc_arrival if r.soc_arrival is not None else r.arrival_soc(self.id)
                )
                if returned_soc >= env_params["ready_soc"]:
                    self.full_batteries[r.installed_class - 1] += 1
                else:
                    self.empty_batteries[r.installed_class - 1] += 1
                self.busy_until = r.departure
                self.services.append(r)

        # Drop processed entries, settle completed swaps, refresh stock.
        self.charging = [job  for job in self.charging if job[0] > until_time]
        self.queue = [r  for r in self.queue if r.begin_time > until_time]
        for r in self.services:
            if r.departure <= until_time and not r.complete:
                r.complete = 1
                self.completed[r.requested_class - 1] += 1
        self.services = [r for r in self.services if r.departure > until_time]

        self.refresh_state()

    def calculate_charging(self, current_time, quantities, discounted=True):
        """Return the CNY electricity cost of a charging batch (optionally discounted)."""
        # Full power for whole intervals; integrate over time-of-use prices.
        total = 0.0

        interval_energy = [float(p) * self.delta_time / 60 for p in self.charging_power]

        # Integrate each class over its full-power intervals.
        for l, number in enumerate(quantities):
            if not number:
                continue
            # number batteries x interval_energy per interval; the reserve-to-ready gap pins duration.
            for k in range(self.batch_steps(l)):
                date = current_time + timedelta(minutes=k * self.delta_time)
                peak = (
                    env_params["peak_electricity"][0]
                    <= date.hour
                    < env_params["peak_electricity"][1]
                )
                price = env_params["electric_price_peak" if peak else "electric_price_off"]
                # Discount by gamma**k; revenue/subsidies accrue undiscounted.
                total += (
                    int(number)
                    * interval_energy[l]
                    * price
                    * (self.gamma ** k if discounted else 1)
                )
        return total

    def simulate_bundle(
            self,
            current_time,
            requests,  # candidate bundle (the orders to serve)
            quantities,  # candidate charging starts (low, high)
            next_step,  # next decision index
            next_hour,
            next_demand  # next-step candidate demand for the new_order feature of the observation
    ):
        """Return a candidate's next station observation using independent shadow state."""
        # Shadow copy: reserve, commit charging/queue, advance one interval.
        schedule = self.reserve(current_time, requests)

        agent_copy = copy.copy(self)
        agent_copy.full_batteries, agent_copy.empty_batteries = (
            self.full_batteries.copy(),
            self.empty_batteries.copy(),
        )
        agent_copy.charging_batteries = self.charging_batteries.copy()
        agent_copy.active_power = float(self.active_power)
        agent_copy.inventory_position = self.inventory_position.copy()
        agent_copy.charged, agent_copy.completed = self.charged.copy(), self.completed.copy()
        agent_copy.charging = list(self.charging)
        agent_copy.queue = [copy.copy(r) for r in self.queue]
        agent_copy.services = [copy.copy(r) for r in self.services]

        projected_requests = [copy.copy(r) for r in requests]
        # Price and return shadow batteries from the same arrival SOC used by step().
        for request in projected_requests:
            request.soc_arrival = request.arrival_soc(self.id)
            request.service_fee = request.fee_at(self.id)
        agent_copy.start_charging(current_time, quantities)
        agent_copy.implement_schedule(projected_requests, schedule)
        agent_copy.advance(current_time + timedelta(minutes=self.delta_time))
        return agent_copy.get_observe(
            next_step,
            next_hour,
            next_demand,
            current_time + timedelta(minutes=self.delta_time),
        )
