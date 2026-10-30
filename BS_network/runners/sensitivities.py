"""Parameter sensitivity analysis on one selected station network.
Each invocation runs RHO, fairDRL, optDRL on 30 episodes in one environment:
the default config or the non-default level of one parameter."""

import shutil
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import data_params, env_params
from common.outputs import atomic_to_csv, scenario as scenario_tag, scenario_size
from simulator.BS_env import BatterySwapEnv
from simulator.bundling import Bundling
from baselines.fair_RHO import RHOPolicy
from DRL.fairDRL import FairDRL
from DRL.opt_DRL import OptimumDRL


# Parameter -> (group, key, low, high); one level equals the config default.
PARAMETERS = {
    "detour_cap": (env_params, "max_detour_minutes", 15.0, 30.0),
    "wait_cap": (env_params, "max_wait_time", 15.0, 30.0),
    "poisson_rate": (data_params, "scale", 1.0, 2.0),
    "low_class_prob": (data_params, "installed_low_probability", 0.8, 0.9),
}

# Column order of every sensitivity CSV (fairDRL main outcomes + comparisons).
SENSITIVITY_COLUMNS = (
    "episode parameter level total_demand rejection rejection_low rejection_high "
    "total_rewards total_revenues total_profits normalized_reward normalized_profit "
    "mean_subsidy mean_station_que mean_wait mean_detour action_seconds_mean impr gap"
).split()


class Sensitivities:
    """Run one 30-episode environment with the three-policy comparison.
    parameter=None: default config; otherwise the non-default level of that
    parameter. scenario: None base network, 0/8 plus8, 1/16 plus16."""

    def __init__(self, parameter=None, episodes=30, resume=True, scenario=None):
        if parameter is not None and parameter not in PARAMETERS:
            raise ValueError(
                f"Unknown sensitivity parameter {parameter!r}; "
                f"choose from {sorted(PARAMETERS)} or None for the default config"
            )
        self.parameter = parameter
        if parameter is None:
            self.group = self.key = self.low = self.high = None
            self.default = None
        else:
            self.group, self.key, self.low, self.high = PARAMETERS[parameter]
            self.default = self.group[self.key]
        # Scenario mirror: None base, 0/8 plus8, 1/16 plus16; stored as size,
        # encoded in the output filename tag.
        if scenario is None:
            self.scenario = None
        else:
            size = scenario_size(scenario)
            if size not in data_params["scenarios"]:
                raise ValueError(
                    f"Unknown sensitivity scenario {scenario!r}; choose None "
                    f"(base network) or one of {data_params['scenarios']}"
                )
            self.scenario = size
        self.tag_suffix = "" if self.scenario is None else scenario_tag(self.scenario)
        self.network_label = "base" if self.scenario is None else f"+{self.scenario}"
        self.episodes = int(episodes)
        self.resume = bool(resume)  # continue an interrupted run when True
        self.seed = int(data_params["seed"])
        self.rng = np.random.default_rng(self.seed)

    def _resample_installed_class(self, env, target):
        """Flip installed classes toward ``target`` and refresh reachability;
        flips stay non-upgrade and use initialize_instance's feasibility rules."""
        base = float(self.default)
        for request in env.all_demand:
            if target < base and request.installed_class == 1:
                if self.rng.random() < (base - target) / base:
                    request.installed_class = request.requested_class = 2
            elif target > base and request.installed_class == 2:
                if self.rng.random() < (target - base) / (1.0 - base):
                    request.installed_class = request.requested_class = 1

        reserve = float(env_params["reserve_soc"])
        consumption = float(env_params["consumption_kwh_per_km"])
        energy_tol = float(env_params["energy_tolerance"])
        detour_cap = float(env_params["max_detour_minutes"]) + float(
            env_params["time_tolerance"]
        )
        kwh = env_params["battery_kwh"]
        for request in env.all_demand:
            usable = (request.SOC - reserve) * kwh[request.installed_class - 1]
            distances = np.asarray(request.travel_distances)
            times = np.asarray(request.travel_times)
            feasible = (
                (distances * consumption <= usable + energy_tol)
                & ((times - request.nearest_time) <= detour_cap)
            )
            request.feasible_stations = tuple(
                env.Agents[j].id for j in np.flatnonzero(feasible)
            )

    def _learners(self, env):
        """Build the three policies; DRL learners load their newest checkpoint."""
        learners = [
            ("RHO", RHOPolicy(env)),
            ("fairDRL", FairDRL(env)),
            ("optDRL", OptimumDRL(env, {"steps": env.total_steps}, envy_weight=0.0)),
        ]
        for _, learner in learners:
            if hasattr(learner, "load_checkpoint"):
                learner.load_checkpoint()
        return learners

    def _output(self, result_dir, tag):
        """Return (path, next_episode, rows) of the newest dated run for tag."""
        candidates = sorted(result_dir.glob(f"sensitivity_{tag}_*.csv"))
        if not candidates:
            return (
                result_dir / f"sensitivity_{tag}_{date.today().isoformat()}.csv",
                0,
                [],
            )
        path = candidates[-1]
        frame = pd.read_csv(path)
        rows = (
            frame.to_dict("records")
            if not frame.empty and "episode" in frame.columns else []
        )
        start = int(frame["episode"].max()) + 1 if rows else 0
        return path, start, rows

    def _run_level(self, level, tag):
        """Run the 30 episodes at one environment (level=None: default) and write its CSV."""
        saved = self.group[self.key] if self.group is not None else None
        saved_mix = None
        # low_class_prob flips demand after generation; others apply their level before loading.
        flip_target = level if self.parameter == "low_class_prob" else None
        if flip_target is None and self.group is not None:
            self.group[self.key] = level
        if flip_target is not None and flip_target != self.default:
            # 0.9 -> 0.8 raises high-class probability; rescale high-class
            # demand/queue denominators to 16.0 for comparable value inputs.
            saved_mix = (
                tuple(env_params["max_demand"]),
                tuple(env_params["max_queue"]),
            )
            env_params["max_demand"] = (
                env_params["max_demand"][0],
                16.0,
            )
            env_params["max_queue"] = (
                env_params["max_queue"][0],
                16.0,
            )
        saved_detour_denominators = None
        if self.parameter == "detour_cap" and level != self.default:
            # detour_cap=30 widens reachable station sets; the wider demand
            # denominator keeps feature_scales from clipping (queue unchanged).
            saved_detour_denominators = (
                tuple(env_params["max_demand"]),
                tuple(env_params["max_queue"]),
            )
            env_params["max_demand"] = (25.0, 10.0)
            env_params["max_queue"] = (16.0, 8.0)
        try:
            env = BatterySwapEnv(seed=None)
            if self.scenario is None:
                env.get_network("training", None)
            else:
                env.get_network("test", self.scenario)
            learners = self._learners(env)
            result_dir = Path(data_params["result_dir"])
            result_dir.mkdir(parents=True, exist_ok=True)
            output, start, rows = self._output(result_dir, tag)
            if not self.resume:
                # Fresh run: restart from episode 0 into today's dated family.
                output = result_dir / f"sensitivity_{tag}_{date.today().isoformat()}.csv"
                output.unlink(missing_ok=True)
                start, rows = 0, []
            if start >= self.episodes:
                print(
                    f"Sensitivity {self.parameter or 'default'} (level {level}) "
                    f"network={self.network_label} "
                    f"already complete: {start}/{self.episodes} episodes.",
                    flush=True,
                )
                return pd.read_csv(output)
            print(
                f"================ Sensitivity {self.parameter or 'default'} "
                f"level={level} network={self.network_label} "
                f"({self.episodes} episodes, resume from {start}) "
                f"================",
                flush=True,
            )
            # Redirect request_records to a scratch folder so results/ is not flooded.
            scratch = Path(tempfile.mkdtemp(prefix="sensitivity_scratch_"))
            try:
                for episode in range(start, self.episodes):
                    env.initialize_instance(
                        data_params["scale"], self.seed + 2 * episode, episode
                    )
                    if flip_target is not None and flip_target != self.default:
                        # Per-episode flip stream: a resumed run flips the same orders.
                        self.rng = np.random.default_rng(self.seed + 2 * episode)
                        self._resample_installed_class(env, flip_target)
                    bundling = Bundling(env)
                    summaries = {}
                    for name, learner in learners:
                        try:
                            data_params["result_dir"] = scratch
                            result = learner.scheduling(
                                bundling,
                                train_flag=False,
                                episode=episode,
                                name=f"{name}_sensitivity_{date.today().isoformat()}",
                            )
                        finally:
                            data_params["result_dir"] = result_dir
                        summaries[name] = result
                    rho, fair, opt = (
                        summaries["RHO"], summaries["fairDRL"], summaries["optDRL"]
                    )
                    impr = (
                        (fair["total_profits"] - rho["total_profits"])
                        / rho["total_profits"]
                        if rho["total_profits"] else np.nan
                    )
                    gap = (
                        (opt["total_profits"] - fair["total_profits"])
                        / opt["total_profits"]
                        if opt["total_profits"] else np.nan
                    )
                    print(
                        f"**************** Sensitivity {self.parameter or 'default'} "
                        f"(level {level}): episode {episode} ****************",
                        flush=True,
                    )
                    print(
                        f'RHO normalized profits={rho["normalized_profit"]:.2f}, '
                        f'fairDRL normalized profits={fair["normalized_profit"]:.2f}, '
                        f'optDRL normalized profits={opt["normalized_profit"]:.2f}; '
                        f'impr={impr:.4f}, gap={gap:.4f}. \n'
                        f'fairDRL: normalized profit={fair["normalized_profit"]:.2f}, '
                        f'total demands=({fair["low_demand"]}, {fair["high_demand"]}, '
                        f'{fair["upgrade_demand"]}), '
                        f'rejection={fair["rejection"]:.2f} '
                        f'(low {fair["rejection_low"]:.2f}, high {fair["rejection_high"]:.2f}); \n'
                        f'mean subsidy={fair["mean_subsidy"]:.2f}, '
                        f'mean wait={fair["mean_wait"]:.2f} min, '
                        f'mean detour={fair["mean_detour"]:.2f} min; \n'
                        f'action time={fair["action_seconds_max"]:.2f} s, '
                        f'run time={fair["run_time"]:.2f} min.',
                        flush=True,
                    )
                    rows.append(
                        {
                            "episode": episode,
                            "parameter": self.parameter
                            if self.parameter is not None else "default",
                            "level": level if level is not None else "default",
                            "total_demand": fair["total_demand"],
                            "rejection": fair["rejection"],
                            "rejection_low": fair["rejection_low"],
                            "rejection_high": fair["rejection_high"],
                            "total_rewards": fair["total_rewards"],
                            "total_revenues": fair["total_revenues"],
                            "total_profits": fair["total_profits"],
                            "normalized_reward": fair["normalized_reward"],
                            "normalized_profit": fair["normalized_profit"],
                            "mean_subsidy": fair["mean_subsidy"],
                            "mean_station_que": fair["mean_station_que"],
                            "mean_wait": fair["mean_wait"],
                            "mean_detour": fair["mean_detour"],
                            "action_seconds_mean": fair["action_seconds_mean"],
                            "impr": impr,
                            "gap": gap,
                        }
                    )
                    # Upsert one row per episode so an interrupted sweep resumes.
                    atomic_to_csv(
                        pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)
                        .drop_duplicates("episode", keep="last")
                        .sort_values("episode"),
                        output,
                    )
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
            frame = pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)
            return frame
        finally:
            if saved_mix is not None:
                env_params["max_demand"], env_params["max_queue"] = saved_mix
            if saved_detour_denominators is not None:
                env_params["max_demand"], env_params["max_queue"] = saved_detour_denominators
            if self.group is not None:
                self.group[self.key] = saved

    def run(self):
        """Run exactly one 30-episode environment; writes
        sensitivity_{default|parameter}{tag}_{date}.csv (tag: ""/_plus8/_plus16)."""
        if self.parameter is None:
            return self._run_level(None, f"default{self.tag_suffix}")
        alternate = self.low if self.low != self.default else self.high
        return self._run_level(alternate, f"{self.parameter}{self.tag_suffix}")
