"""Trainer entry points: baselines, DRL training/resume, and evaluation runs."""

import csv
import os
import platform
import sys
from datetime import date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import data_params, env_params, RL_params
from common.optimization import Optimizer
from common.outputs import (
    atomic_to_csv,
    atomic_write,
    latest_file,
    run_stem,
    scenario as scenario_tag,
)
from simulator.bundling import Bundling
from baselines.fair_Greedy import GreedyPolicy
from baselines.fair_RHO import RHOPolicy
from DRL.base_DRL import BaseDRL
from DRL.fairDRL import FairDRL
from DRL.fairDRL_fixed import FairDRLFixed
from DRL.fairDRL_local import FairDRLLocal
from DRL.fairDRL_one import FairDRLOne
from DRL.opt_DRL import OptimumDRL


def _record_problem(problems, condition, message):
    """Append message to problems when the condition fails."""
    if not condition:
        problems.append(message)


def _collect_mapping_rows(rows, path, value):
    """Recurse containers as markers; escape '~' and '/' in path segments."""
    if isinstance(value, (dict, list, tuple)):
        rows.append((path, type(value).__name__, ""))
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in items:
            segment = str(key).replace("~", "~0").replace("/", "~1")
            _collect_mapping_rows(rows, f"{path}/{segment}", item)
    else:
        # ISO-8601 datetimes; empty cell for None.
        scalar = value.isoformat() if isinstance(value, (date, datetime)) else value
        rows.append((path, type(value).__name__, "" if value is None else scalar))


def _write_mapping_file(path, rows):
    """Write the flattened mapping rows as CSV to ``path``."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("key_path", "data_type", "value"))
        writer.writerows(rows)


class Trainer:
    """Orchestrates policy runs: metadata, seed tables, episode loops, and result writing."""

    def __init__(self, env):
        """Keep the shared environment and its time-step normalization size."""
        self.environment = env  # BatterySwapEnv shared by every policy of this run
        self.state_dims = {"steps": env.total_steps}  # dict[str, int] of network input sizes
        self.validate_settings()

    def validate_settings(self):
        """Pre-check parameter settings once per trainer (all parameter safety checks live here)."""
        problems = []
        _record_problem(problems,
                        data_params["horizon"] == (data_params["start_time"].hour, data_params["end_time"].hour),
                        "data_params['horizon'] must equal (start_time.hour, end_time.hour)")
        _record_problem(problems, data_params["scale"] > 0, "data_params['scale'] must be positive")
        span = (data_params["end_time"] - data_params["start_time"]).total_seconds() / 60
        cutoff = (data_params["decision_end_time"] - data_params["start_time"]).total_seconds() / 60
        delta = float(env_params["delta_time"])
        _record_problem(problems, data_params["start_time"] < data_params["decision_end_time"] <= data_params["end_time"],
                        "start < decision_end <= end must hold")
        _record_problem(problems, delta > 0 and span > 0 and span % delta == 0 and cutoff % delta == 0,
                        "start/order-cutoff/end times must lie on the decision grid")
        _record_problem(problems, 0 < float(env_params["gamma"]) <= 1, "env_params['gamma'] must lie in (0, 1]")
        _record_problem(problems, int(env_params["battery_levels"]) == 2, "env_params['battery_levels'] must be 2")
        for key in ("battery_kwh", "charging_power_kw", "max_demand", "max_queue", "max_capacity"):
            values = env_params[key]
            _record_problem(problems, len(values) == 2 and all(np.isfinite(values)) and all(v > 0 for v in values),
                            f"env_params['{key}'] must hold two finite positive values")
        _record_problem(problems, 0 <= float(env_params["reserve_soc"]) < float(env_params["ready_soc"]) <= 1,
                        "reserve_soc/ready_soc must satisfy 0 <= reserve < ready <= 1")
        for key in ("max_upgrade", "max_size", "max_bundle_size", "max_bundles", "swap_time",
                    "max_wait_time", "max_detour_minutes"):
            _record_problem(problems, float(env_params[key]) > 0, f"env_params['{key}'] must be positive")
        _record_problem(problems,
                        float(env_params["fairness_tolerance"])
                        >= 10 * float(env_params["solver_tolerance"]) > 0,
                        "fairness_tolerance must be >= 10x solver_tolerance > 0 "
                        "(solver tolerance must stay well below the fairness bound)")
        _record_problem(problems, all(v > 0 for v in env_params["vehicle_speed"].values()),
                        "vehicle_speed values must be positive")
        _record_problem(problems, len(RL_params["base_stocks"]) == 2 and all(0 < v <= 1 for v in RL_params["base_stocks"]),
                        "RL_params['base_stocks'] must hold two values in (0, 1]")
        _record_problem(problems, 0 < float(RL_params["fixed_level"]) <= 1,
                        "RL_params['fixed_level'] must lie in (0, 1]")
        for key in ("step_embed_dims", "gat_dims", "batch_size", "buffer_size", "update_freq",
                    "candidate_batch_size", "torch_threads", "checkpoint_freq"):
            _record_problem(problems, int(RL_params[key]) > 0, f"RL_params['{key}'] must be a positive integer")
        _record_problem(problems, int(RL_params["buffer_size"]) >= int(RL_params["batch_size"]),
                        "RL_params['buffer_size'] must be >= batch_size")
        _record_problem(problems, float(RL_params["gat_negative_slope"]) >= 0, "gat_negative_slope must be nonnegative")
        _record_problem(problems, all(int(h) > 0 for h in RL_params["hidden_dims"]), "hidden_dims must be positive integers")
        _record_problem(problems, 0 < float(RL_params["replay_time_decay"]) <= 1, "replay_time_decay must lie in (0, 1]")
        for key in ("initial_learning", "final_learning", "learning_decay", "epsilon_decay"):
            _record_problem(problems, float(RL_params[key]) > 0, f"RL_params['{key}'] must be positive")
        _record_problem(problems, 0 < float(RL_params["final_learning"]) <= float(RL_params["initial_learning"]),
                        "final_learning must be <= initial_learning")
        for key in ("initial_epsilon", "final_epsilon"):
            _record_problem(problems, 0 <= float(RL_params[key]) <= 1, f"RL_params['{key}'] must lie in [0, 1]")
        _record_problem(problems, float(RL_params["final_epsilon"]) <= float(RL_params["initial_epsilon"]),
                        "final_epsilon must be <= initial_epsilon")
        _record_problem(problems, float(RL_params["clip_norm"]) > 0, "clip_norm must be positive")
        _record_problem(problems, float(RL_params["envy_weight"]) >= 0, "envy_weight must be nonnegative")
        sigma = float(RL_params.get("explore_value_sigma", 0.0))
        _record_problem(problems, np.isfinite(sigma) and sigma >= 0, "explore_value_sigma must be finite and nonnegative")
        # Warn instead of blocking: training may start before configure_solver writes the backend.
        if RL_params["solver"] not in ("gurobi", "copt"):
            print(
                f"WARNING: RL_params['solver']={RL_params['solver']!r} is not configured; "
                "solving BMIPs/sub LP requires setting it to 'gurobi' or 'copt' "
                "(Optimizer.configure_solver preferred='gurobi' or 'copt').",
                flush=True,
            )
        _record_problem(problems, set(RL_params["gat_focal_features"]).issubset({"que_order", "inventory_position"}),
                        "gat_focal_features keys must be que_order/inventory_position")
        if problems:
            raise ValueError("Invalid parameter settings:\n- " + "\n- ".join(problems))

    @staticmethod
    def write_mapping_csv(stem, name, data):
        """Write {stem}_{name}.csv: flattened (key_path, data_type, value) rows."""
        rows = []
        _collect_mapping_rows(rows, "", data)
        filename = Path(data_params["result_dir"]) / f"{stem}_{name}"
        atomic_write(filename, lambda target: _write_mapping_file(target, rows))

    def write_run_metadata(self, stem, learner, training):
        """Write {stem}_configuration.csv and {stem}_runtime.csv once per run."""
        configuration = self.environment.training_configuration()
        configuration.update(
            policy=learner.name,
            training=training,
            seed_schedule="parity-v1",
            effective_time_limit=Optimizer.solver_time_limit(),
        )
        configuration["policy_settings"] = dict(
            type=learner.type,
            envy_weight=getattr(learner, "envy_weight", None),
            state_dims=getattr(learner, "state_dims", None),
            scenario=self.environment.scenario,
        )
        if isinstance(learner, BaseDRL):
            # All DRL variants record the same effective feature/attention contract.
            configuration["value_inputs"] = learner.value_net.input_configuration()
        if isinstance(learner, BaseDRL) and learner.type == "Fair":
            ablation = dict(
                variant=learner.ablation, bundle_limit=learner.bundle_limit,
                use_gat=learner.value_net.use_gat,
                attention=learner.value_net.attention_kind,
            )
            if learner.value_net.use_gat:
                # Record which GAT node/message feature layouts produced this run.
                ablation["node_dims"] = learner.value_net.node_dims
                ablation["sender_dims"] = learner.value_net.sender_dims
            configuration["ablation"] = ablation
        self.write_mapping_csv(stem, "configuration.csv", configuration)

        dependencies = {}
        for name in ("numpy", "pandas", "scipy", "torch", "matplotlib", "gurobipy", "coptpy"):
            try:
                dependencies[name] = version(name)
            except PackageNotFoundError:
                dependencies[name] = "not installed"
        runtime = dict(
            python=sys.version,
            platform=platform.platform(),
            machine=platform.machine(),
            dependencies=dependencies,
            python_hash_seed=os.environ.get("PYTHONHASHSEED", "unset"),
            configured_torch_threads=RL_params["torch_threads"],
            torch_threads=torch.get_num_threads(),
            torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            solver_threads=RL_params["solver_threads"],
            device="cpu",
        )
        self.write_mapping_csv(stem, "runtime.csv", runtime)

    def record_episode_seed(self, stem, learner, training):
        """Upsert the current episode's seed row in {stem}_seeds.csv (one row/episode)."""
        env = self.environment
        row = dict(
            episode=env.episode,
            split="train" if training else "evaluation",
            seed_schedule="parity-v1",
            demand_base_seed=env.seed,
            demand_seed=env.episode_seed,
            learning_seed=learner.seed,
            **{f"{name}_seed": seed for name, seed in learner.seeds.items()},
            solver_threads=RL_params["solver_threads"],
        )

        path = Path(data_params["result_dir"]) / f"{stem}_seeds.csv"
        if path.exists():
            previous = pd.read_csv(path)
            rows = previous.to_dict("records") if "episode" in previous.columns else []
        else:
            rows = []
        rows = [old for old in rows if old["episode"] != env.episode] + [row]
        self._atomic_to_csv(pd.DataFrame(rows).sort_values("episode"), path)

    @staticmethod
    def _atomic_to_csv(frame, path, retries=5, delay=0.5):
        """Atomically write a summary/seed DataFrame (delegates to outputs)."""
        atomic_to_csv(frame, path, retries, delay)

    def run(self, learner, start_episode, training=False, stem=None, manual_start=False):
        """Run the remaining episodes and atomically upsert summaries, seeds,
        and checkpoints; ``manual_start`` skips the checkpoint-boundary guards."""
        # Validate the shared environment and resume index before rewriting files.
        if learner.environment is not self.environment:
            raise ValueError("Trainer and learner must share the same environment")
        if isinstance(start_episode, bool) or int(start_episode) != start_episode or start_episode < 0:
            raise ValueError("start_episode must be a nonnegative integer")
        start_episode = int(start_episode)
        if training and isinstance(learner, BaseDRL) and not manual_start:
            if start_episode != learner.training_episodes:
                raise ValueError("Training must start at the learner's next episode")
            if not learner.resume_supported:
                raise ValueError(
                    "Checkpoint cannot continue training: it lacks a valid training "
                    "configuration or was not saved at an episode boundary"
                )

        directory = Path(data_params["result_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        policy = learner.name

        # Filename tags describe the network the environment actually loaded.
        scenario_id = self.environment.scenario
        # Dated stem per (policy, stream, scenario) run.
        stream = "train" if training else "eval"
        if stem is None:
            stem = run_stem(policy, stream, scenario_id)
        summary_path = directory / f"{stem}_summary.csv"
        results = []
        if training and isinstance(learner, BaseDRL) and start_episode and not manual_start:
            self._verify_resume_boundary(stem, start_episode, learner)

        # Resolve the episode budget here; a start at/past the target makes a
        # repeated resume call a no-op.
        MAX_Eps = RL_params["MAX_Episodes"] if training else RL_params["MAX_Tests"]

        # Retain earlier summaries on resume; clear only rewritable records for a fresh run.
        if start_episode:
            # Never combine an unrelated newer experiment with the loaded checkpoint.
            if summary_path.is_file():
                previous = pd.read_csv(summary_path)
                if "episode" not in previous.columns:
                    raise ValueError(
                        f"Cannot resume {summary_path}: legacy summary lacks an episode column"
                    )
                # One row per episode (last wins): interrupted writes cannot duplicate an index.
                results = (
                    previous.loc[previous.episode < start_episode]
                    .drop_duplicates("episode", keep="last")
                    .to_dict("records")
                )
            else:
                print(
                    f"WARNING: resuming {policy} at episode {start_episode} without a summary "
                    f"table for {stem}; earlier episodes are absent and the new table starts there.",
                    flush=True,
                )
            # A checkpoint may precede records written just before an interruption.
            seed_path = directory / f"{stem}_seeds.csv"
            if seed_path.is_file():
                seeds = pd.read_csv(seed_path)
                if "episode" in seeds:
                    self._atomic_to_csv(
                        seeds.loc[seeds.episode < start_episode], seed_path
                    )
            # Drop stale request tables before the resumed episodes rewrite them.
            for path in directory.glob(f"{stem}_requests_*.csv"):
                suffix = path.stem.rsplit("_", 1)[-1]
                if suffix.isdigit() and int(suffix) >= start_episode:
                    path.unlink()
            # Discard rows beyond the restored checkpoint; they are overwritten when rerun.
            self._atomic_to_csv(pd.DataFrame(results), summary_path)
            if start_episode >= MAX_Eps:
                return results
        else:
            if start_episode >= MAX_Eps:
                return results
            for base in (stem, f"{policy}_{stream}{scenario_tag(scenario_id)}", policy):
                for path in directory.glob(f"{base}_requests_*.csv"):
                    path.unlink()
                (directory / f"{base}_seeds.csv").unlink(missing_ok=True)
                if base != stem:
                    (directory / f"{base}_summary.csv").unlink(missing_ok=True)

        if RL_params["validation"]:
            self.write_run_metadata(stem, learner, training)

        for episode in range(start_episode, MAX_Eps):
            seed = self.environment.get_episode_seed(episode, training)
            self.environment.initialize_instance(data_params["scale"], seed, episode)
            bundling = Bundling(self.environment)

            result = learner.scheduling(
                bundling, train_flag=training, episode=episode, name=stem
            )

            results.append(result)
            # One row per episode (last wins): resumed episodes overwrite earlier records.
            self._atomic_to_csv(
                pd.DataFrame(results)
                .drop_duplicates("episode", keep="last")
                .sort_values("episode"),
                summary_path,
            )
            if RL_params["validation"]:
                self.record_episode_seed(stem, learner, training)
            if training and hasattr(learner, "save_checkpoint") and (
                (episode + 1) % RL_params["checkpoint_freq"] == 0
            ):
                learner.save_checkpoint(name=stem)

            # Exploration/learning rate: DRL exposes both; baselines print n/a.
            epsilon = getattr(learner, "epsilon", None)
            param_groups = getattr(getattr(learner, "optimizer", None), "param_groups", None)
            learning_rate = param_groups[0]["lr"] if param_groups else None
            epsilon_text = f"{epsilon:.4g}" if epsilon is not None else "n/a"
            learning_rate_text = f"{learning_rate:.3g}" if learning_rate is not None else "n/a"

            print(f"**************** Policy {policy}: episode {episode} ****************")
            print(
                f'normalized profit={result["normalized_profit"]:.2f}, total demands=({result["low_demand"]}, '
                f'{result["high_demand"]}, {result["upgrade_demand"]}), '
                f'total charged=({result["charged_low"]}, {result["charged_high"]}); \n'
                f'station demands=({result["max_station_demand_low"]}, {result["max_station_demand_high"]}), '
                f'station queues=({result["max_station_que_low"]}, {result["max_station_que_high"]}); \n'
                f'rejections=({result["rejection_low"]:.2f}, {result["rejection_high"]:.2f}); \n'
                f'mean subsidy={result["mean_subsidy"]:.2f}, mean wait={result["mean_wait"]:.2f} min, '
                f'mean detour={result["mean_detour"]:.2f} min; \n'
                f'bundling numbers=({result["mean_bundle_number"]:.2f}, {result["max_bundle_number"]:.2f}), '
                f'bundle sizes={result["mean_bundle_size"]:.2f}; \n'
                f'bundling time={result["bundle_seconds_max"]:.2f} s, action time={result["action_seconds_max"]:.2f} s, '
                f'run time={result["run_time"]:.2f} min; \n'
                f'loss={result["loss"]:.2f}, epsilon={epsilon_text}, learning rate={learning_rate_text}.',
                flush=True,
            )
            print("----------------------------------------------------------")

        # Final checkpoint regardless of episode: exact resume needs the boundary saved.
        if training and hasattr(learner, "save_checkpoint"):
            learner.save_checkpoint(name=stem)

        return results

    def _verify_resume_boundary(self, stem, next_episode, learner=None):
        """Cross-check the checkpoint/summary/seeds episode boundary before resume."""
        next_episode = int(next_episode)
        if next_episode < 1:
            return
        expected = set(range(next_episode))
        directory = Path(data_params["result_dir"])
        summary_path = directory / f"{stem}_summary.csv"
        seed_path = directory / f"{stem}_seeds.csv"

        if not summary_path.is_file():
            raise ValueError(
                f"Cannot resume from {stem}: summary table {summary_path.name} is "
                f"missing; episodes 0..{next_episode - 1} must be recorded"
            )
        previous = pd.read_csv(summary_path)
        if "episode" not in previous.columns:
            raise ValueError(
                f"Cannot resume from {stem}: summary table lacks an episode column"
            )
        recorded = previous["episode"]
        if (not pd.api.types.is_numeric_dtype(recorded) or recorded.isna().any()
                or (recorded < 0).any() or (recorded % 1 != 0).any()):
            raise ValueError(f"Cannot resume from {stem}: summary episode indices must be nonnegative integers")
        prefix = previous.loc[recorded < next_episode]
        if prefix.episode.duplicated().any():
            raise ValueError(
                f"Cannot resume from {stem}: summary contains duplicate episode rows"
            )
        missing = sorted(expected - set(recorded))
        if missing:
            raise ValueError(
                f"Cannot resume from {stem}: summary is missing episodes {missing} "
                f"required before the checkpoint boundary {next_episode}"
            )
        if "seed" not in prefix or any(
            row.seed != self.environment.get_episode_seed(int(row.episode), True)
            for row in prefix.itertuples()
        ):
            raise ValueError(f"Cannot resume from {stem}: summary demand seeds differ from the current stream")
        if learner is not None:
            last = prefix.loc[prefix.episode == next_episode - 1].iloc[0]
            for field in ("total_steps", "gradient_steps", "sync_count"):
                if field not in last or last[field] != getattr(learner, field):
                    raise ValueError(f"Cannot resume from {stem}: checkpoint and summary {field} disagree")
            if learner.resume_configuration is None:
                raise ValueError(
                    f"Cannot resume exactly from {stem}: legacy checkpoint lacks the effective "
                    "configuration/input fingerprints and seed streams; evaluation remains available"
                )

        if RL_params["validation"]:
            if not seed_path.is_file():
                raise ValueError(
                    f"Cannot resume from {stem}: seeds table {seed_path.name} is "
                    "missing although validation tables are enabled"
                )
            seeds = pd.read_csv(seed_path)
            if "episode" not in seeds.columns:
                raise ValueError(
                    f"Cannot resume from {stem}: seeds table lacks an episode column"
                )
            ids = seeds["episode"]
            if (not pd.api.types.is_numeric_dtype(ids) or ids.isna().any()
                    or (ids < 0).any() or (ids % 1 != 0).any()):
                raise ValueError(f"Cannot resume from {stem}: seed episode indices must be nonnegative integers")
            seed_prefix = seeds.loc[ids < next_episode]
            if seed_prefix.episode.duplicated().any():
                raise ValueError(f"Cannot resume from {stem}: seeds contains duplicate episode rows")
            missing_seeds = sorted(expected - set(seeds["episode"]))
            if missing_seeds:
                raise ValueError(
                    f"Cannot resume from {stem}: seeds table is missing episodes "
                    f"{missing_seeds} required before the checkpoint boundary"
                )
            expected_fields = dict(split="train", seed_schedule="parity-v1",
                                   demand_base_seed=self.environment.seed,
                                   solver_threads=RL_params["solver_threads"])
            if learner is not None:
                expected_fields.update(learning_seed=learner.seed,
                    **{f"{name}_seed": seed for name, seed in learner.seeds.items()})
            for field, value in expected_fields.items():
                if field not in seed_prefix or not (seed_prefix[field] == value).all():
                    raise ValueError(f"Cannot resume from {stem}: seeds table {field} differs from the checkpoint/configuration")
            if "demand_seed" not in seed_prefix or any(
                row.demand_seed != self.environment.get_episode_seed(int(row.episode), True)
                for row in seed_prefix.itertuples()
            ):
                raise ValueError(f"Cannot resume from {stem}: seed table demand seeds differ from the summary/checkpoint")

    def baselines(self, training=False, resume=False, target_fixed=False,
                  start_episode=None):
        """Run Greedy + RHO on one parity stream; resume continues the newest
        summary. ``target_fixed`` forwards the fixed charging switch;
        ``start_episode`` is honored only when ``resume=True``."""
        if start_episode is not None and not resume:
            print(
                f"WARNING: start_episode={start_episode!r} ignored because "
                "resume=False; starting from episode 0.",
                flush=True,
            )
            start_episode = None
        if start_episode is not None and (
            isinstance(start_episode, bool)
            or int(start_episode) != start_episode
            or start_episode < 0
        ):
            raise ValueError("start_episode must be a nonnegative integer")
        start_episode = None if start_episode is None else int(start_episode)
        stream = "train" if training else "eval"
        policies = (
            RHOPolicy(self.environment, target_fixed=target_fixed),
            GreedyPolicy(self.environment, target_fixed=target_fixed),
        )
        results = {}
        for policy in policies:
            print(f"--------------- Start running policy {policy.name} ---------------")
            stem = None
            if resume or start_episode is not None:
                directory = Path(data_params["result_dir"])
                tag = scenario_tag(self.environment.scenario)
                existing = latest_file(
                    directory, f"{policy.name}_{stream}{tag}_", "_summary.csv"
                )
                if existing is None:
                    for candidate in (
                        f"{policy.name}_{stream}_summary.csv",
                        f"{policy.name}_summary.csv",
                    ):
                        legacy = directory / candidate
                        if legacy.is_file():
                            existing = legacy
                            break
                if existing is not None:
                    stem = existing.name[: -len("_summary.csv")]
            episode_start = start_episode
            if episode_start is None:
                episode_start = 0
                if resume and stem is not None:
                    previous = pd.read_csv(existing)
                    episode_start = (
                        int(previous["episode"].max()) + 1
                        if not previous.empty and "episode" in previous
                        else 0
                    )

            results[f"{policy.name}_{stream}"] = self.run(
                policy, episode_start, training=training, stem=stem,
            )
        return results

    def run_DRL_policy(self, learner, training, resume, start_episode=None):
        """Train (resume from checkpoint when asked) or evaluate one DRL learner.
        start_episode is honored only with resume=True (manual start skips the
        boundary checks); resume=False always starts the fresh learner at 0."""
        if training:
            manual_start = False
            if start_episode is not None and not resume:
                print(
                    f"WARNING: start_episode={start_episode!r} ignored because "
                    "resume=False; starting from episode 0.",
                    flush=True,
                )
                start_episode = None
            if resume:
                if start_episode is not None:
                    if (isinstance(start_episode, bool)
                            or int(start_episode) != start_episode
                            or start_episode < 0):
                        raise ValueError("start_episode must be a nonnegative integer")
                    start_episode = int(start_episode)
                    manual_start = True
                    # Manual start: newest checkpoint if any, else the fresh learner (baseline-style).
                    try:
                        learner.load_checkpoint()
                        stem = learner.checkpoint_stem
                    except FileNotFoundError:
                        print(
                            f"WARNING: no checkpoint found for {learner.name}; "
                            f"manual start at episode {start_episode} runs the "
                            "fresh learner.",
                            flush=True,
                        )
                        stem = None
                else:
                    learner.load_checkpoint()
                    stem = learner.checkpoint_stem
                    start_episode = int(learner.training_episodes)
            else:
                start_episode = 0
                stem = None

            # run() resolves the remaining budget; repeated resumes become no-ops.
            return self.run(
                learner, start_episode, training=True, stem=stem,
                manual_start=manual_start,
            )

        # Evaluation replays MAX_Tests episodes with frozen weights.
        learner.load_checkpoint()
        return self.run(learner, 0, training=False)

    def fairDRL(self, training=True, resume=False, start_episode=None):
        """Train/resume the standard fair policy or evaluate its newest checkpoint."""
        print(f"--------------- Start running policy fairDRL ---------------")
        learner = FairDRL(self.environment, self.state_dims)
        return self.run_DRL_policy(learner, training, resume, start_episode)

    def fairDRL_one(self, training=True, resume=False, start_episode=None):
        """Singleton-bundle ablation; same training/resume contract as fairDRL."""
        print(f"--------------- Start running policy fairDRL_one ---------------")
        learner = FairDRLOne(self.environment, self.state_dims)
        return self.run_DRL_policy(learner, training, resume, start_episode)

    def fairDRL_local(self, training=True, resume=False, start_episode=None):
        """No-GAT local-model ablation; same training/resume contract as fairDRL."""
        print(f"--------------- Start running policy fairDRL_local ---------------")
        learner = FairDRLLocal(self.environment, self.state_dims)
        return self.run_DRL_policy(learner, training, resume, start_episode)

    def fairDRL_fixed(self, training=True, resume=False, start_episode=None):
        """Fixed baseline-charging ablation; same training/resume contract as fairDRL."""
        print(f"--------------- Start running policy fairDRL_fixed ---------------")
        learner = FairDRLFixed(self.environment, self.state_dims)
        return self.run_DRL_policy(learner, training, resume, start_episode)

    def optDRL(self, training=True, resume=False, envy_weight=0.0, start_episode=None):
        """Efficiency (envy_weight=0) or weighted-fairness (envy_weight>0) policy."""
        if RL_params["validation"] and (
            not np.isfinite(envy_weight) or envy_weight < 0
        ):
            raise ValueError("envy_weight must be finite and nonnegative")
        learner = OptimumDRL(self.environment, self.state_dims, envy_weight=envy_weight)
        print(f"--------------- Start running policy {learner.name} ---------------")
        return self.run_DRL_policy(learner, training, resume, start_episode)

    def wgt4DRL(self, training=True, resume=False, start_episode=None):
        """Weighted-fairness run: optDRL with the configured envy weight."""
        return self.optDRL(
            training, resume, envy_weight=RL_params["envy_weight"],
            start_episode=start_episode,
        )
