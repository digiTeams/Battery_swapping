"""Test-mode runner: run a selected policy list on one expansion network.
Episodes are deterministic and replayed policy by policy in episode-major
order; resume continues an interrupted sweep (tables upsert per episode)."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import RL_params, data_params
from common.analys import Analyst
from common.generate_instances import episode_statistics
from common.outputs import (
    atomic_to_csv,
    latest_file,
    run_stem,
    scenario as scenario_tag,
)
from simulator.BS_env import BatterySwapEnv, _is_peak_step
from simulator.bundling import Bundling
from baselines.fair_Greedy import GreedyPolicy
from baselines.fair_RHO import RHOPolicy
from DRL.fairDRL import FairDRL
from DRL.fairDRL_one import FairDRLOne
from DRL.fairDRL_local import FairDRLLocal
from DRL.fairDRL_fixed import FairDRLFixed
from DRL.opt_DRL import OptimumDRL


TEST_STREAM = "test"
TEST_EPISODES_PER_NETWORK = 30
ACTION_SAMPLE_FAIRDRL = 300  # fairDRL rows kept per peak/off-peak stratum
# Only the compared pair keeps per-request tables.
REQUEST_POLICIES = {"fairDRL", "optDRL"}


# Policy name -> (learner factory for an env, whether it loads a checkpoint).
POLICIES = {
    "fairGreedy": (GreedyPolicy, False),
    "fairRHO": (RHOPolicy, False),
    "fairDRL": (FairDRL, True),
    "optDRL": (
        lambda env: OptimumDRL(env, {"steps": env.total_steps}, envy_weight=0.0),
        True,
    ),
    "wgt4DRL": (
        lambda env: OptimumDRL(env, {"steps": env.total_steps}, envy_weight=4.0),
        True,
    ),
}

# Column order of the per-episode demand artifact (test_instances_summary{tag}.csv).
INSTANCE_COLUMNS = (
    "scenario",
    "episode",
    "stream",
    "seed",
    "total_demand",
    "low_demand",
    "high_demand",
    "centre_demand",
    "suburb_demand",
    "centre_low",
    "centre_high",
    "suburb_low",
    "suburb_high",
    "upgrade_demand",
    "empty_feasible",
    "mean_soc",
    "min_soc",
    "max_soc",
    "first_submission",
    "last_submission",
    "buckets_with_demand",
    "max_bucket_demand",
)

# Column order of the per-policy time-step sample table
# (fairDRL_action_records{tag}.csv).
ACTION_COLUMNS = (
    "policy",
    "scenario",
    "episode",
    "step",
    "is_peak",
    "bundle_number",
    "bundle_seconds",
    "candidate_seconds",
    "bmip_model_seconds",
    "action_seconds",
    "bmip_requests",
    "bmip_options",
    "bmip_variables",
    "bmip_constraints",
)


class TestRunner:
    """Run ``policies`` on the episodes of one expansion network; an explicit
    empty list is rejected."""

    def __init__(self, policies=None, scenario=0,
                 episodes=TEST_EPISODES_PER_NETWORK,
                 action_samples=ACTION_SAMPLE_FAIRDRL,
                 scale=None, resume=True):
        if policies is None:
            policies = POLICIES
        if not policies:
            raise ValueError(
                "policies must be a non-empty list of policy names"
            )
        self.policies = list(policies)
        self.scenario = scenario  # expansion network: 0/1 or literal 8/16
        self.episodes = int(episodes)
        self.action_samples = int(action_samples)
        # Demand scale; applied to data_params at run time so GAT denominators follow it.
        self.scale = float(data_params["scale"] if scale is None else scale)
        self.resume = bool(resume)  # continue an interrupted sweep when True
        unknown = set(self.policies) - set(POLICIES)
        if unknown:
            raise ValueError(f"Unknown test policies: {sorted(unknown)}")

    @staticmethod
    def _scenario_index(scenario):
        """Normalize a scenario index or a literal expansion size to the 0-based index."""
        sizes = data_params["scenarios"]
        if isinstance(scenario, int) and 0 <= scenario < len(sizes):
            return scenario
        return sizes.index(scenario)

    def _seed(self, env, episode):
        """Deterministic demand seed: base + 2*(30*network_index + episode);
        every expansion network owns a private 30-episode block."""
        return env.seed + 2 * (
            TEST_EPISODES_PER_NETWORK * self._scenario_index(self.scenario) + episode
        )

    def _learners(self, env):
        """Build each requested policy and load DRL checkpoints."""
        learners = []
        for name in self.policies:
            factory, needs_checkpoint = POLICIES[name]
            learner = factory(env)
            if needs_checkpoint:
                learner.load_checkpoint()
            learners.append((name, learner))
        return learners

    def _completed(self, result_dir, name, tag):
        """Return (stem, next_episode, rows) for the newest dated summary family."""
        latest = latest_file(result_dir, f"{name}_{TEST_STREAM}{tag}_", "_summary.csv")
        if latest is None:
            return None, 0, []
        frame = pd.read_csv(latest)
        rows = (
            frame.to_dict("records")
            if not frame.empty and "episode" in frame.columns else []
        )
        start = int(frame["episode"].max()) + 1 if rows else 0
        return latest.name[: -len("_summary.csv")], start, rows

    def run(self):
        """Run the sweep for the selected network and export its tables."""
        saved_scale = data_params["scale"]
        data_params["scale"] = self.scale
        try:
            result_dir = Path(data_params["result_dir"])
            result_dir.mkdir(parents=True, exist_ok=True)
            tag = scenario_tag(self.scenario)

            env = BatterySwapEnv(seed=None)
            env.get_network("test", self.scenario)
            env.capture_action_steps = True
            learners = self._learners(env)

            # Newest dated summary per policy fixes its completed prefix.
            stems, starts, summaries = {}, {}, {}
            for name, _ in learners:
                if self.resume:
                    stem, start, rows = self._completed(result_dir, name, tag)
                else:
                    stem, start, rows = None, 0, []
                stems[name] = stem or run_stem(name, TEST_STREAM, self.scenario)
                starts[name] = start
                summaries[name] = rows
            print(
                f"resume={self.resume}: "
                + ", ".join(
                    f"{name} from episode {starts[name]}" for name, _ in learners
                ),
                flush=True,
            )

            # Drop stale request tables for the pair and every family of non-writing policies.
            for name, _ in learners:
                stem = stems[name]
                if name not in REQUEST_POLICIES:
                    for path in result_dir.glob(
                        f"{name}_{TEST_STREAM}{tag}_*_requests_*.csv"
                    ):
                        path.unlink()
                    continue
                for path in result_dir.glob(f"{stem}_requests_*.csv"):
                    suffix = path.stem.rsplit("_", 1)[-1]
                    if suffix.isdigit() and int(suffix) >= starts[name]:
                        path.unlink()

            # Keep the instance rows of already-completed episodes.
            instance_path = result_dir / f"test_instances_summary{tag}.csv"
            action_path = result_dir / f"fairDRL_action_records{tag}.csv"
            instance_rows = []
            min_start = min(starts.values()) if starts else 0
            # Report the actual episode window instead of the nominal budget so
            # the banner always matches the loop below.
            episode_count = max(0, self.episodes - min_start)
            window = (
                f"episodes {min_start}-{self.episodes - 1} "
                f"({episode_count} of {self.episodes})"
                if episode_count
                else "sweep already complete"
            )
            print(
                f"========== Test network {env.scenario}: ({len(env.Agents)} stations), "
                f"scale={self.scale}; {window} ==========",
                flush=True,
            )
            if self.resume and instance_path.is_file():
                previous = pd.read_csv(instance_path)
                if "episode" in previous.columns:
                    instance_rows = previous.loc[
                        previous.episode < min_start
                    ].to_dict("records")
            if starts.get("fairDRL", 0) == 0:
                # fairDRL restarts from episode 0: start a clean per-step store.
                # Do not clear it when fairDRL already finished and only other
                # policies are being resumed; the historical rows stay intact.
                action_path.unlink(missing_ok=True)

            action_rows = []
            for episode in range(min_start, self.episodes):
                env.initialize_instance(
                    data_params["scale"], self._seed(env, episode), episode
                )
                row = episode_statistics(env, episode)
                row.update(scenario=env.scenario, stream=TEST_STREAM)
                instance_rows.append(row)
                # Incremental instance table: upsert one row per episode.
                atomic_to_csv(
                    pd.DataFrame(instance_rows, columns=INSTANCE_COLUMNS)
                    .drop_duplicates("episode", keep="last")
                    .sort_values("episode"),
                    instance_path,
                )

                bundling = Bundling(env)
                for name, learner in learners:
                    if episode < starts[name]:
                        continue
                    env.write_request_records = name in REQUEST_POLICIES
                    stem = stems[name]
                    result = learner.scheduling(
                        bundling, train_flag=False, episode=episode, name=stem
                    )
                    result["scenario"] = env.scenario
                    summaries[name].append(result)
                    print(
                        f"**************** Test policy={name}: episode {episode} ****************",
                        flush=True,
                    )
                    print(
                        f'normalized profit={result["normalized_profit"]:.2f}, '
                        f'total demands=({result["low_demand"]}, {result["high_demand"]}, '
                        f'{result["upgrade_demand"]}), '
                        f'rejection={result["rejection"]:.2f} '
                        f'(low {result["rejection_low"]:.2f}, high {result["rejection_high"]:.2f}); \n'
                        f'mean subsidy={result["mean_subsidy"]:.2f}, '
                        f'mean wait={result["mean_wait"]:.2f} min, '
                        f'mean detour={result["mean_detour"]:.2f} min; \n'
                        f'action time={result["action_seconds_max"]:.2f} s, '
                        f'run time={result["run_time"]:.2f} min.',
                        flush=True,
                    )
                    if name == "fairDRL":
                        for record in env.action_step_records:
                            record.update(policy=name, scenario=env.scenario)
                        action_rows.extend(env.action_step_records)
                        # Incremental per-step store: append raw fairDRL rows.
                        pd.DataFrame(
                            env.action_step_records, columns=ACTION_COLUMNS
                        ).to_csv(
                            action_path,
                            mode="a",
                            header=not action_path.exists(),
                            index=False,
                        )
                    # Incremental summary: upsert one row per (policy, episode).
                    frame = pd.DataFrame(summaries[name])
                    if not frame.empty and "episode" in frame.columns:
                        frame = frame.drop_duplicates("episode", keep="last").sort_values("episode")
                    atomic_to_csv(frame, result_dir / f"{stems[name]}_summary.csv")

            # fairDRL timing samples: <=300 peak + <=300 off-peak steps from the store.
            candidates = list(action_rows)
            if action_path.is_file():
                candidates = pd.read_csv(action_path).to_dict("records") + candidates
            frame = pd.DataFrame(candidates)
            if not frame.empty and "policy" in frame.columns:
                frame = frame.loc[frame["policy"] == "fairDRL"].drop_duplicates(
                    ["episode", "step"], keep="last"
                )
            rng = np.random.default_rng(int(RL_params["seed"]))
            sampled = []
            if "fairDRL" in self.policies and not frame.empty:
                subset = frame.to_dict("records")
                for peak in (True, False):
                    group = [
                        row for row in subset
                        if _is_peak_step(row["step"]) == peak
                    ]
                    if len(group) > self.action_samples:
                        indices = rng.choice(
                            len(group), size=self.action_samples, replace=False
                        )
                        group = [group[int(index)] for index in sorted(indices)]
                    sampled.extend(group)
            atomic_to_csv(pd.DataFrame(sampled, columns=ACTION_COLUMNS), action_path)

            # Paired fairDRL-vs-optDRL comparison when both ran on this network.
            if {"fairDRL", "optDRL"} <= set(self.policies):
                Analyst().compare_requests(stream=TEST_STREAM, scenario_id=self.scenario)
            return summaries
        finally:
            data_params["scale"] = saved_scale
