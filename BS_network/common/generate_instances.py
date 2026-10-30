"""Generate demand instances and one statistics row per episode to
results/instances_summary.csv via the exact training demand path; rows append
as produced and restarts skip completed episodes."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import RL_params, data_params
from simulator.BS_env import BatterySwapEnv


MAX_TRAIN_EPISODES = RL_params["MAX_Episodes"]

SUMMARY_COLUMNS = (
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


def episode_statistics(env, episode):
    """Return one summary row for the episode already generated in env."""
    requests = env.all_demand
    low = sum(r.requested_class == 1 for r in requests)
    high = sum(r.requested_class == 2 for r in requests)
    centre = sum(r.area == "centre" for r in requests)
    suburb = sum(r.area == "suburb" for r in requests)
    centre_low = sum(r.area == "centre" and r.requested_class == 1 for r in requests)
    centre_high = sum(r.area == "centre" and r.requested_class == 2 for r in requests)
    suburb_low = sum(r.area == "suburb" and r.requested_class == 1 for r in requests)
    suburb_high = sum(r.area == "suburb" and r.requested_class == 2 for r in requests)
    upgrades = sum(r.installed_class < r.requested_class for r in requests)
    empty_feasible = sum(not r.feasible_stations for r in requests)
    socs = np.array([r.SOC for r in requests], dtype=float)
    bucket_sizes = [len(bucket) for bucket in env.demand_by_step]
    return dict(
        episode=episode,
        stream="train",
        seed=env.episode_seed,
        total_demand=len(requests),
        low_demand=low,
        high_demand=high,
        centre_demand=centre,
        suburb_demand=suburb,
        centre_low=centre_low,
        centre_high=centre_high,
        suburb_low=suburb_low,
        suburb_high=suburb_high,
        upgrade_demand=upgrades,
        empty_feasible=empty_feasible,
        mean_soc=float(socs.mean()) if socs.size else np.nan,
        min_soc=float(socs.min()) if socs.size else np.nan,
        max_soc=float(socs.max()) if socs.size else np.nan,
        first_submission=(
            min(r.submission for r in requests).isoformat() if requests else None
        ),
        last_submission=(
            max(r.submission for r in requests).isoformat() if requests else None
        ),
        buckets_with_demand=sum(size > 0 for size in bucket_sizes),
        max_bucket_demand=max(bucket_sizes, default=0),
    )


if __name__ == "__main__":
    """Generate training episodes 0..MAX_TRAIN_EPISODES-1 and append their rows."""
    output = Path(data_params["result_dir"]) / "instances_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Lock file prevents concurrent runs from interleaving their appends.
    lock_path = output.with_name(output.name + ".lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, b"locked")
        os.close(descriptor)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another generate_instances run holds {lock_path}; "
            "concurrent runs would corrupt instances_summary.csv"
        ) from exc

    try:
        # Skip episodes that a previous (interrupted) run already completed.
        completed = set()
        if output.exists():
            previous = pd.read_csv(output)
            if "episode" in previous.columns:
                completed = set(previous["episode"].astype(int))

        env = BatterySwapEnv(seed=None)
        env.get_network("training", None)

        writer = None
        for episode in range(MAX_TRAIN_EPISODES):
            if episode in completed:
                continue
            env.initialize_instance(
                data_params["scale"],
                env.get_episode_seed(episode, training=True),
                episode,
            )
            row = episode_statistics(env, episode)
            frame = pd.DataFrame([row], columns=SUMMARY_COLUMNS)
            frame.to_csv(
                output,
                index=False,
                mode="a",
                header=(writer is None and not completed),
            )
            writer = writer if writer is not None else "open"
            if episode % 50 == 0:
                print(
                    f"Generated episode {episode}: total_demand={row['total_demand']}, "
                    f"low={row['low_demand']}, high={row['high_demand']}, "
                    f"centre={row['centre_demand']}, suburb={row['suburb_demand']}",
                    flush=True,
                )
    finally:
        lock_path.unlink(missing_ok=True)

    print(f"Instances summary written to {output}")



