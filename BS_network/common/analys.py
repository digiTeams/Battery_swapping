"""Timing summaries and paired policy-outcome comparisons."""

import re
import warnings
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RL_params, TIMING_COLUMNS, data_params, env_params
from common.outputs import latest_file, scenario

# Timing column -> plot label.
TIMING_CURVES = {
    "bundle_seconds": "Bundle generation",
    "model_build_seconds": "BMIP construction",
    "model_solve_seconds": "Optimization",
}

# Own aggregate tables, skipped when scanning per-policy tables.
AGGREGATE_BASES = (
    "timing_summary",
    "paired_request_outcomes",
    "competing_group_outcomes",
    "competing_group_summary",
)


def _quantile95(series):
    """Return the 0.95 quantile of a grouped series (for named aggregations)."""
    return series.quantile(0.95)


def _group_period(decision_step, area):
    """Classify a decision step as peak/off-peak for the group's area
    (centre 12:00-17:00; suburb 8:00-10:00 and 18:00-20:00)."""
    if pd.isna(decision_step):
        return "off"
    minutes = 60 * data_params["start_time"].hour + int(decision_step) * int(
        env_params["delta_time"]
    )
    hour = minutes / 60.0
    if area == "centre":
        start, end = data_params["peak_hour_centre"]
        return "peak" if start <= hour < end else "off"
    return "peak" if any(s <= hour < e for s, e in data_params["peak_hour_suburb"]) else "off"


def is_aggregate(name):
    """True for tables written by this module (dated or legacy)."""
    return any(
        name == f"{base}.csv" or name.startswith(f"{base}_") for base in AGGREGATE_BASES
    )


def _dated_key(path):
    """Sort key using the ISO date embedded in a result filename."""
    name = Path(path).name
    match = re.search(r"\d{4}-\d{2}-\d{2}", name)
    return (match.group(0) if match else "", name)


class Analyst:
    """Timing summaries and paired request-outcome comparisons between policies."""

    def __init__(self, window=20):
        """Set the smoothing window."""
        self.window = window  # int: smoothing window for moving averages

    def timing_summary(self):
        """Concatenate every *_summary.csv into timing_summary_{date}.csv."""
        columns = [
            "episode", "steps", "fallback_steps", "solver_failed_steps",
            "positive_gap_steps",
        ] + [
            f"{column}_{suffix}"
            for column in TIMING_COLUMNS
            for suffix in ("mean", "p95", "max", "total")
        ]
        rows = []
        for path in sorted(Path(data_params["result_dir"]).glob("*_summary.csv")):
            # instances_summary.csv is a per-episode demand artifact, not a policy table.
            if is_aggregate(path.name) or path.name == "instances_summary.csv":
                continue
            frame = pd.read_csv(path)

            # Prefix before _summary.csv is the full dated run label.
            frame.insert(0, "policy", path.name[: -len("_summary.csv")])
            rows.append(frame[[name for name in ["policy"] + columns if name in frame]])
        result = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(columns=["policy"] + columns)
        )
        result.to_csv(
            Path(data_params["result_dir"]) / f"timing_summary_{date.today().isoformat()}.csv",
            index=False,
        )
        return result

    @staticmethod
    def _request_tables(directory, policy, stream, tag):
        """Return {episode: path} for the newest dated request-table family."""
        prefix = f"{policy}_{stream}{tag}_"
        pattern = re.compile(
            re.escape(prefix) + r"(\d{4}-\d{2}-\d{2})_requests_(\d{3,})\.csv"
        )
        dated = {}
        for path in directory.glob(f"{prefix}*_requests_*.csv"):
            match = pattern.fullmatch(path.name)
            if match:
                dated.setdefault(match.group(1), {})[int(match.group(2))] = path
        if not dated:
            return {}
        return dated[max(dated)]

    def compare_requests(
        self,
        stream="eval",
        scenario_id=None,
        optional=False,
    ):
        """Pair optDRL and fairDRL request tables on (episode, seed, request)
        and export three competing-group tables; rejected sides are masked to
        NaN and fairness_gap = max_envy_efficient - max_envy_fair."""
        fair_policy, efficient_policy = "fairDRL", "optDRL"
        all_pairs = []
        directory = Path(data_params["result_dir"])
        tags = (
            [scenario(scenario_id)]
            if scenario_id is not None
            else [""] + [scenario(index) for index in range(len(data_params["scenarios"]))]
        )
        fair_files = efficient_files = None
        matched_tag = ""
        for tag in tags:
            fair_candidate = self._request_tables(directory, fair_policy, stream, tag)
            efficient_candidate = self._request_tables(
                directory, efficient_policy, stream, tag
            )
            if fair_candidate and efficient_candidate:
                fair_files, efficient_files, matched_tag = (
                    fair_candidate, efficient_candidate, tag,
                )
                break
        if fair_files is None:
            fair_files, efficient_files = {}, {}
        absent = [
            name
            for name, files in (
                (fair_policy, fair_files),
                (efficient_policy, efficient_files),
            )
            if not files
        ]
        if absent:
            message = (
                f"Comparison {fair_policy} vs {efficient_policy} skipped: missing "
                f"request tables for {', '.join(absent)} in {directory} "
                f"(searched tags: {tags})"
            )
            warnings.warn(message)
            return pd.DataFrame()

        # A policy run may have been interrupted, so the two episode maps need
        # not have the same keys. Compare only episodes present for both
        # policies, and make the dropped episodes visible to the caller.
        fair_episodes = set(fair_files)
        efficient_episodes = set(efficient_files)
        paired_episodes = sorted(fair_episodes & efficient_episodes)
        only_fair = sorted(fair_episodes - efficient_episodes)
        only_efficient = sorted(efficient_episodes - fair_episodes)
        if only_fair or only_efficient:
            warnings.warn(
                f"Comparison {fair_policy} vs {efficient_policy} using tag "
                f"{matched_tag or '<base>'}: episode sets differ. "
                f"Comparing {len(paired_episodes)} common episodes; "
                f"only in {fair_policy}: {only_fair or 'none'}; "
                f"only in {efficient_policy}: {only_efficient or 'none'}."
            )
        if not paired_episodes:
            warnings.warn(
                f"Comparison {fair_policy} vs {efficient_policy} skipped: the two "
                f"policies have no evaluated episodes in common (tag "
                f"{matched_tag or '<base>'})."
            )
            return pd.DataFrame()

        for episode in paired_episodes:
            fair = fair_files[episode]
            other = efficient_files[episode]
            a, b = pd.read_csv(fair), pd.read_csv(other)

            # One-to-one join; unmatched rows (interrupted writes) must not become silent NaNs.
            pairs = a.merge(
                b,
                on=["episode", "seed", "request"],
                suffixes=("_fair", "_efficient"),
                how="outer",
                indicator=True,
                validate="one_to_one",
            )
            unmatched = int((pairs["_merge"] != "both").sum())
            if unmatched:
                warnings.warn(
                    f"Comparison {fair_policy} vs {efficient_policy}, episode "
                    f"{episode}: {unmatched} request row(s) exist in only one "
                    "policy table; those rows are excluded from the paired set."
                )
            pairs = pairs.loc[pairs["_merge"] == "both"].drop(columns="_merge")
            if pairs.empty:
                continue

            pairs["assignment_changed"] = pairs.station_fair != pairs.station_efficient
            pairs["accepted_fair"] = pairs.station_fair >= 0
            pairs["accepted_efficient"] = pairs.station_efficient >= 0
            # Mask rejected sides (raw 0 wait/detour) to NaN so differences and
            # group aggregates exclude rejected orders.
            for column in ("waiting_minutes", "detour_minutes"):
                pairs[f"{column}_fair"] = pairs[f"{column}_fair"].where(
                    pairs["accepted_fair"]
                )
                pairs[f"{column}_efficient"] = pairs[f"{column}_efficient"].where(
                    pairs["accepted_efficient"]
                )
            pairs["extra_wait_minutes"] = (
                pairs.waiting_minutes_fair - pairs.waiting_minutes_efficient
            )
            pairs["extra_detour_minutes"] = (
                pairs.detour_minutes_fair - pairs.detour_minutes_efficient
            )
            pairs["extra_subsidy"] = pairs.subsidy_fair - pairs.subsidy_efficient
            pairs["net_disutility_difference"] = (
                pairs.net_disutility_fair - pairs.net_disutility_efficient
            )
            # Uncompensated envy under the efficient policy (the fair side is 0).
            pairs["fairness_gap"] = pairs.max_envy_efficient - pairs.max_envy_fair
            pairs["same_station"] = (
                (pairs.station_fair >= 0) & (pairs.station_fair == pairs.station_efficient)
            )
            pairs["both_rejected"] = (
                (pairs.station_fair < 0) & (pairs.station_efficient < 0)
            )
            pairs["one_rejected"] = (pairs.station_fair < 0) != (pairs.station_efficient < 0)
            if "nearest_station_fair" in pairs:
                pairs["diverted_fair"] = (
                    (pairs.station_fair >= 0)
                    & (pairs.station_fair != pairs.nearest_station_fair)
                )
                pairs["diverted_efficient"] = (
                    (pairs.station_efficient >= 0)
                    & (pairs.station_efficient != pairs.nearest_station_fair)
                )
                pairs["single_battery_nearest"] = pairs.nearest_stock_fair == 1
            else:
                pairs["diverted_fair"] = pairs["diverted_efficient"] = False
                pairs["single_battery_nearest"] = False
            pairs["same_station_diverted"] = pairs["same_station"] & pairs["diverted_fair"]
            if "decision_step_fair" in pairs:
                pairs["group_size"] = pairs.groupby(
                    ["decision_step_fair", "cluster_fair", "requested_class_fair"]
                )["request"].transform("size")
            else:
                pairs["group_size"] = 0
            all_pairs.append(pairs)
        if not all_pairs:
            warnings.warn(
                f"Comparison {fair_policy} vs {efficient_policy} skipped: no "
                "matching request rows in the common episodes."
            )
            return pd.DataFrame()
        merged = pd.concat(all_pairs, ignore_index=True)
        day = date.today().isoformat()

        # Output (3): up to 600 paired rows (reproducible via the master seed).
        rng = np.random.default_rng(int(RL_params["seed"]))
        if len(merged) > 600:
            indices = sorted(int(i) for i in rng.choice(len(merged), size=600, replace=False))
            sampled = merged.iloc[indices]
        else:
            sampled = merged
        sampled.to_csv(
            directory
            / f"paired_request_outcomes_{fair_policy}_{efficient_policy}{matched_tag}_{day}.csv",
            index=False,
        )

        # Output (1): one row per competing group.
        competing = merged[merged["group_size"] >= 2]
        if competing.empty or "decision_step_fair" not in competing:
            warnings.warn(
                f"Comparison {fair_policy} vs {efficient_policy}: no competing "
                "groups (group_size >= 2) with decision-step information; "
                "competing-group tables are skipped."
            )
            return merged
        group_keys = [
            "episode", "decision_step_fair", "cluster_fair", "requested_class_fair"
        ]
        agg_spec = {
            "area_fair": ("area_fair", "first"),
            "group_size": ("request", "size"),
            "nearest_stock_min": ("nearest_stock_fair", "min"),
            "single_battery_groups": ("single_battery_nearest", "max"),
            "same_station_count": ("same_station", "sum"),
            "same_station_diverted_count": ("same_station_diverted", "sum"),
            "different_station_count": ("assignment_changed", "sum"),
            "assignment_changed": ("assignment_changed", "mean"),
            "one_rejected_count": ("one_rejected", "sum"),
            "both_rejected_count": ("both_rejected", "sum"),
            "mean_wait_fair": ("waiting_minutes_fair", "mean"),
            "max_wait_fair": ("waiting_minutes_fair", "max"),
            "mean_wait_efficient": ("waiting_minutes_efficient", "mean"),
            "max_wait_efficient": ("waiting_minutes_efficient", "max"),
            "mean_extra_wait": ("extra_wait_minutes", "mean"),
            "mean_detour_fair": ("detour_minutes_fair", "mean"),
            "mean_detour_efficient": ("detour_minutes_efficient", "mean"),
            "mean_extra_detour": ("extra_detour_minutes", "mean"),
            "mean_subsidy_fair": ("subsidy_fair", "mean"),
            "mean_envy_efficient": ("max_envy_efficient", "mean"),
            "net_disutility_difference": ("net_disutility_difference", "mean"),
        }
        if "cluster_pre_wait_fair" in competing.columns:
            agg_spec["waiting_pre_action"] = ("cluster_pre_wait_fair", "first")
        group = competing.groupby(group_keys).agg(**agg_spec).reset_index()
        if "waiting_pre_action" not in group.columns:
            group["waiting_pre_action"] = np.nan
        # max() over booleans yields bool; int makes mean/p95 meaningful.
        group["single_battery_groups"] = group["single_battery_groups"].astype(int)
        group["period"] = [
            _group_period(step, area)
            for step, area in zip(group.decision_step_fair, group.area_fair)
        ]
        group_columns = group_keys + [
            "area_fair", "period", "group_size", "nearest_stock_min",
            "single_battery_groups", "waiting_pre_action", "same_station_count",
            "same_station_diverted_count", "different_station_count",
            "assignment_changed", "one_rejected_count", "both_rejected_count",
            "mean_wait_fair", "max_wait_fair", "mean_wait_efficient",
            "max_wait_efficient", "mean_extra_wait", "mean_detour_fair",
            "mean_detour_efficient", "mean_extra_detour", "mean_subsidy_fair",
            "mean_envy_efficient", "net_disutility_difference",
        ]
        group[group_columns].to_csv(
            directory
            / f"competing_group_outcomes_{fair_policy}_{efficient_policy}{matched_tag}_{day}.csv",
            index=False,
        )

        # Output (2): 8 classes x mean/max/p95 of every group field (empty classes kept).
        stat_fields = [
            column for column in group_columns
            if column not in group_keys + ["area_fair", "period"]
        ]
        summary_spec = {
            f"{field}_{suffix}": (field, suffix)
            for field in stat_fields
            for suffix in ("mean", "max")
        }
        summary_spec.update(
            {f"{field}_p95": (field, _quantile95) for field in stat_fields}
        )
        summary = group.groupby(
            ["period", "area_fair", "requested_class_fair"]
        ).agg(**summary_spec).reindex(
            pd.MultiIndex.from_product(
                [("peak", "off"), ("centre", "suburb"), (1, 2)],
                names=["period", "area_fair", "requested_class_fair"],
            )
        )
        summary.to_csv(
            directory
            / f"competing_group_summary_{fair_policy}_{efficient_policy}{matched_tag}_{day}.csv"
        )
        return merged

    def plot_timing(self, policy, optional=False):
        """Plot the episode-level mean timings of one policy."""
        directory = Path(data_params["result_dir"])

        # Mean seconds per decision against the episode index, taken from the episode tables.
        # The evaluation entry point uses short baseline names; trainers use their fair names.
        aliases = {"RHO": "fairRHO", "Greedy": "fairGreedy"}
        name = aliases.get(policy, policy)
        # Timing is stream-agnostic: newest dated summary across train/eval and
        # scenario-tagged families.
        source = None
        for stream in ("train", "eval"):
            prefixes = [f"{name}_{stream}_"] + [
                f"{name}_{stream}{scenario(index)}_"
                for index in range(len(data_params["scenarios"]))
            ]
            for prefix in prefixes:
                candidate = latest_file(directory, prefix, "_summary.csv")
                if candidate is not None:
                    source = candidate if source is None else max(
                        source, candidate, key=_dated_key
                    )
        if source is None:
            for candidate in (f"{name}_summary.csv", f"{policy}_summary.csv"):
                legacy = directory / candidate
                if legacy.is_file():
                    source = legacy
                    break
        if source is None:
            message = f"Missing timing table for policy {policy} in {directory}"
            if optional:
                warnings.warn(message)
                return None
            raise FileNotFoundError(message)
        frame = pd.read_csv(source)
        frame = frame.rename(columns={f"{column}_mean": column for column in TIMING_COLUMNS})
        output = directory / f"{policy}_timing_{date.today().isoformat()}.png"

        fig, axis = plt.subplots(figsize=(9, 3.2), layout="constrained")

        # Plot each available timing series against the episode index, then save the figure.
        for column, label in TIMING_CURVES.items():
            if column in frame:
                axis.plot(frame.episode, frame[column], label=label, linewidth=1)
        axis.set(xlabel="Episode", ylabel="Seconds")
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
        fig.savefig(output, dpi=180)
        plt.close(fig)
        return output
