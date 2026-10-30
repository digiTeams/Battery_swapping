"""Plots of saved policy metrics and training curves."""

import warnings
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import data_params
from common.outputs import latest_file

FILES = {
    # Evaluation-stream baselines share the *_eval demand seeds.
    "greedy": ("fairGreedy", "eval"),
    "rho": ("fairRHO", "eval"),
    # Training-stream baselines share the DRL training demand seeds.
    "greedy_train": ("fairGreedy", "train"),
    "rho_train": ("fairRHO", "train"),
    # Fixed-target variants (target_fixed=True, fixed_level).
    "greedy_fixed": ("fairGreedy_fixed", "eval"),
    "rho_fixed": ("fairRHO_fixed", "eval"),
    "greedy_fixed_train": ("fairGreedy_fixed", "train"),
    "rho_fixed_train": ("fairRHO_fixed", "train"),
    # DRL training runs, named by the policy objects in Trainer.
    "fair": ("fairDRL", "train"),
    "fair_one": ("fairDRL_one", "train"),
    "fair_local": ("fairDRL_local", "train"),
    "fair_fixed": ("fairDRL_fixed", "train"),
    "opt": ("optDRL", "train"),

    # Weighted run: OptimumDRL writes wgt{int(envy_weight)}DRL; key matches main.py's wgt4DRL.
    "wgt4": ("wgt4DRL", "train"),
}


class PolicyEvals:
    """Draws policy comparisons and training curves from saved summary tables."""

    def __init__(self, window, files=None):
        """Set the smoothing window and table mapping."""
        self.window = window  # int: smoothing window for moving averages
        self.files = dict(FILES if files is None else files)  # policy key -> (policy, stream)

    def _read(self, key, required=True):
        """Read the newest dated (or legacy undated) summary; optional tables warn."""
        policy, stream = self.files[key]
        directory = Path(data_params["result_dir"])
        path = latest_file(directory, f"{policy}_{stream}_", "_summary.csv")
        if path is None:
            for candidate in (f"{policy}_{stream}_summary.csv", f"{policy}_summary.csv"):
                legacy = directory / candidate
                if legacy.is_file():
                    path = legacy
                    break
        if path is None:
            if required:
                raise FileNotFoundError(
                    f"Missing policy table for {key} ({policy}, {stream}): {directory}"
                )
            warnings.warn(
                f"Skipping optional policy {key} ({policy}, {stream}): "
                f"no summary table in {directory}"
            )
            return None
        return self._derive(pd.read_csv(path))

    def mean_reference(self, key, column="total_rewards"):
        """Return a column mean on the raw scale, or None when absent."""
        frame = self._read(key, required=False)
        if frame is None or frame.empty or column not in frame:
            return None
        return float(frame[column].mean())

    @staticmethod
    def _derive(frame):
        """Add mean_envy/mean_subsidy (totals / demand) when absent."""
        demand = frame.get("total_demand")
        if demand is not None:
            demand = demand.clip(lower=1)
            if "mean_envy" not in frame and "total_envies" in frame:
                frame["mean_envy"] = frame["total_envies"] / demand
            if "mean_subsidy" not in frame and "total_subsidies" in frame:
                frame["mean_subsidy"] = frame["total_subsidies"] / demand
        return frame

    @staticmethod
    def _series(frame, column, scale=1.0):
        """Return (episodes, scaled column) ndarrays."""
        return frame["episode"].to_numpy(), frame[column].to_numpy(dtype=float) / scale

    def moving_average(self, data):
        """Return trailing means of a series."""
        data = np.asarray(data, dtype=float)
        return [data[max(0, i - self.window + 1) : i + 1].mean() for i in range(len(data))]

    def moving_average_with_ci(self, data):
        """Return trailing means and approximate 1.96*SE bands within each window."""
        # Bands summarize one run's episode window; they are not across-seed intervals.
        data = np.asarray(data, dtype=float)
        means, lower, upper = [], [], []

        for i in range(len(data)):
            window = data[max(0, i - self.window + 1) : i + 1]
            mean = window.mean()
            ci = 1.96 * window.std(ddof=1) / np.sqrt(len(window)) if len(window) > 1 else 0.0
            means.append(mean)
            lower.append(mean - ci)
            upper.append(mean + ci)
        return means, lower, upper

    def _curve(
        self,
        axis,
        episodes,
        values,
        color,
        label,
        *,
        marker="o",
        linestyle="-",
        ci=True,
        bounds=None,
    ):
        """Draw one smoothed curve and collect its displayed range."""
        if ci:
            means, lower, upper = self.moving_average_with_ci(values)
        else:
            means, lower, upper = self.moving_average(values), None, None

        axis.plot(
            episodes, means, marker=marker, markersize=2.5, linestyle=linestyle,
            color=color, label=label,
        )
        if ci:
            axis.fill_between(episodes, lower, upper, color=color, alpha=0.2)

        if bounds is not None:
            if ci:
                bounds.extend(lower)
                bounds.extend(upper)
            else:
                bounds.extend(means)
        return means

    def _finish(self, axis, ylabel, title, bounds, floor=None):
        """Apply axis labels, limits, formatting, and legends."""
        axis.set_ylim(floor if floor is not None else min(bounds), max(bounds))
        axis.set_xlabel("Episode")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
        axis.grid(True)

    def _save(self, figure, name):
        """Save and close a completed figure without blocking on an interactive window."""
        directory = Path(data_params["result_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        # Normalize a caller-supplied .png suffix so the date lands before the extension.
        base = name[:-4] if name.endswith(".png") else name
        output = directory / f"{base}_{date.today().isoformat()}.png"
        figure.savefig(output)
        plt.close(figure)
        return output

    def plot_DRLs(self, base_rewards):
        """Plot DRL rewards and training losses with baseline reward references."""

        # axs: object ndarray (2,) of reward and loss panels; metric series have one value per episode.
        figure, axs = plt.subplots(1, 2, figsize=(10, 5))
        reward_bounds, loss_bounds = [], []

        # Add available baseline reward references.
        if base_rewards.get("Greedy") is not None:
            axs[0].axhline(
                y=base_rewards["Greedy"] / 1e4, color="black", label="Greedy Reward"
            )
        if base_rewards.get("RHO") is not None:
            axs[0].axhline(
                y=base_rewards["RHO"] / 1e4, color="yellow", label="RHO Reward"
            )

        # Curve rows: (table key, label, color, marker, required flag, confidence-band flag).
        curves = [
            ("fair", "Fair DRL", "red", "o", True, True),
            ("fair_one", "Fair DRL (one)", "purple", "v", False, True),
            ("fair_local", "Fair DRL (local)", "orange", "^", False, True),
            ("fair_fixed", "Fair DRL (fixed)", "brown", "*", False, True),
            ("opt", "Optimum DRL (w=0)", "black", "D", True, True),
            ("wgt4", "Weighted DRL (w=4)", "blue", "s", False, True),
        ]
        for key, label, color, marker, required, ci in curves:
            frame = self._read(key, required=required)
            if frame is None:
                continue
            episodes, rewards = self._series(frame, "total_rewards", scale=1e4)
            _, losses = self._series(frame, "loss", scale=1e3)
            for axis, values, bounds in (
                (axs[0], rewards, reward_bounds),
                (axs[1], losses, loss_bounds),
            ):
                self._curve(
                    axis, episodes, values, color, f"Policy {label}",
                    marker=marker, ci=ci, bounds=bounds,
                )

        # Finalize shared formatting and save the reward/loss figure.
        self._finish(axs[0], "Total Rewards (1e4)", "Total Rewards for DRL Policies", reward_bounds)
        self._finish(
            axs[1], "Training Loss (1e3)", "Training Loss for DRL Policies", loss_bounds, floor=0
        )
        return self._save(figure, "DRL_policies.png")

    def plot_policies(self):
        """Plot financial outcomes and envy across policies."""

        # axs: object ndarray (2,) of financial and envy panels.
        figure, axs = plt.subplots(1, 2, figsize=(10, 5))
        profit_bounds, envy_bounds = [], []

        curves = [
            ("greedy_train", "Greedy policy", "green", False, True),
            ("rho_train", "RHO policy", "blue", False, True),
            ("fair", "Fair DRL policy", "red", True, True),
            ("fair_one", "Fair DRL (one) policy", "purple", True, False),
            ("fair_local", "Fair DRL (local) policy", "orange", True, False),
            ("fair_fixed", "Fair DRL (fixed) policy", "brown", False, False),
            ("wgt4", "DRL-optimizer(w=4)", "yellow", True, False),
            ("opt", "DRL-optimizer(w=0)", "black", True, True),
        ]

        # Draw cash profits after both subsidies and charging costs, alongside mean envy.
        for key, label, color, ci, required in curves:
            frame = self._read(key, required=required)
            if frame is None:
                continue
            episodes, profits = self._series(frame, "total_profits", scale=1e4)
            _, envies = self._series(frame, "mean_envy")
            self._curve(
                axs[0], episodes, profits, color, label, ci=ci, bounds=profit_bounds
            )
            self._curve(axs[1], episodes, envies, color, label, ci=ci, bounds=envy_bounds)

        self._finish(axs[0], "Total Profits (1e4)", "Total Profits for DRL policies", profit_bounds)
        self._finish(axs[1], "Request Envies", "Request Envies for DRL policies", envy_bounds)
        return self._save(figure, "all_policies.png")
