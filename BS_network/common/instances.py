"""Seeded Poisson demand sampled from trajectory records."""

from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from config import data_params, env_params
from network.orders import Request

class Instance:
    def __init__(self, dataset, scale, eps_id, seed):
        """Build the scenario RNG and (once) the area/hour/slot source pools."""
        self.order_data = dataset  # mapped trajectory source rows incl. upgrade_days
        self.scale = scale  # demand scale multiplier
        self.episode = eps_id  # episode index
        self.seed = int(seed)  # scenario random seed
        self.rng = np.random.default_rng(self.seed)  # scenario RNG
        self.requests = []  # generated requests

        self.pools = dataset.attrs.get("sampling_pools")  # cached source-row index pools
        if self.pools is None:
            self.pools = {}
            for area in ("centre", "suburb"):
                subset = dataset[dataset.area == area]
                self.pools[(area, "all")] = subset.index.to_numpy()
                for hour, rows in subset.groupby("hour"):
                    self.pools[(area, int(hour))] = rows.index.to_numpy()
                for slot, rows in subset.groupby("slot"):
                    self.pools[(area, "slot", int(slot))] = rows.index.to_numpy()
            dataset.attrs["sampling_pools"] = self.pools

    def set_lambda(self, hour, area):
        """Return requests/minute for the area at the given hour."""
        intervals = (
            (data_params["peak_hour_centre"],)
            if area == "centre"
            else data_params["peak_hour_suburb"]
        )
        peak = any(a <= hour < b for a, b in intervals)
        return (
            self.scale
            * data_params[
                f'rate_{"peak" if peak else "off"}_{"centre" if area == "centre" else "suburb"}'
            ]
        )

    def sample_order(self, current_time, area):
        """Return (source row, fallback level) from slot/hour/area pools in order."""
        slot = (current_time.hour * 60 + current_time.minute) // env_params["delta_time"]
        keys = [(area, "slot", slot), (area, current_time.hour), (area, "all")]
        for fallback, key in enumerate(keys):
            rows = self.pools.get(key, ())
            if len(rows):
                return self.order_data.loc[int(self.rng.choice(rows))], fallback
        return self.order_data.iloc[-1], len(keys) - 1

    def generate_poisson(self, area):
        """Generate the area's arrivals across all decision intervals."""
        current = data_params["start_time"]
        delta = env_params["delta_time"]
        result = []
        # Intervals up to the acceptance cutoff, inclusive of the final decision interval.
        while current < data_params["decision_end_time"]:
            # Poisson count + uniform offsets (minutes) in each interval.
            count = self.rng.poisson(
                self.set_lambda(current.hour + current.minute / 60, area) * delta
            )
            offsets = self.rng.uniform(1e-7, delta, count)
            for offset in offsets:
                submission = current + timedelta(minutes=float(offset))

                row, fallback = self.sample_order(submission, area)
                installed = 1 if self.rng.random() < data_params["installed_low_probability"] else 2
                upgrade = (
                    installed == 1
                    and self.rng.random() < data_params["upgrade_probability"]
                )
                requested = 2 if upgrade else installed
                # upgrade_days only for upgrade orders; non-upgrades and legacy rows carry 0.
                low, high = data_params["soc_range"]
                soc = low + (high - low) * self.rng.beta(*data_params["soc_beta"])
                req = Request(
                    0,
                    row.cluster,
                    area,
                    row.longitude,
                    row.latitude,
                    submission,
                    soc,
                    installed,
                    requested,
                    int(row.id),
                    fallback,
                    row.source_area,
                    upgrade_days=(
                        int(row["upgrade_days"])
                        if upgrade
                        and "upgrade_days" in row
                        and pd.notna(row["upgrade_days"])
                        else 0
                    ),
                )

                result.append(req)
            current += timedelta(minutes=delta)
        return result

    def order_generation(self):
        """Merge centre/suburb streams by time and assign IDs 0..R-1."""
        self.requests = sorted(
            self.generate_poisson("centre") + self.generate_poisson("suburb"),
            key=lambda r: r.submission,
        )
        for rid, request in enumerate(self.requests):
            request.id = rid
        return self.requests

    def order_save(self):
        """Write data_params["sample_orders"] (one row per request)."""
        target = Path(data_params["data_dir"]) / data_params["sample_orders"]
        target.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                dict(
                    episode=self.episode,
                    seed=self.seed,
                    id=r.id,
                    source_id=r.source_id,
                    cluster=r.cluster,
                    area=r.area,
                    source_area=r.source_area,
                    longitude=r.longitude,
                    latitude=r.latitude,
                    submission=r.submission,
                    SOC=r.SOC,
                    installed_class=r.installed_class,
                    requested_class=r.requested_class,
                    sample_fallback=r.sample_fallback,
                )
                for r in self.requests
            ],
            columns=(
                "episode seed id source_id cluster area source_area longitude latitude submission SOC "
                "installed_class requested_class sample_fallback"
            ).split(),
        ).to_csv(target, index=False)
        print('-------------------- Requests --------------------')
        print(
            f'Sample demand with {len(self.requests)} requests saved to: '
            f'{data_params["sample_orders"]}'
        )
