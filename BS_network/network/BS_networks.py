"""Load station specifications and aligned demand source tables."""

import ast
from pathlib import Path
import pandas as pd
import numpy as np
from config import data_params, env_params
from common.outputs import expansion_station_file, scenario_size
from network.grids import Grid
from network.clusters import Cluster
from network.stations import Station

class BSNetwork:
    """Loads grids/clusters/stations and the filtered trajectory table."""
    def __init__(self, seed=None):
        self.city = data_params["city"]  # city label
        self.seed = int(data_params["seed"] if seed is None else seed)  # construction/demand seed

        self.Grids, self.Clusters, self.Stations = [], [], []
        self.order_data = pd.DataFrame()  # mapped trajectory rows incl. upgrade_days
        
    def load_network(self, task="training", scenario=None):
        """Load tables; "test" + scenario selects the matching expansion."""
        grids = pd.read_csv(Path(data_params["data_dir"]) / data_params["grid_data"])
        self.Grids = [
            Grid(
                int(r.id),
                r.type,
                r.station_count,
                ast.literal_eval(r.rate),
                ast.literal_eval(r.center),
                ast.literal_eval(r.bound),
                ast.literal_eval(r.centroid),
                ast.literal_eval(r.boundary_points)
            )
            for r in grids.itertuples()
        ]

        clusters = pd.read_csv(Path(data_params["data_dir"]) / data_params["cluster_data"])
        self.Clusters = [
            Cluster(
                int(r.id),
                r.type,
                ast.literal_eval(r.center),
                ast.literal_eval(r.bound),
                ast.literal_eval(r.centroid),
                ast.literal_eval(r.boundary_points)
            )
            for r in clusters.itertuples()
        ]

        source = data_params["station_data"]
        if task == "test" and scenario is not None:
            added = scenario_size(scenario)
            source = expansion_station_file(added)
        stations = pd.read_csv(Path(data_params["data_dir"]) / source).sort_values("id")

        # Legacy expansion tables inherit current base stock/power at matched locations.
        legacy = source != data_params["station_data"] and not {
            "initial_batteries", "grid_power"
        }.issubset(stations.columns)
        if legacy:
            base = pd.read_csv(Path(data_params["data_dir"]) / data_params["station_data"])
            base = base.set_index("id")
            for column in ("initial_batteries", "grid_power"):
                if column not in stations:
                    stations[column] = pd.Series(index=stations.index, dtype=object)
            for index, row in stations.iterrows():
                if row.id not in base.index:
                    continue
                original = base.loc[row.id]
                for column in ("cluster", "capacity", "swap_time", "initial_batteries", "grid_power"):
                    stations.at[index, column] = original[column]

        # Validate clusters; derive missing stock/power from station size.
        types = {c.id: c.type for c in self.Clusters}
        self.Stations = []
        for row in stations.itertuples():
            cluster = int(row.cluster)
            matches = [
                c.id for c in self.Clusters
                if c.contains_location(row.longitude, row.latitude)
            ]
            if cluster not in matches:
                if matches:
                    cluster = min(matches)
                else:
                    # Assign the nearest boundary cluster.
                    boundary = min(
                        self.Clusters,
                        key=lambda c: (c.distance_to_bounds(row.longitude, row.latitude), c.id),
                    )
                    cluster = boundary.id
            capacity = int(row.capacity)
            size = "small" if capacity < int(env_params["large_size"]) else "large"
            stock = self._optional(row, "initial_batteries")
            stock = (
                env_params[f"initial_batteries_{size}"]
                if stock is None
                else ast.literal_eval(stock)
            )
            power = self._optional(row, "grid_power")
            power = env_params[f"grid_power_{size}"] if power is None else float(power)

            self.Stations.append(
                Station(
                    int(row.id), cluster, row.longitude, row.latitude, row.swap_time,
                    capacity, stock, power
                )
            )

        # raw trajectory: id, location, date/hour/minute, area, cluster.
        columns = ["id", "longitude", "latitude", "date", "hour", "minute", "area", "cluster"]
        raw = pd.read_csv(
            Path(data_params["data_dir"]) / data_params["order_data"], usecols=columns
        )
        # Keep valid rows; reconcile area against the live cluster map; slot = decision interval.
        keep = (
            raw.cluster.isin(types)
            & raw.area.isin(("centre", "suburb"))
            & raw.longitude.notna()
            & raw.latitude.notna()
        )
        self.order_data = raw.loc[keep].copy()
        self.order_data["cluster"] = self.order_data.cluster.astype(int)
        self.order_data["source_area"] = self.order_data["area"]
        self.order_data["area"] = self.order_data["cluster"].map(types)
        self.order_data["slot"] = (
            self.order_data.hour * 60 + self.order_data.minute
        ) // env_params["delta_time"]
        # Per-order upgrade charge period; non-upgrades keep 0.
        self.order_data["upgrade_days"] = np.random.default_rng(self.seed).integers(
            *data_params["upgrade_days_range"], size=len(self.order_data)
        )

        print(f"The BS network in {self.city} city is built with {len(self.Stations)} stations, "
              f"{len(self.Grids)} grids, {len(self.Clusters)} clusters, "
              f"and {len(self.order_data)} orders.")
        return self

    @staticmethod
    def _optional(row, column):
        """Return a nonmissing optional station field, else None."""
        value = getattr(row, column, None)
        return None if value is None or pd.isna(value) else value
