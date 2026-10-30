"""Construct spatial cells, station tables, and mapped trajectories."""
import os
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, box
import geopandas as gpd

from config import data_params, env_params, RL_params
from common.geo import haversine_km
from common.outputs import expansion_station_file
from network import grids, clusters, stations


class BSN:
    """Builds/refreshes spatial cells, station tables, and mapped trajectories."""
    def __init__(self, city=None, seed=None):
        """Set the city, the construction RNG, bounds, and empty containers."""
        self.city = city or data_params["city"]  # str: case-study city label
        self.seed = int(data_params["seed"] if seed is None else seed)  # int: source/network seed
        self.rng = np.random.default_rng(
            self.seed
        )  # Generator: network-construction random stream

        self.bounds = (0, 0, 0, 0)  # float (4,): min/max lat, min/max lon

        self.Grids = []  # list[Grid] of generated coarse grids
        self.Clusters = []  # list[Cluster] of generated fine clusters
        self.Stations = []  # list[Station] of generated stations
        self.order_data = pd.DataFrame()  # DataFrame (N,11): mapped trajectories after build_BSN

    def build_BSN(self):
        """Load geometry/trajectories, build cells, and write all source tables."""
        geojson_file = os.path.join(data_params['data_dir'], data_params['geojson_data'])
        # Validation-gated check; geopandas reports a missing file on its own.
        if RL_params["validation"] and not os.path.exists(geojson_file):
            raise FileNotFoundError(f"####### The GeoJSON file '{geojson_file}' does not exist. #######")
        city_shape = gpd.read_file(geojson_file)

        city_boundary = city_shape.unary_union

        # bounds: (min lat, max lat, min lon, max lon).
        self.bounds = (
            city_boundary.bounds[1],
            city_boundary.bounds[3],
            city_boundary.bounds[0],
            city_boundary.bounds[2],
        )

        station_file = os.path.join(data_params["data_dir"], data_params["NIO_data"])
        station_data = pd.read_csv(station_file)

        allGrids = self.raster_grids(city_boundary, station_data)
        allClusters = self.clustering(city_boundary)

        order_file = os.path.join(data_params["data_dir"], data_params["Driver_data"])
        order_data = pd.read_csv(order_file)
        num_orders = order_data.shape[0]

        self.set_grids(allGrids)
        self.set_clusters(station_data, allClusters)
        self.set_stations(station_data)
        self.set_orders(order_data)
        # Expose the same filtered sampling schema as BSNetwork.load_network.
        types = {c.id: c.type for c in self.Clusters}
        mapped = order_data.copy()
        mapped["cluster"] = mapped["cluster"].astype(int)
        mapped["source_area"] = mapped["area"]
        mapped["area"] = mapped["cluster"].map(types)
        mapped["slot"] = (
            mapped.hour * 60 + mapped.minute
        ) // env_params["delta_time"]
        self.order_data = mapped.loc[
            mapped.cluster.isin(types)
            & mapped.area.isin(("centre", "suburb"))
            & mapped.source_area.isin(("centre", "suburb"))
            & mapped.longitude.notna()
            & mapped.latitude.notna()
        ].copy()
        # Match BSNetwork.load_network without advancing the construction RNG.
        self.order_data["upgrade_days"] = np.random.default_rng(self.seed).integers(
            *data_params["upgrade_days_range"], size=len(self.order_data)
        )

        print(f"The BS network in {self.city} city is built with {len(self.Stations)} stations, "
              f"{len(self.Grids)} grids, {len(self.Clusters)} clusters, and {num_orders} orders.")


    def generate_cells(self, width, height):
        """Return rectangular Polygon cells covering the bounds."""
        min_lat, max_lat, min_lon, max_lon = self.bounds
        lat_steps = np.arange(min_lat, max_lat, height)
        lon_steps = np.arange(min_lon, max_lon, width)

        cells = [box(lon, lat, lon + width, lat + height)   for lat in lat_steps for lon in lon_steps]
        return cells


    def raster_grids(self, city_boundary, station_data):
        """Return dicts for coarse cells that contain stations."""
        width = data_params['grid_width']
        height = data_params['grid_height']
        grid_cells = self.generate_cells(width, height)

        allGrids = []
        gid = 0
        for cell in grid_cells:
            if city_boundary.intersects(cell):
                clipped_cell = city_boundary.intersection(cell)
                if isinstance(clipped_cell, Polygon):
                    # coords: (K,2) longitude/latitude boundary points; centroid: (longitude, latitude).
                    coords = list(clipped_cell.exterior.coords)
                    centroid = (clipped_cell.centroid.x, clipped_cell.centroid.y)

                    points_within_grid = station_data[
                        (station_data['latitude'] >= min([c[1] for c in coords])) &
                        (station_data['latitude'] <= max([c[1] for c in coords])) &
                        (station_data['longitude'] >= min([c[0] for c in coords])) &
                        (station_data['longitude'] <= max([c[0] for c in coords]))
                        ]
                    station_count = len(points_within_grid)

                    if station_count > 0:
                        allGrids.append({
                            "id": gid,
                            "type": 'Unknown',
                            "min_lat": min([c[1] for c in coords]),
                            "max_lat": max([c[1] for c in coords]),
                            "min_long": min([c[0] for c in coords]),
                            "max_long": max([c[0] for c in coords]),
                            "cen_lat": centroid[1],
                            "cen_long": centroid[0],
                            "station_count": station_count,
                            "boundary_points": coords,
                            "centroid": centroid
                        })
                        gid += 1

        # Label by raster position so scan-order changes cannot relabel; Nanjing's
        # seven station cells: row2 suburb/centre/centre, row3 suburb/centre/suburb, row4 ./suburb/.
        min_lat, max_lat, min_long, max_long = self.bounds
        width = data_params['grid_width']
        height = data_params['grid_height']
        grid_types = {
            (2, 0): 'suburb', (2, 1): 'centre', (2, 2): 'centre',
            (3, 0): 'suburb', (3, 1): 'centre', (3, 2): 'suburb',
            (4, 1): 'suburb',
        }
        for grid in allGrids:
            row = int(np.floor((grid["cen_lat"] - min_lat) / height))
            col = int(np.floor((grid["cen_long"] - min_long) / width))
            label = grid_types.get((row, col))
            grid["type"] = label
            grid["boundary_points"] = str(grid["boundary_points"])
            grid["centroid"] = str(grid["centroid"])

        # sizes and diags: float ndarrays (G,), measuring cell width and diagonal distance in kilometres.
        num = len(allGrids)
        sizes = np.zeros(num, dtype=float)
        diags = np.zeros(num, dtype=float)
        for index, cl in enumerate(allGrids):
            lat1 = cl["min_lat"]
            long1 = cl["min_long"]
            lat2 = cl["max_lat"]
            long2 = cl["max_long"]
            sizes[index] = haversine_km(lat1, long1, lat1, long2)
            diags[index] = haversine_km(lat1, long1, lat2, long2)
        max_size = np.max(sizes)
        max_diag = np.max(diags)

        print(f'Initially {num} valid grids are generated with size {max_size:.2f} km '
              f'and diagonal {max_diag:.2f} km.')
        return allGrids


    def search_grid(self, long, lat):
        """Return (grid, True) when a grid contains the location, else (None, False)."""
        grid, flag = None, False
        for grid in self.Grids:
            min_lat = grid.bound[0]
            max_lat = grid.bound[1]
            min_long = grid.bound[2]
            max_long = grid.bound[3]
            slack = float(env_params["spatial_tolerance"])
            if (min_lat - slack <= lat <= max_lat + slack
                    and min_long - slack <= long <= max_long + slack):
                flag = True
                break
        return (grid, True) if flag else (None, False)

    def search_cluster(self, long, lat, station_boundary=False):
        """Return (cluster, found); stations fall back to the nearest boundary."""
        if not np.isfinite([long, lat]).all():
            return None, False
        for cls in sorted(self.Clusters, key=lambda c: c.id):
            if cls.contains_location(long, lat):
                return cls, True

        # Stations keep their location; demand rows still require containment.
        if station_boundary and self.Clusters:
            cls = min(self.Clusters, key=lambda c: (c.distance_to_bounds(long, lat), c.id))
            return cls, True
        return None, False

    def search_station(self, station_data, center):
        """Return km from a location to its nearest station."""
        num = station_data.shape[0]
        dists = np.zeros(num, dtype=float)

        for index, st in enumerate(station_data.itertuples()):
            longitude = st.longitude
            latitude = st.latitude
            dists[index] = haversine_km(latitude, longitude, center[0], center[1])

        return  np.min(dists)

    def overlap_search(self, center, bound):
        """Return (grid, True) for the nearest coarse grid overlapping the cell."""
        cen_lat = center[0]
        cen_long = center[1]
        grid, flag = self.search_grid(cen_long, cen_lat)
        if flag:
            return grid, flag

        X_BL = bound[2]
        Y_BL = bound[0]
        X_UR = bound[3]
        Y_UR = bound[1]

        grid = None
        flag = False
        dists = []
        ID = []
        for grid in self.Grids:
            lat = grid.center[0]
            long = grid.center[1]

            x_BL = grid.bound[2]
            y_BL = grid.bound[0]
            x_UR = grid.bound[3]
            y_UR = grid.bound[1]

            x_overlap = (x_BL < X_UR) and (X_BL < x_UR)
            y_overlap = (y_BL < Y_UR) and (Y_BL < y_UR)
            dist = haversine_km(cen_lat, cen_long, lat, long)
            if x_overlap and y_overlap and dist <= env_params["grid_search_radius"]:
                dists.append(dist)
                ID.append(grid.id)

        # Nearest qualifying grid.
        if dists:
            dists = np.array(dists)
            k = dists.argmin()
            gid = ID[k]

            grid = self.Grids[gid]
            flag = True
        return  grid, flag


    def clustering(self, city_boundary):
        """Return dicts for fine cells intersecting the city boundary."""
        width = data_params['cluster_width']
        height = data_params['cluster_height']
        grid_cells = self.generate_cells(width, height)

        allClusters = []
        cid = 0
        for cell in grid_cells:
            if city_boundary.intersects(cell):
                clipped_cell = city_boundary.intersection(cell)
                if isinstance(clipped_cell, Polygon):
                    coords = list(clipped_cell.exterior.coords)
                    centroid = (clipped_cell.centroid.x, clipped_cell.centroid.y)

                    allClusters.append({
                        "id": cid,
                        "min_lat": min([c[1] for c in coords]),
                        "max_lat": max([c[1] for c in coords]),
                        "min_long": min([c[0] for c in coords]),
                        "max_long": max([c[0] for c in coords]),
                        "cen_lat": centroid[1],
                        "cen_long": centroid[0],
                        "boundary_points": coords,
                        "centroid": centroid
                    })
                    cid += 1

        # Serialize coordinate fields.
        for cls in allClusters:
            cls["boundary_points"] = str(cls["boundary_points"])
            cls["centroid"] = str(cls["centroid"])

        # Cell width/diagonal statistics, km.
        num = len(allClusters)
        sizes = np.zeros(num, dtype=float)
        diags = np.zeros(num, dtype=float)
        for index, cl in enumerate(allClusters):
            lat1 = cl["min_lat"]
            long1 = cl["min_long"]
            lat2 = cl["max_lat"]
            long2 = cl["max_long"]
            sizes[index] = haversine_km(lat1, long1, lat1, long2)
            diags[index] = haversine_km(lat1, long1, lat2, long2)
        max_size = np.max(sizes)
        max_diag = np.max(diags)

        print(f'Initially {num} valid clusters are generated with size {max_size:.2f} km '
              f'and diagonal {max_diag:.2f} km.')
        return  allClusters

    def set_grids(self, allGrids):
        """Keep the configured coarse cells and save grid_data CSV."""
        select = data_params["selected_grids"]
        self.Grids = []
        gid = 0
        for gr in allGrids:
            if gr["id"] in select:
                gtype = gr['type']
                min_lat = gr['min_lat']
                max_lat = gr['max_lat']
                min_long = gr['min_long']
                max_long = gr['max_long']
                cen_lat = gr['cen_lat']
                cen_long = gr['cen_long']
                station_count = gr['station_count']
                boundary = gr['boundary_points']
                centroid = gr["centroid"]

                bound = (min_lat, max_lat, min_long, max_long)
                center = (cen_lat, cen_long)
                if gtype == 'centre':
                    rate = (data_params['rate_peak_centre'],
                            data_params['rate_off_centre'])
                else:
                    rate = (data_params['rate_peak_suburb'],
                            data_params['rate_off_suburb'])

                self.Grids.append(
                    grids.Grid(gid, gtype, station_count, rate, center, bound, centroid, boundary)
                )
                gid += 1

        grid_data = [vars(grid) for grid in self.Grids]
        df = pd.DataFrame(grid_data)
        csv_name = data_params['grid_data']
        output_file = os.path.join(data_params['data_dir'], csv_name)
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f'Grids data has been saved to {csv_name}.')

    def set_clusters(self, station_data, allClusters):
        """Keep clusters near a station + overlapping grid; save cluster_data CSV."""
        centres = data_params["centre_clusters"]
        self.Clusters = []
        cid = 0
        for cls in allClusters:
            min_lat = cls['min_lat']
            max_lat = cls['max_lat']
            min_long = cls['min_long']
            max_long = cls['max_long']
            cen_lat = cls['cen_lat']
            cen_long = cls['cen_long']
            boundary = cls['boundary_points']
            centroid = cls["centroid"]

            center = (cen_lat, cen_long)
            bound = (min_lat, max_lat, min_long, max_long)
            grid, flag = self.overlap_search(center, bound)
            dist = self.search_station(station_data, center)

            if flag and (dist <= env_params['max_distance']):
                if cid in centres:
                    ctype = 'centre'
                else:
                    ctype = 'suburb'

                self.Clusters.append(
                    clusters.Cluster(cid, ctype, center, bound, centroid, boundary)
                )
                cid += 1

        cluster_data = [vars(cluster) for cluster in self.Clusters]
        df = pd.DataFrame(cluster_data)
        csv_name = data_params['cluster_data']
        output_file = os.path.join(data_params['data_dir'], csv_name)
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f'Clusters data has been saved to {csv_name}.')


    def set_stations(self, station_data):
        """Map source rows into grids/clusters and save station_data CSV."""
        self.Stations = []
        sid = 0
        for index, st in station_data.iterrows():
            longitude = st['longitude']
            latitude = st['latitude']
            grid, flag = self.search_grid(longitude, latitude)
            cls, cluster_found = self.search_cluster(longitude, latitude, station_boundary=True)

            # Size, stock (2,), swap time, and grid power.
            if flag and cluster_found:
                cid = cls.id
                swap_time = env_params['swap_time']
                if grid.type == 'centre' and self.rng.random() <= data_params['station_large']:
                    size = 'large'
                else:
                    size = 'small'
                battery_cap = env_params[f'{size}_size']
                initial_batteries = env_params[f'initial_batteries_{size}']
                grid_power = env_params[f'grid_power_{size}']

                self.Stations.append(
                    stations.Station(
                        sid, cid, longitude, latitude, swap_time, battery_cap,
                        initial_batteries, grid_power
                    )
                )
                sid += 1
            elif flag:
                print(
                    f"Station at ({latitude}, {longitude}) falls outside every cluster "
                    "and is skipped."
                )

        station_data = [vars(station) for station in self.Stations]
        df = pd.DataFrame(station_data)
        csv_name = data_params['station_data']
        output_file = os.path.join(data_params['data_dir'], csv_name)
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f'Stations data has been saved to {csv_name}.')

    def assign_area(self, order_data):
        """Return (area labels, cluster IDs) per row; area follows the fine cluster type."""
        areas, clusters = [], []

        for index, req in order_data.iterrows():
            try:
                longitude = float(req['longitude'])
                latitude = float(req['latitude'])

                cls, flc = self.search_cluster(longitude, latitude)
                if flc:
                    area, cid = cls.type, cls.id
                else:
                    area, cid = 'NA', -1

            except (ValueError, KeyError, TypeError) as e:
                print(f"######## Error in processing row {index}: {e} ########")
                area, cid = 'NA', -1
            areas.append(area)
            clusters.append(cid)

        return  areas, clusters

    def set_orders(self, order_data):
        """Add cluster-based area and cluster columns and save order_data CSV."""
        areas, clusters = self.assign_area(order_data)
        order_data['area'] = areas
        order_data['cluster'] = clusters

        csv_name = data_params['order_data']
        output_file = os.path.join(data_params['data_dir'], csv_name)
        order_data.to_csv(output_file, index=False, encoding="utf-8")
        print(f'Orders data has been saved to {output_file}.')


    def extend_network(self, selected, num, seed_n):
        """Add stations to suburban clusters and save the expansion table."""
        num = int(num)

        suburbs = np.array([cls.id  for cls in self.Clusters  if cls.type == 'suburb'])
        clusters = {cluster.id: cluster for cluster in self.Clusters}
        print(f'Suburb clusters: {suburbs}.')

        # Dedicated expansion stream: network RNG, or a fresh one seeded by seed_n.
        rng = self.rng if seed_n is None else np.random.default_rng(int(seed_n))

        if len(selected) == 0:
            selects = rng.choice(suburbs, size=num, replace=False)
        else:
            selects = list(selected)

        selects.sort()
        print(f'The selected suburb clusters: {selects}.')

        # Add a station at each selected cluster center using the configured station-size distributions.
        sid = len(self.Stations) - 1
        for cid in selects:
            cluster = clusters[cid]
            latitude, longitude = cluster.center[0], cluster.center[1]
            print(f'New station in cluster {cid}: latitude {latitude:.4f}, longitude {longitude:.4f}')

            swap_time = env_params['swap_time']
            # Expansion stations use their own large-station probability.
            size = 'large' if rng.random() <= data_params['expansion_station_large'] else 'small'
            battery_cap = env_params[f'{size}_size']
            initial_batteries = env_params[f'initial_batteries_{size}']
            grid_power = env_params[f'grid_power_{size}']

            sid += 1
            self.Stations.append(
                stations.Station(
                    sid, cid, longitude, latitude, swap_time, battery_cap,
                    initial_batteries, grid_power
                )
            )

        # Save one specification row per station to the scenario output table.
        station_data = [vars(station) for station in self.Stations]
        df = pd.DataFrame(station_data)
        csv_name = expansion_station_file(num)
        output_file = os.path.join(data_params['data_dir'], csv_name)
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f'Test data with {len(self.Stations)} stations has been saved to {csv_name}.')
