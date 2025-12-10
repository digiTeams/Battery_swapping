"""
Construct the city map with rasterized grids for the case study
Show the NIO BS network in the grid city map
Show an example of request locations in the grid city map

"""
import os
import folium
import pandas as pd
from datetime import datetime
import numpy as np
from shapely.geometry import Polygon, box
import geopandas as gpd
from geopy.distance import geodesic

from config import data_params, env_params
from network import grids, stations, clusters


class BSNetwork:
    def __init__(self, city, work_dir, seed):
        self.city = city
        self.work_dir = work_dir
        self.seed = seed
        self.bounds = (0, 0, 0, 0)

        self.Stations = []   #Stations: id, longitude, latitude, district, city
        self.Grids = []   #id, type, bound, centroid, station_count, rate
        self.Clusters = []   #id, bound, centroid, boundary, demand, envy, subsidy


    ### Build the BS network with grids
    def build_network(self):
        #Load the geographic shape of the city
        geojson_file = os.path.join(self.work_dir, data_params['geojson_data'])
        if not os.path.exists(geojson_file):
            raise FileNotFoundError(f"The GeoJSON file '{geojson_file}' does not exist.")
        city_shape = gpd.read_file(geojson_file)

        #Get city boundary
        city_boundary = city_shape.unary_union  # Combine all geometries into one shape
        self.bounds = (
            city_boundary.bounds[1],  # min_lat
            city_boundary.bounds[3],  # max_lat
            city_boundary.bounds[0],  # min_lon
            city_boundary.bounds[2],  # max_lon
        )

        # Griding and clustering:
        station_file = os.path.join(self.work_dir, 'Nanjing_NIO_2025.csv')
        station_data = pd.read_csv(station_file)

        allGrids = self.raster_grids(city_boundary, station_data)
        allClusters = self.clustering(city_boundary)

        # Order data
        order_file = os.path.join(self.work_dir, 'Nanjing_drivers_201910.csv')
        order_data = pd.read_csv(order_file)
        num_orders = order_data.shape[0]

        ##Setup stations, grids, and clusters in the network
        #Grids: id  type  station_count  rate  center  bound  centroid  boundary_points
        self.set_grids(allGrids)
        #Clusters: id  type  center  bound  centroid  boundary_points  demand  envy  subsidy  Envies  Subsidies
        self.set_clusters(station_data, allClusters)
        # Stations: id  cluster  longitude  latitude  battery_cap  chargers  charge_time  swap_time
        self.set_stations(station_data)
        # Requests: id  status  longitude  latitude  addcode  district  city  time  date  hour  minute
        self.set_orders(order_data)

        # End
        print(f"The BS network in {self.city} city is built with {len(self.Stations)} stations, "
              f"{len(self.Grids)} grids, {len(self.Clusters)} clusters, and {num_orders} orders.")


    ###Generate grid cells based on bounding box and grid size.
    def generate_cells(self, width, height):
        min_lat, max_lat, min_lon, max_lon = self.bounds
        lat_steps = np.arange(min_lat, max_lat, height)
        lon_steps = np.arange(min_lon, max_lon, width)

        cells = [box(lon, lat, lon + width, lat + height)   for lat in lat_steps for lon in lon_steps]
        num_cells = len(cells)
        #print(f'Initially {num_cells} grid cells are generated.')

        return cells


    ### Generate large grids in the city area
    def raster_grids(self, city_boundary, station_data):
        width = data_params['grid_width']
        height = data_params['grid_height']
        grid_cells = self.generate_cells(width, height)

        allGrids = []
        gid = 0
        for cell in grid_cells:
            if city_boundary.intersects(cell):
                clipped_cell = city_boundary.intersection(cell)
                if isinstance(clipped_cell, Polygon):
                    coords = list(clipped_cell.exterior.coords)
                    centroid = (clipped_cell.centroid.x, clipped_cell.centroid.y) #(119.05, 31.67)

                    # Count the number of station points within this grid cell
                    # ID,  longitude,  latitude,  district,	 city
                    points_within_grid = station_data[
                        (station_data['latitude'] >= min([c[1] for c in coords])) &
                        (station_data['latitude'] <= max([c[1] for c in coords])) &
                        (station_data['longitude'] >= min([c[0] for c in coords])) &
                        (station_data['longitude'] <= max([c[0] for c in coords]))
                        ]
                    station_count = len(points_within_grid)

                    if station_count > 0: ##only show the grids with non-zero stations
                        #self.grid_cells[gid] = clipped_cell
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

        #Set type:  0         1         2         3         4         5         6
        types = ['suburb', 'centre', 'centre', 'suburb', 'centre', 'suburb', 'suburb']
        for grid in allGrids:
            gid = grid["id"]
            grid["type"] = types[gid]
            grid["boundary_points"] = str(grid["boundary_points"])
            grid["centroid"] = str(grid["centroid"])

        # Save the grid data
        output_grid = f'{self.city}_grids_202501.csv'  # Nanjing_grids_2025.csv
        output_grid_file = os.path.join(self.work_dir, output_grid)
        df = pd.DataFrame(allGrids)
        df.to_csv(output_grid_file, index=False, encoding="utf-8")

        ''' 
        with open(output_grid_file, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["id", "type", "min_lat", "max_lat", "min_long", "max_long",
                                                         "cen_lat", "cen_long", "station_count", "boundary_points",
                                                         "centroid"])
            writer.writeheader()
            for grid in allGrids:
                writer.writerow(grid)
        '''
        num = len(allGrids)
        sizes = np.zeros(num, dtype=float)
        diags = np.zeros(num, dtype=float)
        for index, cl in enumerate(allGrids):
            lat1 = cl["min_lat"]
            long1 = cl["min_long"]
            lat2 = cl["max_lat"]
            long2 = cl["max_long"]

            sizes[index] = geodesic((lat1, long1), (lat1, long2)).km
            diags[index] = geodesic((lat1, long1), (lat2, long2)).km

        max_size = np.max(sizes)
        max_diag = np.max(diags)
        print(f'Initially {num} valid grids are generated with size {max_size:.2f} km '
              f'and diagonal {max_diag:.2f} km.')

        return allGrids


    ###search the grid for the station
    def search_grid(self, long, lat):
        # grid.bound = (min_lat, max_lat, min_long, max_long)
        grid, flag = None, False
        for grid in self.Grids:
            min_lat = grid.bound[0]
            max_lat = grid.bound[1]
            min_long = grid.bound[2]
            max_long = grid.bound[3]
            if (lat >= min_lat) & (lat <= max_lat) & (long >= min_long) & (long <= max_long):
                flag = True
                break
        # End
        return  grid, flag


    ###search the cluster for the station
    def search_cluster(self, long, lat):
        # grid.bound = (min_lat, max_lat, min_long, max_long)
        cls, flag = None, False
        for cls in self.Clusters:
            min_lat = cls.bound[0]
            max_lat = cls.bound[1]
            min_long = cls.bound[2]
            max_long = cls.bound[3]
            if (lat >= min_lat) & (lat <= max_lat) & (long >= min_long) & (long <= max_long):
                flag = True
                break
        # End
        return  cls, flag


    ###Set the network of stations for the system
    def search_station(self, station_data, center):
        # center = (cen_lat, cen_long)
        num = station_data.shape[0]
        dists = np.zeros(num, dtype=float)

        for index, st in station_data.iterrows():
            longitude = st['longitude']
            latitude = st['latitude']
            dists[index] = geodesic((latitude, longitude), center).km

        # Return the minimum distance
        return  np.min(dists)


    ##Check intersection of two grids
    def overlap_search(self, center, bound):
        # center = (cen_lat, cen_long)
        cen_lat = center[0]
        cen_long = center[1]
        grid, flag = self.search_grid(cen_long, cen_lat)
        if flag:
            return grid, flag

        # bound = (min_lat, max_lat, min_long, max_long)
        X_BL = bound[2]  # min_long
        Y_BL = bound[0]  # min_lat
        X_UR = bound[3]  # max_long
        Y_UR = bound[1]  # max_lat

        grid = None
        flag = False
        dists = []
        ID = []
        for grid in self.Grids:
            lat = grid.center[0]
            long = grid.center[1]

            x_BL = grid.bound[2]  # min_long
            y_BL = grid.bound[0]  # min_lat
            x_UR = grid.bound[3]  # max_long
            y_UR = grid.bound[1]  # max_lat

            #Check overlap
            x_overlap = (x_BL < X_UR) and (X_BL < x_UR)
            y_overlap = (y_BL < Y_UR) and (Y_BL < y_UR)
            dist = geodesic((cen_lat, cen_long), (lat, long)).km
            if x_overlap and y_overlap and dist <= 12:
                dists.append(dist)
                ID.append(grid.id)

        if dists:
            dists = np.array(dists)
            k = dists.argmin()
            gid = ID[k]

            grid = self.Grids[gid]
            flag = True
        # End
        return  grid, flag


    ###Cluster the city area into small grids
    def clustering(self, city_boundary):
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

        for cls in allClusters:
            cls["boundary_points"] = str(cls["boundary_points"])
            cls["centroid"] = str(cls["centroid"])

        # Save the cluster data
        output_cluster = f'{self.city}_clusters_202501.csv'    #Nanjing_clusters_202501.csv
        output_cluster_file = os.path.join(self.work_dir, output_cluster)
        df = pd.DataFrame(allClusters)
        df.to_csv(output_cluster_file, index=False, encoding="utf-8")
        '''
        with open(output_cluster_file, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["id", "min_lat", "max_lat", "min_long", "max_long",
                                                         "cen_lat", "cen_long", "boundary_points", "centroid"])
            writer.writeheader()
            for cls in allClusters:
                writer.writerow(cls)
        '''
        # End
        num = len(allClusters)
        sizes = np.zeros(num, dtype=float)
        diags = np.zeros(num, dtype=float)
        for index, cl in enumerate(allClusters):
            lat1 = cl["min_lat"]
            long1 = cl["min_long"]
            lat2 = cl["max_lat"]
            long2 = cl["max_long"]

            sizes[index] = geodesic((lat1, long1), (lat1, long2)).km
            diags[index] = geodesic((lat1, long1), (lat2, long2)).km

        max_size = np.max(sizes)
        max_diag = np.max(diags)
        print(f'Initially {num} valid clusters are generated with size {max_size:.2f} km '
              f'and diagonal {max_diag:.2f} km.')
        # End
        return  allClusters


    ### Define the grids of city network
    def set_grids(self, allGrids):
        #Grids:["id", "type", "min_lat", "max_lat", "min_long", "max_long",
        #"cen_lat", "cen_long", "station_count", "boundary_points", "centroid"]
        # Set type:   0         1         2         3         4         5         6
        # types = ['suburb', 'centre', 'centre', 'suburb', 'centre', 'suburb', 'suburb']
        select = [0, 1, 2, 3, 4, 5, 6]
        self.Grids = [] #gid, gtype, station_count, rate, bound, centroid, boundary
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

                grid = grids.Grid(gid, gtype, station_count, rate, center, bound, centroid, boundary)
                self.Grids.append(grid)
                gid += 1

        # Save data
        grid_data = [vars(grid)  for grid in self.Grids]
        df = pd.DataFrame(grid_data)
        # 'grid_data': 'Nanjing_grids_2025.csv',
        csv_name = data_params['grid_data']
        output_file = os.path.join(self.work_dir, csv_name)
        df.to_csv(output_file, index=False, encoding="utf-8")

        ##End
        print(f'Grids data has been saved to {output_file}.')


    ### Define the grids of city network
    def set_clusters(self, station_data, allClusters):
        # "id", "min_lat", "max_lat", "min_long", "max_long", "cen_lat", "cen_long",
        # "boundary_points", "centroid"
        self.Clusters = []
        centres = [16, 21, 22, 23, 28, 29, 30, 35, 36] # 'suburb', 'centre'
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

                cluster = clusters.Cluster(cid, ctype, center, bound, centroid, boundary)
                self.Clusters.append(cluster)
                cid += 1

        # Save data
        cluster_data = [vars(cluster)  for cluster in self.Clusters]
        df = pd.DataFrame(cluster_data)
        # 'cluster_data': 'Nanjing_clusters_2025.csv',
        csv_name = data_params['cluster_data']
        output_file = os.path.join(self.work_dir, csv_name)
        df.to_csv(output_file, index=False, encoding="utf-8")

        ##End
        print(f'Clusters data has been saved to {output_file}.')


    ###Set the network of stations for the system
    def set_stations(self, station_data):
        # ID,  longitude,  latitude,  district,  city
        self.Stations = []
        sid = 0
        for index, st in station_data.iterrows():
            longitude = st['longitude']
            latitude = st['latitude']
            grid, flag = self.search_grid(longitude, latitude)
            cls, _ = self.search_cluster(longitude, latitude)

            if flag:
                cid = cls.id
                charge_time = env_params['charge_time_fast']  # unit: minute
                swap_time = env_params['swap_time_II']  # unit: minute
                if (grid.type == 'centre') and (np.random.uniform(0, 1, 1) <= 0.5):
                    battery_cap = env_params['battery_cap_II']
                    chargers = env_params['chargers_II']
                else:
                    battery_cap = env_params['battery_cap_I']
                    chargers = env_params['chargers_I']

                station = stations.Station(sid, cid, longitude, latitude, battery_cap, chargers,
                                           charge_time, swap_time)
                self.Stations.append(station)
                sid += 1

        # Save data
        station_data = [vars(station) for station in self.Stations]
        df = pd.DataFrame(station_data)
        # 'station_data': 'Nanjing_stations_2025.csv',
        csv_name = data_params['station_data']
        output_file = os.path.join(self.work_dir, csv_name)
        df.to_csv(output_file, index=False, encoding="utf-8")

        # End
        print(f'Stations data has been saved to {output_file}.')


    # 优化区域分配过程
    def assign_area(self, order_data):
        # 'order_data': 'Nanjing_drivers_201910.csv'
        # id  status  longitude  latitude  addcode  district  city  time  date  hour  minute
        """为订单数据分配区域信息"""
        areas, clusters = [], []
        grid_cache = {}  # 添加缓存以提高性能

        for index, req in order_data.iterrows():
            try:
                longitude = float(req['longitude'])
                latitude = float(req['latitude'])

                # Get the grid area
                coord_key = f"{longitude:.4f}_{latitude:.4f}"
                if coord_key in grid_cache:
                    area = grid_cache[coord_key]
                else:
                    grid, flag = self.search_grid(longitude, latitude)
                    area = grid.type  if flag  else 'NA'
                    grid_cache[coord_key] = area  # 添加到缓存
                # Add the data: ["centre", "suburb", "NA"]:
                areas.append(area)

                # Get the cluster id
                if area != 'NA':
                    cls, flc = self.search_cluster(longitude, latitude)
                    cid = cls.id   if flc  else -1
                else:
                    cid = -1
                clusters.append(cid)

            except (ValueError, KeyError) as e:
                print(f"######## Error in processing row {index}: {e} ########")
                areas.append('NA')
                clusters.append(-1)

        return  areas, clusters


    def set_orders(self, order_data):
        #'order_data': 'Nanjing_drivers_201910.csv'
        #id  status  longitude  latitude  addcode  district  city  time  date  hour  minute
        areas, clusters = self.assign_area(order_data)

        # Add the columns into order data
        order_data['area'] = areas
        order_data['cluster'] = clusters

        # Output the data
        #'order_data': 'Nanjing_drivers_2025.csv',
        csv_name = data_params['order_data']
        output_file = os.path.join(self.work_dir, csv_name)
        order_data.to_csv(output_file, index=False, encoding="utf-8")

        # End
        print(f'Orders data has been saved to {output_file}.')


    ####Randomly add new stations in the network for suburb areas.
    def expand_network(self, selected, num, seed_n):
        #CLass: num=4, num=8, num=16
        # Identify suburb grids
        suburbs = np.array([cls.id  for cls in self.Clusters  if cls.type == 'suburb'])
        print(f'Suburb grid clusters: {suburbs}. ')

        # Randomly sample suburb clusters to add new stations
        if seed_n is not None:
            np.random.seed(seed_n)

        if num > len(suburbs):
            raise ValueError(f"###### Sample size {num} is greater than the candidate list length "
                             f"{len(suburbs)} ######")
        if len(selected) == 0:
            selects = np.random.choice(suburbs, size=num, replace=False)
        else:
            selects = selected

        selects.sort()
        print(f'The selected suburb grids {selects}.')

        # cluster: (cid, ctype, center, bound, centroid, boundary)
        # Station: id, cluster, longitude, latitude, battery_cap, chargers, charge_time, swap_time
        sid = len(self.Stations) - 1
        for cid in selects:
            cluster = self.Clusters[cid]
            ''' 
            #bound = (min_lat, max_lat, min_long, max_long)
            min_lat, max_lat = cluster.bound[0], cluster.bound[1]
            min_long, max_long = cluster.bound[2], cluster.bound[3]
            lat = np.random.uniform(min_lat, max_lat, 1) # cluster.center[0]
            long = np.random.uniform(min_long, max_long,1 ) # cluster.center[1]
            latitude, longitude = lat[0], long[0]
            '''
            # center = (cen_lat, cen_long)
            latitude, longitude = cluster.center[0], cluster.center[1]
            print(f'New station in grid {cid}: latitude {latitude:.4f}, longitude {longitude:.4f}')

            charge_time = env_params['charge_time_fast']  # unit: minute
            swap_time = env_params['swap_time_II']  # unit: minute
            if (np.random.uniform(0, 1, 1) <= 0.25):
                battery_cap = env_params['battery_cap_II']
                chargers = env_params['chargers_II']
            else:
                battery_cap = env_params['battery_cap_I']
                chargers = env_params['chargers_I']

            sid += 1
            station = stations.Station(sid, cid, longitude, latitude, battery_cap, chargers,
                                       charge_time, swap_time)
            self.Stations.append(station)

        # Save data
        station_data = [vars(station)  for station in self.Stations]
        df = pd.DataFrame(station_data)
        # 'station_data': 'Nanjing_stations_2025.csv',
        csv_name = f'Nanjing_stations_test_{num}.csv'
        output_file = os.path.join(self.work_dir, csv_name)
        df.to_csv(output_file, index=False, encoding="utf-8")
        ##End
        print(f'Test data with {len(self.Stations)} stations has been saved to {output_file}.')
