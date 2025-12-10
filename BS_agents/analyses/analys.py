
import os
import folium
import pandas as pd
from datetime import datetime
import numpy as np
import geopandas as gpd
import csv
import re
import ast

from config import data_params


class Analysist:
    def __init__(self, city, work_dir, network):
        self.city = city
        self.work_dir = work_dir
        self.network = network


    def plot_maps(self, request_csv, station_csv, grid_csv, cluster_csv, flag):
        match flag:
            case 'Griding':
                self.grid_map(request_csv, station_csv, grid_csv)
            case 'Clustering':
                self.cluster_map(request_csv, station_csv, cluster_csv)
            case 'Combine':
                self.combined_map(request_csv, station_csv, grid_csv, cluster_csv)


    ####Generate a folium map with city shape overlay, grid, and markers.
    def grid_map(self, request_csv, station_csv, grid_csv):
        # ID,  longitude,  latitude,  district,	city
        station_file = os.path.join(self.work_dir, station_csv)
        station_data = pd.read_csv(station_file)

        grid_file = os.path.join(self.work_dir, grid_csv)
        grid_data = pd.read_csv(grid_file)

        # id, longitude, latitude, submission, SOC, gridID
        request_file = os.path.join(self.work_dir, request_csv)
        order_data = pd.read_csv(request_file)

        # Load the geographic shape of the city
        geojson_file = os.path.join(self.work_dir, data_params['geojson_data'])
        if not os.path.exists(geojson_file):
            raise FileNotFoundError(f"The GeoJSON file '{geojson_file}' does not exist.")
        city_shape = gpd.read_file(geojson_file)
        city_center = [city_shape.geometry.centroid.y.mean(), city_shape.geometry.centroid.x.mean()]

        city_map = folium.Map(location=city_center, zoom_start=10)
        folium.GeoJson(city_shape, name="City Boundary",
                       style_function=lambda x: {
                           "color": "blue",  # Boundary line color
                           "weight": 1,  # Line weight
                           "fillColor": "#ADD8E6",  # Light blue fill color
                           "fillOpacity": 0.25  # Adjust opacity for better visibility
                       }
                       ).add_to(city_map)

        # Add grids into the city map
        for index, cell in grid_data.iterrows():
            gid = cell['id']
            # Add the grid polygon to the map
            pattern = r"\((\d+\.\d+),\s*(\d+\.\d+)\)"  # 匹配形如 (x, y) 的数值对
            locs = re.findall(pattern, cell['boundary_points'])
            folium.Polygon(
                # [(118.761383, 31.277681), (118.76006831134019, 31.278097000000002)]
                locations=[[float(y), float(x)] for x, y in locs],
                color="red",
                weight=1.0,
                fill=False,
                fill_opacity=0.25
            ).add_to(city_map)

            # Calculate the center of the grid cell to place the circled marker
            centroid = ast.literal_eval(cell['centroid'])
            folium.CircleMarker(
                location=[centroid[1], centroid[0]],
                radius=10,  # Circle size
                color="orange",
                weight=0.5,
                fill=True,
                fill_color="orange",
                fill_opacity=0.75,
                popup=f"Grid ID: {gid}",  # Popup with the grid ID
            ).add_to(city_map)

            # Add the grid ID as text inside the circle
            folium.map.Marker(
                location=[centroid[1], centroid[0]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 14px; color: black; '
                         f'text-align: center;">{gid}</div>')
            ).add_to(city_map)

        # Add stations into the city map
        icon_file = os.path.join(self.work_dir, data_params['station_icon'])
        # ID,  longitude,  latitude,  district,	city
        for index, st in station_data.iterrows():
            sid = st['id']
            icon = folium.CustomIcon(icon_file, icon_size=(14, 14))
            folium.Marker(
                location=(st['latitude'], st['longitude']),
                icon=icon,
                popup=f"ID: {sid}, Type: {'Station'}"
            ).add_to(city_map)

        # Add requests into city map
        # id	longitude	latitude	submission	SOC	gridID	clusterID
        count = 0
        n = min([order_data.shape[0], 3000])
        for index, req in order_data.iterrows():
            rid = req['id']
            count += 1
            if count <= n:
                folium.CircleMarker(
                    location=(req['latitude'], req['longitude']),
                    radius=2,
                    color="blue",
                    weight=0.25,
                    fill=True,
                    fill_color="blue",
                    fill_opacity=0.5,
                    popup=f"ID: {rid}, Type: {'Request'}"
                ).add_to(city_map)

        # Add a legend directly in the map
        legend_html = '''
                    <div style="position: fixed; 
                                top: 10px; right: 10px; width: 120px; height: auto; 
                                background-color: white; z-index:9999; 
                                font-size:12px; border:1px solid grey; padding: 10px; 
                                border-radius: 1px;">
                        <b>Type of points</b><br>
                        <i style="background:blue; width:10px; height:10px; display:inline-block; margin-right:10px;"></i> Request<br>
                        <i style="background:red; width:10px; height:10px; display:inline-block; margin-right:10px;"></i> Station<br>
                    </div>
                    '''
        city_map.get_root().html.add_child(folium.Element(legend_html))

        x = re.findall(r'\d+', grid_csv)  # 匹配所有连续数字
        csv_date = x[0]
        output_file = f'{self.city}_grid_map_{csv_date}.html'
        output_file_path = os.path.join(self.work_dir, output_file)
        city_map.save(output_file_path)

        # End
        print(f'City map with grids has been saved to {output_file_path}.')


    ####Generate a folium map with city shape overlay, grid, and markers.
    def cluster_map(self, request_csv, station_csv, cluster_csv):
        # ID,  longitude,  latitude,  district,	city
        station_file = os.path.join(self.work_dir, station_csv)
        station_data = pd.read_csv(station_file)

        cluster_file = os.path.join(self.work_dir, cluster_csv)
        cluster_data = pd.read_csv(cluster_file)

        # id, longitude, latitude, submission, SOC, gridID
        request_file = os.path.join(self.work_dir, request_csv)
        order_data = pd.read_csv(request_file)

        # Load the city map data
        geojson_file = os.path.join(self.work_dir, data_params['geojson_data'])
        if not os.path.exists(geojson_file):
            raise FileNotFoundError(f"The GeoJSON file '{geojson_file}' does not exist.")
        city_shape = gpd.read_file(geojson_file)
        city_center = [city_shape.geometry.centroid.y.mean(), city_shape.geometry.centroid.x.mean()]

        # Get the city map layer
        city_map = folium.Map(location=city_center, zoom_start=10)
        folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(city_map)
        ''' 
        # 添加 CartoDB 底图
        folium.TileLayer(
            tiles='CartoDB positron',
            name='Light Map',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ).add_to(city_map)
        '''
        # 添加 Stamen Terrain 图层的调用方式
        folium.TileLayer(
            tiles='Stamen.Terrain',
            name='Terrain',
            attr='Map tiles by <a href="https://stamen.com">Stamen Design</a>, under CC BY 3.0. Data by <a href="https://openstreetmap.org">OpenStreetMap</a>, under ODbL.'
        ).add_to(city_map)

        folium.GeoJson(city_shape, name="City Boundary",
                       style_function=lambda x: {
                           "color": "blue",  # Boundary line color
                           "weight": 1,  # Line weight
                           "fillColor": "#ADD8E6",  # Light blue fill color
                           "fillOpacity": 0.25  # Adjust opacity for better visibility
                       }
                       ).add_to(city_map)

        # Add clusters into the city map
        for index, cell in cluster_data.iterrows():
            cid = cell['id']
            # Add the grid polygon to the map
            pattern = r"\((\d+\.\d+),\s*(\d+\.\d+)\)"  # 匹配形如 (x, y) 的数值对
            locs = re.findall(pattern, cell['boundary_points'])
            folium.Polygon(
                #[(118.761383, 31.277681), (118.76006831134019, 31.278097000000002)]
                locations=[[float(y), float(x)] for x, y in locs],
                color="red",
                weight=1.0,
                fill=False,
                fill_opacity=0.25
            ).add_to(city_map)

            # Calculate the center of the grid cell to place the circled marker
            centroid = ast.literal_eval(cell['centroid'])
            folium.CircleMarker(
                location=[centroid[1], centroid[0]],
                radius=10,  # Circle size
                color="orange",
                weight=1.0,
                fill=True,
                fill_color="orange",
                fill_opacity=0.5,
                popup=f"Cluster ID: {cid}",  # Popup with the grid ID
            ).add_to(city_map)

            # Add the grid ID as text inside the circle
            folium.map.Marker(
                location=[centroid[1], centroid[0]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 14px; color: black; '
                         f'text-align: center;">{cid}</div>')
            ).add_to(city_map)


        # Add stations into the city map
        icon_file = os.path.join(self.work_dir, data_params['station_icon'])
        # ID,  longitude,  latitude,  district,	city
        for index, st in station_data.iterrows():
            sid = st['id']
            icon = folium.CustomIcon(icon_file, icon_size=(14, 14))
            folium.Marker(
                location=(st['latitude'], st['longitude']),
                icon=icon,
                popup=f"ID: {sid}, Type: {'Station'}"
            ).add_to(city_map)

        # Add requests into city map
        # id	longitude	latitude	submission	SOC	gridID	clusterID
        count = 0
        n = min([order_data.shape[0], 3000])
        for index, req in order_data.iterrows():
            rid = req['id']
            count += 1
            if count <= n:
                folium.CircleMarker(
                    location=(req['latitude'], req['longitude']),
                    radius=2,
                    color="blue",
                    weight=0.25,
                    fill=True,
                    fill_color="blue",
                    fill_opacity=0.5,
                    popup=f"ID: {rid}, Type: {'Request'}"
                ).add_to(city_map)

        # Add a legend directly in the map
        legend_html = '''
                <div style="position: fixed; 
                            top: 10px; right: 10px; width: 120px; height: auto; 
                            background-color: white; z-index:9999; 
                            font-size:12px; border:1px solid grey; padding: 10px; 
                            border-radius: 1px;">
                    <b>Type of points</b><br>
                    <i style="background:blue; width:10px; height:10px; display:inline-block; margin-right:10px;"></i> Request<br>
                    <i style="background:red; width:10px; height:10px; display:inline-block; margin-right:10px;"></i> Station<br>
                </div>
                '''
        city_map.get_root().html.add_child(folium.Element(legend_html))

        x = re.findall(r'\d+', cluster_csv)  # 匹配所有连续数字
        csv_date = x[0]
        output_file = f'{self.city}_cluster_map_{csv_date}.html'
        output_file_path = os.path.join(self.work_dir, output_file)
        city_map.save(output_file_path)

        # End
        print(f'City map with clusters has been saved to {output_file_path}.')


    #Combine the grids and clusters in the map
    def combined_map(self, request_csv, station_csv, grid_csv, cluster_csv):
        # ID,  longitude,  latitude,  district,	city
        station_file = os.path.join(self.work_dir, station_csv)
        station_data = pd.read_csv(station_file)

        grid_file = os.path.join(self.work_dir, grid_csv)
        grid_data = pd.read_csv(grid_file)

        cluster_file = os.path.join(self.work_dir, cluster_csv)
        cluster_data = pd.read_csv(cluster_file)

        # id, longitude, latitude, submission, SOC, gridID
        request_file = os.path.join(self.work_dir, request_csv)
        order_data = pd.read_csv(request_file)

        # Load the city map data
        geojson_file = os.path.join(self.work_dir, data_params['geojson_data'])
        if not os.path.exists(geojson_file):
            raise FileNotFoundError(f"The GeoJSON file '{geojson_file}' does not exist.")
        city_shape = gpd.read_file(geojson_file)
        city_center = [city_shape.geometry.centroid.y.mean(), city_shape.geometry.centroid.x.mean()]

        # Get the city map layer
        city_map = folium.Map(location=city_center, zoom_start=10)
        folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(city_map)
        ''' 
        # 添加 CartoDB 底图
        folium.TileLayer(
            tiles='CartoDB positron',
            name='Light Map',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ).add_to(city_map)
        '''
        # 添加 Stamen Terrain 图层的调用方式
        folium.TileLayer(
            tiles='Stamen.Terrain',
            name='Terrain',
            attr='Map tiles by <a href="https://stamen.com">Stamen Design</a>, under CC BY 3.0. Data by <a href="https://openstreetmap.org">OpenStreetMap</a>, under ODbL.'
        ).add_to(city_map)

        folium.GeoJson(city_shape, name="City Boundary",
                       style_function=lambda x: {
                           "color": "blue",  # Boundary line color
                           "weight": 1,  # Line weight
                           "fillColor": "#ADD8E6",  # Light blue fill color
                           "fillOpacity": 0.25  # Adjust opacity for better visibility
                       }
                       ).add_to(city_map)

        # Add grids into the city map
        for index, cell in grid_data.iterrows():
            gid = cell['id']
            # Add the grid polygon to the map
            pattern = r"\((\d+\.\d+),\s*(\d+\.\d+)\)"  # 匹配形如 (x, y) 的数值对
            locs = re.findall(pattern, cell['boundary_points'])
            folium.Polygon(
                # [(118.761383, 31.277681), (118.76006831134019, 31.278097000000002)]
                locations=[[float(y), float(x)] for x, y in locs],
                color="red",
                weight=1.0,
                fill=False,
                fill_opacity=0.25
            ).add_to(city_map)

            # Calculate the center of the grid cell to place the circled marker
            centroid = ast.literal_eval(cell['centroid'])
            folium.CircleMarker(
                location=[centroid[1], centroid[0]],
                radius=2.5,  # Circle size
                color="red",
                weight=0.5,
                fill=True,
                fill_color="orange",
                fill_opacity=0.5,
                popup=f"Cluster ID: {gid}",  # Popup with the grid ID
            ).add_to(city_map)

            # Add the grid ID as text inside the circle
            folium.map.Marker(
                location=[centroid[1], centroid[0]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 12px; color: black; '
                         f'text-align: center;">{gid}</div>')
            ).add_to(city_map)

        # Add clusters into the city map
        for index, cell in cluster_data.iterrows():
            cid = cell['id']
            # Add the grid polygon to the map
            pattern = r"\((\d+\.\d+),\s*(\d+\.\d+)\)"  # 匹配形如 (x, y) 的数值对
            locs = re.findall(pattern, cell['boundary_points'])
            folium.Polygon(
                #[(118.761383, 31.277681), (118.76006831134019, 31.278097000000002)]
                locations=[[float(y), float(x)] for x, y in locs],
                color="green",
                weight=1.0,
                fill=False,
                fill_opacity=0.25
            ).add_to(city_map)

            # Calculate the center of the grid cell to place the circled marker
            centroid = ast.literal_eval(cell['centroid'])
            folium.CircleMarker(
                location=[centroid[1], centroid[0]],
                radius=2.5,  # Circle size
                color="green",
                weight=0.5,
                fill=True,
                fill_color="green",
                fill_opacity=0.75,
                popup=f"Cluster ID: {cid}",  # Popup with the grid ID
            ).add_to(city_map)

            ''' 
            # Add the grid ID as text inside the circle
            folium.map.Marker(
                location=[centroid[1], centroid[0]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 12px; color: black; '
                         f'text-align: center;">{cid}</div>')
            ).add_to(city_map)
            '''

        # Add stations into the city map
        icon_file = os.path.join(self.work_dir, data_params['station_icon'])
        # ID,  longitude,  latitude,  district,	city
        for index, st in station_data.iterrows():
            sid = st['id']
            icon = folium.CustomIcon(icon_file, icon_size=(14, 14))
            folium.Marker(
                location=(st['latitude'], st['longitude']),
                icon=icon,
                popup=f"ID: {sid}, Type: {'Station'}"
            ).add_to(city_map)

        # Add requests into city map
        # id	longitude	latitude	submission	SOC	gridID	clusterID
        count = 0
        n = min([order_data.shape[0], 3000])
        for index, req in order_data.iterrows():
            rid = req['id']
            count += 1
            if count <= n:
                folium.CircleMarker(
                    location=(req['latitude'], req['longitude']),
                    radius=2,
                    color="blue",
                    weight=0.25,
                    fill=True,
                    fill_color="blue",
                    fill_opacity=0.5,
                    popup=f"ID: {rid}, Type: {'Request'}"
                ).add_to(city_map)

        # Add a legend directly in the map
        legend_html = '''
                <div style="position: fixed; 
                            top: 10px; right: 10px; width: 120px; height: auto; 
                            background-color: white; z-index:9999; 
                            font-size:12px; border:1px solid grey; padding: 10px; 
                            border-radius: 1px;">
                    <b>Type of points</b><br>
                    <i style="background:blue; width:10px; height:10px; display:inline-block; margin-right:10px;"></i> Request<br>
                    <i style="background:red; width:10px; height:10px; display:inline-block; margin-right:10px;"></i> Station<br>
                </div>
                '''
        city_map.get_root().html.add_child(folium.Element(legend_html))

        x = re.findall(r'\d+', grid_csv)  # 匹配所有连续数字
        csv_date = x[0]
        output_file = f'{self.city}_map_{csv_date}.html'
        output_file_path = os.path.join(self.work_dir, output_file)
        city_map.save(output_file_path)

        # End
        print(f'City map with grids and clusters has been saved to {output_file_path}.')


