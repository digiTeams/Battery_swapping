###Import packages
import os
import copy
import math
import ast
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from config import data_params
from network import instances, BS_networks
from analyses import analys
from network.clusters import Cluster

work_dir = 'D:/Projects/BS_network'
#work_dir = 'C:/OneDrive/Team/BS_network'


def test_network(city, network):
    print('-------------------- Stations --------------------')
    for station in network.Stations:
        print(station)

    print('-------------------- Grids --------------------')
    for grid in network.Grids:
        print(grid)

    print('-------------------- Clusters --------------------')
    for cluster in network.Clusters:
        print(cluster)

    # Analyze the input data and the maps
    anals = analys.Analysist(city, work_dir, network)
    request_csv = 'Request_data_sample.csv'

    #Preprocessing maps
    ''' 
    station_csv = 'Nanjing_NIO_2025.csv'
    grid_csv = 'Nanjing_grids_202501.csv'
    cluster_csv = 'Nanjing_clusters_202501.csv'
    anals.plot_maps(request_csv, station_csv, grid_csv, cluster_csv, 'Griding')
    anals.plot_maps(request_csv, station_csv, grid_csv, cluster_csv, 'Combine')
    anals.plot_maps(request_csv, station_csv, grid_csv, cluster_csv, 'Clustering')
    '''

    #Finalized maps
    station_csv = data_params['station_data']  #'Nanjing_stations_2025.csv'
    grid_csv = data_params['grid_data']
    cluster_csv = data_params['cluster_data']
    anals.plot_maps(request_csv, station_csv, grid_csv, cluster_csv, 'Clustering')
    # anals.plot_maps(request_csv, station_csv, grid_csv, cluster_csv, 'Combine')


def parse_tuple(value: str) -> tuple:
    """安全解析坐标字符串为元组"""
    try:
        return ast.literal_eval(value.strip())
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"坐标解析失败: {value}") from e


def test_orders(num_clusters, scale, seed_r):
    order_file = os.path.join(work_dir, data_params['order_data'])
    order_data = pd.read_csv(order_file)

    episode = 0
    inst = instances.Instance(order_data, scale, episode, seed_r)
    inst.order_generation(num_clusters)
    inst.order_save(check=True)
    # inst.order_profile()


if __name__ == '__main__':
    city = 'Nanjing'
    print('----------- Set up the Nanjing NIO BS network ----------')
    network = BS_networks.BSNetwork(city, work_dir, seed=None)

    ''' '''
    network.build_network()
    # Verify the networks setting
    test_network(city, network)

    ''' 
    Expand the network with new stations
    CLass: num=4, num=8, num=16
    num, seed_n = 4, None  #None
    selected = [7, 26, 31, 44]
    
    num, seed_n = 8, None 
    selected = [0, 4, 8, 10, 27, 31, 37, 44]
    
    num, seed_n = 16, None 
    selected = [1, 2, 5, 11, 12, 14, 26, 31, 32, 33, 34, 42, 43, 44, 45, 48]
    '''
    num, seed_n = 4, None  # None
    selected = [7, 26, 31, 44]
    network.expand_network(selected, num, seed_n)

    # Verify the orders setting
    # num_clusters = len(network.Clusters)
    # test_orders(num_clusters, scale=1.0, seed_r=None)

    anals = analys.Analysist(city, work_dir, network)
    request_csv = 'Request_data_sample.csv'
    cluster_csv = data_params['cluster_data']
    #station_data: 'Nanjing_stations_2025.csv', 'Nanjing_stations_test_5.csv'
    station_csv = f'Nanjing_stations_test_{num}.csv'

    anals.cluster_map(request_csv, station_csv, cluster_csv)



