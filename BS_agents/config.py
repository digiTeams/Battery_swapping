import os
from datetime import datetime, timedelta

### Request data generation parameters
data_params = {
    'data_dir': 'data',
    'result_dir': 'results',

    'city': 'Nanjing',
    'geojson_data': 'Nanjing.geoJson',
    'grid_width': 0.25,  # 经度间隔, Approx. 20 kilometers
    'grid_height': 0.25,  # 纬度间隔, Approx. 20 kilometers
    'cluster_width': 0.11,  # 经度间隔, Approx. 5 kilometers
    'cluster_height': 0.08,  # 纬度间隔, Approx. 5 kilometers
    'station_icon': 'BS.png',

    'order_data': 'Nanjing_drivers_2025.csv',
    'station_data': 'Nanjing_stations_2025.csv',
    'grid_data': 'Nanjing_grids_2025.csv',
    'cluster_data': 'Nanjing_clusters_2025.csv',

    'start_time': datetime.strptime('07:00', '%H:%M'),  # Start time: 7:00 AM
    'end_time': datetime.strptime('23:00', '%H:%M'),  # End time: 23:00 PM
    'horizon': (7, 23),
    'peak_hour_centre': (12, 17),
    'peak_hour_suburb': [(8, 10), (18, 20)],

    'rate_peak_centre': 2.0,
    'rate_off_centre': 1.0,
    'rate_peak_suburb': 1.5,
    'rate_off_suburb': 0.5,

    'scenarios': [4, 8, 16],
    'scale': 1.0,  # 1.0, 2.0
}

''' 
    'rate_peak_centre': 2.0,
    'rate_off_centre': 1.0,
    'rate_peak_suburb': 1.5,
    'rate_off_suburb': 0.5,

    'rate_peak_centre': 2.5,
    'rate_off_centre': 1.5,
    'rate_peak_suburb': 2.0,
    'rate_off_suburb': 1.0,
'''

### Environment simulator parameters
env_params = {
    'test_mode': True,  # True, False
    'delta_time': 5,  # minute
    'gamma': 0.99,

    'battery_cap_I': 13,
    'chargers_I': 6,
    'swap_time_I': 5,  # minute

    'battery_cap_II': 22,
    'chargers_II': 12,
    'swap_time_II': 4,  # minute

    'charge_time_slow': 90,  # minute
    'charge_time_fast': 60,  # minute

    'max_nearby': 10, # maximum number of nearby stations
    'max_rate': 20,  # maximum demand rate
    'max_queue': 10,  # maximum queue length
    'max_cluster': 10,  # maximum clusters with demand

    'vehicle_speed': 20,  # km / h
    'travel_time_factor': 1.0,

    'max_distance': 20,  # km
    'max_wait_time': 15,  # 30, 15 minute
    'max_detour': 5,  # km
    'max_assign': 3,  # maximum new demand
    'max_bundle': 3,  # maximum size of bundle

    'swap_fee_off': 0.4,  # CNY/kWh
    'swap_fee_peak': 0.6,  # CNY/kWh

    'peak_electricity': (8, 22),
    'electric_price_off': 0.58,  # CNY/kwh
    'electric_price_peak': 1.35,  # CNY/kwh
    'power_capacity': 75,  # kwh

    # 平均小时工资约为 51.7元/小时
    'VOT': 0.85,  # CNY/min
    'driving_cost': 2.0,  # CNY/km

    # generalized cost for third-party service: (30 + 5*60/20)*0.85 = 38.25
    'penalty': 40,
    'big_M': 1e4,

    # weight factor of fairness: revenues vs. envies
    'weight': 4,   #0, 4, 8

}

# Learning parameters
RL_params = {
    'check_dir': 'neurals',

    # GUROBI parameters
    'GUROBI_TIME_LIMIT': 10,  # seconds
    'GUROBI_MIPGAP': 1e-3,
    'GUROBI_Threads': 16,
    'GUROBI_SEED': 14,

    # Charging policy
    'base_stocks': (0.5, 1.0),
    'MAX_Episodes': 3001,  # 1001, 2001
    'MAX_Tests': 10,
    'distributed': False,  # True, False
    'num_cores': 16,  # 32, 16

    # State representation
    'use_station': False,  # True, False
    'cluster_hot': False,  # True, False
    'station_embed_dims': 16,
    'cluster_embed_dims': 16,
    'step_embed_dims': 16, # 16, 32
    'hour_embed_dims': 8, #[7, 23]
    'hash_buckets': 1024,
    'num_hashes': 2,

    # MLP structure
    'hidden_dims': (256, 128, 64, 32),  #512, 256, 128, 64; 256, 128, 64, 32
    # Buffer parameters
    'batch_size': 256,  # 256, 512
    'buffer_size': 25600,

    # Training parameters: learning rate decays slow
    'initial_learning': 1e-3,
    'final_learning': 1e-5,  # slow: 1e-5; fast: 1e-4,
    'learning_decay': 0.99,  # slow: 0.95; fast: 0.99,

    'initial_epsilon': 1.0,
    'final_epsilon': 1e-3,
    'epsilon_decay': 0.99,

    # Target network parameters
    'update_freq': 32,
    'tau': 0.9,  # 软更新系数
    'clip_norm': 10,  # 5, 10

}