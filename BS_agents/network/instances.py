'''
Generate an episode by sampling from the request data file
Simulate the demand arrival process in the BS network
'''

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter

from config import data_params, env_params
from network import orders

work_dir = 'D:/Projects/BS_network'
#work_dir = 'C:/OneDrive/Team/BS_network'

class Instance:
    def __init__(self, dataset, scale, eps_id, seed):
        self.scale = scale
        self.episode = eps_id #id of the episode

        self.rng = None
        if seed is not None:
            np.random.seed(seed)
            self.rng = np.random.RandomState(seed)

        # Load the order dataset
        # ID status longitude latitude addcode district city time date hour minute
        self.order_data = dataset  #pd.DataFrame

        self.requests = []  #the set of sampled requests
        self.cluster_demand = np.array([])  # the demand in each cluster


    def __str__(self):
        return (f"Episode ID {self.episode}: totally {len(self.requests)} requests; \n "
                f"cluster demand: {self.cluster_demand}")


    ###Set the Poisson arrival rate
    def set_lambda(self, hour, area):
        '''
        'peak_hour_centre': (13, 17),
        'peak_hour_suburb': [(7, 9), (18, 20)],

        'rate_peak_centre': 2.5,
        'rate_off_centre': 1.0,
        'rate_peak_suburb': 1.5,
        'rate_off_suburb': 0.5,
        '''

        if area == 'centre':
            peak = data_params['peak_hour_centre']  #'peak_hour_centre': (12, 16)
            if peak[0] <= hour <= peak[1]:  # peak time
                lamb = self.scale * data_params['rate_peak_centre']
            else:  # off-peak time
                lamb = self.scale * data_params['rate_off_centre']
        else:
            peaks = data_params['peak_hour_suburb']  #'peak_hour_suburb': [(7, 9), (18, 20)]
            peak_m = peaks[0]
            peak_e = peaks[1]
            if (peak_m[0] <= hour <= peak_m[1]) or (peak_e[0] <= hour <= peak_e[1]):  # peak time
                lamb = self.scale * data_params['rate_peak_suburb']
            else:
                lamb = self.scale * data_params['rate_off_suburb']
        # End
        return  lamb


    ###Sample random demands from the dataset matching arrival times.
    def sample_order(self, current_time, area):
        hr_ = current_time.hour
        min_ = current_time.minute

        #data set: id  status  longitude  latitude  addcode  district  city  time  date  hour  minute  area cluster
        eps = 2.5  # unit: minute
        lower_min = max(0, min_ - eps)
        upper_min = min(59, min_ + eps)

        #filter with clamped minute range
        filter_data = self.order_data[
            (self.order_data['hour'] == hr_)
            & (self.order_data['minute'] >= lower_min)
            & (self.order_data['minute'] <= upper_min)
            & (self.order_data['area'] == area)
            & (self.order_data['cluster'] >= 0)
            ]

        if not filter_data.empty:
            sample = filter_data.sample(n=1, random_state=self.rng) if self.rng is not None \
                else filter_data.sample(n=1)
            sample['submission'] = current_time
            columns = ["id", "cluster", "area", "longitude", "latitude", "submission"]
            sample = sample[columns]
            req = sample.iloc[0].to_dict()

        else:
            req = {}
            #print(f'###### No sample data for grid {grid.id} is available at time: {hr_} : {min_} ######')

        return  req


    ##Generate stochastic dynamic demand following a Poisson arrival process with time-varying rate.
    def generate_poisson(self, area):
        '''
        'start_time': datetime.strptime('07:00', '%H:%M')
        'end_time': datetime.strptime('23:00', '%H:%M')
        'horizon': (7, 23),
        delta_time = env_params['delta_time'] #5 minutes
        '''
        current_time = data_params['start_time']
        latest_time = data_params['end_time'] - timedelta(minutes=env_params['delta_time'])

        reqs = []
        while current_time <= latest_time:
            lamb = self.set_lambda(current_time.hour, area)
            inter_arrival = np.random.exponential(1 / lamb) * 60
            current_time += timedelta(seconds = inter_arrival)

            # "id", "cluster", "area", "longitude", "latitude", "submission"
            req = self.sample_order(current_time, area)
            if (current_time <= latest_time) and req:
                rid = req["id"]
                cid = req["cluster"]
                area = req["area"]
                long = req["longitude"]
                lat = req["latitude"]
                submission = req["submission"]
                SOC = np.random.uniform(10, 20)

                order = orders.Request(rid, cid, area, long, lat, submission, SOC)
                reqs.append(order)
                self.cluster_demand[cid] += 1

        #End
        return  reqs


    ###Combine independent Poisson processes into a single order profile
    def order_generation(self, num_clusters):
        t0 = time.perf_counter()
        all_req = []
        self.cluster_demand = np.zeros(num_clusters, dtype=int)
        print(f'Generating requests by Poisson processes ... ...')

        #Demand in centre areas
        req = self.generate_poisson('centre')
        all_req.extend(req)
        # Demand in suburb areas
        req = self.generate_poisson('suburb')
        all_req.extend(req)

        # Combine and sort by arrival time
        self.requests = sorted(all_req, key=lambda x: x.submission)
        # Preprocess the request data
        for i, req in enumerate(self.requests):
            req.id = i  # 设 id 从 0 开始

        # End
        t1 = time.perf_counter()
        run_time = (t1 - t0) / 60  # time in minutes
        print(f'The runtime for generating requests: {run_time:.2f} minutes.')


    def order_save(self, check):
        #id, cluster, area, longitude, latitude, submission, SOC
        # 将订单列表转换为DataFrame
        data = {
            'id': [req.id  for req in self.requests],
            'cluster': [req.cluster  for req in self.requests],
            'area': [req.area  for req in self.requests],
            'longitude': [req.longitude  for req in self.requests],
            'latitude': [req.latitude  for req in self.requests],
            'submission': [req.submission  for req in self.requests],
            'SOC': [req.SOC  for req in self.requests]
        }
        df = pd.DataFrame(data)

        # 导出为CSV文件
        csv_name = f'Request_data_sample.csv'
        if check:
            output_file = os.path.join(work_dir, csv_name)
        else:
            output_file = os.path.join(data_params['result_dir'], csv_name)

        df.to_csv(output_file, index=False)
        print(f'Request data {csv_name} is saved to {output_file}.')


    def order_profile(self):
        '''
        #id, cluster, area, longitude, latitude, submission, SOC
        'start_time': datetime.strptime('07:00', '%H:%M')
        'end_time': datetime.strptime('23:00', '%H:%M')
        'horizon': (7, 23),
        '''
        print(f"Episode {self.episode}: totally {len(self.requests)} requests; \n "
         f"cluster demand: {self.cluster_demand}")

        horizon = data_params['horizon']
        try:
            arrival_times = [req.submission  for req in self.requests if hasattr(req, 'submission')]
            arrival_hours = [t.hour  for t in arrival_times if horizon[0] <= t.hour < horizon[1]]  # 限定时间范围
        except AttributeError as e:
            print("Error: Ensure all requests have a valid 'submission' timestamp.")
            return

        # 统计每小时订单数量
        order_counts = Counter(arrival_hours)
        valid_hours = list(range(horizon[0], horizon[1]))  # 06:00 - 24:00 小时列表
        df = pd.DataFrame({'hour': valid_hours, 'orders': [order_counts.get(h, 0) for h in valid_hours]})

        # 直方图展示结果
        plt.figure(figsize=(10, 6))
        plt.bar(df["hour"], df["orders"], color='blue', edgecolor='black', alpha=0.7)

        # 添加数值标签
        for i, v in enumerate(df["orders"]):
            plt.text(df["hour"].iloc[i], v + 0.5, str(v), ha='center', fontsize=10)

        # 设置标题，包含总订单数和平均订单数
        plt.title(f'Hourly Demand Distribution: totally {len(self.requests)} requests')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Requests')
        plt.xticks(valid_hours)  # 确保 x 轴刻度正确
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        current_datetime = datetime.now().strftime('%m-%d')
        fig = f'hourly_demand_{current_datetime}.png'
        output_file = os.path.join(work_dir, fig)
        plt.savefig(output_file)

        plt.show()
        plt.close()




