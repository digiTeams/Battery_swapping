'''
The environment simulator for a battery swapping network operation
'''
import os
import copy
import math
import ast
from datetime import datetime, timedelta
from geopy.distance import geodesic, great_circle
import numpy as np
import pandas as pd


from config import data_params, env_params, RL_params
from network.instances import Instance
from network.clusters import Cluster
from simulator.station_agents import StationAgent

###The environment class
class BatterySwapEnv:
    def __init__(self, seed):
        # Set the environment parameters
        self.seed = seed
        self.start_time = data_params['start_time']  #08:00
        self.end_time = data_params['end_time']  #20:00
        self.delta_time = env_params['delta_time'] #5 minutes
        self.gamma = env_params['gamma']

        #Set up the BS network
        # Set up the station agents
        self.agents = []
        #List of grids: id, bound, centroid, boundary, demand, envy, subsidy
        self.Clusters = []  #id: 0, 1, 2, ...
        #Dataframe of orders
        self.order_data = pd.DataFrame()
        self.new_demand = []  # the requests submitted within time [t-1, t)

        # Set the planning parameters
        self.current_time = data_params['start_time'] + timedelta(minutes=self.delta_time)
        self.time_step = 0 #time step
        self.hour = self.current_time.hour #hour of the time

        total_minutes = (self.end_time - self.start_time).total_seconds() / 60
        self.total_steps = math.ceil(total_minutes / self.delta_time)
        horizon = data_params['horizon'] #'horizon': (7, 23)
        self.max_hour = horizon[1] - 1 #22 --> 15

        # Record history logs
        self.total_served = 0  # number of total accepted requests
        self.total_charged = 0  # number of total charged batteries
        self.total_rewards = 0
        self.total_profits = 0
        self.total_revenues = 0
        self.total_subsidies = 0
        self.total_envies = 0

        self.all_demand = [] #set of all requests
        self.cluster_demands =[]
        self.queues = [] #number of maximum queues among the stations
        #'id', 'time', 'cluster', 'station', 'subsidy', 'detour', 'wait', 'envy'
        self.records = [] #record the service outcome of requests


    def parse_tuple(self, value: str) -> tuple:
        """安全解析坐标字符串为元组"""
        try:
            return ast.literal_eval(value.strip())
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"坐标解析失败: {value}") from e


    #Load the network stations, grids, and clusters
    def load_network(self, task, sn):
        '''
        'order_data': 'Nanjing_drivers_201910.csv',
        'station_data': 'Nanjing_stations_2025.csv',
        'cluster_data': 'Nanjing_clusters_2025.csv',
        'cluster_select_data': 'Nanjing_select_clusters_2025.csv',
        '''
        # Load the order dataset
        order_file = os.path.join(data_params['data_dir'], data_params['order_data'])
        self.order_data = pd.read_csv(order_file)

        ##Load cluster data
        #id, type, center, bound, centroid, boundary
        cluster_csv = data_params['cluster_data']
        cluster_file = os.path.join(data_params['data_dir'], cluster_csv)
        cluster_data = pd.read_csv(cluster_file)  #Pandas dataframe
        num_centre, num_suburb = 0, 0  #'suburb', 'centre'
        for index, cls in cluster_data.iterrows():
            cid = int(cls['id'])
            ctype = str(cls['type'])
            center = self.parse_tuple(cls['center'])
            bound = self.parse_tuple(cls['bound'])
            centroid = cls['centroid']
            boundary = cls['boundary_points']
            cluster = Cluster(cid, ctype, center, bound, centroid, boundary)
            self.Clusters.append(cluster)

            if ctype == 'centre':
                num_centre += 1
            else:
                num_suburb += 1

        ##Load station data
        #id, cluster, longitude, latitude, battery_cap, chargers, charge_time, swap_time
        if task == 'Test':
            scenarios = data_params['scenarios']
            num = int(scenarios[sn])
            station_csv = f'Nanjing_stations_test_{num}.csv'
        else:
            station_csv = data_params['station_data']

        station_file = os.path.join(data_params['data_dir'], station_csv)
        station_data = pd.read_csv(station_file)  #Pandas dataframe
        for index, st in station_data.iterrows():
            sid = int(st['id'])  # values in 0, 1, ..., num_station-1
            cid = int(st['cluster'])
            longitude = float(st['longitude'])
            latitude = float(st['latitude'])
            capacity = int(st['battery_cap'])
            chargers = int(st['chargers'])
            swap_time = float(st['swap_time'])
            charge_time = float(st['charge_time'])

            ctype = self.Clusters[cid].type
            agent = StationAgent(sid, cid, ctype, longitude, latitude, capacity, chargers, swap_time, charge_time)
            self.agents.append(agent)

        # Process nearby stations for agents
        num_nearby = np.zeros(len(self.agents), dtype=int)
        for j, agent in enumerate(self.agents):
            org = (agent.latitude, agent.longitude)
            num = 0
            for st in self.agents:
                dest = (st.latitude, st.longitude)
                dist = geodesic(org, dest).km
                if (dist <= env_params['max_detour']) and (agent.id != st.id):
                    num += 1
            agent.nearby_stations = num
            num_nearby[j] = num
        # Maximum nearby
        max_nearby = np.max(num_nearby)

        #End
        print(f'Environment is setup with {self.order_data.shape[0]} requests, {len(self.agents)} stations, '
              f'{len(self.Clusters)} clusters (centre {num_centre}, suburb {num_suburb}), '
              f'maximum nearby: {max_nearby}.')


    # Randomly generate an instance of request data
    def initialize_instance(self, scale, seed_r, episode):
        inst = Instance(self.order_data, scale, episode, seed_r)
        num_clusters = len(self.Clusters)
        inst.order_generation(num_clusters)
        if episode == 0:
            inst.order_save(check=False)

        self.all_demand[:] = inst.requests
        self.cluster_demands = inst.cluster_demand

        # requests in the entire time
        cids = [cls.id   for cls in self.Clusters]
        for req in self.all_demand:
            req.station = -1  # default: not assigned
            req.complete = 0

            nearest_station, nearest_distance, nearest_time = self.get_nearest_station(req)
            req.nearest_station = nearest_station
            req.nearest_distance = nearest_distance
            req.nearest_time = nearest_time

            if req.cluster not in cids:
                print(f'########### Error: request {req.id} has invalid cluster {req.cluster}! ###########')

        #record data statistics
        print(f'************ Initialize environment in episode {episode}: scale {scale:.2f}, '
              f'total demand {len(self.all_demand)} ***********')


    #Filter the requests within time [t-1, t)
    def get_new_requests(self, decision_time):
        """ 筛选当前决策时间范围内的新订单 """
        pre_time = decision_time - timedelta(minutes=self.delta_time)
        new_demand = []
        for req in self.all_demand:
            if pre_time <= req.submission < decision_time:
                req.decision_time = decision_time
                new_demand.append(req)
        #End
        return  new_demand


    #Check the nearest station for each request
    def get_nearest_station(self, req):
        dists = [(st.id, st.get_travel_distance(req))   for st in self.agents]
        nearest_station, nearest_distance = min(dists, key=lambda x: x[1])

        st = self.agents[nearest_station]
        nearest_time = st.get_travel_time(req) # minutes

        # End
        return  nearest_station, nearest_distance, nearest_time


    #Count the requests within acceptable distance for each station
    def get_agent_demand(self, agent, new_demand):
        num_new, num_clusters = 0, 0
        cls, demand = [], [] #available requests for the stations
        for req in new_demand:
            dist = geodesic((req.latitude, req.longitude), (agent.latitude, agent.longitude)).km
            if dist <= req.nearest_distance + env_params['max_detour']:
                num_new += 1
                demand.append(req)
                cls.append(req.cluster)

        # End
        CL = set(cls)
        num_clusters = len(CL)
        return  num_new, num_clusters, demand


    #Reset the environment to the initial states
    def reset(self):
        # Set the initial state S_{1}
        self.current_time = data_params['start_time'] + timedelta(minutes=self.delta_time)
        self.time_step = 0  # time step
        self.hour = self.current_time.hour  # hour of the time

        #Reset the status of requests
        for req in self.all_demand:
            req.station = -1  # assigned station id
            req.complete = 0  # completion flag: 1 = service completed
            req.arrival = data_params['end_time']  # arrival time
            req.begin_time = data_params['end_time']  # service begin time
            req.departure = data_params['end_time']  # departure time
            req.detour_time = 0.0  # travelling detour time
            req.wait_time = 0.0  # waiting time
            req.subsidy = 0
            req.envy = 0

        #Get new demand data and set decision time for requests
        self.new_demand[:] = self.get_new_requests(self.current_time) #requests within time [t-1, t)

        #Set the initial state of agents
        for agent in self.agents:
            num_new, num_clusters, demand = self.get_agent_demand(agent, self.new_demand)
            agent.reset_state(num_new, num_clusters)

        #Set the cluster status
        for cluster in self.Clusters:
            reqs = [req   for req in self.new_demand  if req.cluster == cluster.id]
            cluster.demand = len(reqs)
            cluster.envy = 0  #average envies among the users
            cluster.subsidy = 0
            cluster.Envies.clear()
            cluster.Subsidies.clear()

        # Initialize the environment
        self.total_served = 0
        self.total_charged = 0
        self.total_rewards = 0
        self.total_profits = 0
        self.total_revenues = 0
        self.total_subsidies = 0
        self.total_envies = 0

        self.queues.clear()
        self.records.clear()

        #End
        state = self.get_states()
        return  state


    #store the state information in single dictionary data
    def get_states(self):
        '''
        State space at each time t:
        # Agent: ('time_step', 'hour', 'cluster', 'nearby_stations', 'new_order',
        # 'cluster_demand', 'que_order', 'avg_waiting', 'full_batteries', 'inventory_position')
        '''
        state = {
            'time_step': self.time_step,
            'agents': {st.id: st.get_observe(self.time_step, self.hour)   for st in self.agents}
        }

        #End
        return  state


    # Process an agent local observation
    def process_observe(self, scale_, state_dims, observe):
        '''
        State space at each time t:
        # Agent: ('time_step', 'hour', 'cluster', 'nearby_stations', 'new_order',
        # 'cluster_demand', 'que_order', 'avg_waiting', 'full_batteries', 'inventory_position')
        self.state_dims = {'steps': steps, 'hours': hours, 'stations': num_agents,
                           'clusters': num_clusters, 'observe_dim': 10,
                           'discrete': 3, 'continuous': 7}

        'max_nearby': 10, # maximum number of nearby stations
        'max_rate': 20,  # maximum demand rate
        'max_queue': 10,  # maximum queue length
        'max_cluster': 10,  # maximum clusters with demand
        '''
        observe_dim = state_dims['observe_dim']
        processed = np.zeros(observe_dim, dtype=float)
        if scale_ < 2.0:
            max_demand = int(env_params['max_rate'])
        else:
            max_demand = int(env_params['max_rate']) * 2

        # Set upper limits
        horizon = data_params['horizon']
        max_que = int(env_params['max_queue'])
        max_nearby = int(env_params['max_nearby'])
        max_cluster = int(env_params['max_cluster'])
        max_wait = int(env_params['max_wait_time'])
        max_cap = int(env_params['battery_cap_II'])

        #Processing
        # Agent: ('time_step', 'hour', 'cluster', 'nearby_stations', 'new_order',
        # 'cluster_demand', 'que_order', 'avg_waiting', 'full_batteries', 'inventory_position')
        processed[0] = observe['time_step']
        processed[1] = observe['hour'] - horizon[0] #7, 8, 9, ..., 22, 23 --> 0, 1, 2, ..., 15, 16
        processed[2] = observe['cluster']
        processed[3] = observe['nearby_stations'] / max_nearby
        processed[4] = observe['new_order']/max_demand
        processed[5] = observe['cluster_demand']/max_cluster
        processed[6] = observe['que_order']/max_que
        processed[7] = observe['avg_waiting']/max_wait
        processed[8] = observe['full_batteries']/max_cap
        processed[9] = observe['inventory_position']/max_cap

        #End
        return  processed


    # Process a raw state into the expected input format
    def process_state(self, scale_, state_dims, state):
        ''' State space:
        self.state_dims = {'steps': steps, 'hours': hours, 'stations': num_agents,
                           'clusters': num_clusters, 'observe_dim': 10,
                           'discrete': 3, 'continuous': 7}
        '''
        num_agents = len(self.agents)
        observe_dim = state_dims['observe_dim']
        processed = np.zeros((num_agents, observe_dim), dtype=float)

        # Agent: ('time_step', 'hour', 'id', 'cluster', 'nearby_stations', 'new_order',
        # 'cluster_demand', 'que_order', 'avg_waiting', 'full_batteries', 'inventory_position')
        for j in range(num_agents):
            obs = state['agents'][j]
            process_obs = self.process_observe(scale_, state_dims, obs)
            processed[j,:] = process_obs

        #End
        return  processed


    def set_base_level(self, agent):
        future_time = self.current_time + timedelta(minutes=agent.charge_time)
        hr = future_time.hour  # hour of the time

        if agent.type == 'centre':
            peak_hours = data_params['peak_hour_centre'] ##'peak_hour_centre': (12, 14)
            peak = (peak_hours[0] <= hr <= peak_hours[1])
        else:
            peak_time = data_params['peak_hour_suburb'] ##'peak_hour_suburb': [(8, 9), (18, 19)]
            peak_hours1 = peak_time[0]
            peak1 = (peak_hours1[0] <= hr <= peak_hours1[1])

            peak_hours2 = peak_time[1]
            peak2 = (peak_hours2[0] <= hr <= peak_hours2[1])

            peak = (peak1 or peak2)

        #set the base stock level according to peak
        #'base_stocks': (0.5, 1.0),
        k = 1  if peak  else  0
        #End
        return k


    # Process subsidies actions
    def set_envy(self, rejects):
        # no-demand case
        if not self.new_demand:
            return {}, {}

        # positive demand case
        index = {req.id: i  for i, req in enumerate(self.new_demand)}
        costs, envies = {}, {}

        # Accepted requests
        for agent in self.agents:
            # dictionary {rid: cost}
            costs.update(agent.utilities)
            for req in agent.services:
                rid = req.id
                i = index[rid] #the index of req in the current new demand
                request = self.new_demand[i]

                #update the service result
                request.wait_time = req.wait_time
                request.detour_dist = req.detour_dist
                request.detour_time = req.detour_time

        # Rejected requests
        for req in rejects:
            rid = req.id
            i = index[rid]  # the index of req in the current new demand
            request = self.new_demand[i]

            # dictionary {rid: cost}
            costs[rid] = env_params['penalty']
            # update the service result
            request.wait_time = env_params['max_wait_time']
            request.detour_dist = env_params['max_distance']
            request.detour_time = ((60 * env_params['travel_time_factor'] * env_params['max_distance'])
                                   / env_params['vehicle_speed'])

        # Double check
        if env_params['test_mode']:
            keys_cost = list(costs.keys())
            for req in self.new_demand:
                if req.id not in keys_cost:
                    print(f'########### Request {req.id} is NOT included in utility computation! ###########')

        # Compute envy in the same cluster
        for cls in self.Clusters:
            reqs = [req  for req in self.new_demand if req.cluster == cls.id]
            cs = [costs[req.id]  for req in reqs]

            for req in reqs:
                rid = req.id
                req.envy = costs[rid] - min(cs)
                envies[rid] = costs[rid] - min(cs)

        #Double check
        if env_params['test_mode']:
            keys_envy = list(envies.keys())
            for req in self.new_demand:
                if req.id not in keys_envy:
                    print(f'########### Request {req.id} is NOT included in envy computation! ###########')

        #End
        return  costs, envies


    #Record the service results of all requests
    def update_record(self, rejects, costs, envies):
        # 'id', 'time', 'cluster', 'station', 'subsidy', 'detour', 'wait', 'envy'
        for req in self.new_demand:
            x = {'id': req.id, 'time': req.decision_time, 'cluster': req.cluster, 'area': req.area,
                 'station': req.station, 'detour': req.detour_dist, 'wait': req.wait_time,
                 'subsidy': req.subsidy, 'envy': req.envy}
            self.records.append(x)

        # End
        if env_params['test_mode'] and self.new_demand:
            avg_cost = np.mean(np.array(list(costs.values()), dtype=float))
            avg_envy = np.mean(np.array(list(envies.values()), dtype=float))
            subs = np.array([req.subsidy   for req in self.new_demand], dtype=float)
            avg_sub = np.mean(subs)
            print(f'Requests at time {self.time_step}: rejected {len(rejects)}, utilities {avg_cost:.2f}, '
                  f'envies {avg_envy:.2f}, subsidies {avg_sub:.2f}.')


    #Update the clusters after actions
    def update_cluster(self, subsidies, envies, next_demand):
        #Set the cluster envies and demand
        for cls in self.Clusters:
            if self.new_demand:
                # Update requests in the cluster
                req_id = [req.id   for req in self.new_demand  if req.cluster == cls.id]
                subs = [subsidies[rid]   for rid in req_id]
                envs = [envies[rid]   for rid in req_id]

                # Set average envies among all the requests in the cluster
                cls.Subsidies.extend(subs)
                if cls.Subsidies:
                    cls.subsidy = np.mean(np.array(cls.Subsidies))
                else:
                    cls.subsidy = 0

                cls.Envies.extend(envs)
                if cls.Envies:
                    cls.envy = np.mean(np.array(cls.Envies))
                else:
                    cls.envy = 0

            #Add new requests at time t+1:
            req_id = [req.id   for req in next_demand  if req.cluster == cls.id]
            cls.demand = len(req_id)

        # End
        if env_params['test_mode']:
            envies = np.array([cluster.envy   for cluster in self.Clusters], dtype=float)
            subsidies = np.array([cluster.subsidy   for cluster in self.Clusters], dtype=float)
            avg_envy = np.mean(envies)
            avg_subsidy = np.mean(subsidies)
            print(f'Clusters by time {self.time_step}: envies {avg_envy:.2f}, subsidies {avg_subsidy:.2f}.')


    #Generate transition tuples: (S_{t}, A_{t}, R_{t}, S_{t+1})
    def step(self, action, agent_type):
        """ 执行动作并更新环境状态
        action structure: a_{t}^{ij}, q_{t}^{j}, s_{t}^{i}
        # 'assignment' 编码为 shape=(N,) 的一维Numpy数组，每个值为订单匹配的换电站 ID
        # 'charging' 编码为 shape=(M,) 的一维Numpy数组，每个值为该换电站充电电池数
        # 'subsidy' 编码为 shape=(N,) 的一维Numpy数组，每个值为订单获得的补贴金额
        subsidy = 0 for optimizer agents
        action = {"assignment": assignment, "charging": charging, "subsidy": subsidy}
        """
        ######## Execute actions at time step t ########
        assignment = action["assignment"]  # shape=(new_demand,)
        subsidy = action["subsidy"]  # shape=(new_demand,)

        served = 0  # number of accepted requests at time t
        rejects = []  # set of rejected requests
        subsidies = {}
        if self.new_demand:
            for i, req in enumerate(self.new_demand):
                rid = req.id
                # Process assignment actions:
                sid = assignment[i] #sid = -1: reject; sid >= 0: accept
                req.station = sid
                # Process subsidies actions:
                req.subsidy = subsidy[i]
                subsidies[rid] = subsidy[i]

                #Update agents' queues
                if sid >= 0:  # accepted
                    req.station = sid
                    agent = self.agents[sid]
                    agent.queue.append(req) # add request into the queue
                    served += 1
                else:
                    rejects.append(req)
        self.total_served += served

        # Process charging actions
        charging = action["charging"]  # shape=(num_station,)
        self.total_charged += np.sum(charging)

        # ------------------------------------------------------ #
        ######## Transit to next state S_{t+1} ########
        next_time = self.current_time + timedelta(minutes=self.delta_time)
        next_demand = self.get_new_requests(next_time)  # requests within time [t, t+1)

        # Update the agents
        for j, agent in enumerate(self.agents):
            q = charging[j]
            num_new, num_clusters, _ = self.get_agent_demand(agent, next_demand)
            assign = [req  for req in self.new_demand  if req.station == agent.id]
            agent.update_state(self.time_step, self.current_time, num_new, num_clusters, assign, q)

        # Compute the utilities and envies for all requests at time t:
        costs, envies = self.set_envy(rejects)
        self.update_record(rejects, costs, envies)
        # Update the clusters with current requests' envies and subsidies
        self.update_cluster(subsidies, envies, next_demand)

        # ------------------------------------------------------ #
        # Agents' credit assignment: the rejected requests are attributed to the nearest station
        # revenue = service fees - charging costs
        reward = np.array([agent.current_revenue  for agent in self.agents], dtype=float)
        revenue = np.sum(reward)

        #Types: 'Fair', 'Weighted', 'Optimum'
        # Case: 'Fair'
        if agent_type == 'Fair':
            for i, req in enumerate(self.new_demand):
                j = assignment[i]
                if j >= 0:  # accepted requests
                    reward[j] -= subsidy[i]
                else:  #rejected requests
                    sid = req.nearest_station
                    reward[sid] -= subsidy[i]
        # Case: 'Weighted', 'Optimum'
        else:
            for i, req in enumerate(self.new_demand):
                rid = req.id
                j = assignment[i]
                if j >= 0:  # accepted requests
                    reward[j] -= env_params['weight'] * envies[rid]
                else:  # rejected requests
                    sid = req.nearest_station
                    reward[sid] -= env_params['weight'] * envies[rid]

        #Total accumulated rewards
        self.total_rewards += (self.gamma ** self.time_step) * np.sum(reward)

        # Update system time info
        self.current_time = next_time
        self.time_step += 1
        self.hour = self.current_time.hour  # hour of the time
        self.new_demand[:] = next_demand

        # Check terminal state at time 20:00: step = 143, hour = 20;
        # range: 0 <= t <= 143
        done = (self.time_step >= self.total_steps)
        if done:  #done if t+1 = 144
            self.time_step = self.total_steps - 1
            self.hour = self.max_hour

        # the next state in single dictionary data
        next_state = self.get_states()

        # ------------------------------------------------------ #
        ######## Save the service record ########
        self.total_revenues += revenue
        self.total_subsidies += np.sum(subsidy)
        envy_vals = np.array(list(envies.values()), dtype=float)
        self.total_envies += np.sum(envy_vals)

        #Record queues
        que = np.array([agent.que_order  for agent in self.agents])
        self.queues.append(np.max(que))

        #End
        return  next_state, reward, done


    # Save the record of request service
    def service_records(self, episode):
        # System records
        total_demand = len(self.all_demand)
        rejection = 1 - (self.total_served / total_demand)
        self.total_profits = self.total_revenues - self.total_subsidies

        # Request records
        # 'id', 'time', 'cluster', 'station', 'subsidy', 'detour', 'wait', 'envy'
        Wait = np.array([req['wait']  for req in self.records if req['station'] >= 0],
                        dtype=float)  # waiting time: minutes
        Detour = np.array([req['detour']  for req in self.records if req['station'] >= 0],
                          dtype=float)  # detour distance: km
        Subsidy = np.array([req['subsidy']  for req in self.records],
                           dtype=float)  # subsidy: CNY
        Envy = np.array([req['envy']  for req in self.records],
                        dtype=float)  # envy: CNY

        # {max_wait_time, mean_wait_time, std_wait_time}
        max_wait = np.max(Wait)
        mean_wait = np.mean(Wait)
        std_wait = np.std(Wait)

        # {max_detour_time, mean_detour_time, std_detour_time}
        max_detour = np.max(Detour)
        mean_detour = np.mean(Detour)
        std_detour = np.std(Detour)

        # {max_subsidy, mean_subsidy, std_subsidy}
        max_subsidy = np.max(Subsidy)
        mean_subsidy = np.mean(Subsidy)
        std_subsidy = np.std(Subsidy)

        # {max_envy, mean_envy, std_envy}
        max_envy = np.max(Envy)
        mean_envy = np.mean(Envy)
        std_envy = np.std(Envy)

        # {max_queue, mean_queue, std_queue}
        arr = np.array(self.queues)
        max_queue = np.max(arr)
        mean_queue = np.mean(arr)
        std_queue = np.std(arr)

        # Stations records: 'serve', 'charge'
        serves = np.zeros(len(self.agents), dtype=int)
        charges = np.zeros(len(self.agents), dtype=int)
        for j, agent in enumerate(self.agents):
            serves[j] = agent.served
            charges[j] = agent.charged

        avg_station_serve = np.mean(serves)
        std_station_serve = np.std(serves)
        avg_station_charge = np.mean(charges)
        std_station_charge = np.std(charges)

        # Clusters records: 'envy'
        envies = np.array([cluster.envy   for cluster in self.Clusters], dtype=float)
        avg_cls_envy = np.mean(envies)
        std_cls_envy = np.std(envies)

        # Clusters records: 'subsidy'
        subsidies = np.array([cluster.subsidy  for cluster in self.Clusters], dtype=float)
        avg_cls_subsidy = np.mean(subsidies)
        std_cls_subsidy = np.std(subsidies)

        # Area records: 'envy'
        centres = np.array([req['envy']  for req in self.records if req['area'] == 'centre'],
                           dtype=float)
        avg_centre_envy = np.mean(centres)
        std_centre_envy = np.std(centres)

        suburbs = np.array([req['envy']  for req in self.records if req['area'] == 'suburb'],
                           dtype=float)
        avg_suburb_envy = np.mean(suburbs)
        std_suburb_envy = np.std(suburbs)

        # Area records: 'subsidy'
        centres = np.array([req['subsidy']  for req in self.records if req['area'] == 'centre'],
                           dtype=float)
        avg_centre_subsidy = np.mean(centres)
        std_centre_subsidy = np.std(centres)

        suburbs = np.array([req['subsidy']  for req in self.records if req['area'] == 'suburb'],
                           dtype=float)
        avg_suburb_subsidy = np.mean(suburbs)
        std_suburb_subsidy = np.std(suburbs)

        ''' 
        ['episode', 'total_demand', 'served', 'charged', 'rejection', 'total_rewards', 
         'total_profits', 'total_revenues', 'total_subsidies', 'total_envies',
         'max_wait', 'mean_wait', 'std_wait', 'max_detour', 'mean_detour', 'std_detour',
         'max_envy', 'mean_envy', 'std_envy', 'max_subsidy', 'mean_subsidy', 'std_subsidy', 
         'max_queue', 'mean_queue', 'std_queue', 
         'avg_station_serve', 'std_station_serve', 'avg_station_charge', 'std_station_charge', 
         'avg_cls_envy', 'std_cls_envy', 'avg_cls_subsidy', 'std_cls_subsidy',
         'avg_centre_envy', 'std_centre_envy', 'avg_centre_subsidy', 'std_centre_subsidy',
         'avg_suburb_envy', 'std_suburb_envy', 'avg_suburb_subsidy', 'std_suburb_subsidy',
         ... ...
         'run_time', 'loss']
        '''
        result = {'episode': episode,
                  'total_demand': total_demand,
                  'served': self.total_served,
                  'charged': self.total_charged,
                  'rejection': rejection,
                  'total_rewards': self.total_rewards,
                  'total_profits': self.total_profits,
                  'total_revenues': self.total_revenues,
                  'total_subsidies': self.total_subsidies,
                  'total_envies': self.total_envies,
                  'max_wait': max_wait,
                  'mean_wait': mean_wait,
                  'std_wait': std_wait,
                  'max_detour': max_detour,
                  'mean_detour': mean_detour,
                  'std_detour': std_detour,
                  'max_envy': max_envy,
                  'mean_envy': mean_envy,
                  'std_envy': std_envy,
                  'max_subsidy': max_subsidy,
                  'mean_subsidy': mean_subsidy,
                  'std_subsidy': std_subsidy,
                  'max_queue': max_queue,
                  'mean_queue': mean_queue,
                  'std_queue': std_queue,
                  'avg_station_serve': avg_station_serve,
                  'std_station_serve': std_station_serve,
                  'avg_station_charge': avg_station_charge,
                  'std_station_charge': std_station_charge,
                  'avg_cls_envy': avg_cls_envy,
                  'std_cls_envy': std_cls_envy,
                  'avg_cls_subsidy': avg_cls_subsidy,
                  'std_cls_subsidy': std_cls_subsidy,
                  'avg_centre_envy': avg_centre_envy,
                  'std_centre_envy': std_centre_envy,
                  'avg_centre_subsidy': avg_centre_subsidy,
                  'std_centre_subsidy': std_centre_subsidy,
                  'avg_suburb_envy': avg_suburb_envy,
                  'std_suburb_envy': std_suburb_envy,
                  'avg_suburb_subsidy': avg_suburb_subsidy,
                  'std_suburb_subsidy': std_suburb_subsidy
                  }

        # Subsidy in clusters
        for c, cluster in enumerate(self.Clusters):
            cname = f'cluster_{c}'
            result[cname] = cluster.subsidy

        # End
        return  result


    def request_records(self, policy, episode):
        '''
        x = {'id': req.id, 'time': req.decision_time, 'cluster': req.cluster, 'area': req.area,
             'station': req.station, 'detour': req.detour_dist, 'wait': req.wait_time,
             'subsidy': req.subsidy, 'envy': req.envy}
        '''
        # episode = sn * MAX_Tests + k
        MAX_Tests = int(RL_params['MAX_Tests'])
        sn = episode // MAX_Tests
        eps = episode % MAX_Tests

        # The dataframe
        df = pd.DataFrame(self.records)
        df['scenario'] = sn
        df['episode'] = eps

        # Output the request service records
        csv_name = f'Request_records_{policy}.csv'
        output_file = os.path.join(data_params['result_dir'], csv_name)
        # 检查文件是否存在，决定是否写入表头
        header = not os.path.exists(output_file)
        # 以追加模式写入CSV文件
        df.to_csv(output_file, mode='a', index=False, header=header)


    def station_records(self):
        # Agent: ('time_step', 'hour', 'cluster', 'new_order', 'cluster_demand', 'que_order', 'avg_waiting',
        # 'full_batteries', 'inventory_position', 'short_charging', 'long_charging')
        demands = np.array([agent.new_order  for agent in self.agents],
                           dtype=int)
        max_new = np.max(demands)
        cls = np.array([agent.cluster_demand  for agent in self.agents],
                       dtype=int)
        max_cls = np.max(cls)
        ques = np.array([agent.que_order  for agent in self.agents],
                        dtype=int)
        max_que = np.max(ques)

        # End
        return  max_new, max_cls, max_que
