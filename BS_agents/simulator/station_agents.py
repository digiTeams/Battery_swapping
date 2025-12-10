'''
The data structure for station agent
#id, cluster, type, longitude, latitude, battery_cap, chargers, charge_time, swap_time
'''

import copy
from datetime import datetime, timedelta
import math
import numpy as np
from collections import OrderedDict
from geopy.distance import geodesic, great_circle

from config import data_params, env_params



"""
固定大小的有序字典，当元素超出最大限制时自动删除最早的元素
:param charge_steps: 最大容量限制
"""
class FixedDict(OrderedDict):
    def __init__(self, charge_steps: int):
        super().__init__()
        self.max_size = charge_steps


    def __copy__(self):
        new_copy = type(self)(self.max_size)
        new_copy.update(self)
        return  new_copy


    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        new_copy = type(self)(self.max_size)
        memo[id(self)] = new_copy
        for key, value in self.items():
            new_key = copy.deepcopy(key, memo)
            new_value = copy.deepcopy(value, memo)
            new_copy[new_key] = new_value
        return  new_copy


    def add(self, key, value):
        """ 添加元素，若超出最大限制，则删除最早的元素
        在 collections.OrderedDict 中：
        popitem(last=False) 删除最早插入的元素（队列的左端）。
        popitem(last=True) 删除最新插入的元素（队列的右端）。
        """
        self[key] = value
        if len(self) > self.max_size:
            self.popitem(last=False)  # 删除最早的元素 (FIFO)


    def get_first(self):
        """ 获取 `OrderedDict` 中的第一个元素，若为空则返回 `None, None` """
        if not self:
            return  None, None

        first_key, first_value = next(iter(self.items()))
        return  first_key, first_value


    def sum_all(self):
        total = sum(self.values())
        return  total


    def get_cumsum(self, val):
        """ 计算累加求和，并返回第一个满足条件的 key """
        if val <= 0 or not self:
            return  -1  # 如果要求和小于等于 0，直接返回 -1

        cumsum = 0
        for key, q in self.items():
            cumsum += q
            if cumsum >= val:
                return  key  # 返回第一个满足条件的时间步
        return  -1  # 若未找到，则返回 -1


#################################################
class StationAgent:
    def __init__(self, sid, cid, ctype, longitude, latitude, capacity, chargers, swap_time, charge_time):
        #Temporal info.
        total_minutes = (data_params['end_time'] - data_params['start_time']).total_seconds() / 60
        self.total_steps = math.ceil(total_minutes / env_params['delta_time'])
        horizon = data_params['horizon']  # 'horizon': (8, 20)
        self.max_hour = horizon[1]

        #Station setting
        self.id = sid  #values in 0, 1, ..., num_station-1
        self.cluster = cid  #index of cluster for the station
        self.type = ctype
        self.longitude = longitude
        self.latitude = latitude
        self.capacity = capacity  #maximum available batteries
        self.chargers = chargers  #number of charging slots
        self.swap_time = swap_time  #fixed time for swapping, unit: minute
        self.charge_time = charge_time  #fixed time for charging, unit: minute
        self.charge_steps = math.ceil(charge_time / env_params['delta_time'])

        # the number of stations within detour distance
        self.nearby_stations = 0

        #State space: state at time step t before action
        self.new_order = 0 #新到达附近订单数量
        self.cluster_demand = 0  #number of clusters with valid demand
        self.que_order = 0 #排队等待的订单数量
        self.avg_waiting = 0  #记录当前时刻订单的平均等待时间
        self.full_batteries = capacity  # 当前满电池数量
        self.inventory_position = capacity  # inventory position level before action
        self.short_charging = 0  #in charging batteries with short remaining time: <=3
        self.long_charging = 0  #in charging batteries with long remaining time >6

        #Auxiliary state at time step t after action
        #Action: a_{t}^{ij}, q_{t}^{j} at time step t
        self.queue = []  # 等待换电的订单队列，包含了新匹配的订单
        self.temp_queue = []  # a copy of active queue
        self.services = []
        # 记录前tau_C个时刻充电电池数量记录，包含了新充电电池
        # Dictionary list [{t, q}]: q_{t-tau_C+1}^{j}, q_{t-tau_C+2}^{j}, q_{t-1}^{j}, q_{t}^{j},
        self.pre_charging = FixedDict(self.charge_steps)
        self.charging_batteries = 0  # 正在充电的电池数量，包含了新充电电池
        self.empty_batteries = 0  # 空电池数量
        self.dist = 0

        # reward at the current time t
        self.current_revenue = 0
        self.utilities = {}

        #Service history record
        self.served = 0  #total number of completed requests
        self.charged = 0  #total number of charged batteries
        self.revenues = 0 #total revenues


    #Print the object
    def __str__(self):
        return (f"Agent ID: {self.id}, cluster: {self.cluster}, battery: {self.capacity}, chargers: {self.chargers}, "
                f"charge time: {self.charge_time} min, swap time {self.swap_time} min. ")


    #Reset the station state to be initial
    def reset_state(self, num_new, num_clusters):
        # The initial state values at time step 1
        self.new_order = num_new  # 新到达附近订单数量
        self.cluster_demand = num_clusters
        self.que_order = 0  # 排队等待的订单数量
        self.avg_waiting = 0  # 记录当前时刻订单的平均等待时间
        self.full_batteries = self.capacity  # 当前满电池数量
        self.inventory_position = self.capacity  # inventory position level before action
        self.short_charging = 0  # in charging batteries with short remaining time: <=3
        self.long_charging = 0  # in charging batteries with long remaining time >6

        # Auxiliary state at time step t after action
        self.queue.clear()  # 等待换电的订单队列，包含了新匹配的订单
        self.temp_queue.clear()  # a copy of active queue
        self.pre_charging.clear()
        self.charging_batteries = 0  # 正在充电的电池数量，包含了新充电电池
        self.empty_batteries = 0  # 空电池数量
        self.dist = 0

        self.current_revenue = 0 # reward at the current time t
        self.utilities.clear()  #dictionary {rid: cost}

        # Service history record
        self.served = 0  # accumulated number of completed requests so far
        self.charged = 0  # accumulated number of charging batteries so far


    #return the structured states S_{t}
    def get_observe(self, time_step, hour):
        # Agent: ('time_step', 'hour', 'cluster', 'nearby_stations', 'new_order',
        # 'cluster_demand', 'que_order', 'avg_waiting', 'full_batteries', 'inventory_position')
        observe = {
            'time_step': time_step,
            'hour': hour,
            'cluster': self.cluster,
            'nearby_stations': self.nearby_stations,
            'new_order': self.new_order,
            'cluster_demand': self.cluster_demand,
            'que_order': self.que_order,
            'avg_waiting': self.avg_waiting,
            'full_batteries': self.full_batteries,
            'inventory_position': self.inventory_position
        }
        #End
        return  observe


    #Compute the available time of full batteries for a queueing order
    def available_time(self, current_time, avail_batteries, charging_que, que_len):
        ##full_batteries must be updated at first to include q_{t-C} !!!
        #avail_batteries = self.full_batteries
        if que_len <= 0:
            print(f'########## Error in queue length: {que_len} ########## ')

        if avail_batteries >= que_len:
            avail_time = current_time  #time step t
        else:
            #pre_charging does not include the decision at t
            val = que_len - avail_batteries
            last_step = charging_que.get_cumsum(val)

            '''
            Case: time_step t < C
            self.pre_charging: t-3: q_{t-3}; t-2: q_{t-2}; t-1: q_{t-1}; t: q_{t}            
            
            Case: time_step t >= C
            self.pre_charging: q_{t-(C-1)}, q_{t-(C-2)}, ..., q_{t-1}, q_{t}
            
            avail = start_time + (t+C) * delta_time
            '''
            if last_step >= 0:
                ct = env_params['delta_time'] * (last_step + self.charge_steps + 1)
                avail_time = data_params['start_time'] + timedelta(minutes=ct)
            else:
                avail_time = None  #datetime.strptime('23:59', '%H:%M')

        #End
        return avail_time


    # State update procedure for the selected action
    # Given initial state S_{t}, update the next state S_{t+1} before new demand arrives
    def simul_queue(self, current_time, full_batteries, queue, charging_que, checking):
        """
        At the current time step t, after action taken, new orders already join in queue;
        For each order in the queue, compute the service time, departure time, and waiting time
        """
        if queue:
            #Double-check the arrival time
            num_processing = 0  #number of requests in processing before time t
            for req in queue:
                travel_dist = self.get_travel_distance(req)
                travel_time = self.get_travel_time(req)

                req.detour_dist = np.max([travel_dist - req.nearest_distance, 0])
                req.detour_time = np.max([travel_time - req.nearest_time, 0])

                req.arrival = req.decision_time + timedelta(minutes=travel_time)
                if (req.decision_time < current_time) and (req.begin_time < current_time):
                    num_processing += 1

            # 按到达时间排序，先到先服务
            queue.sort(key=lambda x: x.arrival)
            avail_batteries = full_batteries
            if avail_batteries < 0:
                print(f'########### Error in processing early requests: {full_batteries}, '
                      f'{num_processing}! ###########')

            start_time = current_time
            for k, req in enumerate(queue):  #0, 1, ..., que_size-1
                que_len = k + 1
                #determine the battery available time for the request
                if (req.decision_time < current_time) and (req.begin_time < current_time):
                    #case: earlier request already started service
                    avail_time = req.begin_time
                else:
                    # case: new requests in the queue
                    avail_time = self.available_time(current_time, avail_batteries, charging_que, que_len)

                #start the swapping process
                if (req.decision_time < current_time) and (req.begin_time < current_time):
                    start_serve = req.begin_time
                    complete_time = start_serve + timedelta(minutes=self.swap_time)
                    req.departure = complete_time
                    req.wait_time = (complete_time - req.arrival).total_seconds() / 60

                    # the previous start time for the next order
                    start_time = complete_time

                elif avail_time is not None:
                    start_serve = max(start_time, req.arrival, avail_time)
                    complete_time = start_serve + timedelta(minutes=self.swap_time)
                    req.begin_time = start_serve
                    req.departure = complete_time
                    req.wait_time = (complete_time - req.arrival).total_seconds() / 60

                    # the pre start time for the next order
                    start_time = complete_time

                else:
                    complete_time = data_params['end_time']
                    req.departure = complete_time
                    req.wait_time = 4 * env_params['max_wait_time']
                    if checking:
                        print(f'###### Error@Agent #{self.id}: no feasible service for request {req.id} ######')
                    break

                #Error check
                if checking and (req.wait_time > env_params['max_wait_time']):
                    print(f'###### WARNING@Agent #{self.id}: over-limit waiting time {req.wait_time:.2f} of '
                          f'request {req.id} ######')
        #End
        return  queue


    #State update for station at the next time step
    #current_time: t; next_time: t+1
    def update_state(self, time_step, current_time, num_new, num_clusters, assign, charge):
        #Chekc battery status: full_batteries already include the ones full at time t
        if self.full_batteries < 0:
            print(f'######## Error: negative initial full batteries: {self.full_batteries} ########')

        #Remove q_{t-c} from the queue
        charging_que = copy.deepcopy(self.pre_charging)
        charging_que.add(time_step, 0)

        #simulate the queuing process
        assign_id = [req.id   for req in assign]
        checking = True
        self.queue[:] = self.simul_queue(current_time, self.full_batteries, self.queue, charging_que, checking)
        self.services[:] = [req   for req in self.queue  if req.id in assign_id]
        self.served += len(self.services)

        # Double check services
        if env_params['test_mode']:
            serve_id = [req.id   for req in self.services]
            que_id = [req.id   for req in self.queue]
            for rid in assign_id:
                if (rid not in serve_id) or (rid not in que_id):
                    print(f'######### Error: assigned request {rid} is NOT included in services #########')

        # process charging state
        self.pre_charging.add(time_step, charge)
        self.charged += charge  #record charging
        self.charging_batteries = int(self.pre_charging.sum_all())
        self.empty_batteries -= charge
        if self.charging_batteries > self.chargers:
            print(f'######## Error: Charging exceeds maximum slots! ########')

        # Process the queuing orders
        wt = 0
        if self.queue:
            # the average waiting time for the queuing orders
            wait = np.array([req.wait_time  for req in self.queue])
            wt = np.mean(wait)

            # check any request completed by time t+1
            next_time = current_time + timedelta(minutes=env_params['delta_time'])  # t+1
            completed = [req   for req in self.queue  if req.departure <= next_time]
            self.full_batteries = self.full_batteries - len(completed)
            self.empty_batteries += len(completed)

            #The updated queue after removing completed requests
            self.queue[:] = [req   for req in self.queue  if req.departure > next_time]

        #full batteries at time t+1 without including q_{t+1-C}
        avail_batteries = self.full_batteries

        #State at time t+1
        next_time_step = time_step + 1
        self.new_order = num_new  # 新到达附近订单数量
        self.cluster_demand = num_clusters
        self.que_order = len(self.queue)
        self.avg_waiting = wt

        short_charging, long_charging = self.battery_to_full(self.pre_charging, next_time_step)
        self.short_charging = short_charging
        self.long_charging = long_charging

        if time_step + 1 >= self.charge_steps:  # nonempty case
            t0, qt = self.pre_charging.get_first() # t0 = time_step + 1 - C
            # add the full battery just completed at time step t if any
            self.full_batteries += qt

        if self.full_batteries < 0:
            print(f'######## Error at station {self.id}: negative full battery! ########')

        self.inventory_position = (avail_batteries + self.pre_charging.sum_all()
                                   - self.que_order)

        #reward at time t
        revenue, costs = self.calculate_reward(current_time, self.services, charge)
        if env_params['test_mode'] and assign and len(costs) == 0:
            serve_id = [req.id   for req in self.services]
            print(f'########### Service error of agent {self.id}: requests {serve_id}; '
                  f'costs {costs} ###########')

        self.current_revenue = revenue
        self.utilities.clear()
        self.utilities.update(costs)
        self.revenues += revenue


    #Calculate rewards
    def calculate_reward(self, current_time, services, charge):
        # 判断当前时间是否在高峰时段
        peak_hours = env_params['peak_electricity']
        hour = current_time.hour  # 计算当前时间在小时维度的表示
        is_peak = (hour >= peak_hours[0]) and (hour <= peak_hours[1])
        swap_fee = env_params['swap_fee_peak']  if is_peak else env_params['swap_fee_off']
        charge_fee = env_params['electric_price_peak']  if is_peak else env_params['electric_price_off']

        # Service fees
        fares = (swap_fee + charge_fee) * env_params['power_capacity'] * len(services)

        #Charging costs
        charge_cost = self.calculate_charging(current_time, charge)
        #Reward: service fees - charge costs
        revenue = fares - charge_cost

        # Service revenues and time costs
        costs = {}  # dictionary {rid: cost}
        for req in services:
            rid = req.id
            costs[rid] = env_params['VOT'] * (req.wait_time + req.detour_time)
        ''' 
        if (revenue < 0) and (len(services) > 0):
            print(f'Revenue calculation: agent {self.id}, fares {fares:.2f} CNY, '
                  f'charged {charge}, electricity costs {charge_cost:.2f} CNY.')
        '''
        #End
        return  revenue, costs


    #Get the number of batteries in each time to full level
    def battery_to_full(self, pre_charging, time_step):
        # self.pre_charging.add(time_step, charge)
        # pre_charging: [(t-c+1, q_{t-c+1}), ((t-c+2, q_{t-c+2})), ..., (t,q_{t})]
        # State before action at time t
        # Levels: short k<=t-9; mid t-9<k<=t-3; long k >= t-3
        short_charging, long_charging = 0, 0
        if not pre_charging:
            return  short_charging, long_charging

        #charging time = 60 minutes = 12 steps
        mid = math.floor(self.charge_steps/2)

        for t, q in pre_charging.items():
            if t <= time_step - mid:  #t-8
                short_charging += 1
            else:
                long_charging += 1

        return  short_charging, long_charging


    # Check the assignment feasibility by simulate the queuing process
    def temporal_feasibility(self, time_step, current_time, temp_queue):
        # Copy the charging queue
        full_battery = self.full_batteries
        charging_que = copy.deepcopy(self.pre_charging)
        charging_que.add(time_step, 0)
        if env_params['test_mode'] and (len(charging_que) > self.charge_steps):
            print('######## Error: charging queue exceeds length limit! ########')

        #sort the requests by arrival time
        num_que = len(temp_queue)
        if num_que > env_params['max_queue']:
            return False

        checking = False
        temp_queue[:] = self.simul_queue(current_time, full_battery, temp_queue, charging_que, checking)

        #Check if all requests' waiting time are feasible
        wait_times = np.array([req.wait_time   for req in temp_queue])
        temp_feasb = (np.max(wait_times) <= env_params['max_wait_time'])

        return  temp_feasb


    ##Simulate the queuing process for a bundle assignment
    def simulate_bundle(self, time_step, current_time, bundle_reqs, charge, num_new, num_clusters):
        # Check full_batteries
        full_battery = self.full_batteries

        # Process charging actions
        charging_que = copy.deepcopy(self.pre_charging)
        charging_que.add(time_step, 0)
        if len(charging_que) > self.charge_steps:
            print('######## Error: charging queue exceeds length limit! ########')

        # Execute temporary actions: if new demand and a bundle exist, add corresponding requests
        temp_que = copy.deepcopy(self.queue)
        temp_que.extend(bundle_reqs)

        # Simulate queue processing
        assign_id = [req.id   for req in bundle_reqs]
        checking = True
        temp_que[:] = self.simul_queue(current_time, full_battery, temp_que, charging_que, checking)
        services = [req   for req in temp_que if req.id in assign_id]

        # Check services
        if env_params['test_mode']:
            serve_id = [req.id   for req in services]
            que_id = [req.id   for req in temp_que]
            for rid in assign_id:
                if (rid not in serve_id) or (rid not in que_id):
                    print(f'######### Error: assigned request {rid} is NOT included in services #########')

        # Update the charging queue
        charging_que[time_step] = charge
        charging_batteries = int(charging_que.sum_all())
        if charging_batteries > self.chargers:
            print(f'######## Error: Charging exceeds maximum slots! ########')

        #Service results
        avg_waiting = 0
        next_time = current_time + timedelta(minutes=env_params['delta_time'])
        if temp_que:
            wait = np.array([req.wait_time   for req in temp_que])
            avg_waiting = np.mean(wait)

            completed = [req   for req in temp_que if req.departure <= next_time]
            full_battery = self.full_batteries - len(completed)

            # The updated queue after removing completed requests
            temp_que[:] = [req   for req in temp_que  if req.departure > next_time]

        # full batteries at time t+1 without including q_{t+1-C}
        avail_batteries = full_battery

        #Next state
        next_time_step = time_step + 1
        next_hour = next_time.hour
        que_order = len(temp_que)
        short_charging, long_charging = self.battery_to_full(charging_que, next_time_step)

        if next_time_step >= self.charge_steps:  # nonempty case
            t0, qt = charging_que.get_first() # t0 = time_step + 1  - C
            # add the full battery just completed at time step t if any
            full_battery += qt

        inventory_position = (avail_batteries + charging_que.sum_all() - que_order)

        # Reward at time t
        revenue, utilities = self.calculate_reward(current_time, services, charge)

        # Check terminals status
        done = (next_time_step >= self.total_steps)
        if done:
            next_time_step = self.total_steps - 1
            next_hour = self.max_hour

        # State at time t+1
        # Agent: ('time_step', 'hour', 'cluster', 'nearby_stations', 'new_order',
        # 'cluster_demand', 'que_order', 'avg_waiting', 'full_batteries', 'inventory_position')
        next_obs = {'time_step': next_time_step,
                    'hour': next_hour,
                    'cluster': self.cluster,
                    'nearby_stations': self.nearby_stations,
                    'new_order': num_new,
                    'cluster_demand': num_clusters,
                    'que_order': que_order,
                    'avg_waiting': avg_waiting,
                    'full_batteries': full_battery,
                    'inventory_position': inventory_position
                    }

        # End
        return  next_obs, done, revenue, utilities


    # Estimate the travel distance from request to station
    def get_travel_distance(self, req):
        dist = geodesic((req.latitude, req.longitude),
                        (self.latitude, self.longitude)).km
        return  dist


    def get_travel_time(self, req):
        dist = geodesic((req.latitude, req.longitude),
                        (self.latitude, self.longitude)).km
        #'vehicle_speed': 20 km / h
        trv = env_params['travel_time_factor'] * (dist / env_params['vehicle_speed']) * 60 # minutes
        return  trv


    # Calculate the electricity charging cost
    def calculate_charging(self, current_time, charged):
        peak_hours = env_params['peak_electricity']
        gamma = env_params['gamma']

        #Calculate the charging power per time step
        delta_power = env_params['power_capacity'] / self.charge_steps #75 kwh, 75 / 12 = 6.25 kwh

        start_time = current_time
        end_time = current_time + timedelta(minutes=self.charge_time)
        price = 0.0  # electricity charging cost for a single battery

        while start_time < end_time:
            hour = start_time.hour  # 计算当前时间在小时维度的表示
            next_time = start_time + timedelta(minutes=env_params['delta_time'])  # 以delta_time为步长

            # 判断当前时间是否在高峰时段
            is_peak = (hour >= peak_hours[0]) and (hour <= peak_hours[1])
            rate = env_params['electric_price_peak']  if is_peak else env_params['electric_price_off']

            price += gamma * rate * delta_power
            start_time = next_time

        charge_cost = price * charged
        return  charge_cost


    def final_summary(self):
        print(f"Station {self.id}: orders served {self.served}, batteries charged {self.charged}, "              
              f"total revenues {self.revenues}.")

