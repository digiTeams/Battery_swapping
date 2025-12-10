'''
This is the algorithm for greedy baseline policy:
At each time, search feasible nearest station, and charge as many as possible
'''

from geopy.distance import geodesic
import copy
import time
import math
import numpy as np

# Load configuration parameters (assumed to be defined in config)
from config import data_params, env_params, RL_params



class GreedyPolicy:
    def __init__(self, env):
        self.environment = env  # 绑定仿真环境
        self.gamma = env_params['gamma']
        self.type = 'Fair'


    #Find the stations within a given distance
    def assign_station(self, assigned, request):
        candidates = []
        for agent in self.environment.agents:
            dist = geodesic((request.latitude, request.longitude),
                            (agent.latitude, agent.longitude)).km
            if dist <= request.nearest_distance + env_params['max_detour']:
                agent.dist = dist
                candidates.append(agent)

        if candidates:
            #sort the stations by distance and id for stablized order
            candidates.sort(key=lambda x: (x.dist, x.id))

            # 计算每个站点的等待时间
            for agent in candidates:
                j = agent.id
                feasb = self.assign_feasibility(assigned, agent, request)
                # Select the nearest feasible station
                if feasb:  #feasible
                    return j
        # End
        return  -1  #no feasible


    # Check the assignment feasibility by simulate the queuing process
    def assign_feasibility(self, assigned, agent, request):
        j = agent.id
        if assigned[j] >= env_params['max_assign']:
            return  False

        # Temporarily add into the queue
        queue_copy = copy.deepcopy(agent.temp_queue)
        queue_copy.append(request)
        temp_feasb = agent.temporal_feasibility(self.environment.time_step,
                                                self.environment.current_time,
                                                queue_copy)
        #End
        return  temp_feasb


    # Greedy policy for assigning stations to EVs
    def greedy_assign(self):
        # requests within time [t-1, t)
        num_reqs = len(self.environment.new_demand)
        num_stations = len(self.environment.agents)
        assigned = np.zeros(num_stations, dtype=int)

        assignment = np.full(num_reqs, -1)  #empty if num_reqs = 0
        bundle_assign = {}  # j: bundle
        if num_reqs == 0:
            return  assignment, bundle_assign

        #Set the temporary queue and charging of agents
        for agent in self.environment.agents:
            agent.temp_queue[:] = copy.deepcopy(agent.queue)

        for i, req in enumerate(self.environment.new_demand):
            sid = self.assign_station(assigned, req)
            if sid >= 0:
                assignment[i] = sid
                assigned[sid] += 1
                #update the queue of agents
                agent = self.environment.agents[sid]
                agent.temp_queue.append(req)

        # Check solution
        for j, agent in enumerate(self.environment.agents):
            bundle = [req.id   for i, req in enumerate(self.environment.new_demand)
                      if assignment[i] == j]
            if bundle:
                bundle_assign[j] = bundle  #list data

        if env_params['test_mode'] and self.environment.new_demand:
            print(f'Greedy station-bundle assignments: {bundle_assign}.')

        #End
        return  assignment, bundle_assign


    # Greedy policy for charging
    def greedy_charging(self):
        num_stations = len(self.environment.agents)
        base_stocks = RL_params['base_stocks']

        charging = np.zeros(num_stations, dtype=int)
        for j, agent in enumerate(self.environment.agents):
            k = self.environment.set_base_level(agent)
            base_level = math.ceil(base_stocks[k] * agent.capacity)
            replen = max(base_level - agent.inventory_position, 0)
            idle_slot = max(agent.chargers - agent.charging_batteries, 0)
            charging[j] = min(agent.empty_batteries, idle_slot, replen)

        # End
        return  charging


    # Greedy policy for set subsidies by a linear program
    def greedy_subsidy(self, bundling, assignment, bundle_assign):
        num_reqs = len(self.environment.new_demand)
        #Case: empty demand
        if num_reqs == 0:
            return  np.array([], dtype=float)

        # Case: nonempty demand
        subsidy = bundling.set_subsidies(assignment, bundle_assign)

        # End
        if env_params['test_mode'] and self.environment.new_demand:
            print(f'Greedy subsidies: mean {np.mean(subsidy):.2f}, '
                  f'max {np.max(subsidy):.2f}.')
        return  subsidy


    def greedy_schedule(self, bundling):
        assignment, bundle_assign = self.greedy_assign()
        charging = self.greedy_charging()
        subsidy = self.greedy_subsidy(bundling, assignment, bundle_assign)

        action = {"assignment": assignment,
                  "charging": charging,
                  "subsidy": subsidy}

        # End
        return  action


    # Running process in an episode
    def scheduling(self, bundling, train_flag, episode):
        start_run = time.perf_counter()

        # Initialize simulator
        self.environment.reset()
        state = self.environment.get_states()
        done = False

        # Start simulation process
        while not done:
            #Get the beginning state
            max_new, max_cls, max_que = self.environment.station_records()
            print(f'---------- State at time {self.environment.time_step}, episode {episode} ----------\n'
                  f'new demand {len(self.environment.new_demand)}, agents max demand {max_new}, '
                  f'clusters {max_cls}, queue {max_que}.')

            # select a greedy schedule action
            action = self.greedy_schedule(bundling)

            # execute action and transit to next state
            next_state, reward, done = self.environment.step(action, self.type)
            state = next_state

        ### Ending the greedy baseline policy ###
        end_run = time.perf_counter()
        run_time = (end_run - start_run) / 60 # running time
        avg_loss = 0

        # Result
        result = self.environment.service_records(episode)
        result.update({'run_time': run_time, 'loss': avg_loss})
        if not train_flag:
            self.environment.request_records('greedy', episode)

        # Summary
        print(f"*************** Fair greedy policy in episode {episode} *************** \n"
              f"total demand {len(self.environment.all_demand)}, rewards {result['total_rewards']:.2f} CNY, "
              f"revenues {result['total_revenues']:.2f} CNY, subsidies {result['total_subsidies']:.2f} CNY; \n"
              f"served {result['served']}, charged {result['charged']}, rejection {result['rejection']:.2f}, "
              f"running time {run_time:.2f} minutes.")

        # End
        return  result
