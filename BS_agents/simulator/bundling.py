'''
This is the data structure of bundles
A bundle is a set of requests
'''

import os
import copy
import time
import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from config import env_params, RL_params


class Bundling:
    def __init__(self, env):
        self.environment = env  #input the environment
        self.max_size = env_params['max_bundle']

        self.bundles = []  # 存储唯一组合包（按订单ID排序的元组）
        self.bundle_stations = []  # 每个bundle对应的可行站点列表
        # bundle -> index 的哈希映射
        self.bundle_to_index = {} #{bundle: bid}

        #Indicator matrices and feasible pairs
        self.BS = {}  # BS[(b, j)]
        self.BR = {}  # BR[(b, i)]
        self.BS_pairs = []  # (b, j)
        self.BR_pairs = []  # (b, i)


    def start_bundling(self, episode):
        t0 = time.perf_counter()
        # Generate bundles for requests
        if self.environment.new_demand:
            self.generate_bundles()
            self.set_indicators()
            self.check_indicators()
        else:
            self.bundles[:] = []
            self.bundle_stations[:] = []
            self.bundle_to_index.clear()

        # End
        t1 = time.perf_counter()
        run_time = (t1 - t0) / 60  # Build time in minutes
        if env_params['test_mode'] or len(self.bundles) >= 500:
            max_new, max_cls, max_que = self.environment.station_records()
            print(f'---------- State at time {self.environment.time_step}, episode {episode} ----------\n'
                  f'new demand {len(self.environment.new_demand)}, bundles {len(self.bundles)}; '
                  f'agents: max demand {max_new}, clusters {max_cls}, queue {max_que}.')
            print(f'The runtime for generating bundles: {run_time:.2f} minutes.')


    ## 生成所有可行组合包，并构建BS和BR矩阵
    def generate_bundles(self):
        '''
        A bundle consists of request ids in new_demand
        new_demand:
        index: 0, 1, 2, ..., num_reqs-1
        id: id0, id0+1, id0+2, id0+num_reqs-1
        '''
        self.bundles[:] = []  # 存储唯一组合包（按订单ID排序的元组）
        self.bundle_stations[:] = []  # 每个bundle对应的可行站点列表
        self.bundle_to_index.clear()

        for agent in self.environment.agents:
            ##available requests for the stations
            num_new, num_clusters, demand = self.environment.get_agent_demand(agent, self.environment.new_demand)
            if num_new == 0:
                continue

            # 提取订单ID并生成所有组合
            req_ids = [req.id   for req in demand]
            agent_bundles = []
            for k in range(1, self.max_size + 1):
                agent_bundles.extend(itertools.combinations(req_ids, k))  #tuple data

            # 检查每个组合的可行性
            for bundle in agent_bundles:
                #等待时间可行性
                temp_feasib = self.bundle_feasibility(agent, bundle)
                if temp_feasib:
                    sorted_bundle = tuple(sorted(bundle))
                    #check membership in keys
                    if sorted_bundle not in self.bundle_to_index:
                        bid = len(self.bundles)
                        self.bundles.append(sorted_bundle)
                        self.bundle_to_index[sorted_bundle] = bid
                        #初始化一个列表，将当前站点 agent.id 作为第一个支持该组合包的站点
                        self.bundle_stations.append([agent.id])
                    else:
                        #获取当前组合包 sorted_bundle 在全局唯一组合包列表 bundles 中的索引 bid
                        bid = self.bundle_to_index[sorted_bundle]
                        # 检查当前站点 agent.id 是否已记录在 bundle_stations[bid] 中
                        # 若未记录，将 agent.id 添加到 bundle_stations[bid]
                        if agent.id not in self.bundle_stations[bid]:
                            self.bundle_stations[bid].append(agent.id)
        #End


    # Check the bundle assignment feasibility by simulate the queuing process
    def bundle_feasibility(self, agent, bundle):
        ''' bundle can be either a list or tuple data of requests ids '''
        index = {req.id: i   for i, req in enumerate(self.environment.new_demand)}
        temp_queue = copy.deepcopy(agent.queue)
        ##Double check list data
        reqs = [self.environment.new_demand[index[rid]]   for rid in bundle]
        temp_queue.extend(reqs)

        #Check time feasibility
        temp_feasib = agent.temporal_feasibility(self.environment.time_step,
                                                 self.environment.current_time,
                                                 temp_queue)
        #End
        return  temp_feasib


    ##Construct the indicator matrix for bundles, stations, and requests
    def set_indicators(self):
        '''
        bundles = [(1, 2), (3, 4)]
        bundle_stations = [[0, 1], [2]]  # 组合包 (1,2) 被站点0和1支持
        new_demand:
        index: 0, 1, 2, ..., num_reqs-1
        id: id0, id0+1, id0+2, id0+num_reqs-1
        '''
        index = {req.id: i   for i, req in enumerate(self.environment.new_demand)}

        #Sparse data structure
        self.BS = {}  #BS[(b, j)]
        self.BR = {}  #BR[(b, i)]
        self.BS_pairs = []  #(b, j)
        self.BR_pairs = []  #(b, i)

        #Case: nonempty bundles
        if self.bundles:
            for b, (bundle, stations) in enumerate(zip(self.bundles, self.bundle_stations)):
                for j in stations:
                    self.BS[(b, j)] = 1
                    self.BS_pairs.append((b, j))

                for rid in bundle:  #order id is arranged within the new demand
                    i = index[rid] #index in range(num_reqs)
                    self.BR[(b, i)] = 1
                    self.BR_pairs.append((b, i))


    def check_indicators(self):
        '''
        BS: (num_stations, num_bundles)
        BR: (num_reqs, num_bundles)
        '''
        num_bundles = len(self.bundles)
        num_stations = len(self.environment.agents)
        num_reqs = len(self.environment.new_demand)

        for b in range(num_bundles):
            bundle = self.bundles[b]
            station_ids = self.bundle_stations[b]
            for j in range(num_stations):
                case_1 = (self.BS.get((b, j), 0) == 0) and (j in station_ids)
                case_2 = (self.BS.get((b, j), 0) > 0) and (j not in station_ids)
                if case_1 or case_2:
                    print(f'###### Error in BS indicator {(b, j)}: '
                          f'{self.BS.get((b, j), 0)}; {station_ids} ######')

            for i in range(num_reqs):
                req = self.environment.new_demand[i]
                rid = req.id  # index in range(num_reqs)
                case_1 = (self.BR.get((b, i), 0) == 0) and (rid in bundle)
                case_2 = (self.BR.get((b, i), 0) > 0) and (rid not in bundle)
                if case_1 or case_2:
                    print(f'###### Error in BR indicator {(b, i, rid)}: '
                          f'{self.BR.get((b, i), 0)}; {bundle} ######')


    #Replace element rm with ri in a given bundle
    def replace_bundle(self, bundle, ri, rm):
        #Case: i and m are both in the bundle
        #b = self.bundles.index(bundle)  #tuple data
        b = self.bundle_to_index[bundle]

        if (rm in bundle) and (ri in bundle):
            rbundle = bundle
            bid = b
            flag = True
            return  flag, bid, rbundle

        #Case: i and m are NOT in the bundle, construct the replaced bundle
        #Replace rm with ri in bundle
        if (rm in bundle) and (ri not in bundle):
            ind = bundle.index(rm)
            bx = bundle[:ind] + (ri,) + bundle[ind + 1:] #A new list
            rbundle = tuple(sorted(bx))
            flag = (rbundle in self.bundles)
        else:  #Exception case: not useful
            rbundle = ()
            flag = False

        if flag:
            #bid = self.bundles.index(rbundle)
            bid = self.bundle_to_index[rbundle]
        else:
            bid = -1  # replaced bundle is not feasible

        #End
        checking = (rm not in self.bundles[b]) or (ri not in self.bundles[bid])
        if flag and checking:
            print(f'############ Error: bundles are NOT correct for replace {rm} --> {ri} ############')
            print(f'Initial bundle from {rm}: {bundle}, {self.bundles[b]};')
            print(f'Replaced bundle for {ri}: {rbundle}, {self.bundles[bid]}.')

        # End
        return  flag, bid, rbundle


    #Compute the cost of bundle after replacement
    def replace_cost(self, costs):
        ''' i = index[rid]; costs[(b,j,i)] = utilities[rid] '''
        index = {req.id: i  for i, req in enumerate(self.environment.new_demand)}
        rcosts = {}  #[(b, j, i, m)]

        for cls in self.environment.Clusters:
            req_id = [req.id   for req in self.environment.new_demand if req.cluster == cls.id]
            num = len(req_id)

            if num > 1:
                for ki in range(num):
                    ri = req_id[ki]
                    i = index[ri]

                    for km in range(num):
                        if ki != km:
                            rm = req_id[km]
                            m = index[rm]

                            #Search the costs for replacing m with i in the bundle
                            for (b, j) in self.BS_pairs:
                                bundle = self.bundles[b]
                                flag, bid, rbundle = self.replace_bundle(bundle, ri, rm)
                                if flag and self.BS.get((bid, j), 0) > 0:
                                    # print(f'Costs: {costs[(b, j, m)]}, {costs[(bid, j, i)]}')
                                    # costs[(bid, j, i)]
                                    rcosts[(b, j, i, m)] = costs.get((bid,j,i), env_params['big_M'])
                                else:
                                    rcosts[(b, j, i, m)] = env_params['big_M']
        #End
        return  rcosts


    # Check the solution for validity
    def check_solution(self, bundle_assign):
        #bundle_assign: {j: bundle (tuple data)}
        num_bundles = len(self.bundles)
        num_stations = len(self.environment.agents)
        num_reqs = len(self.environment.new_demand)
        index = {req.id: i   for i, req in enumerate(self.environment.new_demand)}

        station_cap = np.zeros(num_stations, dtype=int)
        req_cap = np.zeros(num_reqs, dtype=int)
        bundle_cap = np.zeros(num_bundles, dtype=int)

        #bundle_assign {j: bundle}
        for j, bundle in bundle_assign.items():
            b = self.bundle_to_index[bundle]
            station_cap[j] += 1
            bundle_cap[b] += 1
            for rid in bundle:
                i = index[rid]
                req_cap[i] += 1

        if np.max(station_cap) > 1:
            print(f'###### Station capacity violation: {station_cap} ######')

        if np.max(req_cap) > 1:
            print(f'###### Request capacity violation: {req_cap} ######')

        if np.max(bundle_cap) > 1:
            print(f'###### Bundle capacity violation: {bundle_cap} ######')


    #Set utilities from assignment solution
    def set_utilities(self, assignment, bundle_assign):
        penalty = env_params['penalty']
        num_reqs = len(self.environment.new_demand)
        index = {req.id: i for i, req in enumerate(self.environment.new_demand)}
        costs, rcosts = {}, {}  # dictionary {rid: cost}

        # Case: empty demand
        if num_reqs == 0:
            return  costs, rcosts

        # Case: nonempty demand
        for j, agent in enumerate(self.environment.agents):
            ## bundle = [req.id:  i, req, assignment[i] == j]
            bundle = bundle_assign.get(j, []) # {j: bundle}
            bundle_reqs = [self.environment.new_demand[index[rid]]   for rid in bundle]

            charge, num_new, num_clusters = 0, 0, 0
            if bundle_reqs:
                _, _, _, utilities = agent.simulate_bundle(self.environment.time_step,
                                                        self.environment.current_time,
                                                        bundle_reqs, charge, num_new,
                                                        num_clusters)
                # dictionary {rid: cost}
                costs.update(utilities)

        # Rejected requests
        for i, req in enumerate(self.environment.new_demand):
            rid = req.id
            if assignment[i] < 0:
                costs[rid] = penalty

        # Double check
        keys_cost = list(costs.keys())
        for req in self.environment.new_demand:
            if req.id not in keys_cost:
                print(f'########### Request {req.id} is NOT included in greedy utility computation! ###########')

        # Compute the costs of replacement in assignment
        for cls in self.environment.Clusters:
            req_id = [req.id   for req in self.environment.new_demand  if req.cluster == cls.id]
            num = len(req_id)

            if num > 1:
                for ki in range(num):
                    ri = req_id[ki]
                    i = index[ri]

                    for km in range(num):
                        if ki != km:
                            rm = req_id[km]
                            m = index[rm]

                            # replaced bundle for i, m
                            jm = assignment[m]
                            if jm >= 0:
                                agent = self.environment.agents[jm]
                                bundle = copy.deepcopy(bundle_assign[jm]) #List data
                                rbundle = list(bundle)
                                rbundle.remove(rm)
                                rbundle.append(ri)
                            else:
                                agent = self.environment.agents[0]
                                rbundle = [ri]

                            #Check feasibility and utilities
                            if (jm >= 0) and self.bundle_feasibility(agent, rbundle):
                                bundle_reqs = [self.environment.new_demand[index[rid]]  for rid in rbundle]
                                charge, num_new, num_clusters = 0, 0, 0
                                _, _, _, utilities = agent.simulate_bundle(self.environment.time_step,
                                                                           self.environment.current_time,
                                                                           bundle_reqs, charge,
                                                                           num_new, num_clusters)
                                # dictionary {rid: cost}
                                rcosts[(i, m)] = utilities[ri]
                            else:
                                rcosts[(i, m)] = env_params['big_M']
        #End
        return  costs, rcosts


    # Set subsidies by a linear program
    def set_subsidies(self, assignment, bundle_assign):
        M = env_params['big_M']
        num_reqs = len(self.environment.new_demand)
        index = {req.id: i  for i, req in enumerate(self.environment.new_demand)}

        #Case: empty demand
        if num_reqs == 0:
            return  np.array([], dtype=float)

        # Case: nonempty demand
        costs, rcosts = self.set_utilities(assignment, bundle_assign)

        # Set up the model
        LP = gp.Model("LP")

        # Define decision variables
        p = {}  # subsidy
        for i in range(num_reqs):
            p[i] = LP.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_{i}")
        LP.update()

        # Constraints: envy-freeness with subsidies
        for cls in self.environment.Clusters:
            req_id = [req.id   for req in self.environment.new_demand if req.cluster == cls.id]
            num = len(req_id)

            if num > 1:
                for ki in range(num):
                    ri = req_id[ki]
                    i = index[ri]
                    cost_i = costs.get(ri, M)

                    for km in range(num):
                        if ki != km:
                            rm = req_id[km]
                            m = index[rm]
                            cost_m = costs.get(rm, M)

                            if cost_i > cost_m:
                                rcosts_im = rcosts.get((i, m), 1e6)
                                LP.addConstr(
                                    cost_i - p[i] <= rcosts_im - p[m],
                                    name=f"envy_{i}_{m}_constraint"
                                )

        # Objective function
        LP.setObjective(
            gp.quicksum(p[i] for i in range(num_reqs)),
            GRB.MINIMIZE
        )

        # Set optimization parameters
        LP.Params.OutputFlag = 0
        LP.Params.TimeLimit = RL_params['GUROBI_TIME_LIMIT']
        LP.Params.Threads = int(RL_params['GUROBI_Threads'])
        LP.Params.Seed = int(RL_params['GUROBI_SEED'])

        # Solve the model
        try:
            LP.optimize()

        except gp.GurobiError as e:
            print(f"######## Gurobi optimization failed: {e} ########")
            return  np.array([], dtype=float)

        status_dict = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.TIME_LIMIT: "TIME LIMIT"
        }
        sol_status = status_dict.get(LP.status, "Other status")

        # Get the solution
        subsidy = np.zeros(num_reqs, dtype=float)
        if (LP.status == GRB.OPTIMAL) or (LP.status == GRB.SUBOPTIMAL):
            # Get the subsidy solution
            for i in range(num_reqs):
                pi = p[i].X
                if abs(p[i].X) < 1e-4:
                    pi = 0
                subsidy[i] = pi
        else:
            print(f'######## Gurobi optimization status for subsidy: {sol_status} ########')

        # End
        return  subsidy

