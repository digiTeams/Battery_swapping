#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
the distributed DRLPolicy class for speeding up deep reinforcement learning training.
"""


import os
import math
import time
from datetime import datetime, timedelta
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gurobipy as gp
from gurobipy import GRB
from sympy import print_fcode

# Load configuration parameters (assumed to be defined in config)
from config import data_params, env_params, RL_params
from DRL.valueNet import ValueNetwork

num_cores = os.cpu_count()

# ---------------------------
# Replay Buffer Classes
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity, batch_size, num_agents, obs_dim):
        self.capacity = capacity
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.buffer = deque(maxlen=capacity)


    def push(self, state_array: np.ndarray, reward: np.ndarray,
             next_state_array: np.ndarray, done: bool):
        """
        Store a transition:
            state_array: shape [num_agents, obs_dim]
            reward:      shape [num_agents]
            next_state_array: shape [num_agents, obs_dim]
            done:        scalar bool
        """
        self.buffer.append((state_array, reward, next_state_array, done))


    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, rewards, next_states, dones = zip(*batch)

        # Stack and convert to torch tensors
        state_batch = torch.tensor(np.stack(states), dtype=torch.float32)  # [B, num_agents, obs_dim]
        reward_batch = torch.tensor(np.stack(rewards), dtype=torch.float32)  # [B, num_agents]
        next_state_batch = torch.tensor(np.stack(next_states), dtype=torch.float32)  # [B, num_agents, obs_dim]
        done_batch = torch.tensor(dones, dtype=torch.float32)  # [B]

        #End
        return  state_batch, reward_batch, next_state_batch, done_batch


    def __len__(self):
        return  len(self.buffer)


# ---------------------------
# DRLPolicy Class
# ---------------------------
class FairnessDRL:
    def __init__(self, env, state_dims):
        # Bind the simulation environment (not included in this file)
        self.environment = env
        self.gamma = env_params['gamma']
        self.type = 'Fair'

        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

        ''' 
        State space at each time t:
        # Agent: ('time_step', 'hour', 'id', 'cluster', 'new_order', 'cluster_demand', 'que_order', 
        # 'avg_waiting', 'full_batteries', 'inventory_position', 'short_charging', 'long_charging')
        self.state_dims = {'steps': steps, 'hours': hours, 'stations': num_agents,
                           'clusters': num_clusters, 'observe_dim': 11, 
                           'discrete': 3, 'continuous': 8}
        '''
        self.state_dims = state_dims
        num_agents = int(self.state_dims['stations'])
        obs_dim = int(self.state_dims['observe_dim'])

        #Main network parameters
        self.learning_rate = RL_params['initial_learning']
        self.epsilon = RL_params['initial_epsilon']
        self.clips = RL_params['clip_norm']

        #Target network parameters
        self.tau = RL_params['tau']  #soft update of taget network: 1.0, 0.01
        self.update_freq = RL_params['update_freq'] #period of steps for taget network update

        #Buffer parameters
        buffer_size = int(RL_params['buffer_size'])
        batch_size = int(RL_params['batch_size'])
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size, num_agents, obs_dim)

        # Initialize main and target value networks
        self.value_net = ValueNetwork(state_dims).to(self.device)
        self.target_net = ValueNetwork(state_dims).to(self.device)
        # Copy parameters from value_net to target_net
        self.target_net.load_state_dict(self.value_net.state_dict())

        self.value_net.train() # 确保主网在 train 模式
        self.target_net.eval()  #初始化设置为.eval() 模式

        # Optimizer and loss
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        #Mean Squared Error Loss: loss = (1/(B*N)) * Σ_{b=1}^{B} Σ_{n=1}^{N} (predicts[b,n] - td_target[b,n])^2
        self.loss_fn = nn.MSELoss(reduction='mean')  # sum over agents and batch

        # Internal step counter
        self.step_count = 0
        self.opt_act = 0
        self.rand_act = 0


    # Compute cost coefficients of BILP
    def BILP_costs(self, bundling, scale_):
        t0 = time.perf_counter()
        num_agents = len(self.environment.agents)

        # Collect agents' data
        next_time = self.environment.current_time + timedelta(minutes=self.environment.delta_time)
        next_demand = self.environment.get_new_requests(next_time)
        agent_news = np.zeros(num_agents, dtype=int)
        agent_clusters = np.zeros(num_agents, dtype=int)
        for j, agent in enumerate(self.environment.agents):
            num_new, num_clusters, _ = self.environment.get_agent_demand(agent, next_demand)
            agent_news[j] = num_new
            agent_clusters[j] = num_clusters

        #Bundle assignment costs:
        # qvals[(b, j)] = reward; costs[(b, j, i)] = util;
        # rcosts[(b, j, i, m)] = costs[(bid, j, i)]
        qvals, costs, rcosts = self.bundle_costs(bundling, agent_news, agent_clusters, scale_)

        #Empty bundle assignment costs:
        #cvals[(k, j)] = reward;
        cvals = self.empty_costs(agent_news, agent_clusters, scale_)

        # End
        t1 = time.perf_counter()
        eval_time = (t1 - t0) / 60  # Build time in minutes
        if env_params['test_mode']:
            print(f'The processing time for evaluating bundle costs: {eval_time:.2f} minutes.')

        return  qvals, cvals, costs, rcosts


    # Compute costs of bundles
    def bundle_costs(self, bundling, agent_news, agent_clusters, scale_):
        base_stocks = RL_params['base_stocks']
        num_levels = len(base_stocks)
        index = {req.id: i   for i, req in enumerate(self.environment.new_demand)}

        # Case: empty demand
        if not self.environment.new_demand:
            return  {}, {}, {}

        # Case: none empty demand
        # qvals[(b, j)] = reward; costs[(b, j, i)] = util;
        # rcosts[(b, j, i, m)] = costs[(bid, j, i)]
        qvals, costs, rcosts = {}, {}, {}
        for (b, j) in bundling.BS_pairs:
            bundle = bundling.bundles[b]
            agent = self.environment.agents[j]

            bundle_reqs = [self.environment.new_demand[index[rid]]   for rid in bundle]
            num_new = agent_news[j]
            num_clusters = agent_clusters[j]

            for k in range(num_levels):
                base_level = math.ceil(base_stocks[k] * agent.capacity)
                replen = max(base_level - agent.inventory_position, 0)
                idle_slot = max(agent.chargers - agent.charging_batteries, 0)
                charge = min(agent.empty_batteries, idle_slot, replen)

                #simulate queue process
                next_obs, done, revenue, utilities = agent.simulate_bundle(self.environment.time_step,
                                                                     self.environment.current_time,
                                                                     bundle_reqs, charge,
                                                                     num_new, num_clusters)

                #Q-values of agent
                #reward = service fees - charge costs
                #qval = revenue + gamma * V(next_obs)
                qvals[(b,k,j)] = self.agent_Qval(scale_, next_obs, done, revenue)

                # per-request cost map
                for rid, util in utilities.items():
                    i = index[rid]
                    costs[(b,j,i)] = util

        ##Compute the envy costs for replacing m with i in bundle b
        #rcosts[(b, j, i, m)] = costs[(bid, j, i)]
        rcosts = bundling.replace_cost(costs)

        # End
        return  qvals, costs, rcosts


    # Compute costs of empty bundles
    def empty_costs(self, agent_news, agent_clusters, scale_):
        base_stocks = RL_params['base_stocks']
        num_levels = len(base_stocks)

        cvals = {}
        for j, agent in enumerate(self.environment.agents):
            bundle_reqs = []
            num_new = agent_news[j]
            num_clusters = agent_clusters[j]

            for k in range(num_levels):
                base_level = math.ceil(base_stocks[k] * agent.capacity)
                replen = max(base_level - agent.inventory_position, 0)
                idle_slot = max(agent.chargers - agent.charging_batteries, 0)
                charge = min(agent.empty_batteries, idle_slot, replen)

                # simulate queue process
                next_obs, done, revenue, utilities = agent.simulate_bundle(self.environment.time_step,
                                                                     self.environment.current_time,
                                                                     bundle_reqs, charge,
                                                                     num_new, num_clusters)
                # Q-values of agent
                # reward = service fees - charge costs
                cvals[(k, j)] = self.agent_Qval(scale_, next_obs, done, revenue)

        # End
        return  cvals


    # Determine scheduling and subsidies simultaneously
    def siml_action_BILP(self, bundling, qvals, cvals, costs, rcosts):
        t0 = time.perf_counter()
        base_stocks = RL_params['base_stocks']
        penalty = env_params['penalty']
        index = {req.id: i   for i, req in enumerate(self.environment.new_demand)}

        num_stations = len(self.environment.agents)
        num_bundles = len(bundling.bundles)
        num_levels = len(base_stocks)
        num_reqs = len(self.environment.new_demand)

        # Set up the model
        BILP = gp.Model("DRL ILP")

        # Define decision variables
        x = {}  # assignment-charging (b,k,j)
        x0 = {} # empty bundle assign (k,j)
        y = {}  # rejection
        p = {}  # subsidy
        for j in range(num_stations):
            for b in range(num_bundles):
                for k in range(num_levels):
                    x[b,k,j] = BILP.addVar(vtype=GRB.BINARY, name=f"x_{b}_{k}_{j}")
        for j in range(num_stations):
            for k in range(num_levels):
                x0[k,j] = BILP.addVar(vtype=GRB.BINARY, name=f"x0_{k}_{j}")
        for i in range(num_reqs):
            y[i] = BILP.addVar(vtype=GRB.BINARY, name=f"y_{i}")
            p[i] = BILP.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_{i}")
        BILP.update()

        #Constraints 1: each agent receives one bundle
        for j in range(num_stations):
            BILP.addConstr(
                gp.quicksum(bundling.BS.get((b, j), 0) * x[b,k,j]
                            for b in range(num_bundles)  for k in range(num_levels))
                + gp.quicksum(x0[k,j]   for k in range(num_levels))
                == 1,
                name=f"agent_{j}_constraint"
            )

        # Constraints 2: each bundle is allocated at most once
        for b in range(num_bundles):
            BILP.addConstr(
                gp.quicksum(bundling.BS.get((b,j), 0) * x[b,k,j]
                            for k in range(num_levels)  for j in range(num_stations))
                <= 1,
                name=f"bundle_{b}_constraint"
            )

        # Constraints 3: each request is served at most once
        for i in range(num_reqs):
            BILP.addConstr(
                gp.quicksum(bundling.BR.get((b, i), 0) * x[b,k,j]
                            for (b,j) in bundling.BS_pairs  for k in range(num_levels))
                + y[i] == 1,
                name=f"request_{i}_constraint"
            )

        # Constraints 4: at most one base level can be selected
        for (b, j) in bundling.BS_pairs:
            BILP.addConstr(
                gp.quicksum(x[b,k,j]   for k in range(num_levels))
                <= 1,
                name=f"level_{b}_{j}_constraint"
            )
        for j in range(num_stations):
            BILP.addConstr(
                gp.quicksum(x0[k,j]   for k in range(num_levels))
                <= 1,
                name=f"level_{j}_constraint"
            )

        # Constraints 5: envy-freeness with subsidies
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

                            # Check the replaced bundle b1 = b\{m}U{i}
                            BILP.addConstr(
                                gp.quicksum(
                                    bundling.BR.get((b,i), 0) * costs.get((b,j,i), env_params['big_M']) * x[b,k,j]
                                    for (b, j) in bundling.BS_pairs  for k in range(num_levels))
                                + penalty * y[i] - p[i]
                                <= gp.quicksum(
                                    bundling.BR.get((b,m), 0) * rcosts.get((b,j,i,m), env_params['big_M']) * x[b,k,j]
                                    for (b, j) in bundling.BS_pairs  for k in range(num_levels))
                                + penalty * y[m] - p[m],
                                name=f"envy_{i}_{m}_constraint"
                            )

        # Set objective coefficients
        BILP.setObjective(
            gp.quicksum(qvals.get((b,k,j), 0) * x[b,k,j]
                        for (b, j) in bundling.BS_pairs  for k in range(num_levels))
            + gp.quicksum(cvals.get((k,j), 0) * x0[k,j]
                          for k in range(num_levels)  for j in range(num_stations))
            - gp.quicksum(p[i]   for i in range(num_reqs)),
            GRB.MAXIMIZE
        )

        t1 = time.perf_counter()
        build_time = (t1 - t0) / 60  # Build time in minutes

        #Set optimization parameters
        BILP.Params.OutputFlag = 0
        BILP.Params.TimeLimit = RL_params['GUROBI_TIME_LIMIT']
        BILP.Params.MIPGap = RL_params['GUROBI_MIPGAP']
        BILP.Params.Threads = int(RL_params['GUROBI_Threads'])
        BILP.Params.Seed = int(RL_params['GUROBI_SEED'])

        # Solve the model
        try:
            BILP.optimize()
        except gp.GurobiError as e:
            print(f"######## Gurobi optimization failed: {e} ########")
            action = {}
            return  action

        status_dict = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.TIME_LIMIT: "TIME LIMIT"
        }
        sol_status = status_dict.get(BILP.status, "Other status")

        # Get the solution
        assignment = np.full(num_reqs, -1)
        charging = np.zeros(num_stations, dtype=np.int32)
        subsidy = np.zeros(num_reqs, dtype=float)
        bundle_assign = {}  # j: bundle

        if (BILP.status == GRB.OPTIMAL) or (BILP.status == GRB.SUBOPTIMAL):
            # assignment and charging
            for (b, j) in bundling.BS_pairs:
                bundle = bundling.bundles[b]
                agent = self.environment.agents[j]
                for k in range(num_levels):
                    if x[b,k,j].X > 0.5:
                        bundle_assign[j] = bundle
                        for rid in bundle:
                            i = index[rid]
                            assignment[i] = j

                            base_level = math.ceil(base_stocks[k] * agent.capacity)
                            replen = max(base_level - agent.inventory_position, 0)
                            idle_slot = max(agent.chargers - agent.charging_batteries, 0)
                            charging[j] = min(agent.empty_batteries, idle_slot, replen)

            for j, agent in enumerate(self.environment.agents):
                for k in range(num_levels):
                    if x0[k,j].X > 0.5:
                        base_level = math.ceil(base_stocks[k] * agent.capacity)
                        replen = max(base_level - agent.inventory_position, 0)
                        idle_slot = max(agent.chargers - agent.charging_batteries, 0)
                        charging[j] = min(agent.empty_batteries, idle_slot, replen)

            # subsidy solution
            for i in range(num_reqs):
                pi = p[i].X
                if abs(p[i].X) < 1e-4:
                    pi = 0
                subsidy[i] = pi
        else:
            print(f'######## Gurobi optimization status in DRL: {BILP.status}, {sol_status} ########')

        # Check the solution
        if env_params['test_mode'] and bundle_assign:
            bundling.check_solution(bundle_assign)

        if env_params['test_mode'] and self.environment.new_demand:
            print(f'BILP optimization under DRL: status {sol_status}; build time {build_time:.2f} minutes; '
                  f'runtime {BILP.Runtime:.2f} seconds')
            print(f'fairDRL station-bundle assignments: {bundle_assign}.')
            print(f'fairDRL subsidies: mean {np.mean(subsidy):.2f}, max {np.max(subsidy):.2f}.')

        # End
        action = {"assignment": assignment, "charging": charging, "subsidy": subsidy}
        return  action, sol_status


    # Determine assignment and subsidies sequentially
    def seq_action_BILP(self, bundling, qvals, cvals, costs):
        t0 = time.perf_counter()
        base_stocks = RL_params['base_stocks']
        penalty = env_params['penalty']
        index = {req.id: i   for i, req in enumerate(self.environment.new_demand)}

        num_stations = len(self.environment.agents)
        num_bundles = len(bundling.bundles)
        num_levels = len(base_stocks)
        num_reqs = len(self.environment.new_demand)

        # Set up the model
        BILP = gp.Model("DRL ILP")

        # Define decision variables
        x = {}  # assignment-charging (b,k,j)
        x0 = {} # empty bundle assign (k,j)
        y = {}  # rejection
        for j in range(num_stations):
            for b in range(num_bundles):
                for k in range(num_levels):
                    x[b,k,j] = BILP.addVar(vtype=GRB.BINARY, name=f"x_{b}_{k}_{j}")
        for j in range(num_stations):
            for k in range(num_levels):
                x0[k,j] = BILP.addVar(vtype=GRB.BINARY, name=f"x0_{k}_{j}")
        for i in range(num_reqs):
            y[i] = BILP.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        BILP.update()

        #Constraints 1: each agent receives one bundle
        for j in range(num_stations):
            BILP.addConstr(
                gp.quicksum(bundling.BS.get((b,j), 0) * x[b,k,j]
                            for b in range(num_bundles)  for k in range(num_levels))
                + gp.quicksum(x0[k,j]   for k in range(num_levels))
                == 1,
                name=f"agent_{j}_constraint"
            )

        # Constraints 2: each bundle is allocated at most once
        for b in range(num_bundles):
            BILP.addConstr(
                gp.quicksum(bundling.BS.get((b,j), 0) * x[b,k,j]
                            for k in range(num_levels)  for j in range(num_stations))
                <= 1,
                name=f"bundle_{b}_constraint"
            )

        # Constraints 3: each request is served at most once
        for i in range(num_reqs):
            BILP.addConstr(
                gp.quicksum(bundling.BR.get((b,i), 0) * x[b,k,j]
                            for (b,j) in bundling.BS_pairs  for k in range(num_levels))
                + y[i] == 1,
                name=f"request_{i}_constraint"
            )

        # Constraints 4: at most one base level can be selected
        for (b, j) in bundling.BS_pairs:
            BILP.addConstr(
                gp.quicksum(x[b,k,j]   for k in range(num_levels))
                <= 1,
                name=f"level_{b}_{j}_constraint"
            )
        for j in range(num_stations):
            BILP.addConstr(
                gp.quicksum(x0[k,j]   for k in range(num_levels))
                <= 1,
                name=f"level_{j}_constraint"
            )

        # Set objective coefficients
        BILP.setObjective(
            gp.quicksum(qvals.get((b,k,j), 0) * x[b,k,j]
                        for (b, j) in bundling.BS_pairs  for k in range(num_levels))
            + gp.quicksum(cvals.get((k,j), 0) * x0[k,j]
                          for k in range(num_levels)  for j in range(num_stations))
            - gp.quicksum(bundling.BR.get((b, i), 0) * costs.get((b, j, i), env_params['big_M']) * x[b, k, j]
                          for (b, j) in bundling.BS_pairs  for k in range(num_levels) for i in range(num_reqs))
            - gp.quicksum(penalty * y[i]
                          for i in range(num_reqs)),
            GRB.MAXIMIZE
        )

        t1 = time.perf_counter()
        build_time = (t1 - t0) / 60  # Build time in minutes

        #Set optimization parameters
        BILP.Params.OutputFlag = 0
        BILP.Params.TimeLimit = RL_params['GUROBI_TIME_LIMIT']
        BILP.Params.MIPGap = RL_params['GUROBI_MIPGAP']
        BILP.Params.Threads = int(RL_params['GUROBI_Threads'])
        BILP.Params.Seed = int(RL_params['GUROBI_SEED'])

        # Solve the model
        try:
            BILP.optimize()
        except gp.GurobiError as e:
            print(f"######## Gurobi optimization failed: {e} ########")
            action = {}
            return  action

        status_dict = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.TIME_LIMIT: "TIME LIMIT"
        }
        sol_status = status_dict.get(BILP.status, "Other status")

        # Get the solution
        assignment = np.full(num_reqs, -1)
        charging = np.zeros(num_stations, dtype=np.int32)
        bundle_assign = {}  # j: bundle (tuple data)

        if (BILP.status == GRB.OPTIMAL) or (BILP.status == GRB.SUBOPTIMAL):
            # assignment and charging
            for (b, j) in bundling.BS_pairs:
                bundle = bundling.bundles[b]
                agent = self.environment.agents[j]
                for k in range(num_levels):
                    if x[b,k,j].X > 0.5:
                        bundle_assign[j] = bundle  #tuple data
                        for rid in bundle:
                            i = index[rid]
                            assignment[i] = j

                            base_level = math.ceil(base_stocks[k] * agent.capacity)
                            replen = max(base_level - agent.inventory_position, 0)
                            idle_slot = max(agent.chargers - agent.charging_batteries, 0)
                            charging[j] = min(agent.empty_batteries, idle_slot, replen)

            for j, agent in enumerate(self.environment.agents):
                for k in range(num_levels):
                    if x0[k,j].X > 0.5:
                        base_level = math.ceil(base_stocks[k] * agent.capacity)
                        replen = max(base_level - agent.inventory_position, 0)
                        idle_slot = max(agent.chargers - agent.charging_batteries, 0)
                        charging[j] = min(agent.empty_batteries, idle_slot, replen)
        else:
            print(f'######## Gurobi optimization status in DRL: {BILP.status}, {sol_status} ########')

        # Check the solution
        if env_params['test_mode'] and bundle_assign:
            bundling.check_solution(bundle_assign)

        # Set the subsidies
        subsidy = bundling.set_subsidies(assignment, bundle_assign)

        if env_params['test_mode'] and self.environment.new_demand:
            print(f'BILP optimization under DRL: status {sol_status}; build time {build_time:.2f} minutes; '
                  f'runtime {BILP.Runtime:.2f} seconds')
            print(f'fairDRL station-bundle assignments: {bundle_assign}.')
            print(f'fairDRL subsidies: mean {np.mean(subsidy):.2f}, max {np.max(subsidy):.2f}.')

        # End
        action = {"assignment": assignment, "charging": charging, "subsidy": subsidy}
        return  action, sol_status


    def action_no_demand(self, cvals):
        base_stocks = RL_params['base_stocks']
        num_stations = len(self.environment.agents)
        num_levels = len(base_stocks)

        # Set empty assignment
        assignment = np.array([], dtype=int)

        # Set charging decisions
        charging = np.zeros(num_stations, dtype=int)
        for j, agent in enumerate(self.environment.agents):
            Cval = np.array([cvals.get((k,j), 0)  for k in range(num_levels)], dtype=float)
            k_opt = Cval.argmax()

            base_level = math.ceil(base_stocks[k_opt] * agent.capacity)
            replen = max(base_level - agent.inventory_position, 0)
            idle_slot = max(agent.chargers - agent.charging_batteries, 0)
            charging[j] = min(agent.empty_batteries, idle_slot, replen)

        # Set empty subsidies
        subsidy = np.array([], dtype=float)

        # End
        action = {"assignment": assignment, "charging": charging, "subsidy": subsidy}
        return  action


    def random_action(self, bundling):
        t0 = time.perf_counter()
        base_stocks = RL_params['base_stocks']

        num_stations = len(self.environment.agents)
        num_bundles = len(bundling.bundles)
        num_reqs = len(self.environment.new_demand)

        # Set assignment decisions
        if self.environment.new_demand:
            assignment = np.full(num_reqs, -1)
        else:
            assignment = np.array([])

        bundle_assign = {}
        if bundling.bundles:
            index = {req.id: i   for i, req in enumerate(self.environment.new_demand)}
            current_bundle_ids = list(range(num_bundles))
            current_agent_ids = list(range(num_stations))

            while current_bundle_ids and current_agent_ids:
                bid = random.choice(current_bundle_ids)
                bundle = bundling.bundles[bid]  #tuple data
                feasible_stations = [sid   for sid in bundling.bundle_stations[bid] if sid in current_agent_ids]

                if not feasible_stations:
                    current_bundle_ids.remove(bid)
                    continue

                else:
                    costs = np.zeros(len(feasible_stations), dtype=float)
                    for j, sid in enumerate(feasible_stations):
                        agent = self.environment.agents[sid]
                        costs[j] = (env_params['VOT'] * agent.avg_waiting * len(bundle))
                        for rid in bundle:
                            req = self.environment.all_demand[rid]
                            dist = agent.get_travel_distance(req)
                            travel_time = env_params['travel_time_factor'] * (dist / env_params['vehicle_speed']) * 60 # minutes
                            detour_time = np.max([travel_time - req.nearest_time, 0])
                            costs[j] += (env_params['VOT'] * detour_time)

                    #Select the station with minimum time costs
                    best_j = costs.argmin()
                    j = feasible_stations[best_j]
                    #sid = random.choice(feasible_stations)

                    bundle_assign[j] = bundle
                    for rid in bundle:
                        i = index[rid]
                        assignment[i] = j

                    #Update the sets
                    current_bundle_ids.remove(bid)
                    current_bundle_ids = [bid  for bid in current_bundle_ids  if set(bundling.bundles[bid]).isdisjoint(bundle)]
                    current_agent_ids.remove(j)

        # Set charging decisions
        charging = np.zeros(num_stations, dtype=int)
        for j, agent in enumerate(self.environment.agents):
            k = self.environment.set_base_level(agent)
            base_level = math.ceil(base_stocks[k] * agent.capacity)
            replen = max(base_level - agent.inventory_position, 0)
            idle_slot = max(agent.chargers - agent.charging_batteries, 0)
            charging[j] = min(agent.empty_batteries, idle_slot, replen)

        # Check the solution
        if env_params['test_mode'] and bundle_assign:
            bundling.check_solution(bundle_assign)

        # Set the subsidies
        subsidy = bundling.set_subsidies(assignment, bundle_assign)

        t1 = time.perf_counter()
        run_time = (t1 - t0) / 60  # Build time in minutes
        if bundle_assign and env_params['test_mode']:
            print(f'Random bundle assignments with time {run_time:.2f} minutes: {bundle_assign}')

        # End
        action = {"assignment": assignment, "charging": charging, "subsidy": subsidy}
        return  action


    # Schedule under DRL policy
    def DRL_schedule(self, bundling, train_flag, scale_):
        # Select a random action
        if (np.random.random() <= self.epsilon) and train_flag:
            self.rand_act += 1
            return  self.random_action(bundling)

        # Select an optimal action
        self.opt_act += 1
        qvals, cvals, costs, rcosts = self.BILP_costs(bundling, scale_)
        if self.environment.new_demand:
            act, sol_status = self.siml_action_BILP(bundling, qvals, cvals, costs, rcosts)
            if (sol_status == "OPTIMAL") or (sol_status == "SUBOPTIMAL"):
                action = act
            else:
                print(f'Apply the sequential action model for speeding.')
                action, _ = self.seq_action_BILP(bundling, qvals, cvals, costs)
        else:
            action = self.action_no_demand(cvals)

        # End
        return  action


    # Calculate Q-value for a single agent
    def agent_Qval(self, scale_, next_obs, done, revenue):
        device = self.device

        # Terminal state: t+1 = 144
        # Agent: ('time_step', 'hour', 'cluster', 'nearby_stations', 'new_order',
        # 'cluster_demand', 'que_order', 'avg_waiting', 'full_batteries', 'inventory_position')
        if done: # next_obs is terminal
            return  revenue

        # Process the state through the appropriate method to get the tensor input
        inputs = self.environment.process_observe(scale_, self.state_dims, next_obs)
        # Create tensor and ensure correct device
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            self.value_net.eval()
            value = self.value_net(inputs_tensor).squeeze(0).item()

        # Calculate the Q-value for the agent
        qval = revenue + self.gamma * value

        # End
        return  qval


    # Compute value estimates for each batch:
    def batch_Qvals(self, state_batch: torch.Tensor,
                    neural_net: nn.Module) -> torch.Tensor:
        """
        state_batch: [B, num_agents, obs_dim]
        returns:    [B, num_agents]
        """
        B, N, D = state_batch.shape

        # Flatten to process all agents at once
        flat_states = state_batch.view(B * N, D).to(self.device)  # [B*N, D]
        #neural_net.train(training)
        values = neural_net(flat_states).view(B, N)  # [B, N]

        # End
        return  values


    # Soft-update target network parameters.
    def update_target_network(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )
        # 强制设置为评估模式
        self.target_net.eval()


    # Add a new transition to the replay buffer.
    def store_transition(self, state_array: np.ndarray,
                         reward: np.ndarray,
                         next_state_array: np.ndarray,
                         done: bool):
        # Add a new transition to the replay buffer
        self.replay_buffer.push(state_array, reward, next_state_array, done)


    # Incremental update of learning parameters
    def para_schedule(self, episode):
        if len(self.replay_buffer) >= int(RL_params['batch_size']):
            # Update epsilon value
            epsilon_decay = RL_params['epsilon_decay']
            final_epsilon = RL_params['final_epsilon']
            self.epsilon = max(self.epsilon * epsilon_decay, final_epsilon)

            # Update optimizer learning rate
            learning_decay = RL_params['learning_decay']
            final_learning = RL_params['final_learning']
            if episode <= 1000:
                final_learning = 1e-4
            self.learning_rate = max(self.learning_rate * learning_decay, final_learning)
            self.optimizer.param_groups[0]['lr'] = self.learning_rate


    #加载 value_net, target_net, optimizer 和 replay_buffer，并将所有 tensor 迁移到 self.device。
    def load_checkpoint(self, chkpt_path: str):
        checkpoint = torch.load(chkpt_path, map_location=self.device, weights_only=False)

        # 加载网络参数
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        # 迁移到 device，并设置模式
        self.value_net.to(self.device).train()
        self.target_net.to(self.device).eval()

        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # 把 optimizer.state 中的所有 tensor 都搬到 device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # 恢复经验回放
        self.replay_buffer = checkpoint['replay_buffer']


    # Perform one update of the value network using Double DQN loss.
    def train_step(self):
        t0 = time.perf_counter()
        if len(self.replay_buffer) < int(RL_params['batch_size']):
            return  None

        # Sample a batch
        # Agent: ('time_step', 'hour', 'cluster', 'nearby_stations', 'new_order',
        # 'cluster_demand', 'que_order', 'avg_waiting', 'full_batteries', 'inventory_position')
        states, rewards, next_states, dones = self.replay_buffer.sample()
        # Move data to device
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current values:value_net.train()
        self.value_net.train()  # Force the training mode
        predicts = self.batch_Qvals(states, self.value_net)  # [B, N]

        # Compute next values from target network (with no_grad):target_net.eval()
        with torch.no_grad():
            self.target_net.eval()
            values_next = self.batch_Qvals(next_states, self.target_net)  # [B, N]

        # Compute TD target: r + gamma * V_target(next)
        done_mask = (1 - dones).unsqueeze(1)  # [B, 1]
        td_target = rewards + self.gamma * values_next * done_mask

        # Compute loss: 绝对误差的L2范数（平方和的平方根）
        loss = self.loss_fn(predicts, td_target)
        ms_err = loss.item()

        # Update main network with gradient descent step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Record gradient norms
        total_norm = 0.0
        grad_norms = []
        for p in self.value_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                grad_norms.append(param_norm)
                total_norm += param_norm ** 2
        global_norm = math.sqrt(total_norm)
        grad_norms = np.array(grad_norms, dtype=float)
        min_norm = np.min(grad_norms) if grad_norms.size > 0 else 0
        max_norm = np.max(grad_norms) if grad_norms.size > 0 else 0

        # Clip gradients
        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.clips)
        # Optimizer step
        self.optimizer.step()

        # Periodically update the target network
        self.step_count += 1
        if self.step_count % self.update_freq == 0:
            self.update_target_network()

        # Print training results
        lr_val = self.optimizer.param_groups[0]['lr']
        pred = predicts.mean().item()
        tg = td_target.mean().item()

        t1 = time.perf_counter()
        tt = (t1 - t0) / 60
        if env_params['test_mode']:
            print(f"Training: time {tt:.2f} minutes, lr {lr_val:.6f}, losses {ms_err:.4f}, "
                  f"predicts {pred:.4f}, targets {tg:.4f}; \n"
                  f"gradients: global {global_norm:.4f}, min {min_norm:.4f}, max {max_norm:.4f}.")

        # End
        return  ms_err


    # Simulate scheduling policy
    def scheduling(self, bundling, train_flag, scale_, episode, seed):
        if seed is not None:
            np.random.seed(seed)
            random.Random(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        start_run = time.perf_counter()
        loss_list = []
        self.opt_act, self.rand_act = 0, 0

        # Initialize simulator
        self.environment.reset()
        state = self.environment.get_states()
        state_array = self.environment.process_state(scale_, self.state_dims, state)
        done = False

        # Start simulation process
        while not done:
            # Generate bundles for requests
            bundling.start_bundling(episode)
            num_new = len(self.environment.new_demand)

            # select a schedule action
            action = self.DRL_schedule(bundling, train_flag, scale_)

            # execute action and transit to next state
            next_state, reward, done = self.environment.step(action, self.type)
            if np.max(reward) < 0:
                print(f'########## Warning Negative rewards: ({np.min(reward):.2f}, {np.max(reward):.2f}) '
                      f'at time {self.environment.time_step} with demand {num_new}! ########## ')

            # store transitions
            next_state_array = self.environment.process_state(scale_, self.state_dims, next_state)
            if train_flag:
                self.store_transition(state_array, reward, next_state_array, done)
            state_array = next_state_array

            # training the MLPs
            if train_flag:
                ms_err = self.train_step()
            else:
                ms_err = None

            if ms_err is not None:
                loss_list.append(ms_err)

        ### Ending the DRL policy ###
        end_run = time.perf_counter()
        run_time = (end_run - start_run) / 60 # running time

        avg_loss = np.mean(np.array(loss_list, dtype=float))  if loss_list else 0
        min_loss = np.min(np.array(loss_list, dtype=float))  if loss_list else 0
        max_loss = np.max(np.array(loss_list, dtype=float))  if loss_list else 0

        # Result
        result = self.environment.service_records(episode)
        result.update({'run_time': run_time, 'loss': avg_loss})
        if not train_flag:
            self.environment.request_records('fairDRL', episode)

        # Summary
        print(f"*************** Fair DRL policy in episode {episode} *************** \n"
              f"total demand {len(self.environment.all_demand)}, rewards {result['total_rewards']:.2f} CNY, "
              f"revenues {result['total_revenues']:.2f} CNY, subsidies {result['total_subsidies']:.2f} CNY; \n"
              f"served {result['served']}, charged {result['charged']}, rejection {result['rejection']:.2f}, "
              f"running time {run_time:.2f} minutes.")
        print(f"optimal actions {self.opt_act}, random actions {self.rand_act}, epsilon {self.epsilon:.6f}, "
              f"lr {self.learning_rate:.6f}; \n"
              f"training losses: avg {avg_loss:.4f}, min {min_loss:.4f}, max {max_loss:.4f}." )

        # End
        return  result