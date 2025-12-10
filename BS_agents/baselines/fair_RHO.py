'''
This is the algorithm for rolling-horizon optimization baseline policy:
At each time, solve the myopic bundle ILP for assignment, and charge
'''

import math
import time
from datetime import datetime, timedelta
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Load configuration parameters (assumed to be defined in config)
from config import data_params, env_params, RL_params


class RHOPolicy:
    def __init__(self, env):
        self.environment = env  # 绑定仿真环境
        self.gamma = env_params['gamma']
        self.type = 'Fair'


    # Compute cost coefficients of BILP
    def BILP_costs(self, bundling):
        t0 = time.perf_counter()
        base_stocks = RL_params['base_stocks']
        num_stations = len(self.environment.agents)

        # Collect computing tasks: (b, j, time_step, current_time, agent, bundle_requests, charge)
        charging = np.zeros(num_stations, dtype=int)
        for j, agent in enumerate(self.environment.agents):
            k = self.environment.set_base_level(agent)
            base_level = math.ceil(base_stocks[k] * agent.capacity)
            replen = max(base_level - agent.inventory_position, 0)
            idle_slot = max(agent.chargers - agent.charging_batteries, 0)
            charging[j] = min(agent.empty_batteries, idle_slot, replen)

        # Results
        qvals, costs, rcosts = self.bundle_costs(bundling, charging)

        t1 = time.perf_counter()
        eval_time = (t1 - t0) / 60  # Build time in minutes
        if env_params['test_mode']:
            print(f'The processing time for evaluating bundle costs: {eval_time:.2f} minutes.')

        # End
        return  qvals, costs, rcosts


    # Compute costs of bundles
    def bundle_costs(self, bundling, charging):
        index = {req.id: i   for i, req in enumerate(self.environment.new_demand)}

        #Collect computing tasks: (b, j, time_step, current_time, agent, bundle_requests, charge)
        num_new, num_clusters = 0, 0
        qvals, costs = {}, {}
        for (b, j) in bundling.BS_pairs:
            bundle = bundling.bundles[b]
            agent = self.environment.agents[j]

            #bundle_reqs = [self.environment.all_demand[rid]   for rid in bundle]
            bundle_reqs = [self.environment.new_demand[index[rid]]   for rid in bundle]
            charge = charging[j]
            _, _, reward, utilities = agent.simulate_bundle(self.environment.time_step,
                                                            self.environment.current_time,
                                                            bundle_reqs, charge,
                                                            num_new, num_clusters)
            qvals[(b, j)] = reward
            for rid, util in utilities.items():  # per-request cost map
                i = index[rid]
                costs[(b, j, i)] = util

        ##Compute the envy costs for replacing m with i in bundle b
        #rcosts[(b, j, i, m)] = costs[(bid, j, i)]
        rcosts = bundling.replace_cost(costs)

        #End
        return  qvals, costs, rcosts


    #Determine assignment and subsidies simultaneously
    def siml_action_BILP(self, bundling, qvals, costs, rcosts):
        t0 = time.perf_counter()
        base_stocks = RL_params['base_stocks']
        penalty = env_params['penalty']
        index = {req.id: i for i, req in enumerate(self.environment.new_demand)}

        num_stations = len(self.environment.agents)
        num_bundles = len(bundling.bundles)
        num_reqs = len(self.environment.new_demand)

        #Set up the model
        BILP = gp.Model("Bundle ILP")

        #Define decision variables
        x = {}  #assignment (b,j)
        y = {}  #rejection
        p = {}  #subsidy
        for j in range(num_stations):
            for b in range(num_bundles):
                x[b,j] = BILP.addVar(vtype=GRB.BINARY, name=f"x_{b}_{j}")
        for i in range(num_reqs):
            y[i] = BILP.addVar(vtype=GRB.BINARY, name=f"y_{i}")
            p[i] = BILP.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_{i}")
        BILP.update()

        #Constraints 1: each agent receives at most one bundle
        for j in range(num_stations):
            BILP.addConstr(
                gp.quicksum(bundling.BS.get((b, j), 0) * x[b,j]
                            for b in range(num_bundles))
                <= 1,
                name=f"agent_{j}_constraint"
            )

        # Constraints 2: each bundle is allocated at most once
        for b in range(num_bundles):
            BILP.addConstr(
                gp.quicksum(bundling.BS.get((b, j), 0) * x[b,j]
                            for j in range(num_stations))
                <= 1,
                name=f"bundle_{b}_constraint"
            )

        # Constraints 3: each request is served at most once
        for i in range(num_reqs):
            BILP.addConstr(
                gp.quicksum(bundling.BR.get((b, i), 0) * x[b,j]
                            for (b, j) in bundling.BS_pairs)
                + y[i] == 1,
                name=f"request_{i}_constraint"
            )

        # Constraints 4: envy-freeness with subsidies
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

                            #Check the replaced bundle b1 = b\{m}U{i}
                            BILP.addConstr(
                                gp.quicksum(bundling.BR.get((b,i), 0) * costs.get((b,j,i), env_params['big_M']) * x[b,j]
                                            for (b, j) in bundling.BS_pairs)
                                + penalty * y[i] - p[i]
                                <= gp.quicksum(bundling.BR.get((b,m), 0) * rcosts.get((b,j,i,m), env_params['big_M']) * x[b,j]
                                               for (b, j) in bundling.BS_pairs)
                                + penalty * y[m] - p[m],
                                name=f"envy_{i}_{m}_constraint"
                            )

        #Objective function
        BILP.setObjective(
            gp.quicksum(qvals.get((b, j), 0) * x[b, j]
                        for (b, j) in bundling.BS_pairs)
            - gp.quicksum(p[i]  for i in range(num_reqs)),
            GRB.MAXIMIZE
        )

        t1 = time.perf_counter()
        build_time = (t1 - t0) / 60  # Build time in minutes

        # Set optimization parameters
        BILP.Params.OutputFlag = 0
        BILP.Params.TimeLimit = RL_params['GUROBI_TIME_LIMIT']
        BILP.Params.MIPGap = RL_params['GUROBI_MIPGAP']
        BILP.Params.Threads = int(RL_params['GUROBI_Threads'])
        BILP.Params.Seed = int(RL_params['GUROBI_SEED'])

        #Solve the model
        try:
            BILP.optimize()
        except gp.GurobiError as e:
            print(f"######## Gurobi optimization failed: {e} ########")
            action = {}
            return action

        status_dict = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.TIME_LIMIT: "TIME LIMIT"
        }
        sol_status = status_dict.get(BILP.status, "Other status")

        #Get the solution
        assignment = np.full(num_reqs, -1)
        charging = np.zeros(num_stations, dtype=int)
        subsidy = np.zeros(num_reqs, dtype=float)
        bundle_assign = {}  # j: bundle

        if (BILP.status == GRB.OPTIMAL) or (BILP.status == GRB.SUBOPTIMAL):
            #Get the assignment solution
            for (b, j) in bundling.BS_pairs:
                bundle = bundling.bundles[b]
                if (x[b,j].X > 0.5) and (bundling.BS.get((b,j), 0) > 0):
                    bundle_assign[j] = bundle
                    for rid in bundle:
                        i = index[rid]
                        assignment[i] = j

            # Get the charging solution
            for j, agent in enumerate(self.environment.agents):
                k = self.environment.set_base_level(agent)
                base_level = math.ceil(base_stocks[k] * agent.capacity)
                replen = max(base_level - agent.inventory_position, 0)
                idle_slot = max(agent.chargers - agent.charging_batteries, 0)
                charging[j] = min(agent.empty_batteries, idle_slot, replen)

            # Get the subsidy solution
            for i in range(num_reqs):
                pi = p[i].X
                if abs(p[i].X) < 1e-4:
                    pi = 0
                subsidy[i] = pi
        else:
            print(f'######## Gurobi optimization status in RHO: {BILP.status}, {sol_status} ########')

        #Check the solution
        if env_params['test_mode'] and bundle_assign:
            bundling.check_solution(bundle_assign)

        if env_params['test_mode'] and self.environment.new_demand:
            print(f'BILP optimization under RHO: status {sol_status}; build time {build_time:.2f} minutes; '
                  f'runtime {BILP.Runtime:.2f} seconds')
            print(f'RHO station-bundle assignments: {bundle_assign}.')
            print(f'RHO subsidies: mean {np.mean(subsidy):.2f}, max {np.max(subsidy):.2f}.')

        #End
        action = {"assignment": assignment, "charging": charging, "subsidy": subsidy}
        return  action, sol_status


    #Determine assignment and subsidies sequentially
    def seq_action_BILP(self, bundling, qvals, costs):
        t0 = time.perf_counter()
        base_stocks = RL_params['base_stocks']
        penalty = env_params['penalty']
        index = {req.id: i  for i, req in enumerate(self.environment.new_demand)}

        num_stations = len(self.environment.agents)
        num_bundles = len(bundling.bundles)
        num_reqs = len(self.environment.new_demand)

        #Set up the model
        BILP = gp.Model("Bundle ILP")

        # Define decision variables
        x = {}  # assignment (b,j)
        y = {}  # rejection
        for j in range(num_stations):
            for b in range(num_bundles):
                x[b, j] = BILP.addVar(vtype=GRB.BINARY, name=f"x_{b}_{j}")
        for i in range(num_reqs):
            y[i] = BILP.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        BILP.update()

        #Constraints 1: each agent receives at most one bundle
        for j in range(num_stations):
            BILP.addConstr(
                gp.quicksum(bundling.BS.get((b,j), 0) * x[b,j]
                            for b in range(num_bundles))
                <= 1,
                name=f"agent_{j}_constraint"
            )

        # Constraints 2: each bundle is allocated at most once
        for b in range(num_bundles):
            BILP.addConstr(
                gp.quicksum(bundling.BS.get((b,j), 0) * x[b,j]
                            for j in range(num_stations))
                <= 1,
                name=f"bundle_{b}_constraint"
            )

        # Constraints 3: each request is served at most once
        for i in range(num_reqs):
            BILP.addConstr(
                gp.quicksum(bundling.BR.get((b,i), 0) * x[b,j]
                            for (b,j) in bundling.BS_pairs)
                + y[i] == 1,
                name=f"request_{i}_constraint"
            )

        #Objective function
        BILP.setObjective(
            gp.quicksum(qvals.get((b, j), 0) * x[b, j]
                        for (b, j) in bundling.BS_pairs)
            - gp.quicksum(bundling.BR.get((b,i), 0) * costs.get((b,j,i), env_params['big_M']) * x[b,j]
                        for (b, j) in bundling.BS_pairs  for i in range(num_reqs))
            - gp.quicksum(penalty * y[i]   for i in range(num_reqs)),
            GRB.MAXIMIZE
        )

        t1 = time.perf_counter()
        build_time = (t1 - t0) / 60  # Build time in minutes

        # Set optimization parameters
        BILP.Params.OutputFlag = 0
        BILP.Params.TimeLimit = RL_params['GUROBI_TIME_LIMIT']
        BILP.Params.MIPGap = RL_params['GUROBI_MIPGAP']
        BILP.Params.Threads = int(RL_params['GUROBI_Threads'])
        BILP.Params.Seed = int(RL_params['GUROBI_SEED'])

        #Solve the model
        try:
            BILP.optimize()
        except gp.GurobiError as e:
            print(f"######## Gurobi optimization failed: {e} ########")
            action = {}
            return action

        status_dict = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.TIME_LIMIT: "TIME LIMIT"
        }
        sol_status = status_dict.get(BILP.status, "Other status")

        #Get the solution
        assignment = np.full(num_reqs, -1)
        charging = np.zeros(num_stations, dtype=int)
        bundle_assign = {}  # j: bundle

        if (BILP.status == GRB.OPTIMAL) or (BILP.status == GRB.SUBOPTIMAL):
            #Get the assignment solution
            for (b, j) in bundling.BS_pairs:
                bundle = bundling.bundles[b]
                if (x[b,j].X > 0.5) and (bundling.BS.get((b,j), 0) > 0):
                    bundle_assign[j] = bundle
                    for rid in bundle:
                        i = index[rid]
                        assignment[i] = j

            # Get the charging solution
            for j, agent in enumerate(self.environment.agents):
                k = self.environment.set_base_level(agent)
                base_level = math.ceil(base_stocks[k] * agent.capacity)

                replen = max(base_level - agent.inventory_position, 0)
                idle_slot = max(agent.chargers - agent.charging_batteries, 0)
                charging[j] = min(agent.empty_batteries, idle_slot, replen)

        else:
            print(f'######## Gurobi optimization status in RHO: {BILP.status}, {sol_status} ########')

        #Check the solution
        if env_params['test_mode'] and bundle_assign:
            bundling.check_solution(bundle_assign)

        #Set the subsidies
        subsidy = bundling.set_subsidies(assignment, bundle_assign)

        if env_params['test_mode'] and self.environment.new_demand:
            print(f'BILP optimization under RHO: status {sol_status}; build time {build_time:.2f} minutes; '
                  f'runtime {BILP.Runtime:.2f} seconds')
            print(f'RHO station-bundle assignments: {bundle_assign}.')
            print(f'RHO subsidies: mean {np.mean(subsidy):.2f}, max {np.max(subsidy):.2f}.')

        # End
        action = {"assignment": assignment, "charging": charging, "subsidy": subsidy}
        return  action, sol_status


    #Action in the no demand case
    def action_no_demand(self):
        base_stocks = RL_params['base_stocks']
        num_stations = len(self.environment.agents)

        # Set empty assignment
        assignment = np.array([], dtype=int)

        #Set charging level
        charging = np.zeros(num_stations, dtype=int)
        for j, agent in enumerate(self.environment.agents):
            k = self.environment.set_base_level(agent)
            base_level = math.ceil(base_stocks[k] * agent.capacity)
            replen = max(base_level - agent.inventory_position, 0)

            idle_slot = max(agent.chargers - agent.charging_batteries, 0)
            charge = min(agent.empty_batteries, idle_slot, replen)

            charging[j] = charge

        # Set empty subsidies
        subsidy = np.array([], dtype=float)

        # End
        action = {"assignment": assignment, "charging": charging, "subsidy": subsidy}
        return  action


    # Schedule under RHO policy
    def RHO_schedule(self, bundling):
        '''
        status_dict = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.TIME_LIMIT: "TIME LIMIT"
        }
        '''

        if self.environment.new_demand:
            # Compute cost coefficients
            qvals, costs, rcosts = self.BILP_costs(bundling)
            act, sol_status = self.siml_action_BILP(bundling, qvals, costs, rcosts)
            if (sol_status == "OPTIMAL") or (sol_status == "SUBOPTIMAL"):
                action = act
            else:
                print(f'Apply the sequential action model for speeding.')
                action, _ = self.seq_action_BILP(bundling, qvals, costs)                

        else:
            action = self.action_no_demand()

        # End
        return  action


    # Simulate scheduling policy
    def scheduling(self, bundling, train_flag, episode):
        start_run = time.perf_counter()

        # Initialize simulator
        self.environment.reset()
        state = self.environment.get_states()
        done = False

        # Start simulation process
        while not done:
            #Generate bundles for requests
            bundling.start_bundling(episode)

            # select a RHO schedule action
            action = self.RHO_schedule(bundling)

            # execute action and transit to next state
            next_state, reward, done = self.environment.step(action, self.type)
            state = next_state

        ### Ending the RHO baseline policy ###
        end_run = time.perf_counter()
        run_time = (end_run - start_run) / 60 # running time
        avg_loss = 0

        # Result
        result = self.environment.service_records(episode)
        result.update({'run_time': run_time, 'loss': avg_loss})
        if not train_flag:
            self.environment.request_records('RHO', episode)

        # Summary
        print(f"*************** Fair RHO policy in episode {episode} *************** \n"
              f"total demand {len(self.environment.all_demand)}, rewards {result['total_rewards']:.2f} CNY, "
              f"revenues {result['total_revenues']:.2f} CNY, subsidies {result['total_subsidies']:.2f} CNY; \n"
              f"served {result['served']}, charged {result['charged']}, rejection {result['rejection']:.2f}, "
              f"running time {run_time:.2f} minutes.")

        # End
        return result