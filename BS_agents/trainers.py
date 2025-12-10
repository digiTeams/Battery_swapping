'''
This is the main loop for training on episodes
Call for DQN, Greedy, Rolling-horizon Optimization algorithm in each episode
Compare the performance of algorithms
'''

import os
import math
import random
from datetime import datetime, timedelta
import numpy as np
import torch
import csv

from config import data_params, env_params, RL_params
from simulator.bundling import Bundling
from baselines.fair_Greedy import GreedyPolicy
from baselines.fair_RHO import RHOPolicy
from DRL.fair_DRL import FairnessDRL
from DRL.opt_DRL import OptimumDRL

class Trainer:
    def __init__(self, env):
        self.environment = env  # 绑定仿真环境
        self.show_stations = [0, 5, 10, 15, 20]
        ''' 
        State space at each time t:
        # Agent: ('time_step', 'hour', 'cluster', 'nearby_stations', 'new_order',
        # 'cluster_demand', 'que_order', 'avg_waiting', 'full_batteries', 'inventory_position')
        self.state_dims = {'steps': steps, 'hours': hours, 'stations': num_agents,
                           'clusters': num_clusters, 'observe_dim': 10, 
                           'discrete': 3, 'continuous': 7}
        '''
        total_minutes = (data_params['end_time'] - data_params['start_time']).total_seconds() / 60
        steps = math.ceil(total_minutes / env_params['delta_time'])
        horizon = data_params['horizon']  # 'horizon': (7, 23)
        hours = horizon[1] - horizon[0] + 1 #7, 8, 9, ..., 22, 23 --> 0, 1, 2, ..., 15, 16
        num_agents = len(self.environment.agents)
        num_clusters = len(self.environment.Clusters)

        self.state_dims = {'steps': steps, 'hours': hours, 'stations': num_agents,
                           'clusters': num_clusters, 'observe_dim': 10,
                           'discrete': 3, 'continuous': 7}
        print(f'State dimensions: \n {self.state_dims}.')

        self.MAX_Episodes = RL_params['MAX_Episodes']

        #Output csv file heads
        cls = [f'cluster_{c}' for c in range(len(self.environment.Clusters))]
        self.inst_head = ['episode', 'total_demand']
        self.inst_head.extend(cls)

        self.result_head = ['episode', 'total_demand', 'served', 'charged', 'rejection', 'total_rewards',
                            'total_profits', 'total_revenues', 'total_subsidies', 'total_envies',
                            'max_wait', 'mean_wait', 'std_wait', 'max_detour', 'mean_detour', 'std_detour',
                            'max_envy', 'mean_envy', 'std_envy', 'max_subsidy', 'mean_subsidy', 'std_subsidy',
                            'max_queue', 'mean_queue', 'std_queue',
                            'avg_station_serve', 'std_station_serve', 'avg_station_charge', 'std_station_charge',
                            'avg_cls_envy', 'std_cls_envy', 'avg_cls_subsidy', 'std_cls_subsidy',
                            'avg_centre_envy', 'std_centre_envy', 'avg_centre_subsidy', 'std_centre_subsidy',
                            'avg_suburb_envy', 'std_suburb_envy', 'avg_suburb_subsidy', 'std_suburb_subsidy']
        self.result_head.extend(cls)
        self.result_head.extend(['run_time', 'loss'])


    def write_to_csv(self, result, filename, fields=None):
        fields = fields or list(result.keys())
        file_exists = os.path.exists(filename)

        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)


    #Run the baseline policies
    def baselines(self, sn):
        scale_ = data_params['scale']
        train_flag = True

        #Output instance data file
        current_datetime = datetime.now().strftime('%m-%d')
        csv_name = f'Instances_summary_{current_datetime}.csv'
        Instances_file = os.path.join(data_params['result_dir'], csv_name)

        # Greedy policy
        greedier = GreedyPolicy(self.environment)
        csv_name = f'fairGreedy_policy_{current_datetime}.csv'
        Greedy_file = os.path.join(data_params['result_dir'], csv_name)

        # RHO policy
        RHO = RHOPolicy(self.environment)
        csv_name = f'fairRHO_policy_{current_datetime}.csv'
        RHO_file = os.path.join(data_params['result_dir'], csv_name)

        # Setup bundling for policies
        bundling = Bundling(self.environment)

        # Evaluation process
        start_eps = 0
        for episode in range(start_eps, self.MAX_Episodes):
            #Set seed number
            if sn is None:
                seed_r = episode
            else:
                seed_r = sn

            # Initialize the environment
            self.environment.initialize_instance(scale=scale_, seed_r=seed_r, episode=episode)
            total_demand = len(self.environment.all_demand)
            #inst_head = ['episode', 'total_demand', 'swap_fee', 'driving_cost', 'waiting_cost',
            #             'price_peak', 'price_off']
            row = {'episode': episode, 'total_demand': total_demand}
            for c, d in enumerate(self.environment.cluster_demands):
                cname = f'cluster_{c}'
                row[cname] = d
            # save results
            self.write_to_csv(row, Instances_file, self.inst_head)

            ### the greedy baseline policy ###
            result_G = greedier.scheduling(bundling, train_flag, episode)
            self.write_to_csv(result_G, Greedy_file, self.result_head)
            print("---------------------------------------------------------")

            ### the RHO baseline policy ###
            #self.environment.initialize_instance(scale=scale_, seed_r=seed_r, episode=episode)
            result_R = RHO.scheduling(bundling, train_flag, episode)
            self.write_to_csv(result_R, RHO_file, self.result_head)

            #Summary
            print(f"*************************** Training episode {episode} ***************************")
            print(f'fairGreedy results: rewards {result_G['total_rewards']:.2f}, subsidies {result_G['total_subsidies']:.2f}, '
                  f'time {result_G['run_time']:.2f} minutes; \n'
                  f'requests: wait {result_G['mean_wait']:.2f} min, detour {result_G['mean_detour']:.2f} km, '
                  f'subsidy {result_G['mean_subsidy']:.2f}, envy {result_G['mean_envy']:.2f}; \n'
                  f"station serve {result_G['avg_station_serve']:.2f}, "
                  f"centre envy {result_G['avg_centre_envy']:.2f}, "
                  f"suburb envy {result_G['avg_suburb_envy']:.2f}.")

            print(f'fairRHO results: rewards {result_R['total_rewards']:.2f}, subsidies {result_R['total_subsidies']:.2f}, '
                  f'time {result_R['run_time']:.2f} minutes; \n'
                  f'requests: wait {result_R['mean_wait']:.2f} min, detour {result_R['mean_detour']:.2f} km, '
                  f'subsidy {result_R['mean_subsidy']:.2f}, envy {result_R['mean_envy']:.2f}; \n'
                  f"station serve {result_R['avg_station_serve']:.2f}, "
                  f"centre envy {result_R['avg_centre_envy']:.2f}, "
                  f"suburb envy {result_R['avg_suburb_envy']:.2f}.")
            print("---------------------------------------------------------")


    #Train the AI agents
    def training_fairDRL(self, sn):
        scale_ = data_params['scale']
        train_flag = True

        #Setup value networks and replay buffer
        learner = FairnessDRL(self.environment, self.state_dims)
        current_datetime = datetime.now().strftime('%m-%d')
        csv_name = f'fairDRL_policy_{current_datetime}.csv'
        DRL_file = os.path.join(data_params['result_dir'], csv_name)

        # Setup bundling for policies
        bundling = Bundling(self.environment)

        # Training process
        start_eps = 0
        if start_eps > 0:
            # Load the latest weights and PriorReplayBuffer before starting training
            self.get_neural_nets(learner)
            learner.epsilon = max(RL_params['initial_epsilon'] * (RL_params['epsilon_decay'] ** start_eps),
                                  RL_params['final_epsilon'])
            learner.learning_rate = max(RL_params['initial_learning'] * (RL_params['learning_decay'] ** start_eps),
                                  RL_params['final_learning'])
            learner.optimizer.param_groups[0]['lr'] = learner.learning_rate

        for episode in range(start_eps, self.MAX_Episodes):
            # Set seed number
            if sn is None:
                seed_r = episode
            else:
                seed_r = sn

            # Initialize the environment
            self.environment.initialize_instance(scale=scale_, seed_r=seed_r, episode=episode)

            ###### DRL policies ######
            result_D = learner.scheduling(bundling, train_flag, scale_, episode, seed=None)
            self.write_to_csv(result_D, DRL_file, self.result_head)

            # decay the epsilon value
            # learner.para_schedule(episode)
            self.para_schedule(episode, learner)

            # Save checkpoint at regular intervals
            if (episode % 10 == 0) or (episode == self.MAX_Episodes-1):
                self.save_checkpoint(episode, learner)

            # Summary of results
            print(f"********************* Training episode {episode} *********************")
            print(f'fairDRL results: rewards {result_D['total_rewards']:.2f}, subsidies {result_D['total_subsidies']:.2f}, '
                  f'loss {result_D['loss']:.4f}, time {result_D['run_time']:.2f} minutes; \n'
                  f'requests: wait {result_D['mean_wait']:.2f} min, detour {result_D['mean_detour']:.2f} km, '
                  f'subsidy {result_D['mean_subsidy']:.2f}, envy {result_D['mean_envy']:.2f}; \n'
                  f"station serve {result_D['avg_station_serve']:.2f}, "
                  f"centre envy {result_D['avg_centre_envy']:.2f}, "
                  f"suburb envy {result_D['avg_suburb_envy']:.2f}.")
            print("---------------------------------------------------------")

        #End
        self.save_checkpoint(self.MAX_Episodes-1, learner)


    #Train the AI agents
    def training_optDRL(self, sn):
        scale_ = data_params['scale']
        train_flag = True

        #Setup value networks and replay buffer
        learner = OptimumDRL(self.environment, self.state_dims)
        current_datetime = datetime.now().strftime('%m-%d')
        w = int(env_params['weight'])
        csv_name = f'w{w}DRL_policy_{current_datetime}.csv'
        DRL_file = os.path.join(data_params['result_dir'], csv_name)

        # Setup bundling for policies
        bundling = Bundling(self.environment)

        # Training process
        start_eps = 2781
        if start_eps > 0:
            # Load the latest weights and PriorReplayBuffer before starting training
            self.get_neural_nets(learner)
            learner.epsilon = max(RL_params['initial_epsilon'] * (RL_params['epsilon_decay'] ** start_eps),
                                  RL_params['final_epsilon'])
            learner.learning_rate = max(RL_params['initial_learning'] * (RL_params['learning_decay'] ** start_eps),
                                  RL_params['final_learning'])
            learner.optimizer.param_groups[0]['lr'] = learner.learning_rate

        for episode in range(start_eps, self.MAX_Episodes):
            # Set seed number
            if sn is None:
                seed_r = episode
            else:
                seed_r = sn

            # Initialize the environment
            self.environment.initialize_instance(scale=scale_, seed_r=seed_r, episode=episode)

            ###### DRL policies ######
            result_O = learner.scheduling(bundling, train_flag, scale_, episode, seed=None)
            self.write_to_csv(result_O, DRL_file, self.result_head)

            # decay the epsilon value
            # learner.para_schedule(episode)
            self.para_schedule(episode, learner)

            # Save checkpoint at regular intervals
            if (episode % 10 == 0) or (episode == self.MAX_Episodes-1):
                self.save_checkpoint(episode, learner)

            # Summary of results
            print(f"********************* Training episode {episode} *********************")
            print(f'optDRL(w={env_params['weight']}) results: rewards {result_O['total_rewards']:.2f}, subsidies {result_O['total_subsidies']:.2f}, '
                  f'loss {result_O['loss']:.4f}, time {result_O['run_time']:.2f} minutes; \n'
                  f'requests: wait {result_O['mean_wait']:.2f} min, detour {result_O['mean_detour']:.2f} km, '
                  f'subsidy {result_O['mean_subsidy']:.2f}, envy {result_O['mean_envy']:.2f}; \n'
                  f"centre envy {result_O['avg_centre_envy']:.2f}, "
                  f"suburb envy {result_O['avg_suburb_envy']:.2f}.")
            print("---------------------------------------------------------")

        #End
        self.save_checkpoint(self.MAX_Episodes-1, learner)


    # Incremental update of learning parameters
    def para_schedule(self, episode, learner):
        if len(learner.replay_buffer) >= int(RL_params['batch_size']):
            # Update epsilon value
            epsilon_decay = RL_params['epsilon_decay']
            final_epsilon = RL_params['final_epsilon']
            learner.epsilon = max(learner.epsilon * epsilon_decay, final_epsilon)

            # Update optimizer learning rate
            learning_decay = RL_params['learning_decay']
            if episode <= 1000:
                final_learning = 10 * RL_params['final_learning']  #1e-4
            else:
                final_learning = RL_params['final_learning']  #1e-5
            learner.learning_rate = max(learner.learning_rate * learning_decay, final_learning)
            learner.optimizer.param_groups[0]['lr'] = learner.learning_rate


    # Save a single PyTorch checkpoint containing
    def save_checkpoint(self, episode, learner):
        """
          - learner.value_net state_dict
          - learner.target_net state_dict
          - learner.optimizer state_dict
          - learner.replay_buffer (picklable)
        """
        check_dir = RL_params['check_dir']
        os.makedirs(check_dir, exist_ok=True)
        agent_type = learner.type  # e.g. 'Fair', 'Weighted', 'Optimum'

        # 单个 .pt 文件名
        if agent_type == 'Fair':
            ckpt_name = f"checkpoint_episode_{episode}_{agent_type}.pt"
        else:
            w = int(env_params['weight'])
            ckpt_name = f"checkpoint_episode_{episode}_w{w}_{agent_type}.pt"
        ckpt_path = os.path.join(check_dir, ckpt_name)

        # 打包所有需要保存的状态
        checkpoint = {
            'value_net': learner.value_net.state_dict(),
            'target_net': learner.target_net.state_dict(),
            'optimizer': learner.optimizer.state_dict(),
            'replay_buffer': learner.replay_buffer,  # deque 是可 pickled 的
        }

        # End
        torch.save(checkpoint, ckpt_path)
        print(f"Temporal value networks and replay buffer saved for agent {agent_type} at episode {episode}.")


    # Load the most recent .pt checkpoint for this learner.type.
    def get_neural_nets(self, learner):
        #restore networks, optimizer and replay buffer.
        check_dir = RL_params['check_dir']
        agent_type = learner.type

        # 匹配新格式的文件名: f"checkpoint_episode_{episode}_{agent_type}.pt"
        #checkpoint_episode_3000_Fair.pt
        files = [f  for f in os.listdir(check_dir)
                 if f.startswith("checkpoint_") and f.endswith(f"_{agent_type}.pt")]
        if not files:
            print("Checkpoints NOT found, starting from scratch.")
            return

        chkpt = files[0]
        chkpt_path = os.path.join(check_dir, chkpt)

        #Load the checkpoint file
        learner.load_checkpoint(chkpt_path)
        print(f"Checkpoints of agent {agent_type} loaded from file: {chkpt}.")


    # Test the DRL policy with a single large-scale instance without learning
    def test_Fairs(self, sn):
        #Scenarios: [4, 8, 16]; k = 0, 1, 2
        scale_ = 2.0
        train_flag = False
        MAX_Tests = int(RL_params['MAX_Tests'])

        self.inst_head.insert(0, 'scenario')
        self.result_head.insert(0, 'scenario')

        # Output instance data file
        current_datetime = datetime.now().strftime('%m-%d')
        Instances_file = os.path.join(data_params['result_dir'],
                                      f'Instances_test_{current_datetime}.csv'
                                      )

        ### Greedy policy
        greedier = GreedyPolicy(self.environment)
        Greedy_file = os.path.join(data_params['result_dir'],
                                   f'fairGreedy_test_{current_datetime}.csv'
                                   )

        ### RHO policy
        RHO = RHOPolicy(self.environment)
        RHO_file = os.path.join(data_params['result_dir'],
                                f'fairRHO_test_{current_datetime}.csv'
                                )

        ### fairDRL policy
        fairer = FairnessDRL(self.environment, self.state_dims)
        fairDRL_file = os.path.join(data_params['result_dir'],
                                f'fairDRL_test_{current_datetime}.csv'
                                )
        # Load the latest weights and PriorReplayBuffer before starting training
        self.get_neural_nets(fairer)
        fairer.epsilon = RL_params['final_epsilon']
        fairer.learning_rate = RL_params['final_learning']
        fairer.optimizer.param_groups[0]['lr'] = fairer.learning_rate

        ### wgtDRL policy: weight > 0
        optimizer = OptimumDRL(self.environment, self.state_dims)
        w = int(env_params['weight'])
        csv_name = f'w{w}DRL_test_{current_datetime}.csv'
        optDRL_file = os.path.join(data_params['result_dir'], csv_name)

        # Load the latest weights and PriorReplayBuffer before starting training
        self.get_neural_nets(optimizer)
        optimizer.epsilon = RL_params['final_epsilon']
        optimizer.learning_rate = RL_params['final_learning']
        optimizer.optimizer.param_groups[0]['lr'] = optimizer.learning_rate

        # Setup bundling for policies
        bundling = Bundling(self.environment)

        ### Test on scenario groups
        for k in range(MAX_Tests):
            episode = sn * MAX_Tests + k
            seed_r = episode

            # Initialize the environment
            self.environment.initialize_instance(scale=scale_, seed_r=seed_r, episode=episode)
            total_demand = len(self.environment.all_demand)
            #inst_head = ['episode', 'total_demand', 'swap_fee', 'driving_cost', 'waiting_cost',
            #             'price_peak', 'price_off']
            row = {'scenario': sn, 'episode': episode, 'total_demand': total_demand}
            for c, d in enumerate(self.environment.cluster_demands):
                cname = f'cluster_{c}'
                row[cname] = d
            # save results
            self.write_to_csv(row, Instances_file, self.inst_head)

            ### the greedy baseline policy ###
            result_G = greedier.scheduling(bundling, train_flag, episode)
            result_G.update({'scenario': sn})
            self.write_to_csv(result_G, Greedy_file, self.result_head)
            print("---------------------------------------------------------")

            ### the RHO baseline policy ###
            result_R = RHO.scheduling(bundling, train_flag, episode)
            result_R.update({'scenario': sn})
            self.write_to_csv(result_R, RHO_file, self.result_head)
            print("---------------------------------------------------------")

            ###### fairDRL policies ######
            result_D = fairer.scheduling(bundling, train_flag, scale_, episode, seed=None)
            result_D.update({'scenario': sn})
            self.write_to_csv(result_D, fairDRL_file, self.result_head)
            print("---------------------------------------------------------")

            ###### optDRL policies ######
            result_O = optimizer.scheduling(bundling, train_flag, scale_, episode, seed=None)
            result_O.update({'scenario': sn})
            self.write_to_csv(result_O, optDRL_file, self.result_head)
            print("---------------------------------------------------------")

            # Summary of results
            print(f"********************* Test episode {episode} *********************")
            print(f'fairGreedy results: rewards {result_G['total_rewards']:.2f}, '
                  f'request subsidies {result_G['mean_subsidy']:.2f}, '
                  f"centre envy {result_G['avg_centre_envy']:.2f}, "
                  f"suburb envy {result_G['avg_suburb_envy']:.2f}.")
            print(f'fairRHO results: rewards {result_R['total_rewards']:.2f}, '
                  f'request subsidies {result_R['mean_subsidy']:.2f}, '                               
                  f"centre envy {result_R['avg_centre_envy']:.2f}, "
                  f"suburb envy {result_R['avg_suburb_envy']:.2f}.")
            print(f'fairDRL results: rewards {result_D['total_rewards']:.2f}, '
                  f'request subsidies {result_D['mean_subsidy']:.2f}, '         
                  f"centre envy {result_D['avg_centre_envy']:.2f}, "
                  f"suburb envy {result_D['avg_suburb_envy']:.2f}.")
            print(f'optDRL(w={env_params['weight']}) results: rewards {result_O['total_rewards']:.2f}, '
                  f'request subsidies {result_O['mean_subsidy']:.2f}, '
                  f"centre envy {result_O['avg_centre_envy']:.2f}, "
                  f"suburb envy {result_O['avg_suburb_envy']:.2f}.")


    # Test the DRL policy with a single large-scale instance without learning
    def test_OPT(self, sn):
        #Scenarios: [4, 8, 16]; k = 0, 1, 2
        scale_ = 2.0
        train_flag = False
        MAX_Tests = int(RL_params['MAX_Tests'])

        self.inst_head.insert(0, 'scenario')
        self.result_head.insert(0, 'scenario')

        # Output instance data file
        current_datetime = datetime.now().strftime('%m-%d')
        Instances_file = os.path.join(data_params['result_dir'],
                                      f'Instances_test_{current_datetime}.csv'
                                      )

        ### optDRL policy: weight = 0
        optimizer = OptimumDRL(self.environment, self.state_dims)
        w = int(env_params['weight'])
        csv_name = f'w{w}DRL_test_{current_datetime}.csv'
        optDRL_file = os.path.join(data_params['result_dir'], csv_name)

        # Load the latest weights and PriorReplayBuffer before starting training
        self.get_neural_nets(optimizer)
        optimizer.epsilon = RL_params['final_epsilon']
        optimizer.learning_rate = RL_params['final_learning']
        optimizer.optimizer.param_groups[0]['lr'] = optimizer.learning_rate

        # Setup bundling for policies
        bundling = Bundling(self.environment)

        ### Test on scenario groups
        for k in range(MAX_Tests):
            episode = sn * MAX_Tests + k
            seed_r = episode

            # Initialize the environment
            self.environment.initialize_instance(scale=scale_, seed_r=seed_r, episode=episode)
            total_demand = len(self.environment.all_demand)
            #inst_head = ['episode', 'total_demand', 'swap_fee', 'driving_cost', 'waiting_cost',
            #             'price_peak', 'price_off']
            row = {'scenario': sn, 'episode': episode, 'total_demand': total_demand}
            for c, d in enumerate(self.environment.cluster_demands):
                cname = f'cluster_{c}'
                row[cname] = d
            # save results
            self.write_to_csv(row, Instances_file, self.inst_head)

            ###### optDRL policies ######
            result_O = optimizer.scheduling(bundling, train_flag, scale_, episode, seed=None)
            result_O.update({'scenario': sn})
            self.write_to_csv(result_O, optDRL_file, self.result_head)

            # Summary of results
            print(f"********************* Test episode {episode} *********************")
            print(f'optDRL(w={env_params['weight']}) results: rewards {result_O['total_rewards']:.2f}, '
                  f'request subsidies {result_O['mean_subsidy']:.2f}, '
                  f"centre envy {result_O['avg_centre_envy']:.2f}, "
                  f"suburb envy {result_O['avg_suburb_envy']:.2f}.")