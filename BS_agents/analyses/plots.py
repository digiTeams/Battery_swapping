import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import csv

from config import data_params

work_dir = 'C:/OneDrive/Team/BS_network'



class Plots:
    def __init__(self, window):
        self.window = window
        # Policy comparison data
        self.Greedy_file = 'fairGreedy_policy_09-10.csv'
        self.RHO_file = 'fairRHO_policy_09-10.csv'

        # DRL comparison data
        self.DRL_file = 'fairDRL_policy_09-10a.csv'
        self.paras = 'DRL-fair(default)'

        self.DRL_file_1 = 'fairDRL_policy_09-10b.csv'
        self.paras_1 = 'DRL-fair(clustering)'

        self.DRL_file_2 = 'fairDRL_policy_09-10c.csv'
        self.paras_2 = 'DRL-fair(512, 256, 128, 64)'

        self.OPT_file = 'optDRL_policy_09-10.csv'
        self.paras_opt = 'DRL-optimizer'

        # Test results data
        self.Greedy_test = 'fairGreedy_test_09-01.csv'
        self.RHO_test = 'fairRHO_test_09-01.csv'
        self.DRL_test = 'fairDRL_test_09-01.csv'
        self.OPT_test = 'optDRL_test_09-01.csv'


    def plot_DRLs(self, base_rewards):
        '''
        ['episode', 'total_demand', 'served', 'charged', 'rejection',
        'total_rewards', 'total_revenues', 'total_subsidies',
        'max_wait', 'mean_wait', 'std_wait', 'max_detour', 'mean_detour', 'std_detour',
        'max_subsidy', 'mean_subsidy', 'std_subsidy', 'max_envy', 'mean_envy', 'std_envy',
        'max_queue', 'mean_queue', 'std_queue',  'avg_station_serve', 'std_station_serve',
        'avg_station_charge', 'std_station_charge', 'avg_cls_serve', 'std_cls_serve',
        'avg_cls_envy', 'std_cls_envy',
        'cluster_{c}',
        'run_time', 'loss']
        '''
        greedy_reward = base_rewards['Greedy']
        RHO_reward = base_rewards['RHO']

        # 创建包含两个子图的图像
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Draw horizontal lines for Greedy and RHO total rewards
        if greedy_reward is not None:
            axs[0].axhline(y=greedy_reward, color='black', linestyle='-', label='Greedy Reward')
        if RHO_reward is not None:
            axs[0].axhline(y=RHO_reward, color='yellow', linestyle='-', label='RHO Reward')

        ######### Data file #########
        DRL_path = os.path.join(work_dir, self.DRL_file)
        DRL_results = pd.read_csv(DRL_path)

        # 提取 DRL_results 中的 episode、loss 和 total reward 数据
        episodes = [result['episode']   for index, result in DRL_results.iterrows()]
        rewards = [result['total_rewards']   for index, result in DRL_results.iterrows()]
        losses = [result['loss']   for index, result in DRL_results.iterrows()]
        losses[0] = losses[1]
        # 计算 DRL 策略奖励的移动平均和95%置信区间
        reward_means, reward_lower, reward_upper = self.moving_average_with_ci(rewards)
        loss_means, loss_lower, loss_upper = self.moving_average_with_ci(losses)

        # 子图1：奖励曲线对比
        axs[0].plot(episodes, reward_means, marker='o', markersize=2.5, color='red',
                    label=f'Policy {self.paras}')
        axs[0].fill_between(episodes, reward_lower, reward_upper, color='red', alpha=0.2)
        y1 = reward_lower
        y1.extend(reward_upper)

        # 子图2：训练损失曲线
        axs[1].plot(episodes, loss_means, marker='o', markersize=2.5, color='red',
                    label=f'Policy {self.paras}')
        axs[1].fill_between(episodes, loss_lower, loss_upper, color='red', alpha=0.2)
        y2 = loss_lower
        y2.extend(loss_upper)

        ######### Data file 1 #########
        DRL_path_1 = os.path.join(work_dir, self.DRL_file_1)
        if os.path.exists(DRL_path_1) and os.path.isfile(DRL_path_1):
            DRL_results = pd.read_csv(DRL_path_1)
            # 提取 DRL_results 中的 episode、loss 和 total reward 数据
            episodes_1 = [result['episode']   for index, result in DRL_results.iterrows()]
            rewards = [result['total_rewards']   for index, result in DRL_results.iterrows()]
            losses = [result['loss']   for index, result in DRL_results.iterrows()]
            losses[0] = losses[1]
            # 计算 DRL 策略奖励的移动平均和95%置信区间
            reward_means_1, reward_lower_1, reward_upper_1 = self.moving_average_with_ci(rewards)
            loss_means_1, loss_lower_1, loss_upper_1 = self.moving_average_with_ci(losses)

            # 子图1：奖励曲线对比
            axs[0].plot(episodes_1, reward_means_1, marker='^', markersize=2.5, color='green',
                        label=f'Policy {self.paras_1}')
            axs[0].fill_between(episodes_1, reward_lower_1, reward_upper_1, color='green', alpha=0.2)
            y1.extend(reward_lower_1)
            y1.extend(reward_upper_1)

            # 子图2：训练损失曲线
            axs[1].plot(episodes_1, loss_means_1, marker='^', markersize=2.5, color='green',
                        label=f'Policy {self.paras_1}')
            axs[1].fill_between(episodes_1, loss_lower_1, loss_upper_1, color='green', alpha=0.2)
            y2.extend(loss_lower_1)
            y2.extend(loss_upper_1)

        ######### Data file 2 #########
        DRL_path_2 = os.path.join(work_dir, self.DRL_file_2)
        if os.path.exists(DRL_path_2) and os.path.isfile(DRL_path_2):
            DRL_results = pd.read_csv(DRL_path_2)
            # 提取 DRL_results 中的 episode、loss 和 total reward 数据
            episodes_2 = [result['episode'] for index, result in DRL_results.iterrows()]
            rewards = [result['total_rewards'] for index, result in DRL_results.iterrows()]
            losses = [result['loss'] for index, result in DRL_results.iterrows()]
            losses[0] = losses[1]
            # 计算 DRL 策略奖励的移动平均和95%置信区间
            reward_means_2, reward_lower_2, reward_upper_2 = self.moving_average_with_ci(rewards)
            loss_means_2, loss_lower_2, loss_upper_2 = self.moving_average_with_ci(losses)

            # 子图1：奖励曲线对比
            axs[0].plot(episodes_2, reward_means_2, marker='s', markersize=2.5, color='blue',
                        label=f'Policy {self.paras_2}')
            axs[0].fill_between(episodes_2, reward_lower_2, reward_upper_2, color='blue', alpha=0.2)
            y1.extend(reward_lower_2)
            y1.extend(reward_upper_2)

            # 子图2：训练损失曲线
            axs[1].plot(episodes_2, loss_means_2, marker='s', markersize=2.5, color='blue',
                        label=f'Policy {self.paras_2}')
            axs[1].fill_between(episodes_2, loss_lower_2, loss_upper_2, color='blue', alpha=0.2)
            y2.extend(loss_lower_2)
            y2.extend(loss_upper_2)

        # Final
        lowest = np.min(np.array(y1))
        highest = np.max(np.array(y1))
        axs[0].set_ylim(lowest, highest)
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel(f'Total Rewards (Window={self.window})')
        axs[0].set_title('Total Rewards for DRL Policies')
        axs[0].legend()
        axs[0].grid(True)

        lowest = 0 # np.min(np.array(y2))
        highest = np.max(np.array(y2))
        axs[1].set_ylim(lowest, highest)
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel(f'Training Loss (Window={self.window})')
        axs[1].set_title('Training Loss for DRL Policies')
        axs[1].legend()
        axs[1].grid(True)

        # 保存图像为 PNG 格式
        current_datetime = datetime.now().strftime('%m-%d')
        fig = f'DRL_policy_{current_datetime}.png'
        output_file = os.path.join(data_params['result_dir'], fig)
        plt.savefig(output_file)

        plt.show()
        plt.close()


    def plot_optimizer(self):
        '''
        ['episode', 'total_demand', 'served', 'charged', 'rejection',
        'total_rewards', 'total_revenues', 'total_subsidies',
        'max_wait', 'mean_wait', 'std_wait', 'max_detour', 'mean_detour', 'std_detour',
        'max_subsidy', 'mean_subsidy', 'std_subsidy', 'max_envy', 'mean_envy', 'std_envy',
        'max_queue', 'mean_queue', 'std_queue',  'avg_station_serve', 'std_station_serve',
        'avg_station_charge', 'std_station_charge', 'avg_cls_serve', 'std_cls_serve',
        'avg_cls_envy', 'std_cls_envy',
        'cluster_{c}',
        'run_time', 'loss']
        '''
        # 创建包含两个子图的图像
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        #### Default DRL policy
        DRL_path = os.path.join(work_dir, self.DRL_file)
        DRL_results = pd.read_csv(DRL_path)
        paras = '(256, 128, 64, 32)-Fair'

        episodes = [result['episode']   for index, result in DRL_results.iterrows()]
        revenues = np.array([result['total_revenues']   for index, result in DRL_results.iterrows()], dtype=float)
        subsidies = np.array([result['total_subsidies'] for index, result in DRL_results.iterrows()], dtype=float)
        envies = np.array([result['mean_envy']   for index, result in DRL_results.iterrows()], dtype=float)

        # 计算 DRL 策略奖励的移动平均和95%置信区间
        profits = revenues - subsidies
        profit_means, profit_lower, profit_upper = self.moving_average_with_ci(profits)
        envy_means, envy_lower, envy_upper = self.moving_average_with_ci(envies)

        # 子图1：奖励曲线对比
        axs[0].plot(episodes, profit_means, marker='o', markersize=2.5, color='red',
                    label=f'Policy {self.paras}')
        axs[0].fill_between(episodes, profit_lower, profit_upper, color='red', alpha=0.2)
        y1 = profit_lower
        y1.extend(profit_upper)

        # 子图2：训练损失曲线
        axs[1].plot(episodes, envy_means, marker='o', markersize=2.5, color='red',
                    label=f'Policy {self.paras}')
        axs[1].fill_between(episodes, envy_lower, envy_upper, color='red', alpha=0.2)
        y2 = envy_lower
        y2.extend(envy_upper)

        #### OPT DRL policy
        OPT_path = os.path.join(work_dir, self.OPT_file)
        OPT_results = pd.read_csv(OPT_path)

        episodes_1 = [result['episode'] for index, result in OPT_results.iterrows()]
        revenues = np.array([result['total_revenues'] for index, result in OPT_results.iterrows()], dtype=float)
        subsidies = np.array([result['total_subsidies'] for index, result in OPT_results.iterrows()], dtype=float)
        envies = np.array([result['mean_envy'] for index, result in OPT_results.iterrows()], dtype=float)

        # 计算 DRL 策略奖励的移动平均和95%置信区间
        profits = revenues - subsidies
        profit_means_1, profit_lower_1, profit_upper_1 = self.moving_average_with_ci(profits)
        envy_means_1, envy_lower_1, envy_upper_1 = self.moving_average_with_ci(envies)

        # 子图1：奖励曲线对比
        axs[0].plot(episodes_1, profit_means_1, marker='^', markersize=2.5, color='green',
                    label=f'Policy {self.paras_opt}')
        axs[0].fill_between(episodes_1, profit_lower_1, profit_upper_1, color='green', alpha=0.2)
        y1.extend(profit_lower_1)
        y1.extend(profit_upper_1)

        # 子图2：训练损失曲线
        axs[1].plot(episodes_1, envy_means_1, marker='^', markersize=2.5, color='green',
                    label=f'Policy {self.paras_opt}')
        axs[1].fill_between(episodes_1, envy_lower_1, envy_upper_1, color='green', alpha=0.2)
        y2.extend(envy_lower_1)
        y2.extend(envy_upper_1)

        # Final
        lowest = np.min(np.array(y1))
        highest = np.max(np.array(y1))
        axs[0].set_ylim(lowest, highest)
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel(f'Total Profits (Window={self.window})')
        axs[0].set_title('Total Profits for DRL policies')
        axs[0].legend()
        axs[0].grid(True)

        lowest = np.min(np.array(y2))
        highest = np.max(np.array(y2))
        axs[1].set_ylim(lowest, highest)
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel(f'Request Envies (Window={self.window})')
        axs[1].set_title('Request Envies for DRL policies')
        axs[1].legend()
        axs[1].grid(True)

        # 保存图像为 PNG 格式
        current_datetime = datetime.now().strftime('%m-%d')
        fig = f'Dif_DRL_policy_{current_datetime}.png'
        output_file = os.path.join(data_params['result_dir'], fig)
        plt.savefig(output_file)

        plt.show()
        plt.close()


    def plot_policies(self):
        '''
        ['episode', 'total_demand', 'served', 'charged', 'rejection',
        'total_rewards', 'total_revenues', 'total_subsidies',
        'max_wait', 'mean_wait', 'std_wait', 'max_detour', 'mean_detour', 'std_detour',
        'max_subsidy', 'mean_subsidy', 'std_subsidy', 'max_envy', 'mean_envy', 'std_envy',
        'max_queue', 'mean_queue', 'std_queue',  'avg_station_serve', 'std_station_serve',
        'avg_station_charge', 'std_station_charge', 'avg_cls_serve', 'std_cls_serve',
        'avg_cls_envy', 'std_cls_envy',
        'cluster_{c}',
        'run_time', 'loss']
        '''
        Greedy_path = os.path.join(work_dir, self.Greedy_file)
        RHO_path = os.path.join(work_dir, self.RHO_file)
        DRL_path = os.path.join(work_dir, self.DRL_file)

        Greedy_results = pd.read_csv(Greedy_path)
        RHO_results = pd.read_csv(RHO_path)
        DRL_results = pd.read_csv(DRL_path)

        # greedy_results 中的 rewards 和 subsidies数据
        greedy_episodes = [result['episode']   for index, result in Greedy_results.iterrows()]
        greedy_rewards = [result['total_rewards']   for index, result in Greedy_results.iterrows()]
        greedy_req_subsidies = [result['mean_subsidy']   for index, result in Greedy_results.iterrows()]
        # 计算 greedy 策略奖励和补贴的移动平均和95%置信区间
        greedy_reward_means = self.moving_average(greedy_rewards)
        greedy_subsidy_means = self.moving_average(greedy_req_subsidies)

        # RHO_results 中的 rewards 和 subsidies数据
        RHO_episodes = [result['episode']   for index, result in RHO_results.iterrows()]
        RHO_rewards = [result['total_rewards']   for index, result in RHO_results.iterrows()]
        RHO_req_subsidies = [result['mean_subsidy']  for index, result in RHO_results.iterrows()]
        # 计算 greedy 策略奖励和补贴的移动平均和95%置信区间
        RHO_reward_means = self.moving_average(RHO_rewards)
        RHO_subsidy_means = self.moving_average(RHO_req_subsidies)

        # Compute total (final cumulative) rewards for Greedy and RHO strategies
        arr = np.array(greedy_rewards)
        greedy_reward_mean = np.mean(arr)
        arr = np.array(RHO_rewards)
        rho_reward_mean = np.mean(arr)

        # Compute total (final cumulative) subsidys for Greedy and RHO strategies
        arr = np.array(greedy_req_subsidies)
        greedy_subsidy_mean = np.mean(arr)
        arr = np.array(RHO_req_subsidies)
        rho_subsidy_mean = np.mean(arr)

        # 提取 DRL_results 中的 rewards 和 subsidies数据
        drl_episodes = [result['episode'] for index, result in DRL_results.iterrows()]
        drl_rewards = [result['total_rewards'] for index, result in DRL_results.iterrows()]
        drl_req_subsidies = [result['mean_subsidy']  for index, result in DRL_results.iterrows()]
        # 计算 DRL 策略奖励和补贴的移动平均和95%置信区间
        drl_reward_means, drl_reward_lower, drl_reward_upper = self.moving_average_with_ci(drl_rewards)
        drl_subsidy_means, drl_subsidy_lower, drl_subsidy_upper = self.moving_average_with_ci(drl_req_subsidies)

        # 创建包含两个子图的图像
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # 子图1：奖励曲线对比
        axs[0].plot(greedy_episodes, greedy_reward_means, linestyle=':', marker='+', markersize=2.5, color='green',
                    label='Fair Greedy Policy')
        axs[0].plot(RHO_episodes, RHO_reward_means, linestyle='--', marker='x', markersize=2.5, color='blue',
                    label='Fair RHO Policy')
        # Draw horizontal lines for Greedy and RHO total rewards
        axs[0].axhline(y=greedy_reward_mean, color='green', linestyle='-', label='Fair Greedy Average Reward')
        axs[0].axhline(y=rho_reward_mean, color='blue', linestyle='-', label='Fair RHO Average Reward')

        #fairDRL policy
        axs[0].plot(drl_episodes, drl_reward_means, marker='o', markersize=2.5, color='red',
                    label='Fair DRL Policy')
        axs[0].fill_between(drl_episodes, drl_reward_lower, drl_reward_upper, color='red', alpha=0.2)

        # Set y-axis range and precision for subplot 1
        y = []
        y.extend(greedy_rewards)
        y.extend(RHO_rewards)
        y.extend(drl_reward_lower)
        y.extend(drl_reward_upper)
        lowest = np.min(np.array(y))
        highest = np.max(np.array(y))
        axs[0].set_ylim(lowest, highest)
        axs[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel(f'Total Rewards (Window={self.window})')
        axs[0].set_title('(a) Total rewards of policies')
        axs[0].legend()
        axs[0].grid(True)

        # 子图2：补贴曲线对比
        axs[1].plot(greedy_episodes, greedy_subsidy_means, linestyle=':', marker='+', markersize=2.5, color='green',
                    label='Fair Greedy Policy Subsidies')
        axs[1].plot(RHO_episodes, RHO_subsidy_means, linestyle='--', marker='x', markersize=2.5, color='blue',
                    label='Fair RHO Policy Subsidies')
        # Draw horizontal lines for Greedy and RHO total subsidys
        axs[1].axhline(y=greedy_subsidy_mean, color='green', linestyle=':', label='Greedy Average Subsidy')
        axs[1].axhline(y=rho_subsidy_mean, color='blue', linestyle='--', label='RHO Average Subsidy')

        #DRL policies
        axs[1].plot(drl_episodes, drl_subsidy_means, marker='o', markersize=2.5, color='red',
                    label='Fair DRL Policy Subsidies')
        axs[1].fill_between(drl_episodes, drl_subsidy_lower, drl_subsidy_upper, color='red', alpha=0.2)

        # Set y-axis range and precision for subplot 1
        y = []
        y.extend(greedy_req_subsidies)
        y.extend(RHO_req_subsidies)
        y.extend(drl_subsidy_lower)
        y.extend(drl_subsidy_upper)
        lowest = np.min(np.array(y))
        highest = np.max(np.array(y))
        axs[1].set_ylim(lowest, highest)
        axs[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel(f'Request subsidy (Window={self.window})')
        axs[1].set_title('(b) Average request subsidy')
        axs[1].legend()
        axs[1].grid(True)

        # 保存图像为 PNG 格式
        current_datetime = datetime.now().strftime('%m-%d')
        fig = f'policies_average_{current_datetime}.png'
        output_file = os.path.join(data_params['result_dir'], fig)
        plt.savefig(output_file)

        plt.show()
        plt.close()


    def plot_envies(self):
        '''
        ['episode', 'total_demand', 'served', 'charged', 'rejection',
        'total_rewards', 'total_revenues', 'total_subsidies',
        'max_wait', 'mean_wait', 'std_wait', 'max_detour', 'mean_detour', 'std_detour',
        'max_subsidy', 'mean_subsidy', 'std_subsidy', 'max_envy', 'mean_envy', 'std_envy',
        'max_queue', 'mean_queue', 'std_queue',  'avg_station_serve', 'std_station_serve',
        'avg_station_charge', 'std_station_charge', 'avg_cls_serve', 'std_cls_serve',
        'avg_cls_envy', 'std_cls_envy',
        'cluster_{c}',
        'run_time', 'loss']
        '''
        Greedy_path = os.path.join(work_dir, self.Greedy_file)
        RHO_path = os.path.join(work_dir, self.RHO_file)
        DRL_path = os.path.join(work_dir, self.DRL_file)

        Greedy_results = pd.read_csv(Greedy_path)
        RHO_results = pd.read_csv(RHO_path)
        DRL_results = pd.read_csv(DRL_path)

        # greedy_results 中的 avg_cls_envy 和 mean_envy 数据
        greedy_episodes = [result['episode']   for index, result in Greedy_results.iterrows()]
        greedy_req_subsidies = [result['mean_subsidy']   for index, result in Greedy_results.iterrows()]
        greedy_req_envies = [result['mean_envy']   for index, result in Greedy_results.iterrows()]
        # 计算 greedy 策略奖励和补贴的移动平均和95%置信区间
        greedy_subsidy_means = self.moving_average(greedy_req_subsidies)
        greedy_envy_means = self.moving_average(greedy_req_envies)

        # RHO_results 中的 avg_cls_envy 和 mean_envy 数据
        RHO_episodes = [result['episode']  for index, result in RHO_results.iterrows()]
        RHO_req_subsidies = [result['mean_subsidy']  for index, result in RHO_results.iterrows()]
        RHO_req_envies = [result['mean_envy']  for index, result in RHO_results.iterrows()]
        # 计算 RHO 策略奖励和补贴的移动平均和95%置信区间
        RHO_subsidy_means = self.moving_average(RHO_req_subsidies)
        RHO_envy_means = self.moving_average(RHO_req_envies)

        # 提取 DRL_results 中的 avg_cls_envy 和 mean_envy 数据
        drl_episodes = [result['episode']  for index, result in DRL_results.iterrows()]
        drl_req_subsidies = [result['mean_subsidy']  for index, result in DRL_results.iterrows()]
        drl_req_envies = [result['mean_envy']  for index, result in DRL_results.iterrows()]
        # 计算 DRL 策略奖励和补贴的移动平均和95%置信区间
        drl_subsidy_means, drl_subsidy_lower, drl_subsidy_upper = self.moving_average_with_ci(drl_req_subsidies)
        drl_envy_means, drl_envy_lower, drl_envy_upper = self.moving_average_with_ci(drl_req_envies)

        # 创建包含两个子图的图像
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # 子图1：subsidy 曲线对比
        axs[0].plot(greedy_episodes, greedy_subsidy_means, linestyle=':', marker='+', markersize=2.5, color='green',
                    label='Fair Greedy Policy')
        axs[0].plot(RHO_episodes, RHO_subsidy_means, linestyle='--', marker='x', markersize=2.5, color='blue',
                    label='Fair RHO Policy')

        #fairDRL policy
        axs[0].plot(drl_episodes, drl_subsidy_means, marker='o', markersize=2.5, color='red',
                    label='Fair DRL Policy')
        axs[0].fill_between(drl_episodes, drl_subsidy_lower, drl_subsidy_upper, color='red', alpha=0.2)

        # Set y-axis range and precision for subplot 1
        y = []
        y.extend(greedy_subsidy_means)
        y.extend(RHO_subsidy_means)
        y.extend(drl_subsidy_lower)
        y.extend(drl_subsidy_upper)
        lowest = np.min(np.array(y))
        highest = np.max(np.array(y))
        axs[0].set_ylim(lowest, highest)
        axs[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel(f'Request subsidy (Window={self.window})')
        axs[0].set_title('(a) Average request subsidy')
        axs[0].legend()
        axs[0].grid(True)

        # 子图2：envy 曲线对比
        axs[1].plot(greedy_episodes, greedy_envy_means, linestyle=':', marker='+', markersize=2.5, color='green',
                    label='Fair Greedy Policy Subsidies')
        axs[1].plot(RHO_episodes, RHO_envy_means, linestyle='--', marker='x', markersize=2.5, color='blue',
                    label='Fair RHO Policy Subsidies')

        #fairDRL policies
        axs[1].plot(drl_episodes, drl_envy_means, marker='o', markersize=2.5, color='red',
                    label='Fair DRL Policy Subsidies')
        axs[1].fill_between(drl_episodes, drl_envy_lower, drl_envy_upper, color='red', alpha=0.2,
                            label='Fair DRL Subsidies 95% CI')

        # Set y-axis range and precision for subplot 1
        y = []
        y.extend(greedy_envy_means)
        y.extend(RHO_envy_means)
        y.extend(drl_envy_lower)
        y.extend(drl_envy_upper)
        lowest = np.min(np.array(y))
        highest = np.max(np.array(y))
        axs[1].set_ylim(lowest, highest)
        axs[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel(f'Request envy (Window={self.window})')
        axs[1].set_title('(b) Average request envies')
        axs[1].legend()
        axs[1].grid(True)

        # 保存图像为 PNG 格式
        current_datetime = datetime.now().strftime('%m-%d')
        fig = f'envies_average_{current_datetime}.png'
        output_file = os.path.join(data_params['result_dir'], fig)
        plt.savefig(output_file)

        plt.show()
        plt.close()


    def plot_tests(self):
        Greedy_path = os.path.join(work_dir, self.Greedy_test)
        RHO_path = os.path.join(work_dir, self.RHO_test)
        DRL_path = os.path.join(work_dir, self.DRL_test)
        OPT_path = os.path.join(work_dir, self.OPT_test)

        Greedy_results = pd.read_csv(Greedy_path)
        RHO_results = pd.read_csv(RHO_path)
        DRL_results = pd.read_csv(DRL_path)
        OPT_results = pd.read_csv(OPT_path)

        # greedy_results 中的 rewards 和 subsidies数据
        greedy_episodes = [result['episode'] for index, result in Greedy_results.iterrows()]
        greedy_rewards = [result['total_rewards'] for index, result in Greedy_results.iterrows()]
        greedy_subsidies = [result['total_subsidies'] for index, result in Greedy_results.iterrows()]

        # RHO_results 中的 rewards 和 subsidies数据
        RHO_episodes = [result['episode'] for index, result in RHO_results.iterrows()]
        RHO_rewards = [result['total_rewards'] for index, result in RHO_results.iterrows()]
        RHO_subsidies = [result['total_subsidies'] for index, result in RHO_results.iterrows()]

        # 提取 DRL_results 中的 rewards 和 subsidies数据
        drl_episodes = [result['episode'] for index, result in DRL_results.iterrows()]
        drl_rewards = [result['total_rewards'] for index, result in DRL_results.iterrows()]
        drl_subsidies = [result['total_subsidies'] for index, result in DRL_results.iterrows()]

        # 提取 opt_results 中的 rewards 和 subsidies数据
        opt_episodes = [result['episode'] for index, result in OPT_results.iterrows()]
        opt_rewards = [result['total_rewards'] for index, result in OPT_results.iterrows()]
        opt_subsidies = [result['total_subsidies'] for index, result in OPT_results.iterrows()]

        # 创建包含两个子图的图像
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # 子图1：奖励曲线对比


        # Set y-axis range and precision for subplot 1
        y = []
        y.extend(greedy_rewards)
        y.extend(RHO_rewards)
        y.extend(drl_rewards)
        y.extend(opt_rewards)

        lowest = np.min(np.array(y))
        highest = np.max(np.array(y))
        axs[0].set_ylim(lowest, highest)
        axs[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel(f'Total Rewards (Window={self.window})')
        axs[0].set_title('(a) Reward Curves Comparison')
        axs[0].legend()
        axs[0].grid(True)

        # 子图2：补贴曲线对比


        # Set y-axis range and precision for subplot 1
        y = []
        y.extend(greedy_subsidies)
        y.extend(RHO_subsidies)
        y.extend(drl_subsidies)
        y.extend(opt_subsidies)

        lowest = np.min(np.array(y))
        highest = np.max(np.array(y))
        axs[1].set_ylim(lowest, highest)
        axs[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel(f'Total Subsidies (Window={self.window})')
        axs[1].set_title('(b) Subsidy Curves Comparison')
        axs[1].legend()
        axs[1].grid(True)

        # 保存图像为 PNG 格式
        current_datetime = datetime.now().strftime('%m-%d')
        fig = f'policies_average_{current_datetime}.png'
        output_file = os.path.join(data_params['result_dir'], fig)
        plt.savefig(output_file)

        plt.show()
        plt.close()


    def moving_average_with_ci(self, data):
        '''
        distrDRLPolicy 的奖励曲线：横坐标为 episode，每个 episode 对应前 5 个 episode 的平均总奖励，
        并以填充区域的形式显示 95% 置信区间（计算公式：均值 ± 1.96 * (std / sqrt(n))）。
        - 同时加入 GreedyScheduler 的奖励曲线（采用相同的移动平均处理，不显示置信区间）。
        '''
        means = []
        lower_bounds = []
        upper_bounds = []

        for i in range(len(data)):
            start = max(0, i - self.window + 1)
            window_data = data[start:i + 1]
            m = np.mean(window_data)
            # 计算无偏标准差，若窗口内样本数为1则设为0
            s = np.std(window_data, ddof=1) if len(window_data) > 1 else 0.0
            n = len(window_data)
            # 计算95%置信区间，采用正态分布近似：1.96 * s / sqrt(n)
            ci = 1.96 * s / np.sqrt(n) if n > 1 else 0.0

            means.append(m)
            lower_bounds.append(m - ci)
            upper_bounds.append(m + ci)

        # End
        return  means, lower_bounds, upper_bounds


    def moving_average(self, data):
        ma = []
        for i in range(len(data)):
            start = max(0, i - self.window + 1)
            avg = np.mean(data[start:i + 1])
            ma.append(avg)

        # End
        return  ma

