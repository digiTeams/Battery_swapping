###Import packages
import os
import time
from network import BS_networks, instances
from analyses.plots import Plots
from simulator.BS_env import BatterySwapEnv
from trainers import Trainer

if __name__ == '__main__':
    env = BatterySwapEnv(seed = None)

    ##Train the policies
    task = 'optTest'  # 'Baseline', 'fairDRL', 'optDRL', 'fairTest', 'optTest'
    match task:
        case 'Baseline':
            sn = None  # None, 28
            env.load_network(task, sn)
            trainer = Trainer(env)
            trainer.baselines(sn=sn)

        case 'fairDRL':
            sn = None  # None, 28
            env.load_network(task, sn)
            trainer = Trainer(env)
            trainer.training_fairDRL(sn=sn)

        case 'optDRL':
            sn = None  # None, 28
            env.load_network(task, sn)
            trainer = Trainer(env)
            trainer.training_optDRL(sn=sn)

        case 'fairTest':
            sn = 0
            Scenarios = [4, 8, 16]
            env.load_network(task, sn)
            trainer = Trainer(env)
            trainer.test_Fairs(sn)

        case 'optTest':
            sn = 2
            Scenarios = [4, 8, 16]
            env.load_network(task, sn)
            trainer = Trainer(env)
            trainer.test_OPT(sn)





