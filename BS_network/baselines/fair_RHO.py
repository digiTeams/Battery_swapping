"""Rolling-horizon joint BMIP policy (one joint solve per 5-minute decision).
``target_fixed`` selects the fixed ``fixed_level`` charging target."""
from time import perf_counter

import numpy as np
from common.optimization import Optimizer
from config import RL_params


class RHOPolicy:
    """Rolling-horizon joint bundle/charging BMIP with hard envy-free
    constraints and subsidies."""
    def __init__(self, env, target_fixed=False):
        """Set the charging-target switch, identity, and solver seed streams."""
        self.environment = env  # BatterySwapEnv: joint demand and station state
        self.type = "Fair"  # str: reward convention
        # Charging-target and result-file naming switch.
        self.target_fixed = bool(target_fixed)  # bool: fixed fixed_level charging targets
        self.name = "fairRHO_fixed" if self.target_fixed else "fairRHO"

        self.seed = int(RL_params["seed"])  # int: root learning seed
        self.seeds = Optimizer.learning_seeds(self.seed)  # dict[str, int] of the four seed streams
        self.rng = np.random.default_rng(self.seeds["exploration"])  # exploration Generator

    def RHO_schedule(self, bundling):
        """Enumerate bundles, build candidates, solve the fair BMIP, record envy."""
        self.environment.last_diagnostics = {}
        bundling.start_bundling()
        start = perf_counter()
        options = bundling.build_options(target_fixed=self.target_fixed)
        self.environment.last_diagnostics["candidate_seconds"] = perf_counter() - start

        action = bundling.bundle_optimize(
            options,
            mode="fair",
            rng=self.rng,
            solver_seed=self.seeds["solver"],
            policy=self.name,
            target_fixed=self.target_fixed,
        )

        return action

    def scheduling(self, bundling, train_flag=False, episode=None, name=None):
        """Play one episode; train_flag exists only for the uniform call signature."""
        episode = self.environment.episode if episode is None else episode
        start = perf_counter()
        self.environment.reset()
        done = False

        while not done:
            action_start = perf_counter()
            action = self.RHO_schedule(bundling)
            self.environment.last_diagnostics["action_seconds"] = perf_counter() - action_start
            _, _, done = self.environment.step(action, self.type)

        result = self.environment.service_records(episode)
        result.update(run_time=(perf_counter() - start) / 60)
        self.environment.request_records(
            name or self.name, episode, evaluation=not train_flag,
            zero_envy=self.type == "Fair",
        )
        return result
