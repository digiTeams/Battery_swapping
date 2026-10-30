"""Manual dispatcher: train one policy, test a policy list, or run one
sensitivity environment."""

from common.optimization import Optimizer
from runners.sensitivities import Sensitivities
from runners.test_runner import TestRunner
from runners.trainers import Trainer
from config import RL_params
from simulator.BS_env import BatterySwapEnv


# Trainer method names; "wgt4DRL" is the weighted-fairness alias of optDRL.
TRAIN_TASKS = (
    "fairDRL",
    "fairDRL_one",
    "fairDRL_local",
    "fairDRL_fixed",
    "optDRL",
    "wgt4DRL",
)

TEST_TASKS = [
    "fairGreedy",
    "fairRHO",
    "fairDRL",
    "optDRL",
    "wgt4DRL",
]


def _run_train(task, resume=False, target_fixed=False, start_episode=None):
    """Run exactly one training policy (a single known name, never a list).
    ``target_fixed`` applies to "Baseline" only; ``start_episode`` is honored
    only with ``resume=True``."""
    if not isinstance(task, str) or (task != "Baseline" and task not in TRAIN_TASKS):
        raise ValueError(
            f"Training mode runs exactly one policy per run; "
            f"expected 'Baseline' or a name in {sorted(TRAIN_TASKS)}, got {task!r}"
        )
    env = BatterySwapEnv(seed=None)
    env.get_network("training", None)
    trainer = Trainer(env)
    if task == "Baseline":
        return trainer.baselines(
            training=True,
            resume=resume,
            target_fixed=target_fixed,
            start_episode=start_episode,
        )
    return getattr(trainer, task)(
        training=True, resume=resume, start_episode=start_episode
    )


def _run_test(tasks, scenario, scale=None, resume=True):
    """Run every policy in the test list on one expansion network; resume
    continues an interrupted sweep from the completed (policy, episode) prefix."""
    if not tasks:
        raise ValueError("Test mode requires a non-empty TEST_TASKS policy list")
    return TestRunner(
        policies=tasks, scenario=scenario, scale=scale, resume=resume
    ).run()


def _run_sensitivities(parameter, scenario=None, resume=True):
    """Run one 30-episode sensitivity environment on the selected network
    (parameter=None: default config; otherwise the non-default level of one
    parameter). ``scenario`` mirrors test mode; resume continues the run."""
    return Sensitivities(parameter=parameter, scenario=scenario, resume=resume).run()


# Valid single-task names: TRAIN_TASKS above plus "Baseline".
task = "fairDRL"
MODE = "test"        # "train" / "test" / "sensitivity"
resume = False       # True: continue an interrupted run; False: start fresh
PARAMETER = None     # None or "detour_cap" / "wait_cap" / "poisson_rate" / "low_class_prob"

if __name__ == "__main__":
    Optimizer.configure_solver(preferred="gurobi")
    print(f"Selected licensed solver: {RL_params['solver']}")
    if MODE == "train":
        _run_train(task, resume, False)
    elif MODE == "test":
        # scenario: 0 (or 8) -> plus8 network; 1 (or 16) -> plus16 network
        _run_test(TEST_TASKS, 1, 2.0, resume)
    elif MODE == "sensitivity":
        # scenario: None -> base network; 0/8 -> plus8; 1/16 -> plus16
        _run_sensitivities(PARAMETER, None, resume)
    else:
        raise ValueError(
            f"Unknown MODE: {MODE!r} (expected 'train', 'test', or 'sensitivity')"
        )

