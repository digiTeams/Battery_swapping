"""Generic value-based DRL learner: value/target networks, replay, TD training, checkpoints."""

import copy
import json
import warnings
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from common.optimization import Optimizer
from config import RL_params
from DRL.checkpoint_checks import (
    configuration_mismatches,
    identity_mismatches,
    value_input_mismatches,
)
from DRL.valueNet import ValueNetwork


__all__ = ["ReplayBuffer", "BaseDRL"]


class ReplayBuffer:
    """Bounded replay of joint transitions with recency-weighted sampling; tensor/container round-trip."""

    # Per-station observation layout (name, width), matching StationAgent.get_observe.
    OBS_LAYOUT = (
        ("time_step", 1),
        ("hour", 1),
        ("new_order", 2),
        ("upgrade_order", 1),
        ("que_order", 2),
        ("avg_waiting", 2),
        ("full_batteries", 2),
        ("inventory_position", 2),
        ("empty_batteries_low", 1),
        ("empty_batteries_high", 1),
        ("next_ready_low", 1),
        ("next_ready_high", 1),
        ("spare_grid", 1),
        ("bay_free", 1),
        ("capacity", 1),
        ("type", 2),
    )
    OBS_WIDTH = sum(width for _, width in OBS_LAYOUT)

    def __init__(self, capacity, batch_size, seed=None, weight_decay=1.0):
        """Initialize bounded transition storage and its sampling random stream."""
        self.memory = deque(maxlen=capacity)  # (encoded state, float32 reward (J,), next state, done)
        self.capacity = int(capacity)         # int: bounded transition count
        self.batch_size = batch_size  # int: transitions sampled per update
        weight_decay = float(weight_decay)
        self.weight_decay = weight_decay  # float: per-transition recency decay (1.0 = uniform)
        self.rng = np.random.default_rng(  # Generator sampling replay indices
            Optimizer.learning_seeds()["replay"] if seed is None else seed
        )

    @classmethod
    def _encode_state(cls, state):
        """Encode one joint-state dict into (time_step, station_ids, obs, memberships, nearest)."""
        observations = state["agents"]
        station_ids = tuple(int(j) for j in sorted(observations))
        rows = []
        for j in station_ids:
            observation = observations[j]
            row = []
            for key, width in cls.OBS_LAYOUT:
                value = observation[key]
                if isinstance(value, (tuple, list, np.ndarray)):
                    row.extend(float(x) for x in value)
                else:
                    row.append(float(value))
            rows.append(row)
        rows = np.asarray(rows, dtype=np.float32)
        memberships = tuple(
            (int(j), tuple(int(x) for x in low), tuple(int(x) for x in high))
            for j, (low, high) in state["memberships"].items()
        )
        nearest = tuple(
            sorted((int(i), int(j)) for i, j in state.get("nearest", {}).items())
        )
        return (
            int(state["time_step"]),
            station_ids,
            rows,
            memberships,
            nearest,
        )

    @classmethod
    def _decode_state(cls, encoded):
        """Decode one compact row back into the joint-state dict contract."""
        time_step, station_ids, rows, memberships, nearest = encoded
        agents = {}
        for index, j in enumerate(station_ids):
            row = rows[index]
            agents[j] = {
                "time_step": int(row[0]),
                "hour": int(row[1]),
                "new_order": (int(row[2]), int(row[3])),
                "upgrade_order": (int(row[4]),),
                "que_order": (int(row[5]), int(row[6])),
                "avg_waiting": (float(row[7]), float(row[8])),
                "full_batteries": (int(row[9]), int(row[10])),
                "inventory_position": (int(row[11]), int(row[12])),
                "empty_batteries_low": int(row[13]),
                "empty_batteries_high": int(row[14]),
                "next_ready_low": float(row[15]),
                "next_ready_high": float(row[16]),
                "spare_grid": float(row[17]),
                "bay_free": float(row[18]),
                "capacity": int(row[19]),
                "type": (int(row[20]), int(row[21])),
            }
        decoded_memberships = {
            int(j): (tuple(int(x) for x in low), tuple(int(x) for x in high))
            for j, low, high in memberships
        }
        return {
            "time_step": int(time_step),
            "agents": agents,
            "memberships": decoded_memberships,
            "nearest": {int(i): int(j) for i, j in nearest},
        }

    def push(self, state, reward, next_state, done):
        """Store an independent, compact copy of a joint transition."""
        # Encode and copy so later environment changes cannot alter stored transitions.
        current, successor = self._encode_state(state), self._encode_state(next_state)
        reward = np.array(reward, dtype=np.float32, copy=True)
        self.memory.append((current, reward, successor, bool(done)))

    def _sample_weights(self):
        """Normalized recency weights (decay**age, newest weight 1); uniform at decay 1.0."""
        size = len(self.memory)
        if self.weight_decay >= 1.0 or size == 0:
            return None
        # Double precision for NumPy choice; observations/rewards stay float32.
        ages = np.arange(size - 1, -1, -1, dtype=np.float64)
        weights = np.power(self.weight_decay, ages)
        return weights / weights.sum()

    def sample(self):
        """Sample joint transitions without replacement (recency-weighted) and decode."""
        # Draw batch_size distinct indices, then rebuild the dict contract for training.
        indices = self.rng.choice(
            len(self.memory),
            self.batch_size,
            replace=False,
            p=self._sample_weights(),
        )
        return [
            (
                self._decode_state(self.memory[i][0]),
                self.memory[i][1],
                self._decode_state(self.memory[i][2]),
                self.memory[i][3],
            )
            for i in indices
        ]

    def __len__(self):
        """Return the number of stored transitions."""
        return len(self.memory)

    def state_dict(self):
        """Serialize the buffer into tensors and plain containers (weights_only-safe)."""
        # Tensors for observations/rewards; plain containers for the rest (weights_only-safe).
        rows = list(self.memory)
        station_ids = rows[0][0][1] if rows else ()
        width = len(station_ids)
        return {
            "capacity": self.capacity,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "rng": self.rng.bit_generator.state,
            "station_ids": station_ids,
            "time_steps": [int(row[0][0]) for row in rows],
            "observations": (
                torch.stack([torch.as_tensor(row[0][2]) for row in rows])
                if rows else torch.empty((0, width, self.OBS_WIDTH))
            ),
            "memberships": [row[0][3] for row in rows],
            "nearest": [row[0][4] for row in rows],
            "rewards": (
                torch.stack([torch.as_tensor(row[1]) for row in rows])
                if rows else torch.empty((0, width))
            ),
            "next_time_steps": [int(row[2][0]) for row in rows],
            "next_observations": (
                torch.stack([torch.as_tensor(row[2][2]) for row in rows])
                if rows else torch.empty((0, width, self.OBS_WIDTH))
            ),
            "next_memberships": [row[2][3] for row in rows],
            "next_nearest": [row[2][4] for row in rows],
            "dones": [bool(row[3]) for row in rows],
        }

    @classmethod
    def from_state_dict(cls, data):
        """Rebuild a buffer from its plain-container representation."""
        # 1) Extract and type-check every payload field.
        capacity = int(data["capacity"])
        batch_size = int(data["batch_size"])
        # Legacy buffers predate recency decay; restore with current config.
        weight_decay = float(
            data.get("weight_decay", RL_params.get("replay_time_decay", 1.0))
        )
        station_ids = tuple(int(j) for j in data["station_ids"])
        time_steps = data["time_steps"]
        next_time_steps = data["next_time_steps"]
        dones = data["dones"]
        observations = data["observations"]
        next_observations = data["next_observations"]
        rewards = data["rewards"]
        memberships = data["memberships"]
        next_memberships = data["next_memberships"]
        nearest = data.get("nearest")
        next_nearest = data.get("next_nearest")
        # 2) Rebuild the deque row by row and restore the sampling RNG state.
        count = len(time_steps)
        if nearest is None:
            nearest = [()] * count
        if next_nearest is None:
            next_nearest = [()] * count
        buffer = cls(capacity, batch_size, seed=0, weight_decay=weight_decay)
        buffer.rng.bit_generator.state = data["rng"]
        for i in range(count):
            buffer.memory.append(
                (
                    (
                        int(time_steps[i]),
                        station_ids,
                        np.asarray(observations[i], dtype=np.float32),
                        memberships[i],
                        tuple((int(a), int(b)) for a, b in nearest[i]),
                    ),
                    np.asarray(rewards[i], dtype=np.float32),
                    (
                        int(next_time_steps[i]),
                        station_ids,
                        np.asarray(next_observations[i], dtype=np.float32),
                        next_memberships[i],
                        tuple((int(a), int(b)) for a, b in next_nearest[i]),
                    ),
                    bool(dones[i]),
                )
            )
        return buffer


class BaseDRL:
    """Base DRL learner shared by every DRL policy: seed/RNG streams, online
    value network + frozen target, Adam, replay, TD(0) training/decay, v5
    checkpoints, the scheduling loop, and GAT scoring + joint BMIP selection."""

    # Explicit BMIP semantics per learner type.
    MODE_BY_TYPE = {"Fair": "fair", "Weighted": "weighted", "Optimum": "efficiency"}

    def __init__(self, env, state_dims=None):
        """Initialize this learner's networks, optimizer, replay, seeds, and counters."""
        self.environment = env  # BatterySwapEnv: joint demand and station state
        self.seed = int(RL_params["seed"])  # int: root learning seed
        self.seeds = Optimizer.learning_seeds(self.seed)  # dict[str, int]: four derived seed streams
        self.rng = np.random.default_rng(self.seeds["exploration"])  # Generator: exploration stream

        # name/type are set by every concrete policy after super().__init__().
        self.ablation: str | None = None  # str or None: structural ablation tag
        self.bundle_limit: int | None = None  # int or None: maximum requests per station bundle
        self.target_fixed: bool = False  # bool: fixed fixed_level charging target; only FairDRLFixed sets True

        # Deterministic torch runtime, then the seeded network construction.
        torch.set_num_threads(RL_params["torch_threads"])
        torch.use_deterministic_algorithms(True)

        self.state_dims = state_dims or {"steps": env.total_steps}  # dict[str, int]: network input sizes

        # fork_rng isolates the init seed from the caller's global torch stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seeds["initialization"])
            self.value_net = self._build_value_network()  # seeded; subclasses override the hook

        # Frozen target copy, online-only Adam optimizer, and the compact replay buffer.
        self.target_net = copy.deepcopy(self.value_net)  # frozen, forward-only
        self.target_net.requires_grad_(False)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(  # Adam over online parameters only
            self.value_net.parameters(), lr=RL_params["initial_learning"]
        )
        self.replay_buffer = ReplayBuffer(  # capacity, batch size, replay seed, recency decay
            RL_params["buffer_size"],
            RL_params["batch_size"],
            self.seeds["replay"],
            weight_decay=float(RL_params.get("replay_time_decay", 1.0)),
        )

        # Training schedule and counters.
        self.epsilon = RL_params["initial_epsilon"]
        self.total_steps = self.gradient_steps = self.sync_count = 0
        self.training_episodes = 0            # completed training episodes (absolute resume start)
        self.envy_weight = None               # float or None: weighted-mode envy coefficient; only OptimumDRL sets it
        self.resume_supported = True          # a fresh learner supports resume by construction
        self.checkpoint_stem = None           # str or None: saved run family retained on resume
        self.checkpoint_path = None           # Path or None: exact destination retained on resume
        self.resume_configuration = None      # dict or None: checkpoint's effective training config
        self.training_time = 0.0              # cumulative pure gradient-update seconds
        self._numerical_error = None          # stop this run until a valid checkpoint is restored

    def _check_finite(self, label, *values):
        """Reject nonfinite values and block further updates/checkpoint
        writes; no-op when validation is disabled."""
        if not RL_params["validation"]:
            return
        if self._numerical_error is not None:
            raise FloatingPointError(self._numerical_error)
        pending = list(values)
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                pending.extend(value.values())
                continue
            if isinstance(value, (tuple, list)):
                pending.extend(value)
                continue
            if torch.is_tensor(value):
                finite = bool(torch.isfinite(value).all())
            elif isinstance(value, (np.ndarray, float, np.floating)):
                finite = bool(np.isfinite(value).all())
            else:
                continue
            if not finite:
                self._numerical_error = (
                    f"{self.name}: nonfinite {label} at episode {self.training_episodes}, "
                    f"transition {self.total_steps}; restore a valid checkpoint before continuing"
                )
                raise FloatingPointError(self._numerical_error)

    # OVERRIDABLE HOOK: value-network factory; called once by BaseDRL.__init__.
    def _build_value_network(self):
        """Build the default shared GAT architecture; local policies override
        this hook (called once by BaseDRL.__init__)."""
        return ValueNetwork(
            self.state_dims,
            self.environment.neighbor_mask,
            self.environment.detour_weights,
            use_gat=True,
        )

    def BILP_costs(self, bundling, explore=False):
        """Build the single completion-time charging-target candidates and
        score them with focal successor-value advantages. ``explore`` keeps
        the uniform hook signature: exploration changes only the BMIP solve
        (randomized coefficients and the quick time limit in
        ``bundle_optimize``), never the charging candidates."""
        # options: list[dict], one station/bundle/charging choice per row.
        options = bundling.build_options(
            enumerated=False,
            max_size=self.bundle_limit,
        )
        return self._score_options(bundling, options)

    def _score_options(self, bundling, options):
        """Score candidates as immediate + gamma * advantage over the station's do-nothing baseline."""
        self._check_finite("immediate candidate values", [option["value"] for option in options])
        # Final step has no successor; keep immediate values.
        if self.environment.time_step == self.environment.total_steps - 1:
            return options

        # Pre-filter: only the top candidates per station are shadow-simulated by the
        # GAT; the rest enter the BMIP with immediate value (successor advantage 0).
        score_limit = RL_params.get("candidate_score_top_k")
        if score_limit:
            per_station = {}
            for index, option in enumerate(options):
                per_station.setdefault(option["station"], []).append(index)
            score_indices = []
            for indices in per_station.values():
                indices.sort(key=lambda i: options[i]["value"], reverse=True)
                score_indices.extend(indices[: int(score_limit)])
            score_indices.sort()
        else:
            score_indices = list(range(len(options)))

        next_step = self.environment.time_step + 1
        next_time = self.environment.current_time + timedelta(minutes=self.environment.delta_time)
        # Oracle one-bucket lookahead: this uses unrevealed realized t+1 demand.
        # A nonanticipative deployment needs forecasts or conditional samples here.
        next_demand = {
            agent.id: [r for r in self.environment.demand_by_step[next_step]
                       if agent.id in r.feasible_stations]
            for agent in self.environment.Agents
        }
        # Do-nothing baseline successor per station (empty bundle + zero charging).
        baseline_observations = {
            agent.id: agent.simulate_bundle(
                self.environment.current_time,
                [],
                np.zeros(2, dtype=int),
                next_step,
                next_time.hour,
                next_demand[agent.id],
            )
            for agent in self.environment.Agents
        }
        next_memberships = None
        next_nearest = None
        if self.value_net.use_gat:
            next_memberships = {
                station_id: tuple(
                    tuple(r.id for r in requests if r.requested_class == l) for l in (1, 2)
                )
                for station_id, requests in next_demand.items()
            }
            next_nearest = {
                r.id: r.nearest_station
                for requests in next_demand.values()
                for r in requests
            }

        batch_size = RL_params["candidate_batch_size"]  # chunked forward control

        self.value_net.eval()
        with torch.no_grad():   # value predictions never participate in gradients
            # Cache baseline sender/edge messages; each candidate supplies receiver + features.
            station_ids = [agent.id for agent in self.environment.Agents]
            baseline_state = {
                "time_step": next_step,
                "agents": baseline_observations,
            }
            if self.value_net.use_gat:
                baseline_state["memberships"] = next_memberships
                baseline_state["nearest"] = next_nearest
            primitives = (
                self.value_net.attention_primitives(baseline_state)
                if self.value_net.use_gat
                else None
            )
            baseline_values = np.asarray(
                self.candidate_values(
                    baseline_state,
                    primitives,
                    [baseline_observations[j] for j in station_ids],
                    station_ids,
                ),
                dtype=np.float32,
            )

            for begin in range(0, len(score_indices), batch_size):
                subset = [options[i] for i in score_indices[begin : begin + batch_size]]

                observations, indices = [], []
                for option in subset:
                    j = option["station"]
                    # Focal station's shadow successor; other stations keep the baseline.
                    observation = self.environment.Agents[j].simulate_bundle(
                        self.environment.current_time,
                        [bundling.requests[rid] for rid in option["bundle"]],
                        np.asarray(option["charging"]),
                        next_step,
                        next_time.hour,
                        next_demand[j],
                    )
                    observations.append(observation)
                    indices.append(j)

                values = self.candidate_values(
                    baseline_state, primitives, observations, indices
                )
                for option, value, j in zip(subset, values, indices):
                    option["value"] += self.environment.gamma * (
                        float(value) - baseline_values[j]
                    )
        self._check_finite("candidate values", [option["value"] for option in options])
        return options

    def candidate_values(self, baseline_state, primitives, observations, indices):
        """Return finite local or GAT candidate values, recomputing focal attention rows."""
        # observations: N station dicts; indices: N station IDs; values: float32 (N,).
        network = self.value_net
        if not network.use_gat:
            values = network.local_forward(observations)
            self._check_finite("candidate predictions", values)
            return values.detach().cpu().numpy()
        # Focal attention rows recomputed per candidate; baseline sender/edge messages reused.
        continuous = network.continuous_features(observations)
        times = network.times_tensor(observations)
        hours = network.hours_tensor(observations)
        device = next(network.parameters()).device
        focal = network.node_projection(
            torch.as_tensor(
                network.node_feature_vectors(observations),
                dtype=torch.float32,
                device=device,
            )
        )
        spatial = torch.cat(
            [network.focal_spatial(primitives, int(j), focal[c : c + 1])[None, :]
             for c, j in enumerate(indices)],
            dim=0,
        )
        values = network.score_parts(times, hours, spatial, continuous)
        self._check_finite("candidate predictions", values)
        return values.detach().cpu().numpy()

    def select_action(self, bundling, train_flag=False):
        """Generate bundles, score the candidates, and solve the joint BMIP allocation."""
        # Reset diagnostics before timed bundle generation.
        self.environment.last_diagnostics = {}
        bundling.start_bundling(max_size=self.bundle_limit)

        # Exploration draws first; bundle_optimize randomizes and quick-solves under exploration.
        explore = train_flag and self.rng.random() < self.epsilon
        start = perf_counter()
        options = self.BILP_costs(bundling, explore=explore)
        self.environment.last_diagnostics["candidate_seconds"] = perf_counter() - start

        # Joint BMIP: exploration randomizes the objective under a quick time limit.
        action = bundling.bundle_optimize(
            options,
            mode=self.MODE_BY_TYPE[self.type],
            randomize=explore,
            quick_solve=explore,
            rng=self.rng,
            solver_seed=self.seeds["solver"],
            policy=self.name,
            envy_weight=getattr(self, "envy_weight", None),
            target_fixed=getattr(self, "target_fixed", False),
        )
        return action

    def store_transition(self, state, reward, next_state, done):
        """Store one independent, compact joint transition in replay."""
        self._check_finite("transition", state["agents"], reward, next_state["agents"])
        self.replay_buffer.push(state, reward, next_state, done)

    def update_target_network(self):
        """Copy the online parameters wholesale to the target network and count the sync."""
        self._check_finite("online network", self.value_net.state_dict())
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.sync_count += 1

    def train_step(self, total_steps):
        """One TD update on a sampled batch; None while the buffer warms up."""
        self._check_finite("training state")
        # Warm-up: no update until a full batch exists.
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return None
        start = perf_counter()

        samples = self.replay_buffer.sample()
        states, rewards, next_states, dones = zip(*samples)

        self.value_net.train()
        # Online predictions (N,) = B x J station values.
        predictions = self.value_net(list(states))  # (N,) = B x J station values
        self._check_finite("online predictions", predictions)
        counts = [len(state["agents"]) for state in states]  # J per sample

        # State-value TD targets from the frozen target network; terminal values are masked.
        with torch.no_grad():
            next_values = self.target_net(list(next_states))
            self._check_finite("target predictions", next_values)
            mask = torch.repeat_interleave(
                torch.tensor([not d for d in dones], dtype=torch.float32), torch.tensor(counts)
            )
            targets = (
                torch.as_tensor(np.concatenate(rewards), dtype=torch.float32)
                + self.environment.gamma * mask * next_values
            )
            self._check_finite("TD targets", targets)

        # Mean squared TD error, then one clipped gradient step.
        errors = (predictions - targets).square()
        loss = errors.mean()
        self._check_finite("loss", loss)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(self.value_net.parameters())
        self._check_finite("gradients", [p.grad for p in parameters])
        try:
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                parameters, RL_params["clip_norm"], error_if_nonfinite=True
            ))
        except RuntimeError as exc:
            self._numerical_error = f"{self.name}: nonfinite gradient norm; restore a valid checkpoint"
            raise FloatingPointError(self._numerical_error) from exc
        self._check_finite("clipped gradients", [p.grad for p in parameters])
        self._check_finite("optimizer state", self.optimizer.state_dict())

        # Roll back a failed optimizer step to the last valid parameters/Adam state.
        saved_parameters = [p.detach().clone() for p in parameters]
        saved_optimizer = copy.deepcopy(self.optimizer.state_dict())
        try:
            self.optimizer.step()
            self._check_finite("updated parameters", self.value_net.state_dict())
            self._check_finite("updated optimizer state", self.optimizer.state_dict())
        except Exception as exc:
            with torch.no_grad():
                for parameter, saved in zip(parameters, saved_parameters):
                    parameter.copy_(saved)
            self.optimizer.load_state_dict(saved_optimizer)
            self.optimizer.zero_grad(set_to_none=True)
            self._numerical_error = self._numerical_error or (
                f"{self.name}: optimizer update failed ({exc}); restore a valid checkpoint"
            )
            raise
        self.gradient_steps += 1
        step_train = (perf_counter() - start)
        self.training_time += step_train      # cumulative gradient-update seconds
        loss_value = float(loss.detach())
        # Diagnostic every 100 transitions.
        if total_steps % 100 == 0:
            print(f"--------------- Training step {total_steps}: ---------------")
            print(
                f"training_time={step_train:.4f} s, "
                f"lr={self.optimizer.param_groups[0]['lr']:.3g}, "
                f"loss={loss_value:.4f}; \n"
                f"mean_prediction={predictions.mean().item():.4f}, "
                f"mean_target={targets.mean().item():.4f}; \n"
                f"gradient_norm={gradient_norm:.4f}.",
                flush=True,
            )

        return loss_value

    def td_learning(self, state, reward, next_state, done):
        """Buffer the transition, run one update, sync on schedule; returns loss or None."""
        # Store first, then learn; warm-up still advances the global step counter.
        self.store_transition(state, reward, next_state, done)
        loss = self.train_step(self.total_steps)
        self.total_steps += 1
        if self.total_steps % RL_params["update_freq"] == 0:
            self.update_target_network()

        return loss

    def decay(self, episode=None):
        """Decay epsilon and learning rate at episode end, floored by config."""
        # Epsilon decays every episode; the learning rate after the warm-up only.
        episode = self.environment.episode if episode is None else int(episode)
        self.epsilon = max(RL_params["final_epsilon"], self.epsilon * RL_params["epsilon_decay"])

        if episode >= RL_params["WarmUP"]:
            for group in self.optimizer.param_groups:
                group["lr"] = max(
                    RL_params["final_learning"], group["lr"] * RL_params["learning_decay"]
                )

    @staticmethod
    def _checkpoint_name(name):
        """Resolve a stem / .pt filename / absolute path below check_dir."""
        filename = Path(name)
        if filename.suffix.lower() != ".pt":
            filename = filename.with_name(filename.name + ".pt")
        if not filename.is_absolute():
            filename = Path(RL_params["check_dir"]) / filename
        return filename.resolve()

    def checkpoint_configuration(self):
        """Portable, weights-only-safe snapshot of the effective training inputs."""
        configuration = dict(
            environment=self.environment.training_configuration(),
            state_dims=self.state_dims,
            policy=dict(name=self.name, type=self.type, ablation=self.ablation,
                        bundle_limit=self.bundle_limit, envy_weight=self.envy_weight),
            learning_seed=self.seed, seeds=self.seeds,
            value_inputs=self.value_net.input_configuration(),
        )
        # Strings/lists keep comparisons OS- and pickle-independent.
        return json.loads(json.dumps(configuration, sort_keys=True, default=str))

    def save_checkpoint(self, name=None):
        """Atomically save the v5 training state (name: stem/.pt/absolute) and return Path."""
        self._check_finite("checkpoint state")
        # Resolve the destination from name, the retained path, or a dated default.
        if name is not None:
            filename = self._checkpoint_name(name)
        elif self.checkpoint_path is not None:
            filename = self.checkpoint_path
        else:
            scenario = getattr(self.environment, "scenario", None)
            tag = "" if scenario is None else f"_plus{scenario}"
            filename = (
                Path(RL_params["check_dir"])
                / f"{self.name}_train{tag}_{date.today().isoformat()}.pt"
            )

        # Pure tensors/containers so the file loads under weights_only=True.
        checkpoint = dict(
            format_version=5,       # tensor/container-only schema; loaded with weights_only=True
            value=self.value_net.state_dict(),          # online network weights
            target=self.target_net.state_dict(),        # frozen target-network weights
            optimizer=self.optimizer.state_dict(),      # Adam state: lr, betas, per-parameter moments
            replay=self.replay_buffer.state_dict(),     # compact buffer: tensors + plain containers
            total_steps=self.total_steps,               # cumulative transitions, warm-up included
            gradient_steps=self.gradient_steps,         # completed gradient updates
            sync_count=self.sync_count,                 # target-network synchronizations
            epsilon=self.epsilon,                       # current exploration rate
            rng=self.rng.bit_generator.state,           # exploration stream state
            training_time=self.training_time,           # cumulative gradient-update seconds
            training_time_unit="seconds",               # explicit cumulative training-time unit
            policy_type=self.type,                      # "Fair"/"Weighted"/"Optimum" identity gate (enforced on load)
            ablation=self.ablation,                     # None/"one"/"local"/"fixed" identity gate (enforced on load)
            gat_attention=self.value_net.attention_kind,  # attention architecture identity gate (enforced on load)
            envy_weight=getattr(self, "envy_weight", None),  # weighted-mode coefficient (OptimumDRL only)
            training_episodes=self.training_episodes,   # absolute resume start index
            episode=self.training_episodes - 1,        # last completed episode (-1 before training)
            training_configuration=self.checkpoint_configuration(),
            episode_boundary=self.resume_supported and getattr(self.environment, "time_step", 0)
            in (0, self.environment.total_steps),       # training-capable state between episodes
        )

        self._check_finite(
            "checkpoint model and optimizer", checkpoint["value"], checkpoint["target"],
            checkpoint["optimizer"], self.epsilon, self.training_time,
        )
        self._check_finite("checkpoint replay", *(
            checkpoint["replay"][key]
            for key in ("observations", "next_observations", "rewards", "weight_decay")
        ))
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Unique sibling temp file, then atomic replace.
        temporary = filename.with_name(
            f".{filename.name}.{np.random.SeedSequence().entropy}.tmp"
        )
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(filename)
        finally:
            temporary.unlink(missing_ok=True)
        self.checkpoint_stem = filename.stem
        self.checkpoint_path = filename
        return filename

    def load_checkpoint(self, name=None):
        """Validate and restore the requested checkpoint; validation is
        identical for evaluation and continuation, and resumability is a
        checkpoint property."""
        # 1) Resolve the file: explicit name or the newest dated candidate.
        directory = Path(RL_params["check_dir"])
        if name is not None:
            filename = self._checkpoint_name(name)
        else:
            prefixes = [f"{self.name}_train_", f"{self.name}_"]
            candidates = []
            for path in directory.iterdir() if directory.is_dir() else ():
                if not path.is_file() or path.suffix.lower() != ".pt":
                    continue
                if path.stem == self.name:
                    candidates.append(path)
                    continue
                for prefix in prefixes:
                    if not path.stem.startswith(prefix):
                        continue
                    suffix = path.stem[len(prefix):]
                    try:
                        valid_date = date.fromisoformat(suffix).isoformat() == suffix
                    except ValueError:
                        valid_date = False
                    if valid_date:
                        candidates.append(path)
                        break
            if not candidates:
                raise FileNotFoundError(
                    f"No checkpoint found for {self.name}_train in {directory}"
                )
            filename = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
        filename = filename.resolve()
        if not filename.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {filename}")
        try:
            checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"Cannot read checkpoint {filename}: {exc}") from exc

        # Explicit format gate: clear message instead of a KeyError in the restore.
        version = checkpoint.get("format_version")
        if version != 5:
            raise ValueError(
                f"Checkpoint {filename.name} has unsupported format_version "
                f"{version!r} (expected 5); retrain or restore a compatible "
                "checkpoint"
            )

        # 2) Validate identity/config before touching weights (readable failure).
        saved_configuration = checkpoint.get("training_configuration")
        problems = identity_mismatches(
            checkpoint,
            self.type,
            self.ablation,
            self.value_net.attention_kind,
            getattr(self, "envy_weight", None),
        )
        if saved_configuration is not None:
            current = self.checkpoint_configuration()
            problems += configuration_mismatches(
                current,
                saved_configuration,
                keys=("state_dims", "policy"),
            )
            if "value_inputs" not in saved_configuration:
                problems.append("value_inputs: missing from the checkpoint")
            else:
                problems += value_input_mismatches(
                    current["value_inputs"], saved_configuration["value_inputs"]
                )
        if problems:
            raise ValueError(
                f"Checkpoint {filename.name} is incompatible with learner "
                f"{self.name}:\n- " + "\n- ".join(problems)
            )

        # 3) Stage the full restore before touching live state (optimizer owns its params).
        resumable = checkpoint["episode_boundary"] is True
        value_net, target_net, optimizer = copy.deepcopy(
            (self.value_net, self.target_net, self.optimizer)
        )
        value_net.load_state_dict(checkpoint["value"])
        target_net.load_state_dict(checkpoint["target"])
        target_net.requires_grad_(False)
        target_net.eval()
        optimizer.load_state_dict(checkpoint["optimizer"])

        replay_data = checkpoint["replay"]
        replay = ReplayBuffer.from_state_dict(replay_data)
        rng = copy.deepcopy(self.rng)
        rng.bit_generator.state = checkpoint["rng"]
        counters = {key: checkpoint.get(key, 0) for key in (
            "total_steps", "gradient_steps", "sync_count", "training_episodes"
        )}
        # episode must equal training_episodes-1; otherwise training_episodes is
        # authoritative and training continues from there.
        saved_episode = checkpoint.get("episode")
        saved_training = checkpoint.get("training_episodes")
        if (
            saved_episode is not None
            and saved_training is not None
            and int(saved_episode) != int(saved_training) - 1
        ):
            warnings.warn(
                f"Checkpoint {filename.name}: episode={saved_episode!r} disagrees "
                f"with training_episodes={saved_training!r}; treating "
                f"training_episodes as authoritative, resetting episode to "
                f"{saved_training!r}, and resuming training from there.",
                stacklevel=2,
            )
            checkpoint["episode"] = int(saved_training)
        epsilon = float(checkpoint["epsilon"])
        training_time = float(checkpoint["training_time"])

        # 4) Commit atomically; torch_rng is not saved (no dropout consumer).
        self.__dict__.update(
            value_net=value_net, target_net=target_net, optimizer=optimizer,
            replay_buffer=replay, rng=rng, epsilon=epsilon, training_time=training_time,
            resume_supported=bool(resumable and saved_configuration is not None),
            checkpoint_stem=filename.stem, checkpoint_path=filename, **counters,
            resume_configuration=saved_configuration,
            _numerical_error=None,
        )
        return filename

    def scheduling(self, bundling, train_flag=False, episode=None, name=None):
        """Play one episode through ``select_action``; train on each transition when train_flag is set."""
        episode = self.environment.episode if episode is None else episode

        # Decision-transition loop until the horizon.
        start = perf_counter()
        state = self.environment.reset()
        done, losses = False, []

        while not done:
            action_start = perf_counter()
            action = self.select_action(bundling, train_flag=train_flag)
            self.environment.last_diagnostics["action_seconds"] = perf_counter() - action_start
            next_state, reward, done = self.environment.step(
                action, self.type, envy_weight=getattr(self, "envy_weight", None)
            )
            if train_flag:
                loss = self.td_learning(state, reward, next_state, done)
                if loss is not None:
                    losses.append(loss)
            state = next_state
        # End-of-episode bookkeeping: decay, index advance, records, summary.
        if train_flag:
            self.decay(episode)
            self.training_episodes = episode + 1

        self.environment.request_records(
            name or self.name, episode, evaluation=not train_flag,
            zero_envy=self.type == "Fair",
        )
        result = self.environment.service_records(
            episode, loss=float(np.mean(losses)) if losses else 0.0
        )
        result.update(
            run_time=(perf_counter() - start) / 60,
            gradient_steps=self.gradient_steps,
            total_steps=self.total_steps,
            sync_count=self.sync_count,
        )
        return result
