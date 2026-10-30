"""GAT value network: station projections, one attention head, and the value MLP."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from config import data_params, env_params, RL_params


L = int(env_params["battery_levels"])  # battery classes

# scale=2.0 (test mode) denominators for the demand/queue features; they are
# used verbatim instead of the env_params max_demand/max_queue/max_upgrade
# bounds so the GAT inputs stay unclipped at the doubled demand scale.
SCALE2_DENOMINATORS = {
    "new_order": (40.0, 16.0),
    "que_order": (20.0, 10.0),
    "upgrade_order": (8.0,),
}


def feature_scales(key):
    """Return the normalization denominator(s) for a feature key: env_params
    bounds verbatim, except scale=2.0 which uses SCALE2_DENOMINATORS."""
    if float(data_params["scale"]) == 2.0 and key in SCALE2_DENOMINATORS:
        return SCALE2_DENOMINATORS[key]
    if key == "new_order":
        return tuple(env_params["max_demand"])
    if key == "que_order":
        return tuple(env_params["max_queue"])
    if key == "upgrade_order":
        return (float(env_params["max_upgrade"]),)
    if key == "capacity":
        return env_params["max_size"]
    if key == "avg_waiting":
        return env_params["max_wait_time"]
    if key in ("full_batteries", "inventory_position"):
        return env_params["max_capacity"]
    if key == "empty_batteries_low":
        return env_params["max_capacity"][0]
    if key == "empty_batteries_high":
        return env_params["max_capacity"][1]
    if key == "next_ready_low":
        return READY_MINUTES[0]
    if key == "next_ready_high":
        return READY_MINUTES[1]
    if key == "spare_grid":
        return 1.0
    if key == "bay_free":
        return env_params["swap_time"]
    if key == "type":
        return (1.0, 1.0)
    raise KeyError(f"unknown feature key {key!r}")


def _ready_minutes(l):
    """Minutes to recharge one class-l battery reserve->ready at full power."""
    interval_kwh = env_params["charging_power_kw"][l] * env_params["delta_time"] / 60.0
    required = (env_params["ready_soc"] - env_params["reserve_soc"]) * env_params["battery_kwh"][l]
    return math.ceil(required / interval_kwh) * env_params["delta_time"]


READY_MINUTES = tuple(_ready_minutes(l) for l in range(L))  # (65.0, 85.0)

CONTINUOUS_FEATURES = (
    ("new_order", L),                 # denominators from feature_scales
    ("upgrade_order", 1),             # denominators from feature_scales
    ("que_order", L),                 # denominators from feature_scales
    ("avg_waiting", L),               # wait scale, minutes
    ("full_batteries", L),            # per-class stock scale
    ("inventory_position", L),        # per-class stock scale
    ("empty_batteries_low", 1),       # depleted low stock
    ("empty_batteries_high", 1),      # depleted high stock
    ("next_ready_low", 1),            # minutes to next ready low battery
    ("next_ready_high", 1),           # minutes to next ready high battery
    ("spare_grid", 1),                # already a fraction of U_power
    ("bay_free", 1),                  # bay idle-until minutes
    ("type", 2),                      # one-hot area: [1,0] centre, [0,1] suburb
)

# Sender message features: capacity and per-class new demand; the receiver
# node vector carries only the candidate-dependent focal block.
SENDER_FEATURES = (
    ("capacity", 1),   # capacity / max_size
    ("new_order", L),  # per-class new demand
)

# Focal receiver features: candidate-dependent per-class queue and net inventory.
_FOCAL_FEATURE_SPECS = {
    "que_order": L,
    "inventory_position": L,
}
NODE_EXTRA_FEATURES = tuple(
    (key, _FOCAL_FEATURE_SPECS[key]) for key in RL_params["gat_focal_features"]
)

class ValueNetwork(nn.Module):
    """Per-station value scorer: node/sender/edge projections, one attention head, value MLP."""
    def __init__(self, state_dims=None, neighbor_mask=None, detour_weights=None, use_gat=True):
        """Build projections and the value MLP; widths derive from config."""
        super().__init__()
        # Time normalization and GAT switch.
        dims = state_dims or {}
        self.total_steps = dims.get(
            "steps",
            int(
                (data_params["end_time"] - data_params["start_time"])
                .total_seconds()
                / (60 * env_params["delta_time"])
            ),
        )  # int: time normalization denominator
        self.use_gat = bool(use_gat)               # bool: aggregate spatial context
        self.station_count = len(neighbor_mask) if neighbor_mask is not None else None  # int|None: J
        gat_dims = RL_params["gat_dims"]           # int: attention embedding width
        # Architecture widths recorded for checkpoint compatibility checks.
        self.gat_dims = int(gat_dims)              # int: attention embedding width
        self.step_embed_dims = int(RL_params["step_embed_dims"])  # int: time-embedding width
        self.hidden_dims = tuple(int(h) for h in RL_params["hidden_dims"])  # tuple[int]: MLP widths

        # Node width = candidate-dependent focal block only; new demand and
        # static capacity are not receiver features.
        levels = int(env_params["battery_levels"])
        self.node_dims = sum(width for _, width in NODE_EXTRA_FEATURES)  # int: node width
        self.sender_dims = sum(width for _, width in SENDER_FEATURES)  # int: capacity + new demand
        self.edge_dims = 2 * levels + 1  # int: overlap (L) + detour (1) + nearest counts (L)
        horizon = data_params["horizon"]
        self.hour_dims = horizon[1] - horizon[0]  # int: hour width

        self.neighbor_mask = neighbor_mask if self.use_gat else None  # (J,J) int8 incoming adjacency
        self.detour_weights = detour_weights if self.use_gat else None  # (J,J) float detour feature
        self._local_senders = None    # int64 (E_local,): edge sources
        self._local_receivers = None  # int64 (E_local,): edge targets
        self._edge_detours = None     # float (E_local,): static detours

        # Self-loop: zero edge features, nonzero sender message; candidate
        # changes enter via the receiver-side block, so each candidate
        # recomputes its attention row.
        if self.use_gat:
            self.node_projection = nn.Linear(self.node_dims, gat_dims, bias=False)
            self.sender_projection = nn.Linear(self.sender_dims, gat_dims, bias=False)
            self.edge_projection = nn.Linear(self.edge_dims, gat_dims, bias=False)
            self.attention = nn.Linear(2 * gat_dims, 1, bias=False)
        self.attention_kind = "additive" if self.use_gat else None
        self.step_embed = nn.Linear(1, RL_params["step_embed_dims"])

        # MLP input = time + hour one-hot + spatial context + local continuous features.
        continuous_dims = sum(entries for _, entries in CONTINUOUS_FEATURES)
        spatial_dims = gat_dims if self.use_gat else 1  # local path keeps capacity only
        layers, size = ([], RL_params["step_embed_dims"] + self.hour_dims + spatial_dims
                        + continuous_dims)
        for hidden in RL_params["hidden_dims"]:
            layers.extend((nn.Linear(size, hidden), nn.ReLU()))
            size = hidden
        layers.append(nn.Linear(size, 1))
        self.value_mlp = nn.Sequential(*layers)

    def input_configuration(self):
        """Return the effective feature scales, attention type, and static graph inputs."""
        return dict(
            total_steps=self.total_steps, hour_dims=self.hour_dims,
            use_gat=self.use_gat, attention_kind=self.attention_kind,
            step_embed_dims=self.step_embed_dims, gat_dims=self.gat_dims,
            hidden_dims=self.hidden_dims,
            continuous=[
                (key, feature_scales(key), width)
                for key, width in CONTINUOUS_FEATURES
            ],
            receiver=[
                (key, feature_scales(key), width)
                for key, width in NODE_EXTRA_FEATURES
            ],
            sender=[
                (key, feature_scales(key), width)
                for key, width in SENDER_FEATURES
            ],
            node_dims=self.node_dims, sender_dims=self.sender_dims,
            neighbor_mask=(None if self.neighbor_mask is None
                           else np.asarray(self.neighbor_mask).tolist()),
            detour_weights=(None if self.detour_weights is None
                            else np.asarray(self.detour_weights).tolist()),
        )

    def _static_edges(self):
        """Return the cached (senders, receivers, detours) edge template (E_local,)."""
        if self._local_senders is not None:
            return self._local_senders, self._local_receivers, self._edge_detours

        senders, receivers = [], []
        for j in range(len(self.neighbor_mask)):  # incoming neighbors m -> j, self-loop included
            for m in np.flatnonzero(self.neighbor_mask[j]):
                senders.append(int(m))
                receivers.append(int(j))
        self._local_senders = np.asarray(senders, dtype=np.int64)
        self._local_receivers = np.asarray(receivers, dtype=np.int64)
        # Static detours at once; demand-dependent overlap is filled in graph_inputs.
        self._edge_detours = np.asarray(
            self.detour_weights[self._local_receivers, self._local_senders], dtype=np.float32
        )
        return self._local_senders, self._local_receivers, self._edge_detours

    def graph_inputs(self, states):
        """Turn joint states into batch graph tensors (nodes/senders/edges/continuous/times/hours)."""
        # N=B*J nodes, E_B=B*E edges; float32 tensors except int64 edge indices/hours.
        states = [states] if isinstance(states, dict) else states
        # Cache the static edge template once; accumulate per-sample features.
        local_senders, local_receivers, _ = self._static_edges()
        nodes, sender_nodes, edges, continuous, times, hours = [], [], [], [], [], []
        senders, receivers, lengths = [], [], []
        for b, state in enumerate(states):
            station_ids = range(len(self.neighbor_mask))
            observations = [state["agents"][j] for j in station_ids]
            memberships = state.get("memberships")
            lengths.append(len(observations))

            # One (J, *) matrix per feature group instead of scalar loops over stations.
            nodes.append(self.node_feature_vectors(observations))
            sender_nodes.append(self.sender_feature_vectors(observations))
            continuous.append(self._continuous_vectors(observations))
            times.append(
                np.asarray([observation["time_step"] for observation in observations], dtype=np.float32)[:, None]
                / self.total_steps
            )
            hour_index = (
                np.asarray([observation["hour"] for observation in observations], dtype=int)
                - data_params["horizon"][0]
            )
            hours.append(np.clip(hour_index, 0, self.hour_dims - 1))

            # Offset edge indices per sample; edge features are built per sample below.
            offset = b * len(observations)
            senders.append(local_senders + offset)
            receivers.append(local_receivers + offset)
            membership_sets = {
                key: (set(low), set(high)) for key, (low, high) in memberships.items()
            }
            nearest = state.get("nearest")
            edges.append(self._edge_feature_matrix(membership_sets, nearest))

        # Convert accumulators to tensors on the module's device.
        device = next(self.parameters()).device
        nodes = torch.as_tensor(np.concatenate(nodes, axis=0), dtype=torch.float32, device=device)
        sender_nodes = torch.as_tensor(np.concatenate(sender_nodes, axis=0), dtype=torch.float32, device=device)
        edges = torch.as_tensor(np.concatenate(edges, axis=0), dtype=torch.float32, device=device)
        continuous = torch.as_tensor(np.concatenate(continuous, axis=0), dtype=torch.float32, device=device)
        times = torch.as_tensor(np.concatenate(times, axis=0), dtype=torch.float32, device=device)
        senders = torch.cat([torch.as_tensor(x, dtype=torch.long, device=device) for x in senders])
        receivers = torch.cat([torch.as_tensor(x, dtype=torch.long, device=device) for x in receivers])
        hours = torch.as_tensor(np.concatenate(hours), dtype=torch.long, device=device)
        return nodes, sender_nodes, edges, continuous, times, senders, receivers, hours, lengths

    def _edge_feature_matrix(self, membership_sets, nearest):
        """Overlap (L), detour (1), and nearest-receiver counts (L) per edge; self-loops zeroed."""
        # membership_sets[j][l]: request IDs; output float32 (E, 2L+1).
        senders, receivers, detours = self._static_edges()
        edges = np.empty((len(senders), self.edge_dims), dtype=np.float32)
        classes = (self.edge_dims - 1) // 2  # per-class overlap columns; detour sits after them
        stations = range(len(self.neighbor_mask))
        for l in range(classes):
            groups = [
                np.asarray(sorted(membership_sets[j][l]), dtype=np.int64)
                for j in stations
            ]
            counts = np.asarray([len(group) for group in groups], dtype=np.float32)
            if counts.sum() == 0:
                overlap = np.zeros(
                    (len(self.neighbor_mask), len(self.neighbor_mask)), dtype=np.float32
                )
            else:
                flat = np.concatenate([group for group in groups if len(group)])
                unique, inverse = np.unique(flat, return_inverse=True)
                indicator = np.zeros((len(self.neighbor_mask), unique.size), dtype=np.float32)
                start = 0
                for j, group in enumerate(groups):
                    stop = start + len(group)
                    indicator[j, inverse[start:stop]] = 1.0
                    start = stop
                overlap = indicator @ indicator.T
            denominator = np.maximum(counts, 1.0)
            edges[:, l] = overlap[receivers, senders] / denominator[receivers]
        edges[:, classes] = detours
        for l in range(classes):
            counts = np.asarray(
                [
                    sum(1 for i in membership_sets[j][l] & membership_sets[m][l]
                        if nearest.get(i) == j)
                    for m, j in zip(senders, receivers)
                ],
                dtype=np.float32,
            )
            edges[:, classes + 1 + l] = np.clip(
                counts / feature_scales("new_order")[l], 0.0, 1.0
            )
        edges[senders == receivers] = 0.0
        return edges

    def node_feature_vectors(self, observations):
        """Normalized receiver node features (the candidate-dependent focal block)."""
        columns = []
        for key, width in NODE_EXTRA_FEATURES:
            values = np.asarray([observation[key] for observation in observations], dtype=np.float32)
            scale = np.asarray(feature_scales(key), dtype=np.float32)
            columns.append(np.clip(values / scale, 0.0, 1.0))
        return np.concatenate(columns, axis=1)

    def sender_feature_vectors(self, observations):
        """Message features: sender capacity and per-class new demand."""
        capacity = (
            np.asarray([observation["capacity"] for observation in observations], dtype=np.float32)[:, None]
            / float(feature_scales("capacity"))
        )
        demand = np.asarray([observation["new_order"] for observation in observations], dtype=np.float32)
        demand = np.clip(
            demand / np.asarray(feature_scales("new_order"), dtype=np.float32),
            0.0,
            1.0,
        )
        return np.concatenate((capacity, demand), axis=1)

    def _continuous_vectors(self, observations):
        """Return the (N, continuous_dims) normalized local features of a batch."""
        columns = []
        for key, width in CONTINUOUS_FEATURES:
            values = np.asarray([observation[key] for observation in observations], dtype=np.float32)
            if values.ndim == 1:  # width-1 scalar fields need an explicit column axis
                values = values[:, None]
            scale = np.asarray(feature_scales(key), dtype=np.float32)
            columns.append(np.clip(values / scale, 0.0, 1.0))
        return np.concatenate(columns, axis=1)

    def continuous_features(self, observations):
        """Return (N, continuous_dims) normalized local features for a batch."""
        device = next(self.parameters()).device
        return torch.as_tensor(
            self._continuous_vectors(observations),
            dtype=torch.float32,
            device=device,
        )

    def times_tensor(self, observations):
        """Return the (N, 1) normalized time-step tensor of a batch."""
        device = next(self.parameters()).device
        values = (
            np.asarray([observation["time_step"] for observation in observations], dtype=np.float32)[:, None]
            / self.total_steps
        )
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    def hours_tensor(self, observations):
        """Return the (N,) clipped hour-index tensor of a batch."""
        device = next(self.parameters()).device
        hour = np.asarray([observation["hour"] for observation in observations], dtype=int)
        hour = np.clip(hour - data_params["horizon"][0], 0, self.hour_dims - 1)
        return torch.as_tensor(hour, dtype=torch.long, device=device)

    def score_parts(self, times, hours, spatial, continuous):
        """Feed the four assembled feature groups through the shared value MLP."""
        features = torch.cat(
            (
                self.step_embed(times),
                F.one_hot(hours, self.hour_dims).float(),
                spatial,
                continuous,
            ),
            dim=1,
        )
        return self.value_mlp(features).squeeze(-1)

    def _attention_scores(self, focal_receivers, messages):
        """Return additive receiver-message logits after LeakyReLU."""
        # Inputs: float32 (E,d) pairs; nn.Linear stores attention weights (1,2*d).
        return F.leaky_relu(
            self.attention(torch.cat((focal_receivers, messages), dim=1)).squeeze(-1),
            negative_slope=RL_params["gat_negative_slope"],
        )

    def attention_primitives(self, state):
        """Cache action-invariant messages and the baseline spatial context."""
        nodes, sender_nodes, edges, _, _, senders, receivers, _, _ = self.graph_inputs([state])
        messages = self.sender_projection(sender_nodes)[senders] + self.edge_projection(edges)
        spatial = self._spatial_context(nodes, sender_nodes, edges, senders, receivers)
        receivers_np = receivers.detach().cpu().numpy()
        incoming = {
            int(j): np.flatnonzero(receivers_np == j) for j in range(len(nodes))
        }
        return {
            "messages": messages,
            "spatial": spatial,
            "senders": senders,
            "edges": edges,
            "receivers": receivers,
            "incoming": incoming,
        }

    def focal_spatial(self, primitives, station, focal_new):
        """Recompute one station's attention row from its projected candidate receiver."""
        # focal_new: float32 (1,d); incoming messages: float32 (degree_j,d).
        inc = primitives["incoming"][int(station)]
        messages = primitives["messages"][inc]
        receiver = focal_new.expand(len(inc), -1)
        scores = self._attention_scores(receiver, messages)
        scores = scores - scores.max()
        alpha = torch.exp(scores)
        alpha = alpha / alpha.sum()
        return (alpha[:, None] * messages).sum(0).relu()

    def _spatial_context(self, nodes, sender_nodes, edges, senders, receivers):
        """Return spatial context (N,d) from GAT message aggregation."""
        # Sender messages = sender projection + edge features; the receiver
        # enters only the attention scores.
        focal = self.node_projection(nodes)
        messages = self.sender_projection(sender_nodes)[senders] + self.edge_projection(edges)

        scores = self._attention_scores(focal[receivers], messages)

        # Stable per-receiver softmax (score-max shift).
        maximum = torch.full((len(nodes),), -torch.inf, device=nodes.device)
        maximum.scatter_reduce_(0, receivers, scores, reduce="amax", include_self=True)
        exp = torch.exp(scores - maximum[receivers])
        denominator = torch.zeros(len(nodes), device=nodes.device).index_add_(0, receivers, exp)
        alpha = exp / denominator[receivers]

        return torch.zeros_like(focal).index_add_(0, receivers, alpha[:, None] * messages).relu()

    def local_forward(self, observations):
        """Predict values without graph construction (capacity in the spatial slot)."""
        device = next(self.parameters()).device

        continuous = self.continuous_features(observations)
        capacity = torch.as_tensor(
            np.asarray([observation["capacity"] for observation in observations], dtype=np.float32)[:, None]
            / float(env_params["max_size"]),
            dtype=torch.float32,
            device=device,
        )
        times = self.times_tensor(observations)
        hours = self.hours_tensor(observations)
        return self.score_parts(times, hours, capacity, continuous)

    def forward(self, state):
        """Return station values (N,) for one joint state or a list of states."""
        # Local ablation bypasses the graph entirely.
        states = [state] if isinstance(state, dict) else state
        if not self.use_gat:
            observations = []
            for sample in states:
                station_ids = (range(self.station_count) if self.station_count is not None
                               else sorted(sample["agents"]))
                observations.extend(sample["agents"][j] for j in station_ids)
            return self.local_forward(observations)

        # GAT path: build the graph, aggregate messages, score the MLP.
        nodes, sender_nodes, edges, continuous, times, senders, receivers, hours, lengths = (
            self.graph_inputs(states)
        )
        spatial = self._spatial_context(nodes, sender_nodes, edges, senders, receivers)

        return self.score_parts(times, hours, spatial, continuous)
