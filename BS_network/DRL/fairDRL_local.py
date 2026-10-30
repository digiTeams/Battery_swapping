"""Fair DRL variant with full bundles and a local value model that contains no GAT."""


from DRL.base_DRL import BaseDRL
from DRL.valueNet import ValueNetwork


class FairDRLLocal(BaseDRL):
    """Local value model without the GAT; the spatial slot holds the capacity."""

    def __init__(self, env, state_dims=None):
        """Build the shared learner, then set the local identity."""
        super().__init__(env, state_dims)

        self.name: str = "fairDRL_local"  # output/checkpoint prefix
        self.type: str = "Fair"  # environment reward convention
        self.ablation: str = "local"  # ablation tag checked when loading checkpoints
        self.bundle_limit: int | None = None  # configured full bundle bound

    def _build_value_network(self):
        """Build the no-GAT network (no graph arrays read)."""
        network = ValueNetwork(self.state_dims, use_gat=False)
        network.station_count = len(self.environment.Agents)
        return network

