"""Fair DRL variant with one request per station bundle and the standard GAT value model."""


from DRL.base_DRL import BaseDRL


class FairDRLOne(BaseDRL):
    """Shared GAT learner with at most one request per station bundle."""

    def __init__(self, env, state_dims=None):
        """Build the shared learner, then set the singleton identity."""
        super().__init__(env, state_dims)

        self.name: str = "fairDRL_one"  # output/checkpoint prefix
        self.type: str = "Fair"  # environment reward convention
        self.ablation: str = "one"  # ablation tag checked when loading checkpoints
        self.bundle_limit: int = 1  # one request per nonempty station bundle
