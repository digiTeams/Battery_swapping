"""Fair DRL variant whose charging target is the fixed ``fixed_level`` target."""

from DRL.base_DRL import BaseDRL


__all__ = ["FairDRLFixed"]


class FairDRLFixed(BaseDRL):
    """GAT learner with the fixed ``fixed_level`` charging target; only
    ``BILP_costs`` differs (enumerated=False, target_fixed=True)."""

    def __init__(self, env, state_dims=None):
        """Build the shared learner, then set the fixed-charging identity."""
        super().__init__(env, state_dims)

        self.name: str = "fairDRL_fixed"  # output/checkpoint prefix
        self.type: str = "Fair"  # environment reward convention
        self.ablation: str = "fixed"  # ablation tag checked when loading checkpoints
        self.bundle_limit: int | None = None  # configured full bundle bound
        self.target_fixed: bool = True  # candidates and all-reject fallback use fixed_level

    def BILP_costs(self, bundling, explore=False):
        """Build candidates with one fixed fixed_level vector per station
        (enumerated=False, target_fixed=True); scoring unchanged."""
        options = bundling.build_options(
            enumerated=False,
            target_fixed=True,
            max_size=self.bundle_limit,
        )
        return self._score_options(bundling, options)
