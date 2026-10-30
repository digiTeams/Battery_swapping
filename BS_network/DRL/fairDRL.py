"""Fair DRL policy: the base fair identity on the shared BaseDRL learner."""

from DRL.base_DRL import BaseDRL


__all__ = ["FairDRL"]


class FairDRL(BaseDRL):
    """Fair value-based DRL with standard GAT scoring and fair BMIP;
    defines the base identity name='fairDRL' / type='Fair'."""

    def __init__(self, env, state_dims=None):
        """Build the shared learner, then set the fair-policy identity."""
        super().__init__(env, state_dims)

        self.name: str = "fairDRL"  # output/checkpoint prefix
        self.type: str = "Fair"  # environment reward convention
