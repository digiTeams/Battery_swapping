"""Efficiency and weighted-fairness DRL policies sharing the common GAT learner."""


from DRL.base_DRL import BaseDRL


class OptimumDRL(BaseDRL):
    """Efficiency (envy_weight=0) or weighted-fairness (envy_weight>0) DRL sharing the GAT learner."""

    def __init__(self, env, state_dims=None, envy_weight=0.0):
        """Build the shared learner, then derive the weighted identity from envy_weight."""
        weight = float(envy_weight)
        super().__init__(env, state_dims)

        self.envy_weight: float = weight  # soft envy penalty in BMIP objective and reward
        weighted = weight > 0
        weight_tag = str(int(weight)) if weight.is_integer() else repr(weight)
        self.name: str = (
            f"wgt{weight_tag}DRL" if weighted else "optDRL"
        )  # output/checkpoint prefix
        self.type: str = "Weighted" if weighted else "Optimum"  # environment reward convention
        self.ablation: str | None = None  # standard GAT architecture
        self.bundle_limit: int | None = None  # configured full bundle bound
