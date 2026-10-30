"""Plot and compare completed runs: reward/loss and profit/envy figures,
timing summaries, and paired request comparisons against optDRL."""

from common import analys, plots

WINDOW = 32  # smoothing window for all policy curves

# Canonical policy timing prefixes; plot_timing maps legacy "Greedy"/"RHO" aliases.
TIMING_POLICIES = (
    "fairGreedy",
    "fairRHO",
    "fairGreedy_fixed",
    "fairRHO_fixed",
    "fairDRL",
    "fairDRL_one",
    "fairDRL_local",
    "fairDRL_fixed",
    "optDRL",
    "wgt4DRL",
)

if __name__ == '__main__':
    evaluator = plots.PolicyEvals(window=WINDOW)
    evaluator.plot_DRLs({
        "Greedy": evaluator.mean_reference("greedy_train"),
        "RHO": evaluator.mean_reference("rho_train"),
    })
    evaluator.plot_policies()

    analysis = analys.Analyst(window=WINDOW)
    analysis.timing_summary()

    # fairRHO timing required; the others skip when tables are missing.
    for policy in TIMING_POLICIES:
        analysis.plot_timing(policy, optional=policy != "fairRHO")

    analysis.compare_requests()


