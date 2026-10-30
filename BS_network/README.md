# BS_network

Research simulator for fairness-aware battery-swap scheduling on the Nanjing
NIO network. Eight policies share the same station physics, timetable
reservations, service costs, fairness comparisons, and episode records. The
mathematical model and the GAT/DRL input-output derivation live in
`docs/MODEL.md`; the test-mode output schema and its audit live in
`docs/TEST_MODE_OUTPUTS.md` (with the machine-readable field catalog
`docs/test_mode_field_catalog.csv` and the read-only consistency check
`docs/verify_test_outputs.py`).

## 1. Project structure

```text
BS_network/
|-- config.py, main.py,       parameters; train/test mode dispatch;
|   README.md, requirements.txt
|-- network/                   input loaders and request/station/cluster models
|-- simulator/                 BatterySwapEnv (BS_env.py), Bundling (bundling.py:
|                              bundles, joint BMIP, subsidy LP), StationAgent
|                              (station_agents.py)
|-- baselines/                 fairGreedy (fair_Greedy.py), fairRHO (fair_RHO.py)
|-- DRL/                       base_DRL.py (generic TD learner), fairDRL.py and
|                              variants (fairDRL_one, fairDRL_local,
|                              fairDRL_fixed), opt_DRL.py, valueNet.py,
|                              checkpoint_checks.py
|-- common/                    optimization.py (solvers), analys.py, plots.py,
|                              evals.py, generate_instances.py, outputs.py
|-- runners/                   trainers.py, test_runner.py, setup.py,
|                              sensitivities.py
|-- data/                      input tables (section 3)
|-- results/                   generated tables and figures (section 4)
|-- neurals/                   version-5 checkpoints
`-- docs/                      MODEL.md, TEST_MODE_OUTPUTS.md,
                               test_mode_field_catalog.csv,
                               verify_test_outputs.py
```

## 2. Strategies

Every DRL policy extends `BaseDRL` (GAT candidate scoring plus the joint
BMIP); `GreedyPolicy` and `RHOPolicy` are standalone baselines. Two charging
helpers are shared by all policies: `set_base_level(current_time)` selects
one `(total, high)` target from the anticipated charging-completion time
(peak or off-peak `base_stocks` fraction), and `enum_base_levels()`
enumerates the 2x2 `base_stocks` target grid (currently unused by any
policy). Setting
`target_fixed=True` pins the target to `RL_params['fixed_level']` (fraction
of the cumulative stock, set in `config.py`). The candidate builder `Bundling.build_options(enumerated,
max_size, target_fixed)` enumerates the 2x2 grid when `enumerated=True`;
otherwise it builds the single completion-time `set_base_level` target per
station, pinned to `fixed_level` when `target_fixed=True`. Every current
policy calls the builder with `enumerated=False`, so the single
completion-time target is the only charging candidate in use; the 2x2 grid
path remains available but unused.

| Strategy | File | Runtime name | Definition |
|---|---|---|---|
| `GreedyPolicy` | `baselines/fair_Greedy.py` | `fairGreedy` (`target_fixed=True` -> `fairGreedy_fixed`) | assigns each revealed request to the time-nearest feasible station (bundle size <= 3), charges from the `set_base_level` target, and repairs fairness with a minimum-subsidy LP |
| `RHOPolicy` | `baselines/fair_RHO.py` | `fairRHO` (`target_fixed=True` -> `fairRHO_fixed`) | solves one joint fair BMIP per decision interval on immediate values (no value network), with `set_base_level` charging targets |
| `FairDRL` | `DRL/fairDRL.py` | `fairDRL` | base fair DRL: GAT candidate scoring + fair joint BMIP |
| `FairDRLOne` | `DRL/fairDRL_one.py` | `fairDRL_one` | the same learner with at most one request per station bundle |
| `FairDRLLocal` | `DRL/fairDRL_local.py` | `fairDRL_local` | the same learner with a local (no-GAT) value network |
| `FairDRLFixed` | `DRL/fairDRL_fixed.py` | `fairDRL_fixed` | the same learner whose charging target is always the fixed `fixed_level` |
| `OptimumDRL` | `DRL/opt_DRL.py` | `optDRL` (envy weight 0) / `wgt4DRL` (envy weight 4) | efficiency / weighted-fairness BMIP objective on the shared learner |

Exploration of the DRL strategies `fairDRL`, `fairDRL_one`, `fairDRL_local`,
`optDRL`, and `wgt4DRL`: with probability `epsilon` the BMIP objective is
randomized (candidate values times `U(-0.5, 2)`, rejection coefficients times
`rejection_time_cost * U(0, 4)`, subsidy coefficients `U(0, 4)`) and solved
under the short exploration time limit; the charging candidates are the same
single completion-time `set_base_level` target in every step.

Common interface. Every policy plays one episode through
`scheduling(bundling, train_flag, episode, name)` and returns the summary row
dict; one decision is the action dict
`{"assignment": (R,) with -1 = reject, "charging": (J,2), "subsidy": (R,)}`,
and the environment step returns `(next_state, reward (J,), done)`. Fairness
is enforced within `(cluster, requested_class)` groups.

## 3. Input data

| File | Content |
|---|---|
| `data/Nanjing.geoJson` | city boundary |
| `data/NIO_stations_2025.csv` | raw station source table |
| `data/Driver_records_2025.csv` | raw trajectory source table |
| `data/Nanjing_stations_2026.csv` | base 38-station table (id, cluster, area, capacity, initial stocks, grid power) |
| `data/Nanjing_grids_2026.csv` | 7 coarse grids |
| `data/Nanjing_clusters_2026.csv` | 47 fine clusters with centre/suburb labels |
| `data/Nanjing_drivers_2026.csv` | mapped order rows: coordinates, SOC, battery classes, `upgrade_days` |
| `data/Nanjing_stations_test_{8,16}.csv` | expansion station tables (+8/+16 stations) |
| `data/Sample_orders.csv` | small sampled scenario used by the map layer |

Station/grid/cluster IDs are consecutive and zero-based; simulator-generated
request IDs are consecutive from zero. Units: WGS84 degrees, kW, kWh,
minutes, CNY, SOC fractions. `python -m runners.setup` rebuilds the derived
tables from the raw sources.

## 4. Output data

| Output | Producer | Content |
|---|---|---|
| `results/{stem}_summary.csv` | every run (`Trainer` / `TestRunner`) | one row per episode: profits/rewards, serving, rejections, fairness, charging, waiting, detours, subsidies, timings, diagnostics; test-mode rows additionally carry per-step BMIP size statistics (`bmip_requests` / `bmip_options` / `bmip_variables` / `bmip_constraints` as mean/p95/max/total) and `bundle_number_p95/total` |
| `results/{stem}_requests_{episode:03d}.csv` | evaluation runs | one row per request: submission/decision step, nearest station and its reveal-time stock, assignment (station cluster/area and its reveal-time stock), detour, waiting, SOC, fee, subsidy, max envy (test mode writes them for fairDRL and optDRL only) |
| `results/{stem}_seeds.csv` | `RL_params['validation']=True` | one row per episode with the seed streams |
| `results/{stem}_configuration.csv` | `RL_params['validation']=True` | flattened `key_path, data_type, value` run metadata |
| `results/{stem}_runtime.csv` | `RL_params['validation']=True` | flattened Python/dependency/runtime rows |
| `results/instances_summary.csv` | `python -m common.generate_instances` | one row per training episode of demand statistics |
| `results/test_instances_summary{tag}.csv` | `TestRunner` | one row per generated test episode |
| `results/{policy}_test{tag}_{date}_summary.csv` | `TestRunner` | summary schema plus `scenario`, the test-mode BMIP size statistics, and `bundle_number_p95/total`, one row per episode |
| `results/fairDRL_action_records{tag}.csv` | `TestRunner` | fairDRL per-step action timings: up to 300 peak-period + 300 off-peak-period steps |
| `results/sensitivity_default{tag}_{date}.csv` | `Sensitivities` | one row per episode: fairDRL main outcomes plus `impr` (vs RHO) and `gap` (vs optDRL); `{tag}` is empty for the base network and `_plus8`/`_plus16` for the expansion networks |
| `results/sensitivity_{param}{tag}_{date}.csv` | `Sensitivities` | the same columns under the changed parameter level |
| `results/timing_summary_{date}.csv` | `Analyst.timing_summary` | per-policy stage timing aggregates |
| `results/paired_request_outcomes_{a}_{b}{tag}_{date}.csv` | `Analyst.compare_requests` | up to 600 randomly sampled paired-request rows (`{tag}` is `_plus8`/`_plus16` or empty for the base network) |
| `results/competing_group_outcomes_{a}_{b}{tag}_{date}.csv` | `Analyst.compare_requests` | one row per competing `(episode, decision step, cluster, class)` group with `group_size >= 2`: group size, nearest stocks, pre-action cluster waiting, same-station/diverted/split/rejected counts, waiting/detour/subsidy/envy/disutility aggregates |
| `results/competing_group_summary_{a}_{b}{tag}_{date}.csv` | `Analyst.compare_requests` | the 8 `(peak/off-peak, centre/suburb, high/low)` classes with mean/max/p95 of every competing-group field |
| `neurals/{stem}.pt` | `BaseDRL.save_checkpoint` | version-5 binary checkpoint |
| `results/*.png`, `results/*_map_*.html` | `common.evals` / `runners.setup` | figures and the city map |

`{stem}` is `{policy}_{stream}_{date}` (for example
`fairDRL_train_2026-10-27`), `{tag}` is the expansion scenario tag.

## 5. Running

Environment: Python 3.10+; the project environment is `C:\Vens\envRL`
(`C:\Vens\envRL\Scripts\Activate.ps1`, then `pip install -r
requirements.txt`). A licensed `gurobi` or `copt` backend solves every BMIP
and subsidy LP (`Optimizer.configure_solver(preferred="gurobi")` probes the
available licenses).

`main.py` switches modes manually:

| Setting | Meaning |
|---|---|
| `MODE` | `"train"`, `"test"`, or `"sensitivity"` |
| `task` | one training policy per run: `Baseline`, `fairDRL`, `fairDRL_one`, `fairDRL_local`, `fairDRL_fixed`, `optDRL`, `wgt4DRL` |
| `resume` | training: DRL continues from the newest checkpoint, `Baseline` after the newest summary; test: continue an interrupted sweep from the completed (policy, episode) prefix |
| `target_fixed` | `task="Baseline"` only: `True` runs the fixed-target baselines (`fairGreedy_fixed`, `fairRHO_fixed`), `False` the scheduled peak-window variants (`fairGreedy`, `fairRHO`) |
| `TEST_TASKS` | policy list run in test mode (subset of the eight runtime names) |
| `PARAMETER` | `None` runs the default config; otherwise `detour_cap` / `wait_cap` / `poisson_rate` / `low_class_prob` (non-default level) |
| sensitivity scenario | the second argument of `_run_sensitivities(PARAMETER, scenario, resume)` in `main.py` (currently hardcoded to `None` = base network); `0`/`8` and `1`/`16` select the 8/16-station expansion networks |

```bash
python main.py                    # train the configured `task` or run the test sweep
python -m common.evals            # timing/paired tables and figures
python -m runners.setup           # optional: rebuild data/ tables from raw inputs
python -m common.generate_instances  # optional: per-episode demand statistics
```

Training runs exactly one policy per invocation on the default base
38-station network; `MAX_Episodes` is the episode budget and DRL checkpoints
are saved every `checkpoint_freq` episodes and at the final episode. Test mode
runs `TEST_TASKS` on expansion network 1 (54 = 38 + 16 stations) at demand
scale 2.0: each expansion network owns a private block of 30 deterministic
episodes and every policy in the list runs on each episode in turn, so all
policies see byte-identical demand; DRL policies load their newest checkpoint
before the sweep, and `resume=True` continues an interrupted sweep from the
earliest incomplete episode instead of restarting.
