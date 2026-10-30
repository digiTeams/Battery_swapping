"""Input paths, simulation parameters, and learning settings."""

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Paths, scenario geometry, and the daily demand clock.
data_params = {
    "data_dir": ROOT / "data",                       # input tables
    "result_dir": ROOT / "results",                  # output tables/figures
    "city": "Nanjing",                               # case-study city label
    'geojson_data': 'Nanjing.geoJson',               # city boundary GeoJSON

    'grid_width': 0.25,                              # coarse cell width, degrees longitude
    'grid_height': 0.25,                             # coarse cell height, degrees latitude
    'cluster_width': 0.11,                           # fine cell width, degrees longitude
    'cluster_height': 0.08,                          # fine cell height, degrees latitude

    'selected_grids': (0, 1, 2, 3, 4, 5, 6),         # retained coarse-grid IDs
    'centre_clusters': (14, 20, 21, 26, 27, 28, 29, 35),  # fine-cluster IDs labelled centre

    'station_icon': 'BS.png',                        # station map marker icon
    "Driver_data": "Driver_records_2025.csv",        # raw trajectory source
    "NIO_data": "NIO_stations_2025.csv",             # raw station source
    "order_data": "Nanjing_drivers_2026.csv",        # mapped trajectory rows
    "sample_orders": "Sample_orders.csv",            # sample scenario used by the map request layer
    "station_data": "Nanjing_stations_2026.csv",     # base station table
    "grid_data": "Nanjing_grids_2026.csv",           # coarse grid table
    "cluster_data": "Nanjing_clusters_2026.csv",     # fine cluster table

    "start_time": datetime(2025, 1, 1, 7),           # first decision clock (t=0 at 07:05)
    "end_time": datetime(2025, 1, 1, 23),            # final decision (t=T-1) at 23:00; charging only
    "decision_end_time": datetime(2025, 1, 1, 23, 0),   # last acceptance cutoff; episode ends once clock passes end_time
    "horizon": (7, 23),                              # hour one-hot window; must match the clock
    "peak_hour_centre": (12, 17),                    # centre peak window (start, end)
    "peak_hour_suburb": ((8, 10), (18, 20)),         # suburb peak windows (start, end)

    "rate_peak_centre": 2.0,                         # centre peak arrivals per minute
    "rate_off_centre": 1.0,                          # centre off-peak arrivals per minute
    "rate_peak_suburb": 1.0,                         # suburb peak arrivals per minute
    "rate_off_suburb": 0.5,                          # suburb off-peak arrivals per minute

    "station_large": 0.8,                            # P(centre station is large)
    "expansion_station_large": 0.5,                  # P(added suburb station is large)
    "installed_low_probability": 0.9,                # P(request installed low-class battery)
    "upgrade_probability": 0.05,                     # P(low-to-high upgrade request)
    "upgrade_days_range": (1, 4),                    # per-order upgrade charge period, days: [low, high)
    "soc_range": (0.10, 0.20),                       # remaining SOC fraction range
    "soc_beta": (2.0, 2.0),                          # Beta(a,b) shape inside soc_range
    "scenarios": (8, 16),                            # expansion sizes (stations added)
    "scale": 1.0,                                    # demand scale; also rescales GAT normalization bounds
    "seed": 20261001,                                # master demand/network seed
}

# Station physics, travel feasibility, charging, and service costs.
env_params = {
    "delta_time": 5,                                 # decision interval, minutes
    "gamma": 0.99,                                   # per-step discount factor

    "battery_levels": 2,                             # battery classes (low, high)
    "battery_kwh": (75.0, 100.0),                    # usable energy per class, kWh
    "charging_power_kw": (62.5, 62.5),               # charging power per class, kW
    "small_size": 13,                                # small station compartments
    "large_size": 21,                                # large station compartments
    "initial_batteries_small": (10, 3),              # initial (low, high) stock, small
    "initial_batteries_large": (16, 5),              # initial (low, high) stock, large
    "grid_power_small": 550.0,                       # small station connection limit, kW
    "grid_power_large": 630.0,                       # large station connection limit, kW
    "ready_soc": 0.93,                               # full battery SOC fraction
    "reserve_soc": 0.05,                             # depleted SOC floor; returned batteries recharge to ready_soc

    "max_demand": (20.0, 10.0),                      # per-class new-demand scale (low, high)
    "max_queue": (16.0, 8.0),                       # per-class queue length scale (low, high)
    "max_upgrade": 5.0,                              # per-step upgrade-demand scale (low-to-high orders)
    "max_capacity": (21.0, 5.0),                     # per-class stock scale (low, high)
    "max_size": 21,                                  # compartment-count normalization scale (large-station size)

    "vehicle_speed": {                               # average road speed by focal-station area, km/h
        "centre": 25.0,
        "suburb": 30.0,
    },
    "road_distance_factor": 1.6,                     # straight-line to road distance multiplier
    "consumption_kwh_per_km": 0.16,                  # energy per km, kWh/km
    "swap_time": 4.0,                                # bay service duration, minutes
    "max_wait_time": 30.0,                           # arrival-to-completion cap, including swap, minutes
    "max_detour_minutes": 15,                        # detour cap beyond the time-nearest station, minutes

    "max_distance": 20,                              # station-to-cluster retention radius, km
    "grid_search_radius": 12.0,                      # fallback radius for locating a nearby coarse grid, km
    "max_bundle_size": 3,                            # max requests per station bundle
    "max_bundles": 200,                              # per-station bundle enumeration cap; not a global total

    "swap_fee_off": 0.4,                             # off-peak service fee, CNY/kWh
    "swap_fee_peak": 0.6,                            # peak service fee, CNY/kWh
    "peak_electricity": (8, 22),                     # peak electricity window (start, end)
    "electric_price_off": 0.58,                      # off-peak electricity, CNY/kWh
    "electric_price_peak": 1.35,                     # peak electricity, CNY/kWh
    "upgrade_fee_per_day": 100.0,                     # upgrade charge, CNY/day
    "VOT": 0.85,                                     # value of time, CNY/minute
    "rejection_time_cost": 50.0,                     # rejection penalty, CNY
    "fairness_tolerance": 1e-4,                      # envy bound slack, CNY; used by both the
                                                     # BMIP fairness rows and the subsidy LP
    "time_tolerance": 1e-3,                          # schedule slack, minutes
    "power_tolerance": 1e-3,                         # grid power slack, kW
    "energy_tolerance": 1e-3,                        # energy slack, kWh
    "fairness_warning_threshold": 1e-3,              # residual (CNY) above which step() prints a violation warning
    "solver_tolerance": 1e-6,                        # solver tolerance; must stay well below fairness_tolerance
    "spatial_tolerance": 1e-3,                       # degrees slack for grid containment checks
}

# Nested charging targets, value model, replay, and solver settings.
RL_params = {
    "check_dir": ROOT / "neurals",                   # checkpoint directory
    "base_stocks": (0.6, 1.0),                       # set_base_level target fractions (off-peak, peak)
    "fixed_level": 0.6,                              # fixed charging-target fraction; fairDRL_fixed always, baselines with target_fixed=True
    "MAX_Episodes": 1000,                            # training episodes
    "WarmUP": 0,                                     # episodes before learning-rate decay
    "MAX_Tests": 30,                                 # evaluation episodes
    "checkpoint_freq": 5,                            # episodes between training checkpoints

    "validation": False,                             # True enables runtime safety checks and runtime tables
    "solver": None,                                  # "gurobi"/"copt"; written once by configure_solver
    "solver_time_limit": 100.0,                       # per-BMIP wall-clock seconds; None disables; COPT uses 1.5x
    "exploration_time_limit": 30.0,                  # per-BMIP wall-clock cap under exploration; None falls back to solver_time_limit
    "solver_mip_gap": 1e-2,                          # relative optimality tolerance
    "solver_node_limit": None,                       # optional per-BMIP node cap
    "solver_threads": 8,                             # MIP solver threads; wall-clock limits can still change incumbents
    "greedy_repair": True,                           # Greedy: repair infeasible subsidy LPs with local rejection;

    # GAT value model. Receiver node vector = gat_focal_features only
    "step_embed_dims": 8,                           # time-step embedding width
    "gat_dims": 16,                                  # GAT projection width
    "gat_negative_slope": 0.2,                       # LeakyReLU slope
    "gat_focal_features": ("que_order", "inventory_position"),  # queue + net-inventory attributes entering the focal node
    "hidden_dims": (256, 128, 64, 32),               # value MLP hidden widths

    "batch_size": 256,                               # transitions per gradient step
    "buffer_size": 25600,                            # replay capacity
    "replay_time_decay": 0.9999,                     # sampling weight = decay**age; 0 < decay <= 1; 1 means uniform
    "initial_learning": 1e-3,                        # Adam learning rate start
    "final_learning": 2e-5,                          # Adam learning rate floor
    "learning_decay": 0.995,                         # per-episode learning-rate decay
    "initial_epsilon": 1.0,                          # exploration start
    "final_epsilon": 1e-3,                           # exploration floor
    "epsilon_decay": 0.99,                           # per-episode exploration decay
    "explore_value_sigma": 25.0,                     # validated value-noise scale; currently unused (exploration randomizes in bundling.bundle_optimize)

    "update_freq": 32,                               # transitions between target syncs
    "clip_norm": 10.0,                               # gradient clip norm
    "torch_threads": 4,                              # torch intra-op threads (1-4)

    "candidate_batch_size": 1024,                    # max candidates scored per forward chunk
    "candidate_score_top_k": 100,                    # DRL: candidates scored per station (value pre-filter); None scores all
    "envy_weight": 4.0,                              # dimensionless multiplier on CNY envy
    "seed": 14,                                      # master learning/solver seed
}

# Per-step stage timings (seconds), one sample per decision step.
TIMING_COLUMNS = (
    "bundle_seconds",
    "candidate_seconds",
    "model_build_seconds",
    "model_solve_seconds",
    "action_seconds",
)

# Per-step BMIP sizes: requests, option rows, variables, constraints (test mode only).
MODEL_SIZE_COLUMNS = (
    "bmip_requests",
    "bmip_options",
    "bmip_variables",
    "bmip_constraints",
)

# Shared per-step samples (BatterySwapEnv.step_samples); empty series report zeros.
STEP_SAMPLE_COLUMNS = TIMING_COLUMNS + MODEL_SIZE_COLUMNS + ("bundle_number",)
