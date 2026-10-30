"""Request inputs, travel feasibility, and service outcomes."""

from config import env_params


class Request:
    """Request features, travel feasibility, pricing, and service outcomes."""
    def __init__(self, rid, cid, area, long, lat, submission, SOC, installed_class, requested_class,
        source_id=None, sample_fallback=0, source_area=None, upgrade_days=None):
        """Set exogenous features and reset service outcomes."""
        self.id = int(rid)  # unique request identifier
        self.cluster = int(cid)  # fine cluster ID of the source
        self.area = area  # area type ("centre"/"suburb")
        self.source_area = area if source_area is None else source_area  # original trajectory area label
        self.longitude, self.latitude = float(long), float(lat)  # degrees
        self.submission, self.SOC = submission, float(SOC)  # request time | remaining SOC fraction
        self.installed_class, self.requested_class = int(installed_class), int(requested_class)  # battery classes (1/2)
        self.source_id = source_id  # raw source-table trajectory ID
        self.sample_fallback = sample_fallback  # sampling pool fallback (0=slot, 1=hour, 2=area)
        self.upgrade_days = int(upgrade_days) if upgrade_days is not None else 0  # upgrade charge days; 0 for non-upgrades
        self.nearest_station, self.nearest_time = -1, 0.0  # time-nearest station and its driving time

        self.travel_distances = ()  # per-station road distance, km
        self.travel_times = ()  # per-station travel minutes
        self.feasible_stations = ()  # station IDs this request can reach
        self.reset_service()

    def reset_service(self):
        """Clear outcomes, keeping exogenous features."""
        self.station, self.complete = -1, 0  # station ID (-1 unassigned) | completion flag
        self.decision_time = None  # decision interval revealing this request
        self.decision_step = None  # decision index revealing this request
        self.arrival = self.begin_time = self.departure = None  # committed service timestamps
        self.wait_time = self.detour_time = 0.0  # minutes: arrival-to-departure | extra travel
        self.subsidy = self.envy = 0.0  # CNY: compensation | maximum positive net envy
        self.soc_reserved = self.SOC  # booking-time SOC
        self.soc_arrival = None  # set once the request is assigned
        self.service_fee = self.reject_cost = 0.0  # CNY, repriced at reveal
        self.energy_rate = 0.0  # CNY/kWh, requoted at reveal
        self.upgrade_charge = 0.0  # CNY, requoted at reveal

    def arrival_soc(self, station):
        """Return booking SOC minus travel consumption at the selected station."""
        # Reachability is checked before assignment; the floor only absorbs energy-tolerance slack.
        consumed = (
            self.travel_distances[station] * env_params["consumption_kwh_per_km"]
            / env_params["battery_kwh"][self.installed_class - 1]
        )
        return max(
            float(env_params["reserve_soc"]),
            float(self.soc_reserved) - float(consumed),
        )

    def fee_at(self, station):
        """Return the service fee on delivered energy (full requested-class battery
        minus residual energy at arrival) plus the upgrade charge."""
        energy = (
            env_params["ready_soc"] * env_params["battery_kwh"][self.requested_class - 1]
            - self.arrival_soc(station) * env_params["battery_kwh"][self.installed_class - 1]
        )
        return float(energy * self.energy_rate + self.upgrade_charge)

    def __str__(self):
        return (f"Request {self.id}: class {self.installed_class}->{self.requested_class}, "
                f"SOC={self.SOC:.3f}")
