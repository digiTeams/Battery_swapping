"""Static station specifications."""

class Station:
    """Fixed specification of one station."""
    def __init__(self, sid, cid, long, lat, swap_time, battery_cap, initial_batteries, grid_power):
        self.id, self.cluster = int(sid), int(cid)  # station id and home cluster ID
        self.longitude, self.latitude = float(long), float(lat)  # degrees
        self.swap_time = float(swap_time)  # swap service minutes
        self.capacity = int(battery_cap)  # compartment count

        # initial_batteries: tuple[int, int] (low, high); grid_power is the connection limit in kW.
        self.initial_batteries = tuple(int(x) for x in initial_batteries)  # initial ready stock (low, high)
        self.grid_power = float(grid_power)  # grid connection limit, kW

    def __str__(self):
        return (
            f"Station {self.id}: inventories={self.initial_batteries}, "
            f"grid={self.grid_power} kW"
        )
