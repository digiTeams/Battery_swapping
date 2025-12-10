''''
The data structure of BS station
id  cluster  longitude  latitude  battery_cap  chargers  charge_time  swap_time
'''


class Station:
    def __init__(self, sid, cid, long, lat, battery_cap, chargers, charge_time, swap_time):
        self.id = sid  #values in 0, 1, ..., num_station-1self.cluster = cid  # id of the cluster containing the station
        self.cluster = cid  # id of the cluster containing the station
        self.longitude = long
        self.latitude = lat

        self.battery_cap = battery_cap  #capcity of batteries
        self.chargers = chargers #number of charging bays
        self.charge_time = charge_time  #unit: minutes (integral multiples of time steps)
        self.swap_time = swap_time  #unit: minutes


    def __str__(self):
        return (f"Station ID: {self.id}, cluster: {self.cluster}, battery: {self.battery_cap}, "
                f"chargers: {self.chargers}, charge_time: {self.charge_time} minutes. ")


