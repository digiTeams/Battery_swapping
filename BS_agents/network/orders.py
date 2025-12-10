''''
The data structure of EV requests
id, cluster, area, longitude, latitude, submission, SOC
'''

from datetime import datetime, timedelta
from config import data_params, env_params


class Request:
    def __init__(self, rid, cid, area, long, lat, submission, SOC):
        self.id = rid  #values in 0, 1, ..., num_req-1
        self.cluster = cid  # the id of cluster for GPS location
        self.area = area  # centre, or suburb
        self.longitude = long
        self.latitude = lat
        self.submission = submission  # submission time
        self.SOC = SOC  #initial state-of-charge

        self.decision_time = datetime.strptime('00:00', '%H:%M')
        self.nearest_station = -1
        self.nearest_distance = env_params['max_distance']
        self.nearest_time = 1.5 * (env_params['max_distance'] / env_params['vehicle_speed']) * 60

        #service status record
        self.station = -1  # assigned station id
        self.complete = 0  # completion flag: 1 = service completed
        self.arrival = data_params['end_time']  #arrival time
        self.begin_time = data_params['end_time'] # service begin time
        self.departure = data_params['end_time']  #departure time
        self.detour_dist = 0.0  # travelling detour distance
        self.detour_time = 0.0 #travelling detour time
        self.wait_time = 0.0  # waiting time
        self.subsidy = 0
        self.envy = 0


    def __str__(self):
        return f"Request ID: {self.id}, gridID: {self.cluster}, submission: {self.submission}. "


    ###Compute the waiting time of request
    def check_wait(self):
        wait = 0
        if self.station > 0:
            wait = self.departure - self.arrival
        return wait


    def set_arrival(self, arr_time):
        self.arrival = arr_time


    def set_departure(self, depart_time):
        self.departure = depart_time


    def set_station(self, station):
        self.station = station




