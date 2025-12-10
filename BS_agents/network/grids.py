''''
The data structure of Grid:
id  type  station_count  rate  center  bound  centroid  boundary_points
'''

class Grid:

    def __init__(self, gid, gtype, station_count, rate, center, bound, centroid, boundary):
        self.id = gid #0, 1, 2, ...
        self.type = gtype  #centre or suburb area
        self.station_count = station_count  #number of BS stations
        self.rate = rate  #(rate_peak, rate_off), Peak and off hour Poisson rate
        self.center = center
        self.bound = bound  # (min_lat, max_lat, min_long, max_long)
        self.centroid = centroid  # (cen_lat, cen_long)
        self.boundary_points = boundary


    def __str__(self):
        return (f"Grid ID: {self.id}, type {self.type}, boundary {self.bound}, "
                f"rates {self.rate}. ")
