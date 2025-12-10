''''
The data structure of Cluster:
id  type  center  bound  centroid  boundary_points  demand  envy  subsidy  Envies  Subsidies
'''

class Cluster:

    def __init__(self, cid, ctype, center, bound, centroid, boundary):
        self.id = cid
        self.type = ctype
        self.center = center
        self.bound = bound  #(min_lat, max_lat, min_long, max_long)
        self.centroid = centroid  #(cen_lat, cen_long)
        self.boundary_points = boundary

        self.demand = 0
        self.envy = 0  #maximum envies among the users
        self.subsidy = 0  #average subsidy of the users

        self.Envies = []
        self.Subsidies = []


    def __str__(self):
        return  f"Cluster ID: {self.id}, type {self.type}, boundary {self.bound}, envy {self.envy}."
