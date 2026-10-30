"""Fine spatial cluster specifications."""

from common.geo import haversine_km
from config import env_params


class Cluster:
    """Geometry and area type of one fine cluster."""
    def __init__(self, cid, ctype, center, bound, centroid, boundary):
        self.id, self.type = int(cid), ctype  # cluster ID | area type

        # center: (latitude, longitude); bound: tuple (4,), min/max latitude then min/max longitude.
        self.center, self.bound = center, bound

        # centroid: (longitude, latitude); boundary_points: K coordinate pairs in that order.
        # Geometry fields may be serialized strings during source-table construction.
        self.centroid, self.boundary_points = centroid, boundary

        # Rectangular cell extents in km and road-adjusted driving minutes at the area speed.
        min_lat, max_lat, min_lon, max_lon = self.bound
        latitude = (min_lat + max_lat) / 2.0
        longitude = (min_lon + max_lon) / 2.0
        self.length_distance = float(
            haversine_km(min_lat, longitude, max_lat, longitude)
        )  # km, north-south extent
        self.width_distance = float(
            haversine_km(latitude, min_lon, latitude, max_lon)
        )  # km, east-west extent
        self.diagonal_distance = float(
            haversine_km(min_lat, min_lon, max_lat, max_lon)
        )  # km, diagonal extent
        speed = float(env_params["vehicle_speed"][self.type])  # area road speed, km/h
        factor = float(env_params["road_distance_factor"])  # straight-line to road multiplier
        self.length_driving_time = self.length_distance * factor / speed * 60.0
        self.width_driving_time = self.width_distance * factor / speed * 60.0
        self.diagonal_driving_time = self.diagonal_distance * factor / speed * 60.0

    def contains_location(self, longitude, latitude, tolerance=1e-8):
        """Check a location against the inclusive cluster bounds."""
        return (
            self.bound[0] - tolerance <= latitude <= self.bound[1] + tolerance
            and self.bound[2] - tolerance <= longitude <= self.bound[3] + tolerance
        )

    def distance_to_bounds(self, longitude, latitude):
        """Return kilometres to the closest clamped point on the cluster bounds."""
        # Clamp coordinates to the rectangle without changing the station's location.
        edge_latitude = min(max(latitude, self.bound[0]), self.bound[1])
        edge_longitude = min(max(longitude, self.bound[2]), self.bound[3])
        return float(haversine_km(latitude, longitude, edge_latitude, edge_longitude))

    def __str__(self):
        return f"Cluster {self.id}: {self.type}, {self.centroid}"
