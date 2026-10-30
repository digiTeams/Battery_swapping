"""Coarse spatial grid specifications."""

class Grid:
    """Geometry and demand rates of one coarse grid."""
    def __init__(self, gid, gtype, station_count, rate, center, bound, centroid, boundary):
        self.id = gid  # coarse grid identifier
        self.type = gtype  # area type ("centre"/"suburb")
        self.station_count = station_count  # stations inside this grid

        # rate: tuple (2,), peak/off-peak requests per minute; center: (latitude, longitude).
        # bound: tuple (4,), min/max latitude then min/max longitude.
        self.rate = rate  # peak/off-peak arrival rates per minute
        self.center = center  # cell center (latitude, longitude)
        self.bound = bound  # (min_lat, max_lat, min_lon, max_lon)

        # centroid: (longitude, latitude); boundary_points: K coordinate pairs in that order.
        # Geometry fields may be serialized strings during source-table construction.
        self.centroid = centroid  # centroid (longitude, latitude)
        self.boundary_points = boundary  # boundary ring coordinates

    def __str__(self):
        return (
            f"Grid ID: {self.id}: type {self.type}, centroid {self.centroid}, rates {self.rate}. "
        )
