"""Shared haversine great-circle distances (mean Earth radius)."""

import numpy as np

EARTH_RADIUS_KM = 6371.0088  # mean Earth radius, km


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; degrees, scalars or broadcastable arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = (
        np.sin((lat1 - lat2) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
