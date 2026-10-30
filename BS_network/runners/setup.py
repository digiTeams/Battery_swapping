"""Build the source network and inspect its stations and demand."""
import ast
from collections import Counter

from network import set_networks
from common import network_maps
from common.instances import Instance
from config import data_params

def format_counts(counter, unit=''):
    """Format category counts for display."""
    return ', '.join(
        f'{value}{unit}: {count}'
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    )

def format_grid(grid):
    """Return the grid line with 4-decimal centroid coordinates."""
    longitude, latitude = _centroid_pair(grid.centroid)
    return (
        f"Grid ID: {grid.id}: type {grid.type}, "
        f"centroid ({longitude:.4f}, {latitude:.4f}), rates {grid.rate}."
    )

def format_cluster(cluster):
    """Return the cluster line with 4-decimal centroid coordinates."""
    longitude, latitude = _centroid_pair(cluster.centroid)
    return f"Cluster {cluster.id}: {cluster.type}, ({longitude:.4f}, {latitude:.4f})"

def cluster_size_line(cluster):
    """Return the cluster's length/width/diagonal distances and driving minutes."""
    return (
        f"  size: length {cluster.length_distance:.2f} km "
        f"({cluster.length_driving_time:.2f} min), "
        f"width {cluster.width_distance:.2f} km "
        f"({cluster.width_driving_time:.2f} min), "
        f"diagonal {cluster.diagonal_distance:.2f} km "
        f"({cluster.diagonal_driving_time:.2f} min)"
    )

def _centroid_pair(centroid):
    """Return (longitude, latitude) floats from a tuple or its serialized string."""
    if isinstance(centroid, str):
        centroid = ast.literal_eval(centroid)
    return float(centroid[0]), float(centroid[1])

def station_summary(network):
    """Print/return counts by capacity, stock, power, and area."""
    types = {cluster.id: cluster.type for cluster in network.Clusters}
    areas = Counter(types.get(station.cluster, 'unknown') for station in network.Stations)
    per_cluster = Counter(station.cluster for station in network.Stations)

    summary = {
        'total': len(network.Stations),
        'capacity': Counter(station.capacity for station in network.Stations),
        'stock': Counter(station.initial_batteries for station in network.Stations),
        'grid_power': Counter(station.grid_power for station in network.Stations),
        'area': areas,
        'clusters_used': len(per_cluster),
        'clusters_total': len(network.Clusters),
        'per_cluster': per_cluster,
    }

    counts = list(per_cluster.values()) or [0]
    print('-------------------- Station statistics --------------------')
    print(f"Stations in total      : {summary['total']}")
    print(f'By capacity    : {format_counts(summary["capacity"], "")}')
    print(f'By grid power          : {format_counts(summary["grid_power"], " kW")}')
    print(f'By area type           : {format_counts(areas)}')
    print(f"Clusters with stations : {summary['clusters_used']} of {summary['clusters_total']}")
    print(f'Stations per cluster   : min {min(counts)}, max {max(counts)}, '
          f'mean {sum(counts) / len(counts):.2f}')
    return summary

def check_network(city, network):
    """Print the static network and render its cluster map."""
    print('-------------------- Grids --------------------')
    for grid in network.Grids:
        print(format_grid(grid))
    print('-------------------- Clusters --------------------')
    for cluster in network.Clusters:
        print(format_cluster(cluster))
        print(cluster_size_line(cluster))

    print('-------------------- Stations --------------------')
    for station in network.Stations:
        print(station)
    station_summary(network)

    maps = network_maps.NetworkMap(city)
    maps.plot_maps('Clustering')

def check_demand(episode=0, demand_seed=None, network=None):
    """Generate/save one scenario from the mapped table; builds the network if absent."""
    seed = (
        int(data_params["seed"]) + 2 * episode + 1
        if demand_seed is None
        else int(demand_seed)
    )

    instance = Instance(network.order_data, data_params["scale"], episode, seed)
    instance.order_generation()
    instance.order_save()

    # Statistics over the sampled orders: area, classes, upgrades, SOC, fallback level.
    class_name = {1: "low", 2: "high"}
    areas = Counter(r.area for r in instance.requests)
    installed = Counter(class_name.get(r.installed_class, str(r.installed_class))
                        for r in instance.requests)
    requested = Counter(class_name.get(r.requested_class, str(r.requested_class))
                        for r in instance.requests)
    upgrades = sum(r.installed_class < r.requested_class for r in instance.requests)
    fallbacks = Counter(r.sample_fallback for r in instance.requests)
    socs = [r.SOC for r in instance.requests]
    print('-------------------- Sampled order statistics --------------------')
    print(f'By cluster type : {format_counts(areas)}')
    print(f'Installed class : {format_counts(installed)}')
    print(f'Requested class : {format_counts(requested)}')
    print(f'Upgrades (low->high): {upgrades}')
    print(f'Source fallback : {format_counts(fallbacks)}')
    if socs:
        print(f'SOC             : min={min(socs):.3f}, max={max(socs):.3f}, '
              f'mean={sum(socs) / len(socs):.3f}')
    else:
        print('SOC             : no orders')

# Build source tables, inspect the network, and regenerate expansion scenarios.
if __name__ == '__main__':
    city = 'Nanjing'
    print('----------- Set up the Nanjing NIO BS network ----------')
    network = set_networks.BSN(city, seed=None)
    network.build_BSN()
    check_network(city, network)
    check_demand(episode=0, demand_seed=None, network=network)

    print('----------- Expand the Nanjing NIO BS network ----------')
    base_stations = list(network.Stations)
    for added in data_params["scenarios"]:
        network.Stations = list(base_stations)
        network.extend_network(
            selected=[], num=added, seed_n=int(data_params["seed"]) + int(added)
        )
