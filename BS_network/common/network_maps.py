"""City, station, and request maps from source tables."""

import ast
import base64
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd

from config import data_params

# Station icon dimensions in pixels.
STATION_ICON_SIZE = 16
# ID label centred on the pin; font capped below the icon size.
STATION_LABEL_FONT = 11

# Configure city-boundary line and fill appearance.
BOUNDARY_STYLE = {
    "color": "blue",
    "weight": 1,
    "fillColor": "#ADD8E6",
    "fillOpacity": 0.25,
}

# Build the fixed map legend from HTML rows matching the map markers.
LEGEND_TEMPLATE = """
            <div style="position: fixed;
                        top: 10px; right: 10px; width: 140px; height: auto;
                        background-color: white; z-index:9999;
                        font-size:12px; border:1px solid grey; padding: 10px;
                        border-radius: 1px;">
                <b>Type of points</b><br>
                {rows}
            </div>
            """

DOT_ROW = (
    '<i style="background:{color}; width:10px; height:10px; border-radius:50%; '
    'display:inline-block; margin-right:10px; vertical-align: middle;"></i> {label}<br>'
)

MARKER_COLOR = "orange"
REQUEST_COLOR = "blue"
# Grid centroid colors by area type.
GRID_MARKER_COLORS = {"centre": "red", "suburb": "orange"}
# Every cluster centroid shares the same light-blue icon.
CLUSTER_MARKER_COLOR = "#ADD8E6"
# Cluster area fills by type: centre = light orange, suburb = light purple.
CLUSTER_AREA_COLORS = {"centre": "#FFCC99", "suburb": "#D8BFD8"}


class NetworkMap:
    """Renders cells, stations, and sampled requests on a city map."""
    def __init__(self, city=None):
        """Initialize the city label."""
        self.city = city or data_params["city"]  # str: case-study city label

    def plot_maps(self, flag="Clustering", station_csv=None, show_id=True):
        """Render and save a selected city-network layout (AMap tiles, id + 1 labels)."""
        # Cell layers: (CSV name, label, color, marker radius, label font).
        if flag == "Griding":
            suffix, cells = "grid_map", [(data_params["grid_data"], "Grid", "red", 10, 14)]
        elif flag == "Clustering":
            suffix, cells = "cluster_map", [
                (data_params["cluster_data"], "Cluster", "red", 10, 14)
            ]
        elif flag == "Combine":
            suffix, cells = "map", [
                (data_params["grid_data"], "Grid", "red", 2.5, 12),
                (data_params["cluster_data"], "Cluster", "green", 2.5, None),
            ]

        # Draw the city geometry over the map and add the selected spatial cell layers.
        city_shape = gpd.read_file(Path(data_params["data_dir"]) / data_params["geojson_data"])
        # AMap tiles use GCJ-02, so WGS84 overlays may be offset unless converted.
        city_map = folium.Map(location=self._center(city_shape), zoom_start=10, tiles=None)
        folium.TileLayer(
            tiles=(
                "https://webrd0{s}.is.autonavi.com/appmaptile"
                "?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}"
            ),
            attr='&copy; <a href="https://www.amap.com/">高德地图</a>',
            name="高德地图",
            subdomains="1234",
            max_zoom=18,
        ).add_to(city_map)
        folium.GeoJson(
            city_shape.geometry.__geo_interface__,
            name="City Boundary",
            style_function=lambda _: dict(BOUNDARY_STYLE),
        ).add_to(city_map)

        for csv_name, label, color, radius, label_font in cells:
            self._add_cells(
                city_map, self._read(csv_name), color, radius, label_font, label=label,
            )

        # Overlay station and request locations, then attach their legend.
        self._add_stations(city_map, station_csv, show_id)
        self._add_requests(city_map)
        city_map.get_root().html.add_child(
            folium.Element(self._legend(cells))
        )

        # Choose a layout/scenario-specific output filename and save the HTML map.
        year = Path(data_params["cluster_data"]).stem.split("_")[-1]
        name = f"{self.city}_{suffix}_{year}"
        if station_csv:
            name += "_" + Path(station_csv).stem.replace("Nanjing_stations_", "")
        result_dir = Path(data_params["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        output = result_dir / f"{name}.html"
        city_map.save(str(output))
        print(f"City map with {flag.lower()} layout has been saved.")
        return output

    def _read(self, csv_name):
        """Read a source CSV from the data directory."""
        return pd.read_csv(Path(data_params["data_dir"]) / csv_name)

    def _add_cells(
        self, city_map, table, color, radius, label_font, label="Cell"
    ):
        """Add spatial cells, centroids, and labels to the map."""
        # Clusters: type-based fills + shared icon; grids: plain outlines + type-colored markers.
        is_cluster = label == "Cluster"
        for _, cell in table.iterrows():
            # Map display uses id + 1; internal ids stay 0-based.
            display_id = int(cell["id"]) + 1
            # ring: list of K float pairs (longitude, latitude); convert to latitude/longitude for Folium.
            ring = ast.literal_eval(cell["boundary_points"])
            cell_type = str(cell["type"]).lower() if "type" in table.columns else ""
            if is_cluster:
                # Cluster polygons are filled light orange (centre) or light purple (suburb).
                area_color = CLUSTER_AREA_COLORS.get(cell_type, color)
                fill, fill_color, fill_opacity = True, area_color, 0.3
                marker_color = CLUSTER_MARKER_COLOR
            else:
                area_color = color
                fill, fill_color, fill_opacity = False, area_color, 0.25
                marker_color = GRID_MARKER_COLORS.get(cell_type, MARKER_COLOR)
            folium.Polygon(
                locations=[[float(y), float(x)] for x, y in ring],
                color=area_color,
                weight=1.0,
                fill=fill,
                fill_color=fill_color,
                fill_opacity=fill_opacity,
            ).add_to(city_map)

            # Add a centroid marker and an optional cell-ID label.
            longitude, latitude = ast.literal_eval(cell["centroid"])
            location = [latitude, longitude]
            folium.CircleMarker(
                location=location,
                radius=radius,
                color=marker_color,
                weight=0.5,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.75,
                popup=(
                    f"{label} ID: {display_id} ({cell['type']})"
                    if "type" in table.columns
                    else f"{label} ID: {display_id}"
                ),
            ).add_to(city_map)

            if label_font:
                folium.map.Marker(
                    location=location,
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: {label_font}px; color: black; '
                        f'text-align: center;">{display_id}</div>'
                    ),
                ).add_to(city_map)

    def _add_stations(self, city_map, station_csv=None, show_id=True):
        """Add station locations, icons, and their centred ID labels to the map."""
        stations = self._read(station_csv or data_params["station_data"])
        for _, station in stations.iterrows():
            # Display id + 1; the internal station id stays 0-based.
            display_id = int(station["id"]) + 1
            # Folium elements have one parent, so every marker needs its own icon instance.
            icon = folium.CustomIcon(
                str(Path(data_params["data_dir"]) / data_params["station_icon"]),
                icon_size=(STATION_ICON_SIZE, STATION_ICON_SIZE),
            )
            folium.Marker(
                location=(station["latitude"], station["longitude"]),
                icon=icon,
                popup=f"ID: {display_id}, Type: Station",
            ).add_to(city_map)

            if show_id:
                folium.map.Marker(
                    location=(station["latitude"], station["longitude"]),
                    icon=folium.DivIcon(
                        # Shift the label onto the bottom-anchored pin's centre.
                        html=(
                            f'<div style="font-size: {STATION_LABEL_FONT}px; '
                            f'line-height: 1; color: black; font-weight: bold; '
                            f'white-space: nowrap; '
                            f'text-shadow: -1px -1px 0 white, 1px -1px 0 white, '
                            f'-1px 1px 0 white, 1px 1px 0 white; '
                            f'transform: translate(-50%, '
                            f'calc(-50% - {STATION_ICON_SIZE // 2}px));">'
                            f'{display_id}</div>'
                        ),
                        icon_size=(0, 0),
                    ),
                ).add_to(city_map)

    def _add_requests(self, city_map):
        """Draw the sampled scenario orders (data/Sample_orders.csv) on the map."""
        # The map uses the saved sample scenario directly, not the network order table.
        sample_path = Path(data_params["data_dir"]) / data_params["sample_orders"]
        if not sample_path.is_file():
            raise FileNotFoundError(
                f"Sample orders table missing: {sample_path}; run Instance.order_save() "
                "to generate it"
            )
        requests = self._read(data_params["sample_orders"])
        for _, request in requests.iterrows():
            folium.CircleMarker(
                location=(request["latitude"], request["longitude"]),
                radius=2,
                color=REQUEST_COLOR,
                weight=0.25,
                fill=True,
                fill_color=REQUEST_COLOR,
                fill_opacity=0.5,
                popup=f"ID: {int(request['id']) + 1}, Type: Request",
            ).add_to(city_map)

    def _legend(self, cells):
        """Build the legend for requests, stations, and spatial cells."""
        icon = base64.b64encode(
            (Path(data_params["data_dir"]) / data_params["station_icon"]).read_bytes()
        ).decode("ascii")

        # rows: list[str] of HTML snippets for requests, station icons, and selected cell layers.
        rows = [
            DOT_ROW.format(color=REQUEST_COLOR, label="Request"),
            f'<img src="data:image/png;base64,{icon}" width="{STATION_ICON_SIZE}" '
            f'height="{STATION_ICON_SIZE}" alt="station" '
            f'style="margin-right:10px; vertical-align: middle;"> Station<br>',
        ]
        # Clusters legend the shared icon + both area fills; grids one shared dot.
        for _, label, *_ in cells:
            if label == "Cluster":
                rows.append(DOT_ROW.format(color=CLUSTER_MARKER_COLOR, label="Cluster"))
                rows.append(DOT_ROW.format(color=CLUSTER_AREA_COLORS["centre"], label="Centre cluster"))
                rows.append(DOT_ROW.format(color=CLUSTER_AREA_COLORS["suburb"], label="Suburb cluster"))
            else:
                rows.append(DOT_ROW.format(color=MARKER_COLOR, label=label))
        return LEGEND_TEMPLATE.format(rows="".join(rows))

    @staticmethod
    def _center(city_shape):
        """Return the center of the city bounding box."""

        # total_bounds: float ndarray (4,) in min longitude/latitude, max longitude/latitude order.
        min_lon, min_lat, max_lon, max_lat = city_shape.total_bounds
        return [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
