import sys
from qgis.core import (
    QgsApplication,
    QgsVectorLayer,
    QgsProject,
    QgsDistanceArea,
    QgsPointXY,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform
)

# Initialize QGIS Application
QgsApplication.setPrefixPath("/path/to/qgis", True)
qgs = QgsApplication([], False)
qgs.initQgis()

# Load the OSM data (assuming you have a .shp or .geojson file)
osm_layer_path = "/path/to/osm_data.shp"
osm_layer = QgsVectorLayer(osm_layer_path, "OSM Schools", "ogr")
if not osm_layer.isValid():
    print("Layer failed to load!")
    sys.exit(1)

# Add layer to the project
QgsProject.instance().addMapLayer(osm_layer)

# Define the coordinates to check and buffer distance
target_lon, target_lat = -0.1257, 51.5085  # Example coordinates
buffer_distance = 1000  # Distance in meters

# Create a point from the coordinates
target_point = QgsPointXY(target_lon, target_lat)

# Create a coordinate reference system (CRS)
crs_epsg_4326 = QgsCoordinateReferenceSystem("EPSG:4326")
crs_project = osm_layer.crs()

# Transform the point to the layer's CRS
transform = QgsCoordinateTransform(crs_epsg_4326, crs_project, QgsProject.instance())
transformed_point = transform.transform(target_point)

# Create a distance area object
distance_area = QgsDistanceArea()
distance_area.setSourceCrs(crs_project, QgsProject.instance().transformContext())

# Buffer the point by the specified distance
buffer_geom = QgsGeometry.fromPointXY(transformed_point).buffer(buffer_distance, 5)

# Check for intersections with the OSM schools layer
for feature in osm_layer.getFeatures():
    if buffer_geom.intersects(feature.geometry()):
        print(f"School found: {feature['name']}")

# Exit QGIS
qgs.exitQgis()
