from shapely.geometry import Polygon
import geopandas as gpd

# Define the dimensions of the large rectangle
width, height = 3000, 3000  # in meters
cols, rows = 3, 3  # Grid specification

# Function to create subrectangles as polygons
def create_subrectangles(width, height, cols, rows):
    subrectangles = []
    subwidth = width / cols
    subheight = height / rows
    for i in range(cols):
        for j in range(rows):
            lower_left_x = i * subwidth
            lower_left_y = j * subheight
            # Define the corners of the rectangle
            rectangle = Polygon([
                (lower_left_x, lower_left_y),
                (lower_left_x + subwidth, lower_left_y),
                (lower_left_x + subwidth, lower_left_y + subheight),
                (lower_left_x, lower_left_y + subheight),
                (lower_left_x, lower_left_y)  # Close the polygon by repeating the first point
            ])
            subrectangles.append(rectangle)
    return subrectangles

# Create the subrectangles
subrectangles = create_subrectangles(width, height, cols, rows)

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(geometry=subrectangles)

# Save the GeoDataFrame to shapefiles
shapefile_path = 'simple_rectangle_shapefile'
gdf.to_file(shapefile_path, driver='ESRI Shapefile')

print(f"Shapefile saved to {shapefile_path}")