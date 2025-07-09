## This code is intended to be a self-contained implementation that supports
## use of existing state data and creations of custom states. Currently only
## allows user to specify the parameters of a custom state and generates all
## of the supporting files for optimization

from shapely.geometry import Polygon
from shapely.geometry import Polygon as ShapelyPolygon, LineString
from shapely.ops import split

import geopandas as gpd
import numpy as np
import geopandas
import json
import random
import sys
import os
import glob
import networkx as nx
import cvxpy as cp
import scipy.sparse as sps
import datetime

import pandas
import geopandas
from libpysal.weights import Rook
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb
import tkinter as tk

from OT_solver import solve_Kantorovich_with_details

# Function to dynamically generate window size based on monitor resolution 
def screen_size(scale_factor=0.8):
    """
    Dynamically calculate the figure size for Matplotlib plots based on the screen resolution.

    Parameters:
    - scale_factor: float, proportion of the screen size to use (default is 0.8 or 80%)
    - dpi: int, dots per inch for screen resolution (default is 100)

    Returns:
    - figsize: tuple (width_in_inches, height_in_inches)
    """
    # Get screen resolution using tkinter
    root = tk.Tk()
    dpi = root.winfo_fpixels('1i')  # Get DPI (dots per inch)
    screen_width = root.winfo_screenwidth()  # Screen width in pixels
    screen_height = root.winfo_screenheight()  # Screen height in pixels
    root.destroy()

    # Calculate figure size in inches
    fig_width = (screen_width / dpi) * scale_factor
    fig_height = (screen_height / dpi) * scale_factor

    return (fig_width, fig_height)

# Function to generate a list of maximally distinct colors that also provide good contrast
# for black text
def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors       # Evenly spaced hues around the color wheel
        saturation = 0.5          # Moderate saturation for pastel shades
        value = 0.95              # High value for light colors
        color = hsv_to_rgb((hue, saturation, value))
        colors.append(color)
    return colors

def get_user_int(prompt_str, min=0, max=-99999):
    """
    Prompt the user to input an integer value within a specified range or from a specified list.

    :param prompt_str: The prompt string to display to the user.
    :param min: Either a single integer value or a list/array of allowable integers.
    :param max: The maximum allowable value (only applicable when `min` is a single integer).
    :return: The validated integer input from the user.
    """
    done = False
    while not done:
        user_input = input(prompt_str)
        
        if user_input.isnumeric():
            user_int = int(user_input)
            
            # Check if `min` is a list or array of allowable integers
            if isinstance(min, (list, tuple, np.ndarray)):
                if user_int in min:
                    done = True
                else:
                    print(f"Error: specified value must be one of {min}. Try again.")
            else:  # `min` is a single integer
                if user_int >= min:
                    if max > min and user_int <= max:
                        done = True
                    elif max < min:
                        done = True
        
        if not done:
            if isinstance(min, (list, tuple, np.ndarray)):
                print(f"Error: specified value must be one of {min}. Try again.")
            else:
                if max > min:
                    print(f"Error: specified value must be an integer between {min} and {max}. Try again.")
                else:
                    print(f"Error: specified value must be an integer >= {min}. Try again.")
    
    return user_int

def get_menu_choice(prompt_str, options):
    """
    Display a menu and get a valid choice from the user.

    :param prompt_str: The prompt string to display before the options.
    :param options: A dictionary where keys are single-character options 
                    and values are their descriptions.
    :return: The chosen option's key.
    """
    # Construct the menu display
    menu_display = f"{prompt_str}\n"
    for key, description in options.items():
        menu_display += f"  {key}. {description}\n"

    # Loop until a valid choice is made
    while True:
        print(menu_display)
        choice = input("Please select an option: ").strip().lower()
        if choice in options:
            return choice
        else:
            print("Invalid choice. Please select a valid option.")

def does_user_agree(prompt):
    """
    Wait for the user to press 'y' or 'n'. Returns True for 'y' and False for 'n'.
    Ignores case and other inputs. User does not need to press Enter.

    :param prompt: The prompt string to display to the user.
    :return: True if the user presses 'y', False if the user presses 'n'.
    """
    print(prompt + " (y/n): ", end="", flush=True)
    
    if os.name == 'nt':  # Windows
        import msvcrt
        while True:
            key = msvcrt.getch().decode('utf-8').lower()  # Read a single key press
            if key in ['y', 'n']:
                print(key)  # Echo the pressed key
                print("\r" + " " * len(prompt + " (y/n): y") + "\r", end="")
                return key == 'y'
    else:  # Unix-like systems
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)  # Set raw mode to capture single key press
            while True:
                key = sys.stdin.read(1).lower()  # Read a single character
                if key in ['y', 'n']:
                    print(key)  # Echo the pressed key
                    # Clear the remaining part of the line and reset the cursor
                    sys.stdout.write("\r" + " " * len(prompt + " (y/n): y") + "\r")
                    return key == 'y'
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def generate_assignment_text(source_id, assignments, transport_details, include_target=True):
    """
    Generate a multi-line string with source-to-target assignments and percentage contributions.

    Args:
        source_id (int): The source district ID.
        assignments (int or list): A single target district ID or a list of target district IDs.
        transport_details (dict): A dictionary where keys are source IDs and values are dictionaries
                                  mapping target IDs to transported masses.

    Returns:
        str: A multi-line string showing source-to-target assignments and percentage contributions.
    """
    # Normalize assignments to always be a list
    if isinstance(assignments, int):
        assignments = [assignments]

    # People from this source district assigned to each target
    source_assignments = transport_details.get(source_id, {})  # Get target assignments for the source district

    # Total people assigned to each target district
    target_totals = {
        target: sum(td.get(target, 0) for td in transport_details.values())
        for target in assignments
    }

    # Percent contributions for this source district
    percent_contributions = {
        target: (source_assignments.get(target, 0) / target_totals[target]) * 100 if target_totals[target] > 0 else 0
        for target in assignments
    }

    # Build assignment string
    if include_target == True:
        assignment_str = str(source_id) + "→" + ", ".join(str(int(x)) for x in assignments)
    else:
        assignment_str = str(source_id)

    # Build percentage string with single decimal place for values >= 1 and 2 decimals otherwise
    percent_str = ", ".join(
        f"{percent_contributions[x]:.2f}%" if percent_contributions[x] < 1 else f"{percent_contributions[x]:.1f}%" 
        for x in assignments
    )

    # Combine into a multi-line string
    text_str = f"{assignment_str}\n{percent_str}"
    
    return text_str

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

def validateGraph(G):
    """
    Validate if the graph G is a valid NetworkX graph with at least
    two nodes and one edge.

    Args:
        G (nx.Graph): The input graph to validate.

    Returns:
        bool: True if the graph is valid, False otherwise.
    """
    # Check if G is a NetworkX graph
    if not isinstance(G, nx.Graph):
        print("The object is not a valid NetworkX graph.")
        return False

    # Check if the graph has at least two nodes
    if len(G.nodes) < 2:
        print("The graph does not have at least two nodes.")
        return False

    # Check if the graph has at least one edge
    if len(G.edges) < 1:
        print("The graph does not have at least one edge.")
        return False

    #print("The graph is valid.")
    return True

def GraphWeights(G,epsilon = 0.25):
    for e in list(G.edges):
        G.edges[e]['weight'] = random.uniform(1-epsilon,1+epsilon)
    return G

def quadratic_distance(x, y):
    return np.sum(np.square(np.array(x) - np.array(y)))

def graph_distance_matrix(graph,targetlist):  
    N = len(graph.nodes)
    k = len(targetlist)
    
    matrix = np.empty((N,k))   
    print('Computing partial distance matrix with networkx')
    for i in range(N):
        for j in range(k):
            matrix[i,j] = nx.shortest_path_length(graph,source = i, target = targetlist[j])
            #matrix[i,j] = nx.shortest_path_length(graph,source = i, target = targetlist[j]) + ep * nx.shortest_path_length(graph,source = i, target = targetlist[j]) ** 2    
        if (N-i)%10==0:
            print(N-i)        
            
    return matrix

def distance_matrix(points):
    N = len(points)
    matrix = np.fromfunction(
        np.vectorize(lambda i, j: quadratic_distance(points[i], points[j])),
        (N, N),
        dtype=int,
    )
    return matrix

def partial_distance_matrix(points,targetlist):
    N = len(points)
    k = len(targetlist)
    
    matrix = np.empty((N,k))   
    
    for i in range(N):
        for j in range(k):
            matrix[i][j] = quadratic_distance(points[i],points[targetlist[j]])
            
    return matrix

def get_graph_from_shapefile(filepath, id_column=None):
    #weights = pysal.rook_from_shapefile(filepath, id_column)
    weights = Rook.from_shapefile(filepath, id_column)    
    return nx.Graph(weights.neighbors)

# Debugging Function
def build_graph_with_debug(df, data_cost_matrix, plan_id):
    # Debugging log
    print("Starting graph construction...")

    # Initialize the graph
    G = nx.Graph()

    # Check if df is a GeoDataFrame
    if not isinstance(df, gpd.GeoDataFrame):
        print("Warning: df is not a GeoDataFrame. Attempting to convert...")
        try:
            df = gpd.GeoDataFrame(df, geometry=df[plan_id])
        except Exception as e:
            print(f"Error converting df to GeoDataFrame: {e}")
            return None

    # Debug: Print CRS before conversion
    print("Original CRS:", df.crs)

    # Convert CRS
    try:
        df = df.to_crs("EPSG:3857")  # Example: Web Mercator
        print("CRS successfully converted to EPSG:3857.")
    except Exception as e:
        print(f"Error converting CRS: {e}")
        return None

    # Debug: Validate geometry
    if 'geometry' not in df.columns or df.geometry.is_empty.any():
        print("Error: GeoDataFrame has missing or invalid geometry.")
        return None

    # Calculate centroids
    try:
        df['centroid'] = df.geometry.centroid  # Ensure geometry column is correctly set
        print("Centroids calculated.")
    except Exception as e:
        print(f"Error calculating centroids: {e}")
        return None

    # Add nodes with centroids as attributes
    for idx, row in df.iterrows():
        try:
            G.add_node(idx, centroid=(row.centroid.x, row.centroid.y))
        except Exception as e:
            print(f"Error adding node {idx}: {e}")

    print(f"Total nodes added: {G.number_of_nodes()}")

    # Add edges based on the data_cost_matrix
    try:
        rows, cols = data_cost_matrix.shape
        print(f"Data cost matrix shape: {rows}x{cols}")
        for i in range(rows):
            for j in range(i + 1, cols):  # Avoid duplicate edges in an undirected graph
                print(data_cost_matrix[i, j])
                if data_cost_matrix[i, j] in [0, 1]:  # Rule: 0 or 1 -> weight = 1
                    G.add_edge(i, j, weight=1)
                elif data_cost_matrix[i, j] == -1:  # Rule: -1 -> weight = -1
                    G.add_edge(i, j, weight=-1)
    except Exception as e:
        print(f"Error adding edges: {e}")
        return None

    print(f"Total edges added: {G.number_of_edges()}")

    # Final validation of the graph
    if len(G.nodes) < 2:
        print("Warning: Graph has less than 2 nodes.")
    if len(G.edges) < 1:
        print("Warning: Graph has no edges.")

    print("Graph construction complete.")
    return G

def distance_matrix_from_shapefile(filepath):
    df = geopandas.read_file(filepath)
    points = [centroid.coords for centroid in df.centroid]
    matrix = distance_matrix(points)
    return matrix

def partial_distance_matrix_from_shapefile(filepath,targetlist):
    df = geopandas.read_file(filepath)
    points = [centroid.coords for centroid in df.centroid]
    #targetlist = np.random.choice(len(points),number_of_targets,replace=False)
    matrix = partial_distance_matrix(points,targetlist)
    #print(targetlist)
    return matrix

def partial_graph_distance_matrix_from_shapefile(filepath,targetlist):
    graph = get_graph_from_shapefile(filepath)
        
    matrix = graph_distance_matrix(graph,targetlist)
    return matrix

# Function to count and identify sources assigned to multiple targets
def count_splitting(coupling_matrix):
    """
    Identifies and counts sources in the coupling matrix assigned to multiple targets.
    
    Args:
        coupling_matrix (np.ndarray): The coupling matrix, where rows represent sources and columns represent targets.
    
    Returns:
        counter (int): Number of sources assigned to multiple targets.
        splitting_sources (list): List of source IDs assigned to multiple targets.
        target_assignments (dict): A dictionary mapping source IDs to their target IDs.
    """
    M = len(coupling_matrix[0])  # Number of targets
    N = len(coupling_matrix[:, 0])  # Number of sources

    counter = 0
    splitting_sources = []
    target_assignments = {}

    for i in range(N):
        # Identify nonzero entries (targets assigned to this source)
        assigned_targets = np.nonzero(coupling_matrix[i])[0]
        
        if len(assigned_targets) > 1:  # Check if assigned to multiple targets
            counter += 1
            splitting_sources.append(i)
            target_assignments[i] = assigned_targets.tolist()  # Store target IDs

    return counter, splitting_sources, target_assignments

#This function counts the number of non-zero rows in a distribution
def count_support(distribution):  
  
  N = len(distribution)
  
  counter = int(0)
  
  for i in range(N):
      if distribution[i] > 0:
          counter = counter+1
  return counter

# Function to check and retrieve assignments
def get_multiple_assignments(district_id, assignments):
    """
    Checks if a district has multiple assignments and retrieves them.
    
    Args:
        district_id (int): The ID of the district to check.
        assignments (dict): The dictionary of assignments.

    Returns:
        tuple: (bool, list) where bool indicates if there are multiple assignments,
               and list contains the assignments (or an empty list if not applicable).
    """
    if district_id in assignments:
        assigned_targets = assignments[district_id]
        if len(assigned_targets) > 1:
            return True, assigned_targets
    return False, []

#The next two functions take an optimal transport plan and if possible returns a corresponding optimal transport map
def plan_to_map(matrix,geoidtable):
    districts = dict()
    maxes = np.argmax(matrix, axis=1)
    #districts.update({'GEOID':'DISTRICT'})
    for i, j in enumerate(maxes):
        #districts.update({geoidtable[i][7:]:j}) #Only for data sets that have a GEOID suffix
        districts.update({geoidtable[i]:j})
    return districts    

def plan_to_map_array(size,matrix,geoidtable):
    districts = np.empty(size)
    maxes = np.argmax(matrix, axis=1)
    #districts.update({'GEOID':'DISTRICT'})
    for i, j in enumerate(maxes):
        #districts.update({geoidtable[i][7:]:j}) #Only for data sets that have a GEOID suffix
        districts[i] = j
    return districts
    
def calculate_divergence(G, a, b):
    # Number of nodes in the graph
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    
    # Initialize the divergence matrix with zeros
    D = np.zeros((num_nodes, num_edges))

    # Edge index for tracking
    edge_index = 0
    
    # Calculate divergence for each edge
    for u, v in G.edges:
        # Update divergence for the nodes connected by the edge
        D[u][edge_index] = -1
        D[v][edge_index] = 1
        edge_index += 1

    return D

def solve_beckmann(G, a, b):
    num_edges = G.number_of_edges()
    print("Graph edges:", num_edges)
    S = cp.Variable(num_edges)

    # We calculate the divergence matrix D...
    D = calculate_divergence(G, a, b)
    #print("Calculated Divergence matrix D:")
    #print(D)

    constraints = [D@S == b-a]
    prob = cp.Problem(cp.Minimize(cp.norm1(S)), constraints)

    prob.solve()

    sol = S.value

    edge_costs = np.array([G[u][v]['weight'] for u, v in G.edges()])
    total_cost = np.sum(edge_costs * np.abs(sol))  # Cost = sum of |flow| * edge cost
    print("Total Flow cost:", total_cost)

    return sol, total_cost

def add_arrow(ax, pos, u, v, flow_value, arrow_color='red', arrow_size=20):
    # Helper function to add an arrow to the graph, reversing direction if flow is negative
    if flow_value < 0:
        posA, posB = pos[v], pos[u]  # Reverse direction
    else:
        posA, posB = pos[u], pos[v]  # Normal direction

    arrow = FancyArrowPatch(posA=posA, posB=posB, arrowstyle='-|>', color=arrow_color,
                            mutation_scale=arrow_size, lw=1, connectionstyle='arc3,rad=0.0')
    ax.add_patch(arrow)

def visualize_transport_process(G, a, b, flow):
    pos = nx.spring_layout(G)  # Layout for the graph

    # Create a composite plot with original, intermediary, and final states
    #fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    fig, ax = plt.subplots(1, 3, figsize=screen_size(0.8))

    # Plot the original graph with initial mass distribution
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=700, ax=ax[0])
    labels = {i: f"{a[i]:.2f}" for i in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=14, font_color="black", ax=ax[0])
    for node, (x, y) in pos.items():
        ax[0].text(x, y - 0.15, s=str(node), horizontalalignment='center', fontsize=12, color='red')
    ax[0].set_title("Initial Mass Distribution")

    # Manually plot the nodes and edges to control layering, with axes turned off
    ax[1].scatter([pos[n][0] for n in G.nodes], [pos[n][1] for n in G.nodes], s=700, c='lightblue', zorder=1)
    for (u, v) in G.edges:
        ax[1].plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 'k-', zorder=1)

    # Draw arrows using FancyArrowPatch with increased size and handle negative flow
    for idx, (u, v) in enumerate(G.edges):
        add_arrow(ax[1], pos, u, v, flow[idx], arrow_size=25)  # Pass the flow value

    # Draw edge labels on top of arrows
    edge_labels = {edge: f"{flow[idx]:.2f}" for idx, edge in enumerate(G.edges)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=14, ax=ax[1])

    for node, (x, y) in pos.items():
        ax[1].text(x, y - 0.15, s=str(node), horizontalalignment='center', fontsize=12, color='red', zorder=3)
    ax[1].set_title("Mass Flow Along Edges")

    # Turn off the axes for the middle subplot to remove the box
    ax[1].axis('off')

    # Calculate and plot the final mass distribution
    final_mass = a.copy()
    for idx, (u, v) in enumerate(G.edges):
        final_mass[u] -= flow[idx]
        final_mass[v] += flow[idx]

    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=700, font_size=16, ax=ax[2])
    labels = {i: f"{final_mass[i]:.2f}" for i in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=14, font_color="black", ax=ax[2])
    for node, (x, y) in pos.items():
        ax[2].text(x, y - 0.15, s=str(node), horizontalalignment='center', fontsize=12, color='red')
    ax[2].set_title("Final Mass Distribution")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def find_nearest_district(event, dataframe):
    """
    Finds the nearest district to the click position.
    Args:
        event: Matplotlib event object containing click information.
        dataframe (GeoDataFrame): GeoDataFrame containing district geometries.
    Returns:
        nearest_district_id (int): ID of the nearest district.
    """
    if event.inaxes:  # Ensure the click is inside the axes
        x, y = event.xdata, event.ydata

        # Extract centroids or representative points
        centroids = dataframe.geometry.representative_point()
        centroid_coords = np.array([[point.x, point.y] for point in centroids])

        # Calculate distances to the click position
        click_coords = np.array([x, y])
        distances = np.linalg.norm(centroid_coords - click_coords, axis=1)

        # Find the nearest district
        nearest_idx = np.argmin(distances)
        nearest_district_id = dataframe.index[nearest_idx]
        #print(f"Nearest district ID: {nearest_district_id}")
        return nearest_district_id
    else:
        #print("Click was outside the axes.")
        return None

def on_click_with_zoom(event, dataframe):
    """
    Callback function to handle click events, find the nearest district, and open a zoomed region.
    Args:
        event: Matplotlib event object containing click information.
        dataframe (GeoDataFrame): GeoDataFrame containing district geometries.
    """
    nearest_district = find_nearest_district(event, dataframe)
    if nearest_district is not None:
        #print(f"Nearest district ID to the click: {nearest_district}")
        open_zoomed_region(dataframe, nearest_district)


def geopdvisual(statename, dataframe, plan_id, outfilename, highlight_ids=None, 
                multiple_assignments=None, transport_details=None, kantorovich_cost=0, 
                beckmann_cost=0, n_grid=1):
    """
    Visualizes state districts with colormap-based color assignments and highlights specific districts.

    Args:
        dataframe (GeoDataFrame): The GeoDataFrame containing the geometry and data.
        plan_id (str): The column used to determine colors for the map.
        outfilename (str): The file name for saving the plot.
        color (str): The colormap to use.
        highlight_ids (list): A list of IDs to highlight.
    """
    # Normalize the plan_id values for colormap
    #norm = mcolors.Normalize(vmin=dataframe[plan_id].min(), vmax=dataframe[plan_id].max())
    #cmap = cm.get_cmap(color)
    
    #cmap = plt.cm.tab10  # Replace with your desired colormap
    #highlight_normalized = np.linspace(0, 1, len(highlight_ids))
    #id_to_color = {highlight_id: cmap(normalized_val) for highlight_id, normalized_val in zip(highlight_ids, highlight_normalized)}

    highlight_colors = generate_colors(len(highlight_ids))
    id_to_color = {highlight_id: col for highlight_id, col in zip(highlight_ids, highlight_colors)}

    # Initialize interactive mode
    plt.ion()

    # Initialize the plot
    #fig, ax = plt.subplots(figsize=(20, 16))
    fig, ax = plt.subplots(figsize=screen_size(0.8))

    minx, miny, maxx, maxy = dataframe.total_bounds # get dimensions of state
    state_width = abs(maxx - minx)
    state_height = abs(maxy - miny)
    #print("state width",state_width,"height",state_height)

    for i in range(len(dataframe)):
        line_width = 0.5
        # Determine color based on colormap and plan_id value
        district_value = dataframe[plan_id].iloc[i]
        #print("district:", district_value, " i:", i)
        #facecolor = cmap(norm(district_value))
        facecolor = id_to_color.get(district_value)
        # Calculate the bounds of the geometry
        minx, miny, maxx, maxy = dataframe.geometry.iloc[i].bounds

        # Calculate width and height
        width = abs(maxx - minx)
        height = abs(maxy - miny)
        #print(i,"width",width/state_width,"height",height/state_height)

        # For smaller districts, don't show the target text
        if width/state_width >= 0.025 and height/state_height >= 0.025:
            # Display text
            display_target_id = True
        else:
            display_target_id = False
            line_width = 0.25

        if highlight_ids is not None and dataframe.index[i] in highlight_ids:
            continue
        else:
            has_multiple, assignments = get_multiple_assignments(i, multiple_assignments)
            if has_multiple:
                #print(f"District {i} DOES have multiple assignments: {assignments}")
                polygon_geometry = dataframe.geometry.iloc[i]
                if hasattr(polygon_geometry, 'exterior'):
                    ax.add_patch(Polygon(
                        polygon_geometry.exterior.coords,  # Exterior coordinates of the polygon
                        facecolor=id_to_color.get(assignments[1]),  # Primary color
                        edgecolor=id_to_color.get(assignments[0]),    # Hatch color
                        linewidth=line_width,        # Outline width
                        hatch="x"            # Hatch pattern for checkered effect
                    ))
                    for offset in range(-20, 20):  # Create a thicker crosshatch effect
                        ax.add_patch(Polygon(
                            [(x + offset*0.5, y + offset*0.5) for x, y in polygon_geometry.exterior.coords],
                            facecolor="none",
                            edgecolor=id_to_color.get(assignments[0]),
                            linewidth=line_width,
                            hatch="x"
                        ))
                    ax.add_patch(Polygon(
                        polygon_geometry.exterior.coords,  # Exterior coordinates of the polygon
                        facecolor="none",  # Primary color
                        edgecolor="white",
                        linewidth=line_width        # Outline width
                    ))
            else:
                #print(f"District {i} does not have multiple assignments.")
                # Use colormap for non-highlighted districts
                gpd.GeoSeries(dataframe.geometry.iloc[i]).plot(
                    ax=ax, facecolor=facecolor, edgecolor="white", linewidth=line_width
                )

        # Annotate the polygon with its ID
        centroid = dataframe.geometry.iloc[i].centroid
        if width/state_width <= 0.005 or height/state_height <= 0.005:
            text_str = ''
            line_width = 0.10
        elif has_multiple:
            text_str = generate_assignment_text(dataframe.index[i], assignments, transport_details, display_target_id)
            #assignment_str = str(dataframe.index[i]) + "→" + ", ".join(str(int(x)) for x in assignments)
        else:
            #assignment_str = str(dataframe.index[i]) + "→" + str(int(district_value))
            text_str = generate_assignment_text(dataframe.index[i], int(district_value), transport_details, display_target_id)
        ax.text(
            centroid.x, centroid.y, text_str,  # Replace with specific column if needed
            fontsize=6, color="black", ha="center", va="center"
        )

    for i in range(len(dataframe)):
        # Determine color based on colormap and plan_id value
        district_value = dataframe[plan_id].iloc[i]
        #facecolor = cmap(norm(district_value))
        facecolor = id_to_color.get(district_value)

        if width/state_width <= 0.005 or height/state_height <= 0.005:
            continue

        if highlight_ids is not None and dataframe.index[i] in highlight_ids:
            # Highlighted districts in red
            gpd.GeoSeries(dataframe.geometry.iloc[i]).plot(
                ax=ax, facecolor=facecolor, edgecolor="black", linewidth=1, hatch="."
            )
        else:
            continue

        # Annotate the polygon with its ID
        centroid = dataframe.geometry.iloc[i].centroid
        #assignment_str = str(dataframe.index[i]) + "→" + str(int(district_value))
        text_str = generate_assignment_text(dataframe.index[i], int(district_value), transport_details, display_target_id)
        ax.text(
            centroid.x, centroid.y, text_str,  # Replace with specific column if needed
            fontsize=6, color="black", ha="center", va="center"
        )

    # Add titles and labels
    plt.title(f"{statename} Voting District Assignments (click on map to open magnified view)")
    ax.axis('off')
    note = f"Kantorovich Cost: {kantorovich_cost}\nBeckmann Cost: {beckmann_cost}\n$\cdot$ represents target voting center district\nx represents districts assigned multiple voting centers\n% represents this district population divided by total population assigned to the common target district"
    plt.text(0.1, 0.1, note, transform=plt.gcf().transFigure, fontsize=14, 
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(f"{outfilename}_composite", dpi=600, bbox_inches="tight")
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click_with_zoom(event, dataframe, id_to_color, plan_id, highlight_ids, multiple_assignments, transport_details))
    plt.show(block=False)

def plot_districts(ax, dataframe, id_to_color, plan_id, highlight_ids=None, multiple_assignments=None, transport_details=None, fontsize=8):
    """
    Plots districts on the given axes with consistent formatting, colors, and patterns.

    Args:
        ax: Matplotlib axes to plot on.
        dataframe: GeoDataFrame containing district geometries and data.
        id_to_color: Dictionary mapping district IDs to colors.
        plan_id: Column name for district values.
        highlight_ids: List of IDs to highlight.
        multiple_assignments: Dictionary of districts with multiple assignments.
        transport_details: Dictionary of transport details for district assignments.
        fontsize: Font size for district labels.
    """
    for i in range(len(dataframe)):
        district_value = dataframe[plan_id].iloc[i]
        facecolor = id_to_color.get(district_value, "lightgray")
        geometry = dataframe.geometry.iloc[i]

        # Check if the district has multiple assignments
        has_multiple, assignments = get_multiple_assignments(dataframe.index[i], multiple_assignments) if multiple_assignments else (False, None)

        if has_multiple and assignments:
            # Cross-hatched pattern for multiple assignments
            primary_color = id_to_color.get(assignments[1], "lightgray")  # Primary color
            hatch_color = id_to_color.get(assignments[0], "black")       # Hatch color
            
            # Add the cross-hatched polygons
            ax.add_patch(Polygon(
                geometry.exterior.coords, facecolor=primary_color, edgecolor=hatch_color,
                linewidth=0.5, hatch="x"
            ))

            # Overlay a transparent white outline for contrast
            ax.add_patch(Polygon(
                geometry.exterior.coords, facecolor="none", edgecolor="white", linewidth=0.5
            ))
        elif highlight_ids is not None and dataframe.index[i] in highlight_ids:
            # Add dotted pattern for highlighted districts
            gpd.GeoSeries(geometry).plot(
                ax=ax, facecolor=facecolor, edgecolor="black", linewidth=1, hatch="."
            )
        else:
            # Plot regular districts
            gpd.GeoSeries(geometry).plot(
                ax=ax, facecolor=facecolor, edgecolor="white", linewidth=0.5
            )

        # Annotate districts
        centroid = geometry.representative_point()
        if has_multiple:
            # Generate text for multiple assignments
            text_str = generate_assignment_text(dataframe.index[i], assignments, transport_details)
        else:
            # Generate text for single assignment
            text_str = generate_assignment_text(dataframe.index[i], int(district_value), transport_details)

        ax.text(
            centroid.x, centroid.y, text_str,
            fontsize=fontsize, color="black", ha="center", va="center"
        )

def open_zoomed_region(dataframe, clicked_district_id, id_to_color, plan_id, highlight_ids=None, multiple_assignments=None, transport_details=None, zoom_factor=8):
    # Calculate bounds for zoomed region
    minx, miny, maxx, maxy = dataframe.total_bounds
    zoom_width = (maxx - minx) / zoom_factor
    zoom_height = (maxy - miny) / zoom_factor

    centroid = dataframe.loc[clicked_district_id].geometry.centroid
    zoom_minx = max(minx, centroid.x - zoom_width / 2)
    zoom_maxx = min(maxx, centroid.x + zoom_width / 2)
    zoom_miny = max(miny, centroid.y - zoom_height / 2)
    zoom_maxy = min(maxy, centroid.y + zoom_height / 2)

    # Filter districts within the zoomed region
    zoomed_region_df = dataframe.cx[zoom_minx:zoom_maxx, zoom_miny:zoom_maxy]

    # Plot the zoomed region
    #fig, ax = plt.subplots(figsize=(14, 14))
    fig, ax = plt.subplots(figsize=screen_size(0.8))
    ax.set_xlim(zoom_minx, zoom_maxx)
    ax.set_ylim(zoom_miny, zoom_maxy)
    ax.axis('off')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plot_districts(ax, zoomed_region_df, id_to_color, plan_id, highlight_ids, multiple_assignments, transport_details)
    plt.title(f"Zoomed Region Around District {clicked_district_id}")
    ax.set_xlim(zoom_minx, zoom_maxx)
    ax.set_ylim(zoom_miny, zoom_maxy)
    plt.show()

def on_click_with_zoom(event, dataframe, id_to_color, plan_id, highlight_ids, multiple_assignments, transport_details):
    nearest_district = find_nearest_district(event, dataframe)
    if nearest_district is not None:
        #print(f"Nearest district ID: {nearest_district}")
        num_source_districts = len(transport_details)
        zoom_level = int((num_source_districts+1000)/1000)
        if zoom_level <= 2:
            zoom_level = 2
        open_zoomed_region(dataframe, nearest_district, id_to_color, plan_id, highlight_ids, multiple_assignments, transport_details, zoom_level)
    
def select_custom_shapefile():
    customList = glob.glob('custom_*_shapefile_*')
    menuLength = len(customList)
    print("\nExisting custom shapefiles:")
    for i in range(0, menuLength):
        menuItem = str(i+1) + ". " + customList[i]
        print(menuItem)
    menuChoice = get_user_int("\nType the number of the custom state you wish to load and hit enter: ", 1, menuLength)
    shapeName = customList[menuChoice-1]
    print("You have chosen to load the custom state:", shapeName)
    # Determine number of districts based on shapefile name
    words = shapeName.split('_')
    rectangle_word = words[1]
    rectangle_word = rectangle_word.replace('rectangle', '')
    m_by_n = rectangle_word.split('x')
    total_districts = int(m_by_n[0]) * int(m_by_n[1])
    # print(total_districts)
    return shapeName, total_districts

def create_custom_shapefile():
    print("\nYou have chosen to create a custom (artificial) state.")
    print("Note that artificial states are rectangles with equally")
    print("partitioned district grids.\n")
    print("Let's get some parameters for the custom state (numerical values only)...")

    width = get_user_int("How WIDE should the state be (in meters): ", 1)
    height = get_user_int("How TALL should the state be (in meters): ", 1)
    num_district_rows = get_user_int("How many district rows in state: ", 1)
    num_district_cols = get_user_int("How many district columns in state: ", 1)
    max_voting_districts = num_district_cols * num_district_rows
    prompt_text = "How many districts contain voting centers (max = " + str(max_voting_districts) + "): "
    voting_districts = get_user_int(prompt_text, 1, max_voting_districts)

    ## Now create the shapefiles
    # First, the subrectangles
    subrectangles = create_subrectangles(width, height, num_district_cols, num_district_rows)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=subrectangles)

    # Save the GeoDataFrame to shapefiles
    shapefile_folder = "custom_" + str(num_district_rows) + "x" + str(num_district_cols) + "rectangle_shapefile_" + str(voting_districts)
    gdf.to_file(shapefile_folder, driver='ESRI Shapefile')

    print(f"Custom shapefile saved to {shapefile_folder}")

    # Now calculate costs associated with transport from each district to randomly assigned voting centers
    #shapefile_path = "./simple_5x5rectangle_shapefile/simple_5x5rectangle_shapefile.shp"
    shapefile_path = "./" + shapefile_folder + "/" + shapefile_folder + ".shp"

    # Determine number of districts based on shapefile
    df = geopandas.read_file(shapefile_path)
    geometry = df['geometry'].values
    total_districts = len(geometry)
    print(total_districts, "districts identified in shapefile; mapping to", voting_districts, "targets")

    words = shapefile_path.split('/')
    last_word = words[-1]
    print(last_word)
    prefix = last_word.replace('.shp', '')
    matrix_outpath_g = "./" + shapefile_folder + "/" + prefix + ".json"
    matrix_outpath_g_targets = "./" + shapefile_folder + "/" + prefix + "_targets.json"

    points = [centroid.coords for centroid in df.centroid]
    N = len(points)
    
    print(N,"points mapping to",voting_districts,"targets")
    targetlist = np.random.choice(N,voting_districts,replace=False)        
    print("targetlist:",targetlist)    

    matrix_g = partial_graph_distance_matrix_from_shapefile(shapefile_path,targetlist)   
    print(matrix_g.tolist())
    print("Size of cost matrix")
    print(matrix_g.shape)
    
    with open(matrix_outpath_g, "w") as f:
        json.dump(matrix_g.tolist(), f)
       
    #This saves the list of targets
    with open(matrix_outpath_g_targets, "w") as f:
        json.dump(targetlist.tolist(), f)

    return shapefile_folder, total_districts

def compute_districts(cost_matrix, population,number_of_districts,geoidtable,targetlist):
    total_population = np.sum(population, dtype=float)
    N = len(population)
        
    c = cost_matrix.astype(float)
    print("Cost matrix:")
    print(c)
    
    mu = population
    
    print("targetlist type:")
    print(type(targetlist))
    if isinstance(targetlist, np.ndarray):
        print("Targetlist was passed")
        print("total population",total_population)
        total_targets = targetlist.shape[0]
        print(total_targets)
        if total_targets > 0:
            population_per_target = int(total_population/total_targets)
        else:
            population_per_target = total_population
        total_districts = population.shape[0]
        nu = np.zeros(total_districts)
        #print("Print population shape:", population.shape[0])
        remaining_population = total_population
        for i in range(0,total_targets):
            if i < (total_targets - 1):
                nu[targetlist[i]] = population_per_target
                print("Target:",targetlist[i], "Assigned:",population_per_target)
                remaining_population = remaining_population - population_per_target
            else:
                nu[targetlist[i]] = remaining_population
                print("Target:",targetlist[i], "Assigned:",remaining_population)
                remaining_population = 0
    else:
        print("No target list defined")
        #    array_list = [0, 0, 30000, 0, 0, 30000, 0, 0, 30000]
        #    nu = np.array(array_list) # Convert Python list to Numpy array
        #    array_list = [1000000, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #    nu = np.array(array_list)
        nu = np.ones(number_of_districts) * total_population/number_of_districts
    
    print("mu:", mu)
    print("nu:", nu)
    
    # total_cost, solution = solve_Kantorovich(c, mu, nu)
    total_cost, ot_matrix, transport_details = solve_Kantorovich_with_details(c, mu, nu)
    #print("Transport Details:")
    #for source, targets in transport_details.items():
    #    print(f"Source {source}:")
    #    for target, flow in targets.items():
    #        print(f"  -> Target {target}: {flow}")

    #ot_matrix = solution.reshape(c.shape)
    
    print('')
    #print('Number of source points assigned to multiple targets')
    #print(count_splitting_old(ot_matrix))
    multiple_assignments, source_ids, target_ids = count_splitting(ot_matrix)
    print(multiple_assignments,'source districts assigned to multiple targets:', source_ids)
    if multiple_assignments > 0:
        print("Details of the assignments:", target_ids)
    print('')
    non_zero_districts = count_support(mu)
    print('Number of tracts/groups/blocks/districts with non-zero population:', non_zero_districts)
    print('')
        
    #print(ot_matrix)
        
    ####Transport plan to transport map
    #districts = plan_to_map(ot_matrix,geoidtable)
    districts = plan_to_map_array(N,ot_matrix,geoidtable)
        
    return districts, mu, nu, total_cost, target_ids, transport_details

def demo_optimal_transport(shape, num_districts=1, num_targets=4, remap_data=None, remap_type=None):
    state_geoid = 0
    if type(shape) == int:
        state_geoid = shape
        if state_geoid == 42: #Pennsylvania
            # Columns: [STATEFP10, COUNTYFP10, VTDST10, GEOID10, VTDI10, NAME10, NAMELSAD10, LSAD10, MTFCC10, FUNCSTAT10, ALAND10, AWATER10, INTPTLAT10, INTPTLON10, ATG12D, ATG12R, GOV10D, GOV10R, PRES12D, PRES12O, PRES12R, SEN10D, SEN10R, T16ATGD, T16ATGR, T16PRESD, T16PRESOTH, T16PRESR, T16SEND, T16SENR, USS12D, USS12R, GOV, TS, HISP_POP, TOT_POP, WHITE_POP, BLACK_POP, NATIVE_POP, ASIAN_POP, F2014GOVD, F2014GOVR, 2011_PLA_1, REMEDIAL_P, 538CPCT__1, 538DEM_PL, 538GOP_PL, 8THGRADE_1, gdk18_1, gdk9_1, gdk15_1, gdk2_1, gdk3_1, gdk4_1, gdk18_2, gdk18_dual, gdk18_du_1, gdk18_prim, gdk18_pr_1, gdk18_pr_2, gdk18_pr_3, gdk18_pr_4, gdk18_du_2, gdk18_pr_5, gdk3_prima, gdk3_dual_, gdk3_pri_1, gdk3_dua_1, gdk3_pri_2, m, geometry]
            #num_districts = 18
            num_districts = 8921
            state_name = 'PA'
            metric_path = "distances/PA_vtds_%s.json" %num_districts
            #metric_path = "./data/PA_vtds_%s.json" %num_districts # (missing)
            filepath = "./data/PA_VTD.shp"
            #fip_heading = "STATEFP10"
            fip_heading = 'COUNTYFP10'
            population_heading = 'TOT_POP'
        elif state_geoid == 48: # Texas
            state_name = 'TX'
            num_districts = 36 # 38 is current number, but was 36 at time data was obtained
            metric_path = "./data/TX_vtds_%s.json" %num_districts
            filepath = "./data/TX_vtds.shp"
            fip_heading = 'FIPS'
            population_heading = 'TOTPOP'
        elif state_geoid == 19: # Iowa
            # Columns: [STATEFP10, COUNTYFP10, GEOID10, NAME10, NAMELSAD10, ALAND10, AWATER10, INTPTLAT10, INTPTLON10, TOTPOP, NH_WHITE, NH_BLACK, NH_AMIN, NH_ASIAN, NH_NHPI, NH_OTHER, NH_2MORE, HISP, H_WHITE, H_BLACK, H_AMIN, H_ASIAN, N_NHPI, H_OTHER, H_2MORE, VAP, HVAP, WVAP, BVAP, AMINVAP, ASIANVAP, NHPIVAP, OTHERVAP, 2MOREVAP, TOTVOT00, PRES00D, PRES00R, PRES00G, PRES00OTH, TOTVOT04, PRES04D, PRES04R, PRES04OTH, TOTVOT08, PRES08D, PRES08R, PRES08OTH, TOTVOT12, PRES12D, PRES12R, PRES12OTH, TOTVOT16, PRES16D, PRES16R, PRES16OTH, CD, geometry]
            #num_districts = 4
            num_districts = 99 # (in full distance calculations)
            state_name = 'IA'
            #metric_path = "./data/IA_counties_%s.json" %num_districts
            metric_path = "distances/IA_vtds_%s.json" %num_districts
            filepath = "./data/IA_counties.shp"
            fip_heading = 'COUNTYFP10'
            population_heading = 'TOTPOP'
        elif state_geoid == 51: # Virginia
            state_name = 'VA'
            num_districts = 11
            metric_path = "./data/VA_%s.json" %num_districts
            filepath = "./data/VA.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 17: # Illinois
            state_name = 'IL'
            num_districts = 18
            metric_path = "./data/IL_%s.json" %num_districts
            filepath = "./data/IL.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 55: # Wisconsin
            state_name = 'WI'
            num_districts = 8
            metric_path = "./data/WI_%s.json" %num_districts
            filepath = "./data/WI.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 12: # Florida
            state_name = 'FL'
            num_districts = 27
            metric_path = "./data/FL_%s.json" %num_districts
            filepath = "./data/FL.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 13: # Georgia
            state_name = 'GA'
            num_districts = 14
            metric_path = "./data/GA_%s.json" %num_districts
            filepath = "./data/GA.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 16: # Idaho
            state_name = 'ID'
            num_districts = 2
            metric_path = "./data/ID_%s.json" %num_districts
            filepath = "./data/ID.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 18: # Indiana
            state_name = 'IN'
            num_districts = 9
            metric_path = "./data/IN_%s.json" %num_districts
            filepath = "./data/IN.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 20: # Kansas
            state_name = 'KS'
            num_districts = 4
            metric_path = "./data/KS_%s.json" %num_districts
            filepath = "./data/KS.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 21: # Kentucky
            state_name = 'KY'
            num_districts = 6
            metric_path = "./data/KY_%s.json" %num_districts
            filepath = "./data/KY.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 22: # Louisiana
            state_name = 'LA'
            num_districts = 6
            metric_path = "./data/LA_%s.json" %num_districts
            filepath = "./data/LA.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 23: # Maine
            state_name = 'ME'
            num_districts = 2
            metric_path = "./data/ME_%s.json" %num_districts
            filepath = "./data/ME.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 24: # Maryland
            state_name = 'MD'
            num_districts = 8
            metric_path = "./data/MD_%s.json" %num_districts
            filepath = "./data/MD.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 25: # Massachusetts
            state_name = 'MA'
            num_districts = 9
            metric_path = "./data/MA_%s.json" %num_districts
            filepath = "./data/MA.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 26: # Michigan
            state_name = 'MI'
            num_districts = 14
            metric_path = "./data/MI_%s.json" %num_districts
            filepath = "./data/MI_precincts.shp"
            fip_heading = 'CountyFips'
            population_heading = 'TOTPOP'
        elif state_geoid == 27: # Minnesota
            state_name = 'MN'
            num_districts = 8
            metric_path = "./data/MN_%s.json" %num_districts
            filepath = "./data/MN.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 28: # Mississippi
            state_name = 'MS'
            num_districts = 4
            metric_path = "./data/MS_%s.json" %num_districts
            filepath = "./data/MS.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 29: # Missouri
            state_name = 'MO'
            num_districts = 8
            metric_path = "./data/MO_%s.json" %num_districts
            filepath = "./data/MO.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 30: # Montana
            state_name = 'MT'
            num_districts = 1
            metric_path = "./data/MT_%s.json" %num_districts
            filepath = "./data/MT.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 31: # Nebraska
            state_name = 'NE'
            num_districts = 3
            metric_path = "./data/NE_%s.json" %num_districts
            filepath = "./data/NE.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 32: # Nevada
            state_name = 'NV'
            num_districts = 4
            metric_path = "./data/NV_%s.json" %num_districts
            filepath = "./data/NV.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 33: # New Hampshire
            state_name = 'NH'
            num_districts = 2
            metric_path = "./data/NH_%s.json" %num_districts
            filepath = "./data/NH.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 34: # New Jersey
            state_name = 'NJ'
            num_districts = 12
            metric_path = "./data/NJ_%s.json" %num_districts
            filepath = "./data/NJ.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 35: # New Mexico
            state_name = 'NM'
            num_districts = 3
            metric_path = "./data/NM_%s.json" %num_districts
            filepath = "./data/NM.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 36: # New York
            state_name = 'NY'
            num_districts = 27
            metric_path = "./data/NY_%s.json" %num_districts
            filepath = "./data/NY.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 37: # North Carolina
            state_name = 'NC'
            num_districts = 13
            metric_path = "./data/NC_%s.json" %num_districts
            filepath = "./data/NC.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 38: # North Dakota
            state_name = 'ND'
            num_districts = 1
            metric_path = "./data/ND_%s.json" %num_districts
            filepath = "./data/ND.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 39: # Ohio
            state_name = 'OH'
            num_districts = 16
            metric_path = "./data/OH_%s.json" %num_districts
            filepath = "./data/OH.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 40: # Oklahoma
            state_name = 'OK'
            num_districts = 5
            metric_path = "./data/OK_%s.json" %num_districts
            filepath = "./data/OK.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 41: # Oregon
            state_name = 'OR'
            num_districts = 5
            metric_path = "./data/OR_%s.json" %num_districts
            filepath = "./data/OR.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 43: # Rhode Island
            state_name = 'RI'
            num_districts = 2
            metric_path = "./data/RI_%s.json" %num_districts
            filepath = "./data/RI.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 44: # South Carolina
            state_name = 'SC'
            num_districts = 7
            metric_path = "./data/SC_%s.json" %num_districts
            filepath = "./data/SC.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 45: # South Dakota
            state_name = 'SD'
            num_districts = 1
            metric_path = "./data/SD_%s.json" %num_districts
            filepath = "./data/SD.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        elif state_geoid == 46: # Tennessee
            state_name = 'TN'
            num_districts = 9
            metric_path = "./data/TN_%s.json" %num_districts
            filepath = "./data/TN.shp"
            fip_heading = 'GEOID10'
            population_heading = 'TOTPOP'
        else:
            raise ValueError('Invalid state_geoid. Must be 42 (PA), 48 (TX), or 19 (IA)')
    else:
        # Custom shapefile was created
        state_name = shape
        metric_path = "./" + shape + "/" + shape + ".json"
        #metric_path = "./data/simple_rectangle_shapefile.json"
        #filepath = "./data/simple_rectangle_shapefile.shp"
        filepath = "./" + shape + "/" + shape + ".shp"
        fip_heading = 'GEOID'
        population_heading = 'population'
        
    # determine the number of columns in shapefile
    with open(metric_path) as f:
        temp_matrix = np.array(json.load(f))
        matrix_dimensions = temp_matrix.shape
        print("Read the following cost matrix dimensions:", matrix_dimensions)
        matrix_rows, matrix_columns = matrix_dimensions
        del temp_matrix # we got the info we needed, so we can delete the matrix now
        f.close()

    # Are we re-mapping from a previous run?
    if remap_data is not None:
        print(f"Type of remap_data: {type(remap_data)}")
        print(f"Value of remap_data: {remap_data}")
        targetlist = remap_data['targetlist']
        if remap_type == "remap_targets":
            new_targetlist = np.array([remap_data.get(item, item) for item in targetlist])
            targetlist = new_targetlist
    else:
        # does a target file exist?
        target_path = metric_path.replace('.json', '_targets.json')
        print(target_path)
        targetlist = False
        if os.path.isfile(target_path):
            # Read target file
            with open(target_path) as tf:
                targetlist = np.array(json.load(tf))
                print("Targets defined:", targetlist)
                tf.close()

            # Select 4 random elements without replacement
            selected_elements = np.random.choice(targetlist, size=num_targets, replace=False)

            # Assign it to a similar data structure (NumPy array)
            new_targetlist = np.array(selected_elements)

            print("Original Target List:", targetlist)
            print("Selected Target List:", new_targetlist)
            targetlist = new_targetlist
        else:
            print("No target file found (using defaults)")

    #for num_districts_k in range(1,num_districts+1):
    num_districts_k = num_districts
    print("Optimizing for %s districts" % num_districts_k)
    #### Loads distance matrix
    #num_districts_k = 4 # for testing
    with open(metric_path) as f:
        #only load the first num_districts_k rows
        #data_cost_matrix = np.array(json.load(f))
        if (state_geoid > 0):
            data_cost_matrix = np.array(json.load(f))[:matrix_rows,:num_districts_k] # for Texas 8941 x 36
        else:
            data_cost_matrix = np.array(json.load(f))[:matrix_rows,:matrix_rows] # Test state with 9 partitions
        #data_cost_matrix = np.array(json.load(f))[:matrix_rows,:matrix_columns]
        print("data_cost_matrix size:", data_cost_matrix.shape)
        print("data_cost_matrix:\n", data_cost_matrix)
        #data_cost_matrix = np.rot90(data_cost_matrix)
        print(data_cost_matrix)
        print(data_cost_matrix.shape)

    plan_id = 'gedk%s_1' % num_districts

    #### Sets up data frame from shapefile
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_colwidth', None)
    
    # Read selection of lines from data file
    #df = geopandas.read_file(filepath, rows=10)
    df = geopandas.read_file(filepath)

    # Distribute populations
    if (state_geoid > 0):
        print('Data headings:')
        print(df.head(0))
        
        #### Temporary population gatherer
        #tracts_graph = data_provider.tracts_graph(state_geoid)
        #data = tracts_graph.data #data is a dictionary     
        #population = np.array([feature["properties"]["2013_population_estimate"] for feature in data["features"]], dtype=float)   
            
        #block_id_table = df['GEOID'].values            
        
        #### Reads population and GEOID table from the dataframe
        block_id_table = df[fip_heading].values        
        population = df[population_heading].values.astype(float)
        #population = np.ones(len(block_id_table))
        Pop_district = np.sum(population) / num_districts
        g = np.ones(num_districts) * Pop_district
        delta = Pop_district * (0.01)
    else:
        Pop_district = 10000
        g = np.ones(num_districts) * Pop_district
        population = np.ones(num_districts_k) * Pop_district
        delta = Pop_district * (0.01)

    #### Call compute_districts to solve OT problem
    #num_districts_k = num_districts
    #if state_geoid == 100:
    #    districts = compute_districts(data_cost_matrix, population, 100, state_geoid, targetlist)
    #elif state_geoid == 105:
    #    districts = compute_districts(data_cost_matrix, population, 25, state_geoid, targetlist)
    #elif state_geoid > 0:
    districts, mu, nu, transport_cost, multiple_assignments, transport_details  = compute_districts(data_cost_matrix, population, num_districts_k, state_geoid, targetlist)
    #print("Highlight Target List:", targetlist)
    #else:
    #    districts = compute_districts(data_cost_matrix, population, 9, state_geoid, targetlist)

    #print("Data type of `districts` variable:", type(districts))
    #print('District assignments:', districts)
    if remap_type == 'remap_assignments':
        # Backup the original districts for reassignment logic
        original_districts = districts.copy()

        # Process reassignments
        for source_id, new_target_id in remap_data.items():
            try:
                # Convert source_id and new_target_id to integers
                source_int = int(source_id)
                new_target_int = int(new_target_id)

                if source_int in transport_details and len(transport_details[source_int]) > 1:
                    # This district was previously assigned to multiple targets, so we need to
                    # identify how the population was split between assignments
                    for previous_target_id, transported_mass in transport_details[source_int].items():
                        # Subtract the transported mass from the previous target population
                        nu[previous_target_id] -= transported_mass
                    nu[new_target_int] += mu[source_int] # now reassign entire source population to new target district

                    # Update the districts array to reflect the new target assignment
                    districts[source_int] = new_target_int
                    print(multiple_assignments)
                    print(type(multiple_assignments))
                    #multiple_assignments -= 1 # one fewer district that is assigned to multiple target districts
                    del multiple_assignments[source_int]
                else: 
                    previous_target_id = int(original_districts[source_int])
                    previous_target_population = nu[previous_target_id]
                    nu[previous_target_id] = previous_target_population - mu[source_int]
                    nu[new_target_int] += mu[source_int]               
                    districts[source_int] = new_target_int # Update the districts array with the new target assignment
                del transport_details[source_int]
                transport_details[source_int] = {new_target_int: mu[source_int]}

                # Update ot_matrix to reflect the reassignment
                # Zero out the row corresponding to the source district
                #ot_matrix[source_int, :] = 0

                # Assign the entire population to the new target column
                #ot_matrix[source_int, new_target_int] = mu[source_int]

            except (ValueError, IndexError) as e:
                # Handle invalid source/target IDs or out-of-bounds errors
                print(f"Skipping invalid reassignment: {source_id} → {new_target_id}. Error: {e}")

        #for source_id, target_id in remap_data.items():
        #    try:
        #        # Convert source_id and target_id to integers
        #        source_int = int(source_id)
        #        target_int = int(target_id)

        #        # Update districts array
        #        if 0 <= source_int < len(districts):
        #            districts[source_int] = target_int
        #    except (ValueError, TypeError):
        #        # Skip non-numeric or invalid entries
        #        print(f"Skipping invalid remap_data entry: {source_id} -> {target_id}")
        #    # we also need to adjust the population capacity at each voting center in nu

        # Output updated results
        print("Updated districts array:", districts)
        print("Updated target populations (nu):", nu)
        print('New district assignments:', districts)

        ## DO SOME VALIDATION HERE (LIKE CHECKING THAT POPULATION TOTALS ARE STILL GOOD)
        # Calculate the total population from mu
        total_mu_population = int(np.sum(mu))

        # Calculate the total population from nu
        total_nu_population = int(np.sum(nu))

        # Validation
        if total_mu_population == total_nu_population:
            print(f"Validation passed: Total population in mu ({total_mu_population}) matches nu ({total_nu_population}).")
        else:
            raise ValueError(
                f"Validation failed: Total population in mu ({total_mu_population}) "
                f"does not match nu ({total_nu_population}). Exiting program."
            )
        
    #districts = districts
    #### Adds the districts to the data frame
    df[plan_id] = pandas.Series(districts)

    # Recover the graph data structure from the connectivity matrix
    G = nx.Graph()
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(df, geometry=df[plan_id])

    df = df.to_crs("EPSG:3857")  # Example: Web Mercator

    # Calculate centroids
    df['centroid'] = df.geometry.centroid  # Ensure geometry column is correctly set

    # Add nodes with centroids as attributes
    for idx, row in df.iterrows():
        G.add_node(idx, centroid=(row.centroid.x, row.centroid.y))

    # Add edges directly based on the data_cost_matrix
    rows, cols = data_cost_matrix.shape
    for i in range(rows):
        for j in range(i + 1, cols):  # Avoid duplicate edges in an undirected graph
            if data_cost_matrix[i, j] in [0, 1]:  # Rule: 0 or 1 -> weight = 1
                G.add_edge(i, j, weight=1)
            elif data_cost_matrix[i, j] == -1:  # Rule: -1 -> weight = -1
                G.add_edge(i, j, weight=-1)

    # Print the edges to verify
    #print("Edges in the graph with weights:")
    #for u, v, weight in G.edges(data=True):
    #    print(f"({u}, {v}, weight: {weight['weight']})")
    
    flow_cost = 0
    #G = build_graph_with_debug(df, data_cost_matrix, "plan_id")
    if validateGraph(G) == True:
        optimal_flow, flow_cost = solve_beckmann(G, mu, nu)
    #print("nu", nu)
    #print("Optimal Flow:", optimal_flow)

    ## BEGIN: MANUALLY CALCULATE THE TOTAL TRANSPORT COST (NOT NECESSARILY OPTIMAL)
    # Initialize total travel cost
    total_travel_cost = 0

    # Variables for tracking min and max travel distances and costs
    min_travel_distance = float('inf')
    min_travel_cost = float('inf')
    min_travel_info = None  # To store source and target district info for min
    max_travel_distance = float('-inf')
    max_travel_cost = float('-inf')
    max_travel_info = None  # To store source and target district info for max

    # Step through each source district in the transport details
    for source_district_id, target_flows in transport_details.items():
        for target_district_id, transported_mass in target_flows.items():
            # Step 3: Obtain graph distance from the cost matrix
            graph_distance = data_cost_matrix[source_district_id, target_district_id]

            # Calculate the travel cost for this transport
            district_travel_cost = graph_distance * transported_mass

            # Update the total travel cost
            total_travel_cost += district_travel_cost

            # Update min and max travel distance and cost
            if graph_distance < min_travel_distance:
                min_travel_distance = graph_distance
                min_travel_cost = district_travel_cost
                min_travel_info = (source_district_id, target_district_id)

            if graph_distance > max_travel_distance:
                max_travel_distance = graph_distance
                max_travel_cost = district_travel_cost
                max_travel_info = (source_district_id, target_district_id)

    # Output the results
    #print(f"Total Travel Cost: {total_travel_cost}")
    #print(f"Minimum Travel Distance: {min_travel_distance} (Cost: {min_travel_cost})")
    #print(f"Occurs between Source District {min_travel_info[0]} and Target District {min_travel_info[1]}")
    #print(f"Maximum Travel Distance: {max_travel_distance} (Cost: {max_travel_cost})")
    #print(f"Occurs between Source District {max_travel_info[0]} and Target District {max_travel_info[1]}")

    # Initialize total travel cost (for manually re-assigned district map)
    total_travel_cost = 0

    # Variables for tracking min and max travel distances and costs
    min_travel_distance = float('inf')
    min_travel_cost = float('inf')
    min_travel_info = None  # To store source and target district info for min
    max_travel_distance = float('-inf')
    max_travel_cost = float('-inf')
    max_travel_info = None  # To store source and target district info for max

    # Step through each district in the state
    for source_district_id, source_population in enumerate(mu):
        # Check if the source district has multiple target mappings
        if source_district_id in transport_details and len(transport_details[source_district_id]) > 1:
            # print(source_district_id, 'is assigned multiple targets')
            # Use transport_details for multiple target mappings
            for target_district_id, transported_mass in transport_details[source_district_id].items():
                # Obtain graph distance from the cost matrix
                graph_distance = data_cost_matrix[source_district_id, target_district_id]

                # Calculate the travel cost for this transport
                district_travel_cost = graph_distance * transported_mass

                # Update the total travel cost
                total_travel_cost += district_travel_cost

                # Update min and max travel distance and cost
                if graph_distance < min_travel_distance:
                    min_travel_distance = graph_distance
                    min_travel_cost = district_travel_cost
                    min_travel_info = (source_district_id, target_district_id)

                if graph_distance > max_travel_distance:
                    max_travel_distance = graph_distance
                    max_travel_cost = district_travel_cost
                    max_travel_info = (source_district_id, target_district_id)
        else:
            # Use original logic for single target mappings
            target_district_id = int(districts[source_district_id])  # Ensure integer type

            # Obtain graph distance from the cost matrix
            graph_distance = data_cost_matrix[source_district_id, target_district_id]

            # Calculate the travel cost for this district
            district_travel_cost = graph_distance * source_population

            # Update the total travel cost
            total_travel_cost += district_travel_cost

            # Update min and max travel distance and cost
            if graph_distance < min_travel_distance:
                min_travel_distance = graph_distance
                min_travel_cost = district_travel_cost
                min_travel_info = (source_district_id, target_district_id)

            if graph_distance > max_travel_distance:
                max_travel_distance = graph_distance
                max_travel_cost = district_travel_cost
                max_travel_info = (source_district_id, target_district_id)

    # Output the results
    #print(f"Total Travel Cost: {total_travel_cost}")
    #print(f"Minimum Travel Distance: {min_travel_distance} (Cost: {min_travel_cost})")
    #print(f"Occurs between Source District {min_travel_info[0]} and Target District {min_travel_info[1]}")
    #print(f"Maximum Travel Distance: {max_travel_distance} (Cost: {max_travel_cost})")
    #print(f"Occurs between Source District {max_travel_info[0]} and Target District {max_travel_info[1]}")
    ## END: MANUALLY CALCULATE THE TOTAL TRANSPORT COST (NOT NECESSARILY OPTIMAL)

    filename = state_name + '_k%s_' % num_districts_k

    # Get the current timestamp in a format suitable for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Append the timestamp to the filename
    filename = filename + timestamp

    #print("Generating map... close window and press Crtl+D to continue")
    #n_grid = int((num_districts + 1000)/1000)
    n_grid = 2
    geopdvisual(state_name, df, plan_id, filename, targetlist, multiple_assignments, transport_details, transport_cost, flow_cost, n_grid) #'tab20' 'gist_rainbow' 'hsv'
    plt.show()
    print('')
    print("Transport map visualization is opening in external window (may take a few seconds)")
    print('')

    # Assuming `df` is your GeoDataFrame and `plan_id` contains district assignments
    df['centroid'] = df.geometry.centroid  # Calculate centroids for each polygon

    # Create a graph based on the connecting matrix
    G = nx.Graph()

    # Plot the graph with geometric structure
    #fig, ax = plt.subplots(figsize=(20, 12))

    # Plot the base GeoDataFrame
    #df.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.5)

    # Plot nodes using their centroids
    positions = {node: data['centroid'] for node, data in G.nodes(data=True)}
    #nx.draw(
    #    G, pos=positions, ax=ax, node_size=50, node_color="blue", edge_color="gray", with_labels=True
    #)

    #plt.title("Graph with Geometric Structure")
    #plt.show()

    #visualize_transport_process(G, mu, nu, optimal_flow)

    return targetlist, num_districts

userOption = input("Choice (C to create a custom state, L to load a custom state, or numerical ID of existing state to optimize; Iowa=19, Texas=48, PA=42): ")

#createCustom = input("Would you like to create a custom state to optimize (y or N): ")
#if createCustom == 'y' or createCustom == 'Y':
if userOption == 'C' or userOption == 'c':
    shapefile, districts = create_custom_shapefile()
    targetlist, num_districts = demo_optimal_transport(shapefile, districts)
elif userOption == 'L' or userOption == 'l':
    shapefile, districts = select_custom_shapefile()
    targetlist, num_districts = demo_optimal_transport(shapefile, districts)
else:
    state_geoid = int(userOption)
    user_num_targets = get_user_int('How many districts should contain voting centers (i.e. # of targets): ', 1, 50)
    #state_geoid = get_user_int("Enter the state GeoID # (IA=19, TX=48, PA=): ", min=1, max=60)
    targetlist, num_districts = demo_optimal_transport(state_geoid, 1, user_num_targets) # 1 is dummy val

    continueRunning = True
    while continueRunning:
        # Does user wish to remap any source or targets?
        prompt = "Type the letter of the desired menu option below (or q to quit to terminal):"
        menu_options = {
            'a': 'Remap target district',
            'b': 'Remap source to target',
            'q': 'Quit'
        }

        selected_option = get_menu_choice(prompt, menu_options)
        if selected_option == 'a':
            print("You opted to ", menu_options['a'])
            comma_delimited = ", ".join(targetlist.astype(str))
            remap_data = {}
            is_finished = False
            while is_finished == False:
                # Remap targets
                print("Which of the following target districts would you like to remap: ", comma_delimited)
                old_district_id = get_user_int('Type OLD target district ID and hit enter: ', targetlist)
                print("Which District ID would you like to use to replace District", old_district_id, " (a number between 0 and", num_districts, ")")
                new_district_id = get_user_int('Type NEW target district ID and hit enter: ', 0, num_districts)
                remap_data[old_district_id] = new_district_id
                if does_user_agree('Remap an additional target?') == False:
                    is_finished = True
            plt.close('all')
            print("Remap", remap_data)
            remap_data['targetlist'] = targetlist
            targetlist, num_districts = demo_optimal_transport(state_geoid, 99, remap_data=remap_data, remap_type='remap_targets')
        elif selected_option == 'b':
            print("You opted to ", menu_options['b'])
            comma_delimited = ", ".join(targetlist.astype(str))
            remap_data = {}
            is_finished = False
            while is_finished == False:
                # Remap targets
                print(f"Which source district would you like to reassign (0-{num_districts}):")
                old_district_id = get_user_int('Type OLD SOURCE district ID and hit enter: ', 0, num_districts)
                print("Which target District ID would you like to use to reassign it to:", comma_delimited)
                new_district_id = get_user_int('Type NEW TARGET district ID and hit enter: ', targetlist)
                remap_data[old_district_id] = new_district_id
                if does_user_agree('Remap an additional source?') == False:
                    is_finished = True
            plt.close('all')
            print("Remap", remap_data)
            remap_data['targetlist'] = targetlist
            targetlist, num_districts = demo_optimal_transport(state_geoid, 99, remap_data=remap_data, remap_type='remap_assignments')
        elif selected_option == 'q':
            continueRunning = False
            exit
        else:
            print("Unexpected option! This should not happen.")