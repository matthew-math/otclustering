import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist
from collections import defaultdict
import ot  # POT library
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib import rcParams  # Correctly importing rcParams

# Set global font to sans-serif
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']  # List of fallback sans-serif fonts
rcParams['text.usetex'] = True  # Use LaTeX for rendering text
rcParams['text.latex.preamble'] = r"\usepackage{sfmath}"  # Use sans-serif math fonts for LaTeX

# GLOBAL VARIABLE: Specify the optimizer to use
OPTIMIZER = "POT"  # Options: "POT", "CPLEX", "Gurobi"

# Dynamically generate a list of colors for the target nodes
def generate_colors(m):
    cmap = cm.get_cmap("tab10", m)  # Use a colormap with `m` distinct colors
    return [cmap(i) for i in range(m)]  # Extract `m` colors

def count_islands(islands):
    """
    Count the number of contiguous blocks of islands.

    Parameters:
    - islands: list of tuples, where each tuple is a (start, end) range of an island.

    Returns:
    - int: the total number of distinct islands (contiguous blocks).
    """
    if not islands:
        return 0

    # Sort the island ranges by their start points
    islands = sorted(islands)

    # Merge contiguous ranges
    merged_islands = []
    current_start, current_end = islands[0]

    for start, end in islands[1:]:
        if start <= current_end + 1:  # Overlaps or touches
            current_end = max(current_end, end)  # Extend the current range
        else:
            merged_islands.append((current_start, current_end))
            current_start, current_end = start, end

    merged_islands.append((current_start, current_end))  # Add the last range

    return len(merged_islands)

def draw_island_boxes(ax, islands, source_positions, node_size, target_color, padding=0.01):
    """
    Draw thin boxes around groups of nodes that belong to the same island and label them with an 'I'.

    Parameters:
    - ax: Matplotlib axis object.
    - islands: list of tuples, where each tuple is (start, end) range of an island.
    - source_positions: array of source node x-coordinates.
    - node_size: float, size of each node box in data units.
    - target_color: str, color to use for the 'I' label and visual indication.
    - padding: float, additional padding to add around the box (in data units).
    """
    for start, end in islands:
        # Determine the x-coordinates of the first and last nodes in the island
        start_x = source_positions[start - 1] - node_size / 2 - padding
        end_x = source_positions[end - 1] + node_size / 2 + padding
        center_x = (source_positions[start - 1] + source_positions[end - 1]) / 2  # Center of the island

        # Draw a thin rectangle around the nodes
        width = end_x - start_x
        height = node_size * (1 + padding)  # Make the box slightly taller than the node size
        ax.add_patch(Rectangle(
            (start_x, -height / 2),  # Bottom-left corner
            width, height,  # Width and height
            edgecolor="black", facecolor="none", lw=1, linestyle="--", zorder=3
        ))

        # Add a centered 'I' label above the island
        ax.text(
            center_x, height / 2 + 0.1,  # Slightly above the top of the box
            "I",
            color=target_color, ha="center", va="bottom", fontsize=12, fontweight="bold", zorder=4
        )
        
def assign_targets(start, end, m, mode="EQUAL", cluster_count=3, spacing=1):
    """
    Dynamically calculate target positions based on the assignment mode.

    Parameters:
    - start: int, start of the range
    - end: int, end of the range
    - m: int, total number of target nodes
    - mode: str, one of ["EQUAL", "RANDOM", "CLUSTER"]
    - cluster_count: int, number of clusters to create (only used for "CLUSTER" mode)
    - spacing: int, minimum spacing between targets in a cluster (only for "CLUSTER" mode)

    Returns:
    - targets: list of target node positions
    """
    
    if mode == "EQUAL":
        # Evenly spaced targets
        if m == 1:
            # Special case: only one target, place it at the midpoint
            targets = [start + (end - start) // 2]
        else:
            targets = [start + i * (end - start) // (m - 1) for i in range(m)]
    elif mode == "TERMINAL":
        targets = [1, n]
    elif mode == "RANDOM":
        # Randomly assign targets within [start, end]
        targets = sorted(np.random.choice(range(start, end + 1), size=m, replace=False))
    elif mode == "CLUSTER":
        # Assign targets into clusters
        if cluster_count > m:
            cluster_count = m
            #raise ValueError("Number of clusters cannot exceed the number of target nodes.")
        
        # Divide the range [start, end] into cluster_count subregions
        cluster_size = m // cluster_count
        remainder = m % cluster_count  # Handle extra targets that don't fit evenly
        cluster_ranges = np.linspace(start, end, cluster_count + 1, dtype=int)  # Cluster boundaries
        
        targets = []
        for i in range(cluster_count):
            # Define the cluster range
            cluster_start = cluster_ranges[i]
            cluster_end = cluster_ranges[i + 1] - 1
            cluster_targets = np.linspace(cluster_start, cluster_end, cluster_size + (1 if i < remainder else 0), dtype=int)
            
            # Apply spacing if required
            cluster_targets = list(cluster_targets)
            for j in range(1, len(cluster_targets)):
                if cluster_targets[j] - cluster_targets[j - 1] < spacing:
                    cluster_targets[j] = cluster_targets[j - 1] + spacing
            
            # Add cluster targets to the final list
            targets.extend(cluster_targets)

    else:
        raise ValueError(f"Invalid mode: {mode}. Supported modes are 'EQUAL', 'RANDOM', 'CLUSTER'.")
    
    # Ensure targets are unique
    targets = sorted(set(targets))
    return targets

def characterize_optimal_transport(assignments, transport_plan, n, target_nodes):
    """
    Characterize the optimal transport plan into primary assignments, islands, and sprinkles.

    Parameters:
    - assignments: 1D numpy array of target indices for each source node
    - transport_plan: 2D numpy array, the optimal transport plan
    - n: int, number of source nodes
    - target_nodes: List of actual target node IDs

    Returns:
    - result: Dictionary containing the characterizations
    """
    result = {
        "num_source_nodes": n,
        "num_target_nodes": len(target_nodes),
        "transport_plan": transport_plan,
        "assignments": assignments,
        "targets": defaultdict(lambda: {"primary": None, "islands": [], "sprinkles": []})
    }

    # Map target indices to target node IDs
    unique_targets = np.unique(assignments)
    for target_index in unique_targets:
        target_id = target_nodes[target_index]  # Map target index to node ID

        # Find all contiguous blocks of source nodes assigned to the current target
        blocks = []
        current_start = None
        for i in range(n):
            if assignments[i] == target_index:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    blocks.append((current_start, i - 1))
                    current_start = None
        # Add the final block if it ended at the last node
        if current_start is not None:
            blocks.append((current_start, n - 1))

        # Identify the largest block as the primary assignment
        primary_block = max(blocks, key=lambda x: x[1] - x[0] + 1)
        result["targets"][target_id]["primary"] = (primary_block[0] + 1, primary_block[1] + 1)  # Adjust to 1-based indexing

        # Identify islands and sprinkles
        for block in blocks:
            block_start, block_end = block

            # Check if this block is part of the primary block
            if block != primary_block:
                if block_start == block_end:  # Single-node block
                    i = block_start  # Index of the single node
                    left_neighbor_1 = assignments[i - 1] if i > 0 else None
                    left_neighbor_2 = assignments[i - 2] if i > 1 else None
                    right_neighbor_1 = assignments[i + 1] if i < n - 1 else None
                    right_neighbor_2 = assignments[i + 2] if i < n - 2 else None

                    # Check sprinkle conditions
                    if (i == 0 and right_neighbor_1 != target_index and right_neighbor_2 != target_index) or \
                    (i == n - 1 and left_neighbor_1 != target_index and left_neighbor_2 != target_index) or \
                    (i > 0 and i < n - 1 and
                        left_neighbor_1 != target_index and left_neighbor_2 != target_index and
                        right_neighbor_1 != target_index and right_neighbor_2 != target_index):
                        result["targets"][target_id]["sprinkles"].append(i + 1)  # Adjust to 1-based indexing
                    else:
                        # If it doesn't meet sprinkle criteria, it's an island
                        result["targets"][target_id]["islands"].append((block_start + 1, block_end + 1))  # 1-based indexing
                else:
                    # Multi-node blocks are always islands
                    result["targets"][target_id]["islands"].append((block_start + 1, block_end + 1))  # 1-based indexing

        # Identify sprinkles (single nodes surrounded by different targets or at edges)
        for i in range(n):
            if assignments[i] == target_index:
                # Check if it's a true sprinkle: isolated and surrounded by other targets
                is_isolated = all(
                    not (block[0] <= i <= block[1]) for block in blocks
                )
                if is_isolated:
                    # Handle edge cases for first and last nodes
                    if i == 0:  # First node
                        right_neighbor = assignments[i + 1] if i + 1 < n else None
                        if right_neighbor != target_index:
                            result["targets"][target_id]["sprinkles"].append(i + 1)  # Adjust to 1-based indexing
                    elif i == n - 1:  # Last node
                        left_neighbor = assignments[i - 1] if i - 1 >= 0 else None
                        if left_neighbor != target_index:
                            result["targets"][target_id]["sprinkles"].append(i + 1)  # Adjust to 1-based indexing
                    else:  # Middle nodes
                        left_neighbor = assignments[i - 1] if i - 1 >= 0 else None
                        right_neighbor = assignments[i + 1] if i + 1 < n else None
                        if left_neighbor != target_index and right_neighbor != target_index:
                            result["targets"][target_id]["sprinkles"].append(i + 1)  # Adjust to 1-based indexing

    return result

def solve_optimal_transport(cost_matrix, source_population, target_population, normalize=False):
    """
    Solve the optimal transport problem using the specified optimizer.

    Parameters:
    - cost_matrix: 2D numpy array, cost of transporting from sources to targets
    - source_population: 1D numpy array, population distribution of sources
    - target_population: 1D numpy array, population distribution of targets
    - normalize: bool, whether to normalize populations to have equal total mass

    Returns:
    - transport_plan: 2D numpy array, the optimal transport plan
    - total_cost: float, the total transport cost
    """
    # Normalize populations to have the same total mass if required
    if normalize:
        source_sum = np.sum(source_population)
        target_sum = np.sum(target_population)
        if source_sum != target_sum:
            scale = min(source_sum, target_sum)
            source_population = source_population / source_sum * scale
            target_population = target_population / target_sum * scale

    if OPTIMIZER == "POT":
        # Use the POT library for solving optimal transport
        transport_plan = ot.emd(source_population, target_population, cost_matrix)

    elif OPTIMIZER == "CPLEX":
        raise NotImplementedError("CPLEX support is not implemented in this code.")
    elif OPTIMIZER == "Gurobi":
        raise NotImplementedError("Gurobi support is not implemented in this code.")
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}. Supported optimizers are: 'POT'.")

    # Calculate total transport cost
    total_cost = np.sum(transport_plan * cost_matrix)

    return transport_plan, total_cost

def save_transport_summary(characterizations, output_file="transport_summary.txt", grouped=True):
    """
    Save a grouped summary of transport characterizations to a text file.

    Parameters:
    - characterizations: list of tuples (characterization, group_name).
    - output_file: str, the file path for the summary file.
    """
    if grouped:
        # Group characterizations by their group name
        grouped = defaultdict(list)
        for char, group in characterizations:
            grouped[group].append(char)

        # Write to the output file
        with open(output_file, "w") as f:
            for group, chars in grouped.items():
                f.write(f"### {group} ###\n")
                for char in chars:
                    f.write(char + "\n")
                f.write("\n")
    else:
        # Write ungrouped characterizations to the output file
        with open(output_file, "w") as f:
            for char, group in characterizations:
                f.write(char + "\n")

    print(f"Transport summary saved to {output_file}")

def generate_1D_graph(n, d, m, target_nodes, colors, source_population=None, target_population=None, node_size=2, normalize=False, dist_type="EQUAL", output_dir="1doutput"):
    """
    Generate a 1D graph and perform optimal transport assignment.

    Parameters:
    - n: int, total number of nodes
    - d: float, distance between consecutive nodes
    - m: int, total number of target nodes
    - target_nodes: list of length m, IDs of target nodes
    - colors: list of length m, colors corresponding to target nodes
    - source_population: 1D numpy array, population distribution of sources (optional)
    - target_population: 1D numpy array, population distribution of targets (optional)
    - node_size: float, size of each node box (side length in pixels)
    - normalize: bool, whether to normalize populations to have equal total mass
    - output_dir: str, root directory to save the plots.
    """
    # Ensure proper separation between nodes
    if d < 0.1:
        raise ValueError("Node spacing (d) must be a reasonable value greater than 0.")

    # Generate the 1D graph
    source_positions = np.array(range(1, n + 1)) * d  # Positions of source nodes
    target_positions = np.array(target_nodes) * d     # Positions of target nodes

    # Initialize source populations if not provided
    if source_population is None:
        source_population = np.ones(n)  # Equal population for each source node
    
    # Initialize target populations if not provided
    if target_population is None:
        total_source_mass = np.sum(source_population)
        target_population = np.full(m, total_source_mass / m)  # Equal distribution among targets

    # Explicitly check that the source and target masses are equal
    source_mass = np.sum(source_population)
    target_mass = np.sum(target_population)
    if not np.isclose(source_mass, target_mass):
        raise ValueError(
            f"Source mass ({source_mass}) and target mass ({target_mass}) must be equal for optimal transport."
        )

    # Compute pairwise distances between source and target nodes
    cost_matrix = cdist(source_positions.reshape(-1, 1), target_positions.reshape(-1, 1))

    # Solve the optimal transport problem using the selected optimizer
    transport_plan, total_cost = solve_optimal_transport(cost_matrix, source_population, target_population, normalize)

    # Generate assignment information
    assignments = np.argmax(transport_plan, axis=1)
    
    # Characterize the transport plan
    result = characterize_optimal_transport(assignments, transport_plan, n, target_nodes)

    # Print the results
    for target_id, data in result["targets"].items():
        data["num_islands"] = count_islands(data["islands"])
        print(f"Target {target_id}:")
        print(f"  Primary Block: {data['primary']}")
        print(f"  Islands: {data['islands']}")
        print(f"  Sprinkles: {data['sprinkles']}")
    
    # Compute total mass assigned to each target node
    mass_assigned = np.sum(transport_plan, axis=0)
    
    # Check for islands or sprinkles across all targets
    has_islands_or_sprinkles = any(
        data["islands"] or data["sprinkles"] for data in result["targets"].values()
    )
    
    # Count total islands and sprinkles
    #total_islands = sum(len(data["islands"]) for data in result["targets"].values())
    total_islands = sum(data["num_islands"] for data in result["targets"].values())
    total_sprinkles = sum(len(data["sprinkles"]) for data in result["targets"].values())

    # Determine the directory name based on characteristics
    if total_islands > 0 and total_sprinkles > 0:
        dir_name = f"{min(4, total_islands)}_islands_{min(4, total_sprinkles)}_sprinkles"
    elif total_islands > 0:
        dir_name = f"{min(4, total_islands)}_islands"
    elif total_sprinkles > 0:
        dir_name = f"{min(4, total_sprinkles)}_sprinkles"
    else:
        dir_name = "no_islands_no_sprinkles"

    # Create the directory if it doesn't exist
    #save_dir = os.path.join(output_dir, dir_name)
    save_dir = os.path.join(output_dir, dist_type, dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # Visualize only if there are islands or sprinkles
    #if has_islands_or_sprinkles:
    if 1 == 1:
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 2))

        # Determine the scaling factor for converting node size to pixel units
        pixel_to_data = ax.transData.transform([(0, 0), (1, 1)])  # Map 1 data unit to pixel units
        pixel_size = pixel_to_data[1, 1] - pixel_to_data[0, 1]  # Pixel-to-data scaling factor
        data_node_size = node_size / pixel_size  # Convert pixel size to data space

        # Add legend for target nodes
        #for idx, target in enumerate(target_nodes):
        #    plt.scatter([], [], color=colors[idx], label=f"Target {target}")

        # Add legend for target nodes
        print("Target nodes in legend: ", target_nodes)
        for idx, target in enumerate(target_nodes):
            if idx < len(colors):
                plt.scatter([], [], color=colors[idx], label=f"Target {target}")
                #print(f"Legend Entry: Target {target}, Color: {colors[idx]}")
            else:
                print(f"Warning: Missing color for Target {target} (index {idx}).")

        #if n==4 and m==4:
        #    sys.exit()

        if n > 20:
            ith = int(n * 0.1)
        else:
            ith = 5
        #print("n: ", n)
        #print("ith: ", ith)
        #sys.exit()
        for i in range(n):
            node_center_x = source_positions[i]
            node_center_y = 0
            
            # Check if this node is part of an island or sprinkle
            for target_id, data in result["targets"].items():
                islands = data["islands"]  # List of (start, end) tuples for islands
                if islands:
                    # Get the color for the current target
                    target_index = target_nodes.index(target_id)
                    target_color = colors[target_index]
                    draw_island_boxes(ax, islands, source_positions, data_node_size, target_color, padding=0.1)                # Add "I" for islands
                #for island_start, island_end in data["islands"]:
                #    if island_start <= i + 1 <= island_end:  # Check if current node is part of an island
                #        plt.text(
                #            node_center_x, data_node_size / 2 + 0.3, "I",  # Slightly above the box
                #            ha='center', va='bottom', fontsize=10, color='black', fontweight='bold'
                #        )
                # Add "S" for sprinkles
                if i + 1 in data["sprinkles"]:  # Check if current node is a sprinkle
                    plt.text(
                        node_center_x, data_node_size / 2 + 0.3, "S",  # Slightly above the box (higher than islands)
                        ha='center', va='bottom', fontsize=10, color='black', fontweight='bold'
                    )
        
            # Determine the mass distribution for the current node
            mass_distribution = transport_plan[i]  # Mass assigned to each target for this source node
            total_mass = np.sum(mass_distribution)
            proportions = mass_distribution / total_mass if total_mass > 0 else np.zeros_like(mass_distribution)

            # Draw the node with split colors if assigned to multiple targets
            if np.count_nonzero(proportions) > 1:
                # Node is assigned to multiple targets
                left_edge = node_center_x - data_node_size / 2
                for target_index, proportion in enumerate(proportions):
                    if proportion > 0:  # Only draw if mass is assigned
                        width = proportion * data_node_size
                        ax.add_patch(Rectangle(
                            (left_edge, node_center_y - data_node_size / 2),  # Bottom-left corner
                            width, data_node_size,  # Width and height
                            color=colors[target_index], zorder=2
                        ))
                        left_edge += width  # Move to the right for the next segment
            else:
                # Node is assigned to a single target
                target_index = np.argmax(proportions)
                color = colors[target_index]
                ax.add_patch(Rectangle(
                    (node_center_x - data_node_size / 2, node_center_y - data_node_size / 2),  # Bottom-left corner
                    data_node_size, data_node_size,  # Width and height
                    color=color, zorder=2
                ))
            
            # Add vertical text numbering for the first, last, every 10th node, and all target nodes
            if i == 0 or i == n - 1 or (i + 1) % ith == 0 or (i + 1) in target_nodes:
                if (i+1) in target_nodes:
                    # Determine the index of the target node and get its corresponding color
                    target_index = target_nodes.index(i + 1)  # Find the index of the target node
                    target_color = colors[target_index]  # Get the color for this target node
                    # Underline the target node ID
                    plt.text(
                        node_center_x, -data_node_size / 2 - 0.08,  # Slightly below the node
                        f"$\\sf\\underline{{{i+1}}}$",  # Use LaTeX underline
                        color=target_color, ha="center", va="top", rotation=270, fontsize=10, usetex=True  # Match the target color
                    )
                else:       
                    plt.text(
                        node_center_x, -data_node_size / 2 - 0.08, str(i + 1),  # Slightly below the box
                        ha='center', va='top', fontsize=10, rotation=270  # Centered, vertical text (top-to-bottom)
                    )

            # Add mass values above target nodes
            for idx, target in enumerate(target_nodes):
                target_pos = target_positions[idx]  # Position of the target node
                plt.text(
                    target_pos, data_node_size / 2 + 0.05, f"{mass_assigned[idx]:.2f}",  # Slightly above the box
                    ha='center', va='bottom', fontsize=10, color=colors[idx]
                )

        # Customize graph visuals
        # plt.xlabel("1D Graph Nodes")
        plt.yticks([])  # Remove y-axis ticks
        plt.xticks([])  # Remove x-axis ticks
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 0.15), ncol=m)  # Legend above the plot
        dist_type_title = dist_type.title()
        plt.title(f"Optimal Transport Assignments with {dist_type_title} Target Spacing (Total Cost: {total_cost:.2f})")
        plt.box(False)  # Remove the surrounding box
        plt.axis('equal')  # Ensure square boxes maintain aspect ratio
        #plt.show()
        
        # Join the target node IDs as a string, separated by underscores
        target_ids_str = "_".join(map(str, target_nodes))
        
        # Generate the filename, including the target node IDs
        file_name = f"{len(target_nodes)}_targets_{n}_sources_targets_{target_ids_str}.png"
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot to {save_path}")
        
        # Build the text-based characterization
        characterization = f"{save_path}: "
        for i in range(n):
            target_index = assignments[i]
            target_id = target_nodes[target_index]
            label = ""

            # Check for islands and sprinkles
            for data in result["targets"].values():
                if any(start <= i + 1 <= end for start, end in data["islands"]):
                    label = "(I)"
                if i + 1 in data["sprinkles"]:
                    label = "(S)"

            # Include mass for target nodes
            if i + 1 in target_nodes:
                mass = mass_assigned[target_nodes.index(i + 1)]
                characterization += f"{i + 1}->{target_id}(m:{mass:.2f}) "
            else:
                characterization += f"{i + 1}->{target_id}{label} "

        # Return the characterization string
        return characterization, result

# Example usage
n = 50  # Total number of nodes

distribution_types = ["EQUAL", "RANDOM", "CLUSTER"]

# Initialize a list to store characterizations
characterizations = []

m = 2
group = "Terminal Assignments (No Islands or Sprinkles)"
for n in range(5, 505, 5):
    print(f"Total nodes: {n} Target nodes: {m}")
    dist_type = "TERMINAL"
    target_nodes = assign_targets(1, n, m, dist_type)
    number_of_targets = len(target_nodes) # Total number of target nodes
    print(f"Target nodes: {target_nodes}")
    d = 1   # Distance between consecutive nodes (must be >= node_size)
    colors = generate_colors(number_of_targets)  # Colors corresponding to target nodes
    # Source population: Each node has a population of 1
    source_population = np.ones(n)
    
    # Generate the 1D graph and retrieve the characterization and result
    char, result = generate_1D_graph(
        n, d, number_of_targets, target_nodes, colors,
        source_population=source_population, node_size=100, normalize=False, dist_type=dist_type
    )

    # Determine the group for this characterization
    total_islands = sum(len(data["islands"]) for data in result["targets"].values())
    total_sprinkles = sum(len(data["sprinkles"]) for data in result["targets"].values())
    
    # Store the characterization and group   
    characterizations.append((char, group))
    
# Save the summary text file
#save_transport_summary(characterizations, output_file="transport_summary.txt")
save_transport_summary(characterizations, output_file="transport_terminal_summary.txt", grouped=False)

# Initialize a list to store characterizations
characterizations = []

m = 2
group = "Sliding Assignments"
#for n in range(5, 105, 5):
for n in range(5, 25, 5):
    for o in range(1, n):
        print(f"Total nodes: {n} Target nodes: {m}")
        dist_type = "SLIDING"
        #target_nodes = assign_targets(1, n, o, dist_type)
        target_nodes = [o, o+1]
        
        number_of_targets = len(target_nodes) # Total number of target nodes
        print(f"Target nodes: {target_nodes}")
        d = 1   # Distance between consecutive nodes (must be >= node_size)
        colors = generate_colors(number_of_targets)  # Colors corresponding to target nodes
        # Source population: Each node has a population of 1
        source_population = np.ones(n)
        
        # Generate the 1D graph and retrieve the characterization and result
        char, result = generate_1D_graph(
            n, d, number_of_targets, target_nodes, colors,
            source_population=source_population, node_size=100, normalize=False, dist_type=dist_type
        )

        # Determine the group for this characterization
        total_islands = sum(len(data["islands"]) for data in result["targets"].values())
        total_sprinkles = sum(len(data["sprinkles"]) for data in result["targets"].values())
        
        # Store the characterization and group   
        characterizations.append((char, group))
    
# Save the summary text file
#save_transport_summary(characterizations, output_file="transport_summary.txt")
save_transport_summary(characterizations, output_file="transport_sliding_summary.txt", grouped=False)
sys.exit()

for n in range(4, 52, 1):
    for m in range(2, 6, 1):
        if m > n:
            continue # Skip cases where the number of targets exceeds the number of sources
        print(f"Total target nodes: {m}")
        for dist_type in distribution_types:
            print(f"Total source nodes: {n}")
            print(f"Target distribution: {dist_type}")
            # Define the range for n
            start = 1
            end = n

            # Assign target_nodes at equidistant spacing:
            target_nodes = assign_targets(1, n, m, dist_type)
            number_of_targets = len(target_nodes) # Total number of target nodes
            print(f"Target nodes: {target_nodes}")
            d = 3   # Distance between consecutive nodes (must be >= node_size)
            colors = generate_colors(number_of_targets)  # Colors corresponding to target nodes

            # Source population: Each node has a population of 1
            source_population = np.ones(n)
            # target_population = np.array([2, 4, 6]) # for custom target populations

            # Generate the 1D graph and retrieve the characterization and result
            char, result = generate_1D_graph(
                n, d, number_of_targets, target_nodes, colors,
                source_population=source_population, node_size=100, normalize=False, dist_type=dist_type
            )

            # Determine the group for this characterization
            total_islands = sum(len(data["islands"]) for data in result["targets"].values())
            total_sprinkles = sum(len(data["sprinkles"]) for data in result["targets"].values())
            if total_islands == 0 and total_sprinkles == 0:
                group = "No Islands or Sprinkles"
            elif total_islands > 0 and total_sprinkles > 0:
                group = f"{min(4, total_islands)} Islands and {min(4, total_sprinkles)} Sprinkles"
            elif total_islands > 0:
                group = f"{min(4, total_islands)} Islands"
            elif total_sprinkles > 0:
                group = f"{min(4, total_sprinkles)} Sprinkles"

            # Store the characterization and group
            characterizations.append((char, group))
            
# Save the summary text file
save_transport_summary(characterizations, output_file="transport_summary.txt")