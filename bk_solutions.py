import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import scipy.linalg as la
import scipy.stats as stats
import networkx as nx
import json
from OT_solver import solve_Kantorovich
from colorsys import hsv_to_rgb
import random
import pickle
from datetime import datetime
import cvxpy as cp
import scipy.sparse as sps

random.seed(datetime.now().timestamp())
massAssignment = 'random' # 'uniform' or 'random'
#graphSignature = 'T2W8DCNY' # Use signature from past run or leave empty to generate a new graph
graphSignature = 'SH55LTVB'
if graphSignature == '':
    newGraphSignature = ''.join((random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for i in range(8)))
    print('Graph Signature (set graphSignature equal to this string to recreate this graph):', newGraphSignature)
else:
    newGraphSignature = ''

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
    print("Calculated Divergence matrix D:")
    print(D)

    constraints = [D@S == b-a]
    prob = cp.Problem(cp.Minimize(cp.norm1(S)), constraints)

    prob.solve()

    sol = S.value
    return sol
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
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

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

# This function takes as input n,k,sigma1,sigma2, and r
# It outputs a list x of length n*k (comprised of vectors in the plane)
# and a n*k by n*k matrix M
#
# The list is x made out of nk points in the plane generated from a Gaussian Mixture. 
# Concretely, we first sample k points distributed according to a 2D gaussian with mean 0 and variance sigma1^2
# and then for each of these k points one samples n points from a gaussian with mean at the point
# and variance sigma2
#
# The matrix M entries are all either 0 or 1,
# M[i,j] = 1 if distance between x[i] and x[j] is less than r
# M[i,j] = 0 otherwise
# 
def GenerateRandomGraph(n = 100, k = 4, sigma1 = 10, sigma2 = 1, r = 0.25, signature='', new_signature=''):
    if signature == '':
        centers = stats.multivariate_normal.rvs(mean = [0,0], cov = sigma1*np.eye(2), size = k )
        x = {}
        for i in range(0,k):
            for j in range(0,n):
                x[i*n+j] = stats.multivariate_normal.rvs(mean = centers[i], cov = sigma2*np.eye(2))
        M = np.zeros((n*k,n*k))
        for i in range(n*k):
            for j in range(n*k):
                if i != j and la.norm(x[i]-x[j]) < r:
                    M[i,j] = 1
                    # It also might be interesting to have a weight
                    # that is never larger than 1 but can be any number
                    # between 0 and 1 according to the distance between 
                    # x[i] and x[j]
                    #M[i,j] = la.norm(x[i]-x[j]) 
        # Save the graph in case we need to reproduce the results
        with open(new_signature+'.graph.pkl', 'wb') as f:
            pickle.dump((x, M), f)
        return x, M
    else:
        with open(signature+'.graph.pkl', 'rb') as f:
            x, M = pickle.load(f)
        return x, M

# This function takes as input a networkx.graph object G
# and produces a n by n matrix (2d array)
# where matrix[i,j] is the graph distance in G from node i to node j
def graph_distance(graph):
    N = len(graph.nodes)
    
    matrix = np.empty((N,N))   
    print('Computing partial distance matrix with networkx')
    for i in range(N):
        for j in range(N):
            matrix[i,j] = nx.shortest_path_length(graph,source = i, target = j)
        # The following helps track how much is left to complete computation
        if (N-i)%10==0:
            #print(N-i), ' ... ',
            print(str(N-i)+' ... ', end='', flush=True)
    print('done.')
            
    return matrix

def find_n_farthest_nodes(G, n):
    # Compute the shortest path lengths between all pairs of nodes
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    # Find the initial farthest pair of nodes
    max_length = 0
    farthest_pair = (None, None)

    for source, lengths in shortest_path_lengths.items():
        for target, length in lengths.items():
            if source < target and length > max_length:
                max_length = length
                farthest_pair = (source, target)

    # Initialize the list of farthest nodes with the farthest pair
    farthest_nodes = list(farthest_pair)

    # Select the remaining nodes
    while len(farthest_nodes) < n:
        max_min_dist = 0
        next_node = None

        for candidate in G.nodes():
            if candidate not in farthest_nodes:
                # Find the minimum distance from this candidate to the already selected nodes
                min_dist_to_selected = min(shortest_path_lengths[candidate][node] for node in farthest_nodes)
                if min_dist_to_selected > max_min_dist:
                    max_min_dist = min_dist_to_selected
                    next_node = candidate

        if next_node is not None:
            farthest_nodes.append(next_node)

    return farthest_nodes

# Function to generate a list of maximally distinct colors
def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors  # Evenly spaced hues
        saturation = 0.8  # Fixed saturation
        value = 0.9  # Fixed value
        color = hsv_to_rgb(hue, saturation, value)
        colors.append(color)
    return colors

# Function to compute Euclidean distance between two nodes
def euclidean_distance(node1, node2):
    x1, y1 = G.nodes[node1]['pos']
    x2, y2 = G.nodes[node2]['pos']
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def generate_transport_plan(optimal_flow, G):
    """
    Generates a transport plan matrix and computes the total transport cost.

    Parameters:
    - optimal_flow: array of optimal flow values for each edge (from solver)
    - G: networkx graph with edges and corresponding weights
    
    Returns:
    - transport_plan: NxN matrix representing mass transported between nodes
    - total_transport_cost: total cost of transporting the mass
    """
    # Number of nodes in the graph
    num_nodes = G.number_of_nodes()

    # Initialize transport plan matrix
    transport_plan = np.zeros((num_nodes, num_nodes))
    
    # Get the list of edges and corresponding costs (weights)
    edge_list = list(G.edges)
    costs = np.array([G[u][v]['weight'] for u, v in edge_list])
    
    # Construct the transport plan based on optimal flow values and edge_list
    for i, (u, v) in enumerate(edge_list):
        transport_plan[u, v] = optimal_flow[i]
        transport_plan[v, u] = -optimal_flow[i]  # Assuming undirected graph

    # Calculate total transport cost
    total_transport_cost = np.sum(optimal_flow * costs)

    return transport_plan, total_transport_cost


def generate_transport_plan_old(optimal_flow, num_nodes, edge_list=None):
    """
    Converts the optimal flow into a transport plan.

    Args:
        optimal_flow (dict or np.ndarray): Optimal flow on each edge, either as a dictionary
                                           with edge tuples (i, j) as keys and net flow values
                                           as values, or as a 1-dimensional array representing
                                           the flow on edges in edge_list order.
        num_nodes (int): Number of nodes in the graph.
        edge_list (list of tuples): List of edges in the graph, used when optimal_flow is a
                                    1-dimensional array to map the flow values to the transport plan.

    Returns:
        np.ndarray: Transport plan matrix where entry (i, j) represents the amount transported
                    from node i to node j.
    """
    # Initialize a transport plan matrix with zeros
    transport_plan = np.zeros((num_nodes, num_nodes))

    # Check if the optimal flow is a dictionary or a 1-dimensional array
    if isinstance(optimal_flow, dict):
        print("Optimal flow is a dictionary")
        # If optimal_flow is a dictionary, process it as (i, j) -> net_flow pairs
        for (i, j), net_flow in optimal_flow.items():
            if net_flow > 0:
                transport_plan[i, j] = net_flow  # Flow from i to j
            elif net_flow < 0:
                transport_plan[j, i] = -net_flow  # Flow from j to i (reverse flow)
    elif isinstance(optimal_flow, np.ndarray) and edge_list is not None:
        print("Optimal flow is an array")
        # If optimal_flow is a 1-dimensional array, use edge_list to map flows to the transport plan
        for idx, (i, j) in enumerate(edge_list):
            net_flow = optimal_flow[idx]
            if net_flow > 0:
                transport_plan[i, j] = net_flow  # Flow from i to j
            elif net_flow < 0:
                transport_plan[j, i] = -net_flow  # Flow from j to i (reverse flow)
    else:
        raise TypeError("optimal_flow must be a dictionary or a 1-dimensional numpy array with an edge list.")

    return transport_plan

def calculate_total_transport_cost_from_graph(transport_plan, G):
    """
    Calculates the total transport cost given a transport plan and a graph object.

    Args:
        transport_plan (np.ndarray): Transport plan matrix where entry (i, j) represents
                                     the amount transported from node i to node j.
        G (networkx.Graph): Graph object where edge weights represent the cost of transporting
                            mass between connected nodes.

    Returns:
        float: The total transport cost.
    """
    num_nodes = len(G.nodes)
    
    # Initialize the cost matrix with graph distances
    cost_matrix = np.zeros((num_nodes, num_nodes))

    # Compute the shortest path distances between all pairs of nodes to populate the cost matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                try:
                    cost_matrix[i, j] = nx.shortest_path_length(G, source=i, target=j, weight='weight')
                except nx.NetworkXNoPath:
                    cost_matrix[i, j] = np.inf  # If no path exists, set cost to infinity

    # Element-wise multiplication of the transport plan and cost matrix, followed by summation
    total_cost = np.sum(transport_plan * cost_matrix)
    return cost_matrix, total_cost

def fn_node_assignments(transport_plan):
    """
    Generates node assignments based on the transport plan matrix.

    Parameters:
    - transport_plan: NxN matrix representing mass transported between nodes

    Returns:
    - assignments: a list of tuples (node_i, assigned_node_j)
    """
    num_nodes = transport_plan.shape[0]
    assignments = []

    for i in range(num_nodes):
        # Find the node to which the most mass is transported from node i
        # Ignore self-transport (set diagonal to zero temporarily)
        transport_row = transport_plan[i].copy()
        transport_row[i] = 0  # Ensure self-transport is ignored

        # Find the node with the maximum transported mass
        assigned_node = np.argmax(transport_row)

        # If there's no transport, skip assignment
        if transport_row[assigned_node] > 0:
            assignments.append((i, assigned_node))

    return assignments

def propagate_assignments_from_flow(optimal_flow, G):
    """
    Propagates node assignments based on the flow iteratively until they stabilize.

    Parameters:
    - optimal_flow: array of optimal flow values for each edge (from solver)
    - G: networkx graph with edges and corresponding weights

    Returns:
    - assignments: a dictionary where each node is mapped to its assigned node
    """
    edge_list = list(G.edges)
    assignments = {node: node for node in G.nodes}  # Start by assigning each node to itself

    # Track outgoing mass transport from each node
    outgoing_mass = {node: {} for node in G.nodes}

    # Step 1: Analyze the flow values and store them as outgoing mass
    for i, (u, v) in enumerate(edge_list):
        flow_value = optimal_flow[i]

        # If there is mass moving along the edge, register it as outgoing
        if flow_value > 0:
            outgoing_mass[u][v] = flow_value
            outgoing_mass[v][u] = flow_value

    # Step 2: Propagate node assignments based on the flow iteratively
    changed = True
    while changed:
        changed = False
        for u in G.nodes:
            # If node u has outgoing mass, assign it to the node with the most mass outgoing
            if outgoing_mass[u]:
                max_mass_node = max(outgoing_mass[u], key=outgoing_mass[u].get)

                # If the current assignment changes, update it
                if assignments[u] != assignments[max_mass_node]:
                    assignments[u] = assignments[max_mass_node]
                    changed = True

    return assignments


def generate_node_assignments_from_flow(optimal_flow, G):
    """
    Generates node assignments directly based on the optimal flow values.

    Parameters:
    - optimal_flow: array of optimal flow values for each edge (from solver)
    - G: networkx graph with edges and corresponding weights

    Returns:
    - assignments: a list of tuples (node_i, assigned_node_j)
    """
    edge_list = list(G.edges)
    assignments = {}

    # Track outgoing mass transport from each node
    outgoing_mass = {node: {} for node in G.nodes}

    # Analyze the flow values to determine where each node sends mass
    for i, (u, v) in enumerate(edge_list):
        flow_value = optimal_flow[i]

        # Assign flow in both directions (u -> v and v -> u) for undirected graphs
        if flow_value > 0:
            outgoing_mass[u][v] = flow_value
            outgoing_mass[v][u] = flow_value

    # Assign each node to the node to which it sends the most mass
    for u in G.nodes:
        if outgoing_mass[u]:
            assigned_node = max(outgoing_mass[u], key=outgoing_mass[u].get)
            assignments[u] = assigned_node

    return list(assignments.items())

def identify_sources_and_targets(optimal_flow, G, mu, nu, flow_threshold=1e-8):
    """
    Identifies the source nodes (where mass originates) and target nodes (where mass ends up).
    Also constructs a flow matrix for debugging purposes.
    `mu` and `nu` are arrays where node IDs are implicit (the array index represents the node).

    Parameters:
    - optimal_flow: array of optimal flow values for each edge (from solver)
    - G: networkx graph with edges
    - mu: array where non-zero entries represent source nodes' mass
    - nu: array where non-zero entries represent target nodes' mass
    - flow_threshold: the minimum flow value to consider (to filter out small numerical values)

    Returns:
    - source_nodes: list of source node IDs where mass originates
    - target_nodes: list of target node IDs that received mass based on the flow
    - flow_matrix: NxN matrix where each entry (i, j) represents the flow from node i to node j
    """
    num_nodes = G.number_of_nodes()
    
    # Initialize flow matrix to store flow between nodes
    flow_matrix = np.zeros((num_nodes, num_nodes))

    # Identify source nodes as those where mu > 0 (i.e., source mass)
    source_nodes = [i for i, mass in enumerate(mu) if mass > 0]

    # Set to hold target nodes
    target_nodes = set()

    # Iterate through the edges and populate the flow matrix
    edge_list = list(G.edges)
    for i, (u, v) in enumerate(edge_list):
        flow_value = optimal_flow[i]

        # Ignore small and negative flow values
        if abs(flow_value) > flow_threshold:
            print(f"Edge ({u}, {v}) has significant flow: {flow_value}")

            # Add flow to the flow matrix (undirected graph, flow is symmetric)
            flow_matrix[u, v] = flow_value
            flow_matrix[v, u] = flow_value

            # If flow is positive, check if either node u or v is a target in nu
            if flow_value > 0:
                if nu[v] > 0:
                    print(f"Node {v} is a target and receives mass.")
                    target_nodes.add(v)
                if nu[u] > 0:
                    print(f"Node {u} is a target and receives mass.")
                    target_nodes.add(u)

    return source_nodes, list(target_nodes), flow_matrix

def assign_source_to_target(mu, nu, optimal_flow, G):
    # Identify source nodes (nodes with non-zero mass in mu) and target nodes (nodes with non-zero mass in nu)
    source_nodes = np.where(mu > 0)[0]
    target_nodes = np.where(nu > 0)[0]

    # Track the remaining mass capacity for each target node
    target_capacity = {target: nu[target] for target in target_nodes}
    node_assignments = {}

    # Get the list of edges in the order used by the optimal flow
    edge_list = list(G.edges())

    # Accumulate the mass flow from each source node to the target nodes
    mass_sent = {source: {target: 0 for target in target_nodes} for source in source_nodes}
    for idx, (u, v) in enumerate(edge_list):
        flow_value = optimal_flow[idx]
        if flow_value > 0 and u in source_nodes and v in target_nodes:
            mass_sent[u][v] += flow_value
        elif flow_value < 0 and v in source_nodes and u in target_nodes:
            mass_sent[v][u] += -flow_value

    # Iteratively assign source nodes to target nodes based on mass requirements
    for source_node in source_nodes:
        # Filter targets that still have remaining capacity
        available_targets = {target: mass_sent[source_node][target] for target in target_nodes if target_capacity[target] > 0}

        # Find the target node that received the most mass from this source node
        if available_targets:
            target_node = max(available_targets, key=available_targets.get)

            # Assign the source node to this target node
            node_assignments[source_node] = target_node
            target_capacity[target_node] -= 1  # Reduce the remaining capacity for the target node

    return node_assignments

n = 5; k = 2; sigma1 = 2.5; sigma2 = 0.2; r = 1
# To recover a graph from a previous run, pass the signature as last argument
# 'ABEFEAYF'
if graphSignature == '':
    x, M = GenerateRandomGraph(n, k, sigma1, sigma2, r, graphSignature, newGraphSignature)
else:
    x, M = GenerateRandomGraph(n, k, sigma1, sigma2, r, graphSignature)
#print("x:",x)
#print("M:",M)
print("M sum:",np.sum(M))
try: # older versions of networkx
    G = nx.from_numpy_matrix(M)
except: # newer versions of networkx
    G = nx.from_numpy_array(M)

num_centers = 3 # number of node centers farthest apart

# Ensure graph is connected
for experiments in range(1):
    if experiments > 0:
        x = pos_x
        del mu, nu, node_id, target_nodes, target_colors, labels, assignments, ot_matrix, color_map, total_mass, target_mass, split_mass, i
    print("experiments:",experiments) 
    connectedGraph = False
    connectionAttempts = 0
    target_nodes = []
    total_mass = n*k
    split_mass = int(total_mass / num_centers)
    mu = np.ones(n*k)
    nu = np.zeros(n*k)

    while connectedGraph == False and connectionAttempts < 10:
        #print("G:",G)
        #print(G.adj)
        try:
            if experiments == 0:
                if graphSignature == '':
                    pos = nx.spring_layout(G)  # Using spring layout to assign positions for the nodes
                    with open(newGraphSignature+'.node_positions.pkl', 'wb') as f:
                        pickle.dump(pos, f)
                else:
                    with open(graphSignature+'.node_positions.pkl', 'rb') as f:
                        pos = pickle.load(f)
                #print("Positions:",pos)
                # Add positions as node attributes
                for node, coordinates in pos.items():
                    G.nodes[node]['pos'] = coordinates
                # Compute the Euclidean distance for each edge and add it as an edge attribute
                for (u, v) in G.edges():
                    distance = euclidean_distance(u, v)
                    G.edges[u, v]['weight'] = distance
                    #print("Edge:",u,v,"Distance:",distance," ", end='', flush=True)
                dist = graph_distance(G)
                print("Dist:",dist)
                print("Dist sum:",np.sum(dist))
            #target_node = 5
            # Initialize all nodes with a default color
            color_map = ['lightblue'] * len(G.nodes())
            # Set the color of the target node

            # Example: Finding the shortest path based on Euclidean distance
            if experiments == 0:
                source = 0
                target = 5
                shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight')
                shortest_path_length = nx.shortest_path_length(G, source=source, target=target, weight='weight')
                print(f"The shortest path from node {source} to node {target} is {shortest_path} with a length of {shortest_path_length}.")
                    
            # Find and print the n farthest pairs of nodes
            farthest_nodes = find_n_farthest_nodes(G, num_centers)
            print("farthest_nodes", farthest_nodes)
            counter = 0;
            for node_id in farthest_nodes:
                print("farthest node", counter, ":", node_id)
                color_map[node_id] = 'red'
                #if massAssignment == 'uniform':
                if experiments == 0:
                    nu[node_id] = split_mass
                else:
                    upper_mass = total_mass - np.sum(nu)
                    nu[node_id] = random.randint(1, upper_mass)
                counter = counter + 1
                if node_id not in target_nodes:
                    target_nodes.append(node_id)
            
            print("Target nodes: ", target_nodes)
            #nx.draw_networkx_nodes(G,pos=x, node_size = 20, node_color=color_map)
            #nx.draw_networkx_edges(G,pos=x)        
            #plt.draw()
            #plt.show()
            connectedGraph = True
        except Exception as e:
            if experiments == 0:
                # graph isn't connected; identify nodes that are missing a path
                # from generated exception, e.g. No path between 0 and 3.
                error_str = str(e)
                error_str = error_str.replace("No path between ", "");
                error_str = error_str.replace(".", "");
                nodes = error_str.split(" and ")
                node0 = int(nodes[0])
                node1 = int(nodes[1])
                print("Connect node", node0, "and node", node1)
                G.add_edge(node0, node1)
            else:
                connectionAttempts = 100
                print("Unknown exception occurred on subsequent experiments:")
                print(str(e))
                break
            
        connectionAttempts = connectionAttempts + 1

    # Define the colors
    num_colors = num_centers
    colors = generate_colors(num_colors)
    target_colors = {target_nodes[i]: colors[i] for i in range(len(target_nodes))}
    print("target_colors:",target_colors)

    # Assign labels to target nodes
    labels = {node: str(node) for node in target_nodes}

    target_mass = np.sum(nu)
    if target_mass != total_mass:
        # adjust accordingly
        nu[node_id] = nu[node_id] + (total_mass - target_mass)

    np.set_printoptions(threshold=np.inf)    
    print("mu:",mu)
    print("nu:",nu)
    print("Total mass of mu:",np.sum(mu))
    print("Total mass of nu:",np.sum(nu))
    if np.sum(mu) != np.sum(nu):
        print("Source and target masses are not equal")
        raise Exception("Source and target masses are not equal")

    # Cost matrix based on Euclidean distance for each pair of nodes
    num_nodes = len(G.nodes())
    cost_matrix = np.zeros((num_nodes, num_nodes))

    ''''
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                cost_matrix[i, j] = euclidean_distance(i, j)
    '''
    # Print the cost matrix
    #print("Cost Matrix (Euclidean distances between nodes):")
    #print(cost_matrix)
    print("Cost matrix sum: ")
    print(np.sum(cost_matrix))

    #nx.draw_networkx_edges(G,pos=x)
    node_color_dist = n * k - 1
    #nx.draw_networkx_nodes(G,pos=x, node_size = 30, node_color = dist[node_color_dist], cmap = 'plasma')
    #plt.draw()
    #plt.show()

    #This saves the distance function in a json file    
    distance_file_path = "dist.json"

    with open(distance_file_path, "w") as f:
        json.dump(dist.tolist(), f)

    #This saves the vectors forming the vertices in a json file     
    vertices_file_path = "vert.json"
    vectors = np.array([item[1] for item in list(x.items())])        
    with open(vertices_file_path, "w") as f:
        json.dump(vectors.tolist(), f)
    
    ### Solve the Beckmann Formulation using CVXPY
    '''if graphSignature != '':
        num_nodes = G.number_of_nodes()
        #a = np.ones(num_nodes)
        #b = np.zeros(num_nodes)
        #optimal_flow = solve_beckmann(G, a, b)
        optimal_flow = solve_beckmann(G, mu, nu)
        print("Optimal Flow:", optimal_flow)
        # Output the optimal flow
        if optimal_flow is not None:
            print("Optimal Flow:", optimal_flow)
            visualize_transport_process(G, mu, nu, optimal_flow)
        else:
            print("No feasible solution found.")
            
        # Generate the transport plan
        transport_plan = generate_transport_plan(optimal_flow, num_nodes)
        print("Transport Plan Matrix:")
        print(transport_plan)

        exit()'''
    ###
    
    ###
    ### Solve the Beckmann Formulation using CVXPY
if graphSignature != '':
    num_nodes = G.number_of_nodes()
    
    # Solve the Beckmann problem to obtain the optimal flow
    optimal_flow = solve_beckmann(G, mu, nu)
    print("nu", nu)
    print("Optimal Flow:", optimal_flow)

    # Output the optimal flow and generate the transport plan if a solution is found
    if optimal_flow is not None:
        print("Optimal Flow:", optimal_flow)

        # Get the edge list from the graph in the order used by the solver
        edge_list = list(G.edges())

        # Generate the transport plan using the optimal flow and the edge list
        #transport_plan = generate_transport_plan(optimal_flow, num_nodes, edge_list=edge_list)

        #print("Transport Plan Matrix:")
        #print(transport_plan)
        transport_plan, total_transport_cost = generate_transport_plan(optimal_flow, G)
        print("Total Cost:", total_transport_cost)
        print("Solution:")
        print(transport_plan)

        node_assignments = assign_source_to_target(mu, nu, optimal_flow, G)
        print("Node Assignments (source node -> target node):", node_assignments)
        
        # Generate node assignments based on the transport plan
        assignments = fn_node_assignments(transport_plan)

        # Output node assignments
        print("\nOT Node Assignments:")
        for node_i, assigned_node_j in assignments:
            print(f"Node {node_i} is assigned to Node {assigned_node_j}")

        # Generate node assignments based on the optimal flow
        assignments = generate_node_assignments_from_flow(optimal_flow, G)

        # Output node assignments
        print("\nFlow Node Assignments:")
        for node_i, assigned_node_j in assignments:
            print(f"Node {node_i} is assigned to Node {assigned_node_j}")

        # Propagate node assignments based on the optimal flow
        assignments = propagate_assignments_from_flow(optimal_flow, G)

        # Output node assignments
        print("\nPropogate Node Assignments:")
        for node_i, assigned_node_j in assignments.items():
            print(f"Node {node_i} is assigned to Node {assigned_node_j}")

        # Output source and target nodes from the flow
        print("Sanity check: Source and target nodes from the flow")

        # Identify source and target nodes from the flow
        print("mu:", mu)
        print("nu:", nu)
        source_nodes, target_nodes, flow_matrix = identify_sources_and_targets(optimal_flow, G, mu, nu)
        print("flow_matrix:", flow_matrix)

        # Output source and target nodes
        print(f"Source Nodes: {source_nodes}")
        print(f"Target Nodes: {target_nodes}")
        
        # Visualize the transport process
        visualize_transport_process(G, mu, nu, optimal_flow)
    else:
        print("No feasible solution found.")
    ###
    
    node_assignments = assign_source_to_target(mu, nu, optimal_flow, G)
    print("Node Assignments (source node -> target node):", node_assignments)
    
    #exit()
    #transport_plan = generate_transport_plan(optimal_flow, num_nodes, edge_list)
    #cost_matrix, total_cost = calculate_total_transport_cost_from_graph(transport_plan, G)
    #solution = transport_plan
    
    #total_cost, solution = solve_Kantorovich(cost_matrix, mu, nu)
    #print("Total Cost:", total_cost)
    #print("Solution:")
    #print(solution)

    ot_matrix = solution.reshape(cost_matrix.shape)
    print("OT Matrix: ", ot_matrix)

    #color_map = ['lightblue'] * len(G.nodes())
    assignments = [0] * len(G.nodes())  # Initialize the list with default values
    for i in range(num_nodes):
        for j in range(num_nodes):
            if int(ot_matrix[i, j]) == 1:
                #print("Node", i, "maps to node", j)
                assignments[i] = target_colors[j]
            else:
                assignments[i] = 'lightblue'

    print("Color assignments:", assignments)
    # Draw the graph with colored node assignments
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos=x, with_labels=False, node_size=30, node_color=assignments, font_size=16, font_color='white')

    # Draw specific labels with semi-transparent background
    # x.draw_networkx_labels(G, x, labels, font_size=16, font_color='black') # Simple labeling can be hard to read
    pos_x = x
    #ax = plt.gca()
    legend_elements = []
    for node, (x, y) in pos_x.items():
        if node in labels:
            label = labels[node]
            text = ax.text(x, y, label, fontsize=14, color=assignments[node], ha='center', va='center')
            text.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor=assignments[node], boxstyle='round,pad=0.3'))
            text.set_fontweight('bold')
            legend_elements.append(patches.Patch(facecolor=assignments[node], edgecolor='black', label=str(int(nu[node]))+' nodes mapped to #'+str(node)))
        else:
            text = ax.text(x, y, str(node), fontsize=10, color=assignments[node], ha='center', va='center')
            text.set_bbox(dict(facecolor='white', alpha=1, edgecolor=assignments[node], boxstyle='round,pad=0.1'))
    legend_elements.append(patches.Patch(facecolor='black', edgecolor='black', label='Total transport cost= '+str(total_cost)))
    if graphSignature == '':
        legend_elements.append(patches.Patch(facecolor='black', edgecolor='black', label='Graph Signature: '+newGraphSignature))
    else:
        legend_elements.append(patches.Patch(facecolor='black', edgecolor='black', label='Graph Signature: '+graphSignature))

    # Add the legend to the plot
    plt.legend(handles=legend_elements, loc='best')
    plt.savefig('graph' + str(experiments+1) + '.png')
    plt.show()
