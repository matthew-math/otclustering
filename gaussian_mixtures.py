import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.linalg as la
import scipy.stats as stats
import networkx as nx
import json
from OT_solver import solve_Kantorovich
from colorsys import hsv_to_rgb
import mpld3 # This is a library that allows for interactive plots
from mpld3 import plugins
import random
import pickle
from datetime import datetime
random.seed(datetime.now().timestamp())
massAssignment = 'random' # 'uniform' or 'random'
#graphSignature = '8WT6QPV8' # Use signature from past run or leave empty to generate a new graph
graphSignature = ''
if graphSignature == '':
    newGraphSignature = ''.join((random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for i in range(8)))
    print('Graph Signature (set graphSignature equal to this string to recreate this graph):', newGraphSignature)
else:
    newGraphSignature = ''

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

n = 50; k = 4; sigma1 = 2.5; sigma2 = 0.2; r = 1
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

num_centers = 2 # number of node centers farthest apart

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
            nx.draw_networkx_nodes(G,pos=x, node_size = 20, node_color=color_map)
            nx.draw_networkx_edges(G,pos=x)        
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

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                cost_matrix[i, j] = euclidean_distance(i, j)

    # Print the cost matrix
    #print("Cost Matrix (Euclidean distances between nodes):")
    #print(cost_matrix)
    print("Cost matrix sum: ")
    print(np.sum(cost_matrix))

    nx.draw_networkx_edges(G,pos=x)
    node_color_dist = n * k - 1
    #nx.draw_networkx_nodes(G,pos=x, node_size = 100, node_color = dist[80], cmap = 'plasma')
    nx.draw_networkx_nodes(G,pos=x, node_size = 30, node_color = dist[node_color_dist], cmap = 'plasma')
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
        
    total_cost, solution = solve_Kantorovich(cost_matrix, mu, nu)
    print("Total Cost:", total_cost)
    #print("Solution:")
    #print(solution)

    ot_matrix = solution.reshape(cost_matrix.shape)
    #print(ot_matrix)

    #color_map = ['lightblue'] * len(G.nodes())
    assignments = [0] * len(G.nodes())  # Initialize the list with default values
    for i in range(num_nodes):
        for j in range(num_nodes):
            if int(ot_matrix[i, j]) == 1:
                #print("Node", i, "maps to node", j)
                assignments[i] = target_colors[j]

    # Draw the graph with colored node assignments
    nx.draw(G, pos=x, with_labels=False, node_size=30, node_color=assignments, font_size=16, font_color='white')

    # Draw specific labels with semi-transparent background
    # x.draw_networkx_labels(G, x, labels, font_size=16, font_color='black') # Simple labeling can be hard to read
    pos_x = x
    ax = plt.gca()
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
