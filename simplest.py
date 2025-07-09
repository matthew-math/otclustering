import numpy as np
import networkx as nx
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import cvxpy as cp
import scipy.sparse as sps

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

# Define a simple linear graph with 3 nodes
G = nx.Graph()
G.add_edge(0, 1, weight=1.0)
G.add_edge(1, 2, weight=1.0)
#G.add_edge(2, 3, weight=1.0)
#G.add_edge(3, 4, weight=1.0)

a = np.array([0.5, 0.0, 0.5])
b = np.array([0.0, 1.0, 0.0])

# Solve the problem using Beckmann's formulation
optimal_flow = solve_beckmann(G, a, b)

# Output the optimal flow
if optimal_flow is not None:
    print("Optimal Flow:", optimal_flow)
    visualize_transport_process(G, a, b, optimal_flow)
else:
    print("No feasible solution found.")