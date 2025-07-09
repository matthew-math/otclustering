import numpy as np
import geopandas
import json
import random

import networkx
import pysal
import libpysal
from libpysal.weights import Rook

def GraphWeights(G,epsilon = 0.25):
    for e in list(G.edges):
        G.edges[e]['weight'] = random.uniform(1-epsilon,1+epsilon)
    return G

# WANT TO DO GRAPH_DISTANCE_MATRIX

def quadratic_distance(x, y):
    return np.sum(np.square(np.array(x) - np.array(y)))

def graph_distance_matrix(graph,targetlist):
    
    N = len(graph.nodes)
    k = len(targetlist)
    
    matrix = np.empty((N,k))   
    print('Computing graph distance matrix with networkx')
    for i in range(N):
        for j in range(k):
            matrix[i,j] = networkx.shortest_path_length(graph,source = i, target = targetlist[j])
            #matrix[i,j] = networkx.shortest_path_length(graph,source = i, target = targetlist[j]) + ep * networkx.shortest_path_length(graph,source = i, target = targetlist[j]) ** 2    
        if (N-i)%10==0:
            print(N-i)        
            
    return matrix

def full_graph_distance_matrix(graph,targetlist):
    N = len(graph.nodes)
    k = len(targetlist)
    
    matrix = np.empty((N,k))   
    print('Computing full distance matrix with networkx')
    #for i in range(N):
    #    for j in range(k):
    #        matrix[i,j] = networkx.shortest_path_length(graph,source = i, target = j)
    #        #matrix[i,j] = networkx.shortest_path_length(graph,source = i, target = targetlist[j]) + ep * networkx.shortest_path_length(graph,source = i, target = targetlist[j]) ** 2    
    #    if (N-i)%10==0:
    #        print(N-i)        

    # Get all shortest paths as a dictionary of dictionaries (much faster than the above double loop)
    shortest_paths = dict(networkx.shortest_path_length(graph))

    # Populate the matrix by extracting only the distances for target nodes
    for i in range(N):
        for j, target in enumerate(targetlist):
            matrix[i, j] = shortest_paths.get(i, {}).get(target, np.inf)  # Use np.inf if no path exists

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
    return networkx.Graph(weights.neighbors)

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

#targetlist - barycenters
# run dinstance_matrix_from shapefile


def main():

    k = 10

    shapefile_path = "./shapefiles/PA_VTD.shp"
    #shapefile_path = "./shapefiles/IA_counties.shp"

    #matrix_outpath_e = "./distances/TX_vtds_%s.json" %k
    #matrix_outpath_g = "./distances/PA_vtds_%s.json" % k
    #matrix_outpath_g_targets = "./distances/PA_vtds_%s_targets.json" % k
    matrix_outpath_g = "./distances/IA_vtds_%s.json" % k
    matrix_outpath_g_targets = "./distances/IA_vtds_%s_targets.json" % k
        
    df = geopandas.read_file(shapefile_path)

    # Check if the geometry is in a geographic CRS (latitude/longitude)
    # Addresses the Python UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.
    if df.crs and df.crs.is_geographic:
        # Convert to a projected CRS for accurate centroid calculations
        df = df.to_crs(df.estimate_utm_crs())

    points = [centroid.coords for centroid in df.centroid]
    N = len(points)
        
    #targetlist = np.random.choice(N,k,replace=False)
    targetlist = np.arange(N) # Create targetlist as a list of ALL indices, instead of a random subset   
    print(targetlist)

    #matrix_g = partial_graph_distance_matrix_from_shapefile(shapefile_path,targetlist) # this is what we were previously using
    graph = get_graph_from_shapefile(shapefile_path)
    matrix_g = full_graph_distance_matrix(graph,targetlist)        
    #matrix_e = partial_distance_matrix_from_shapefile(shapefile_path,targetlist)
    #matrix_mixed = partial_graph_distance_matrix_from_shapefile(shapefile_path,targetlist)
    
    #### Starts interactive python terminal
    #import code
    #code.interact(local=locals())
    
    with open(matrix_outpath_g, "w") as f:
        json.dump(matrix_g.tolist(), f)
    #with open(matrix_outpath_e, "w") as f:
    #    json.dump(matrix_e.tolist(), f)        
    #with open(matrix_outpath_mixed, "w") as f:
    #    json.dump(matrix_mixed.tolist(), f)        
       
    #This saves the list of targets
    with open(matrix_outpath_g_targets, "w") as f:
        json.dump(targetlist.tolist(), f)
        
    #with open(matrix_outpath_e, "w") as f:
    #    json.dump(matrix_e.tolist(), f)        
    #with open(matrix_outpath_mixed, "w") as f:
    #    json.dump(matrix_mixed.tolist(), f)        
           
        
    print("Done!")


if __name__ == "__main__":
    main()
