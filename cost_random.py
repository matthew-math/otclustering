import numpy as np
import geopandas
import json
import random

import networkx
import pysal
import libpysal
from libpysal.weights import Rook

def GraphWeights(G,epsilon = 0.25):
    if epsilon == 0:
        return G
    else: 
        for e in list(G.edges):
            G.edges[e]['weight'] = random.uniform(epsilon,1/epsilon)
        return G

def graph_distance_matrix(graph,targetlist):
    
    N = len(graph.nodes)
    k = len(targetlist)
    
    matrix = np.empty((N,k))   
    print('Computing partial distance matrix with networkx')
    for i in range(N):
        for j in range(k):
            matrix[i,j] = networkx.shortest_path_length(graph,source = i, target = targetlist[j])
        if (N-i)%10==0:
            print(N-i)        
            
    return matrix

def get_graph_from_shapefile(filepath,id_column=None,epsilon = 0.25):
    #weights = pysal.rook_from_shapefile(filepath, id_column)
    weights = Rook.from_shapefile(filepath, id_column)    
    #Add random weights (a perturbation of constant weight 1)    
    
    print(type(weights))
    
    G = networkx.Graph(weights.neighbors)
    G = GraphWeights(G,epsilon)
    
    return G

def graph_distance_matrix_from_shapefile(filepath,targetlist,epsilon = 0.25):
    graph = get_graph_from_shapefile(filepath,None,epsilon)
        
    matrix = graph_distance_matrix(graph,targetlist)
    return matrix

def main():

    shapefile_path = "./shapefiles/TX_vtds.shp"
    
    df = geopandas.read_file(shapefile_path)
    N = len(df)
    k = 36

    #matrix_outpath = "./distances/TX_vtds_%s_a.json" % k
    matrix_outpath_rand = "./distances/TX_vtds_rand%s_a.json" % k
    matrix_outpath_targets = "./distances/TX_vtds_rand%s_a_targets.json" % k
    
    targetlist = np.random.choice(N,k,replace=False)        
    print(targetlist)    

    #matrix = graph_distance_matrix_from_shapefile(shapefile_path,targetlist,0)        
    matrix_rand = graph_distance_matrix_from_shapefile(shapefile_path,targetlist,1/100)        
    
    #### Starts interactive python terminal
    #import code
    #code.interact(local=locals())
    
    #with open(matrix_outpath, "w") as f:
    #    json.dump(matrix.tolist(), f)
               
    with open(matrix_outpath_rand, "w") as f:
        json.dump(matrix_rand.tolist(), f)        
           
    #This saves the list of targets
    with open(matrix_outpath_targets, "w") as f:
        json.dump(targetlist.tolist(), f)

    
    print("Done!")


if __name__ == "__main__":
    main()
