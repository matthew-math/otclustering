import os
import json
import csv
from collections import defaultdict
from multiprocessing import Value

import numpy as np
import pandas
import geopandas
import folium
import matplotlib.pyplot as plt

#from graph_tools import district_subgraph
#from graph_tools import is_district_connected
#from graph_tools import connected_districts

from CKO_solver import CKOscheme
from OT_solver import solve_Kantorovich

"""

We are trying to take a shapefile that contains population and geometric data (a graph) to 

1) generate a rectangular distance matrix from the entire graph to a subset of k elements (k<100 nodes)
2) use that rectangular data to set up an optimal transport problem from N elements to k elements (N is the size of the graph)
3) save the resulting map as a column in the shapefile 

"""

class OptimizationError(Exception):
    pass

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
        if geoidtable == 0: # test case
            #list = [0, 40000, 0, 0, 10000, 0, 0, 40000, 0]
            #list = [90000, 0, 0, 0, 0, 0, 0, 0, 0]
            #list = [30000, 30000, 30000, 0, 0, 0, 0, 0, 0]
            array_list = [0, 0, 30000, 0, 0, 30000, 0, 0, 30000]
            nu = np.array(array_list) # Convert Python list to Numpy array
        elif geoidtable == 100:
            array_list = [1000000, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            nu = np.array(array_list)
        elif geoidtable == 105:
            array_list = [100000, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 50000, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 100000]
            nu = np.array(array_list)
        else:
            nu = np.ones(number_of_districts) * total_population/number_of_districts
    
    print("mu:", mu)
    print("nu:", nu)
    
    total_cost, solution = solve_Kantorovich(c, mu, nu)

    ot_matrix = solution.reshape(c.shape)
    
    print('')
    print('Number of source points assigned to multiple targets')
    print(count_splitting(ot_matrix))
    print('')
    print('Number of tracts/groups/blocks with non-zero population')
    print(count_support(mu))    
    print('')
        
    #print(ot_matrix)
        
    ####Transport plan to transport map
    #districts = plan_to_map(ot_matrix,geoidtable)
    districts = plan_to_map_array(N,ot_matrix,geoidtable)
        
    return districts
  
#This function counts the number of points in the source assigned to multiple targets  
def count_splitting(coupling_matrix):  
  
  M = len(coupling_matrix[0])
  N = len(coupling_matrix[:,0]) #this is the large one
 
  counter = int(0)
  
  for i in range(N):
      if len(np.nonzero(coupling_matrix[i])[0])>1:
          counter = counter+1
  return counter

#This function counts the number of non-zero rows in a distribution
def count_support(distribution):  
  
  N = len(distribution)
  
  counter = int(0)
  
  for i in range(N):
      if distribution[i] > 0:
          counter = counter+1
  return counter
  
def geopdvisual(dataframe,plan_id,outfilename,color):
    dataframe.plot(column = plan_id, cmap = color)
    #plt.savefig(outfilename, dpi = 1000)
    plt.savefig(outfilename, dpi = 300)
    print('Map saved as %s' % outfilename)
    
def foliumvisual():    
    pass
    
    ####The first line is commented out if an appropriate json file already exists
    #data_provider.shpconverter(filepath,'testjson.json')
    
    #map_center = np.array([pandas.to_numeric(df['INTPTLAT']).mean(),pandas.to_numeric(df['INTPTLON']).mean()])

    #m = folium.Map(location = [map_center[0],map_center[1]], zoom_start = 8, tiles='Mapbox bright' )        
    
    #m.choropleth(??)
    #m.save(outfile = "foliumtest.html")
  
        
def DetermineDistricts(N,c,alpha):

    nodes = np.empty((N,len(alpha)))
    district = np.empty(N)
    for i in range(N):
        Aff = []
        for j in range(len(alpha)):
            Aff.append(c[i,j]-alpha[j])
        nodes[i] = Aff
        
    for i in range(N):
        counter = 0
        
        Aff = nodes[i]
        for j in range(len(alpha)):
            if Aff[j] == min(Aff):
                counter = counter + 1
        if counter == 1:
                district[i] = np.argmin(np.asarray(Aff))
        if counter >1:
                district[i] = len(alpha)

    return district 
  
def demo_optimal_transport(state_geoid=42):

    if state_geoid == 0: # Simple test case
        num_districts = 3
        state_name = 'simple_rectangle'
        metric_path = "./data/simple_rectangle_shapefile.json"
        filepath = "./data/simple_rectangle_shapefile.shp"
        fip_heading = 'GEOID'
        population_heading = 'population'
    elif state_geoid == 100: # Simple test case
        num_districts = 10
        state_name = 'simple_10x10rectangle'
        metric_path = "./data/simple_10x10rectangle_shapefile_100.json"
        filepath = "./data/simple_10x10rectangle_shapefile.shp"
        fip_heading = 'GEOID'
        population_heading = 'population'
    elif state_geoid == 105: # Simple test case
        num_districts = 25
        state_name = 'simple_5x5rectangle'
        metric_path = "./data/simple_5x5rectangle_shapefile.json"
        filepath = "./data/simple_5x5rectangle_shapefile.shp"
        fip_heading = 'GEOID'
        population_heading = 'population'
    elif state_geoid == 42: #Pennsylvania
        # Columns: [STATEFP10, COUNTYFP10, VTDST10, GEOID10, VTDI10, NAME10, NAMELSAD10, LSAD10, MTFCC10, FUNCSTAT10, ALAND10, AWATER10, INTPTLAT10, INTPTLON10, ATG12D, ATG12R, GOV10D, GOV10R, PRES12D, PRES12O, PRES12R, SEN10D, SEN10R, T16ATGD, T16ATGR, T16PRESD, T16PRESOTH, T16PRESR, T16SEND, T16SENR, USS12D, USS12R, GOV, TS, HISP_POP, TOT_POP, WHITE_POP, BLACK_POP, NATIVE_POP, ASIAN_POP, F2014GOVD, F2014GOVR, 2011_PLA_1, REMEDIAL_P, 538CPCT__1, 538DEM_PL, 538GOP_PL, 8THGRADE_1, gdk18_1, gdk9_1, gdk15_1, gdk2_1, gdk3_1, gdk4_1, gdk18_2, gdk18_dual, gdk18_du_1, gdk18_prim, gdk18_pr_1, gdk18_pr_2, gdk18_pr_3, gdk18_pr_4, gdk18_du_2, gdk18_pr_5, gdk3_prima, gdk3_dual_, gdk3_pri_1, gdk3_dua_1, gdk3_pri_2, m, geometry]
        num_districts = 18
        state_name = 'PA'
        metric_path = "./data/PA_vtds_%s.json" %num_districts # (missing)
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
        num_districts = 4
        state_name = 'IA'
        metric_path = "./data/IA_counties_%s.json" %num_districts
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
    
    #filepath = "./data/2012_42_test.shp"
    #metric_path = "./data/TX_vtds_%s.json" %num_districts
    #metric_path = "./data/VA_%s.json" %num_districts    
    #metric_path = "./data/VA_qed_k%s_1.json"  % num_districts
    #metric_path = "./data/graph_distance_2012_42_test_k16.json"    
    plan_id = 'gedk%s_1' % num_districts                
                    
    #### Sets up data frame from shapefile
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_colwidth', None)
    
    # Read selection of lines from data file
    #df = geopandas.read_file(filepath, rows=10)
    df = geopandas.read_file(filepath)
        
    if state_geoid == 100:
        print('Data headings:')
        print(df.head(0))
        population = np.ones(100) * 10000
        print(population)
        Pop_district = np.sum(population) / num_districts
        array_list = [1000000, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        g = np.array(array_list) # Convert Python list to Numpy array
        #g = np.ones(num_districts) * Pop_district
        print(g)
        delta = Pop_district * (0.01)
        #exit(1)
    elif state_geoid == 105:
        print('Data headings:')
        print(df.head(0))
        population = np.ones(25) * 10000
        print(population)
        Pop_district = np.sum(population) / num_districts
        array_list = [120000, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 130000]
        g = np.array(array_list) # Convert Python list to Numpy array
        #g = np.ones(num_districts) * Pop_district
        print(g)
        delta = Pop_district * (0.01)
        #exit(1)
    elif (state_geoid > 0):
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
        
        #import code
        #code.interact(local=locals())
    else:
        population = np.ones(9) * 10000
        print(population)
        Pop_district = np.sum(population) / num_districts
        #list = [0, 0, 0, 30000, 30000, 30000, 0, 0, 0]
        #list = [0, 30000, 0, 0, 30000, 0, 0, 30000, 0]
        list = [30000, 30000, 30000, 0, 0, 0, 0, 0, 0]
        #list = [90000, 0, 0, 0, 0, 0, 0, 0, 0]
        #list = [45000, 0, 0, 0, 0, 0, 0, 0, 45000]
        g = np.array(list) # Convert Python list to Numpy array
        #g = np.ones(num_districts) * Pop_district
        print(g)
        delta = Pop_district * (0.01)
        #exit(1)
    
    print("\npopulation:",population, "num_districts:", num_districts, "state_geoid:",state_geoid)
    
    # determine the number of columns in shapefile
    with open(metric_path) as f:
        temp_matrix = np.array(json.load(f))
        matrix_dimensions = temp_matrix.shape
        print("Read the following cost matrix dimensions:", matrix_dimensions)
        matrix_rows, matrix_columns = matrix_dimensions
        del temp_matrix # we got the info we needed, so we can delete the matrix now
        f.close()

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
    else:
        print("No target file found (using defaults)")
        
    for num_districts_k in range(1,num_districts+1):
        print("Optimizing for %s districts" % num_districts_k)
        #### Loads distance matrix
        #num_districts_k = 4 # for testing
        with open(metric_path) as f:
            #only load the first num_districts_k rows
            #data_cost_matrix = np.array(json.load(f))
            if state_geoid == 105:
                data_cost_matrix = np.array(json.load(f))[:matrix_rows,:matrix_rows] # Test state with 100 partitions
            elif (state_geoid > 0):
                data_cost_matrix = np.array(json.load(f))[:matrix_rows,:num_districts_k] # for Texas 8941 x 36
            else:
                data_cost_matrix = np.array(json.load(f))[:matrix_rows,:matrix_rows] # Test state with 9 partitions
            #data_cost_matrix = np.array(json.load(f))[:matrix_rows,:matrix_columns]
            print("data_cost_matrix size:", data_cost_matrix.shape)
            print("data_cost_matrix:\n", data_cost_matrix)
            #data_cost_matrix = np.rot90(data_cost_matrix)
            print(data_cost_matrix)
            print(data_cost_matrix.shape)
                    
        #### Call compute_districts to solve OT problem
        #num_districts_k = num_districts
        if state_geoid == 100:
            districts = compute_districts(data_cost_matrix, population, 100, state_geoid, targetlist)
        elif state_geoid == 105:
            districts = compute_districts(data_cost_matrix, population, 25, state_geoid, targetlist)
        elif state_geoid > 0:
            districts = compute_districts(data_cost_matrix, population, num_districts_k, state_geoid, targetlist)
        else:
            districts = compute_districts(data_cost_matrix, population, 9, state_geoid, targetlist)

        alpha = np.zeros(num_districts_k)
        alpha[0] = 50000
        print(len(population))
        print('CKOscheme...')
        print(alpha.shape)
        print(population.shape)
        print(data_cost_matrix.shape)
        alpha_sol = CKOscheme(alpha,population,g,data_cost_matrix,delta) 
        
        #districts = DetermineDistricts(len(population),data_cost_matrix,alpha_sol)
            
        #districts = districts
        #### Adds the districts to the data frame
        df[plan_id] = pandas.Series(districts)
            
        #### Writes changes to the data frame file 
        #df.to_file(filepath)
        
        #sub_df = district_subgraph(df,plan_id,district_number)    
        #list = connected_districts(df,plan_id,num_districts)

        ####Now we call a function that generates and saves a map
        ####it uses geopandas, geojson, and pyplot
        if num_districts > 20:
            color_map = 'hsv'
        else:
            color_map = 'tab20'
        filename = state_name + '_qed_k%s_1_unif.png' % num_districts_k
        print("Generating map... close window and hit Crtl+D to iterate to next district count")
        geopdvisual(df, plan_id, filename, color_map) #'tab20' 'gist_rainbow' 'hsv'
        plt.show()
        #with open('test.csv', 'w') as f:
            #fieldnames = ['GEOID', 'DISTRICT']
            #writer = csv.DictWriter(f, fieldnames=fieldnames)
            #writer.writeheader()
            #data = [dict(zip(fieldnames, [k, v])) for k, v in districts.items()]
            #writer.writerows(data)

        import code
        code.interact(local=locals())
        if state_geoid == 0:
            break # just show the first result for synthetic states

    #return districts
    
def main():
    
    #### Starts interactive python terminal
    #import code
    #code.interact(local=locals())
    
    # for tests, use 0 for a 3x3, 100 for 10x10, and 105 for 5x5
    demo_optimal_transport(105) #42 is Pennsylvania, 48 is Texas, 19 is Iowa, 26 is Michigan, 51 is Virginia, 17 is Illinois, 55 is Wisconsin, 12 is Florida, 13 is Georgia, 16 is Idaho, 18 is Indiana, 20 is Kansas, 21 is Kentucky, 22 is Louisiana, 23 is Maine, 24 is Maryland, 25 is Massachusetts, 27 is Minnesota, 28 is Mississippi, 29 is Missouri, 30 is Montana, 31 is Nebraska, 32 is Nevada, 33 is New Hampshire, 34 is New Jersey, 35 is New Mexico, 36 is New York, 37 is North Carolina, 38 is North Dakota, 39 is Ohio, 40 is Oklahoma, 41 is Oregon, 43 is Rhode Island, 44 is South Carolina, 45 is South Dakota, 46 is Tennessee

if __name__ == '__main__':
    main()
