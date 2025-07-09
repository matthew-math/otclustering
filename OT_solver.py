import numpy as np

import cplex

import matplotlib.pyplot as plt
from scipy import sparse

"""
Sets up and solves the Kantorovich problem using the cplex class.
"""

def sparse_constraints_dual(M,N):
    constraints = []

    for i in range(M):
        for j in range(N):
            constraints.append([[str(i),str(M+j)],[1.0,1.0]])
    return constraints

def sparse_constraints_dual_rhs(cost,M,N):
    rhs = []

    for i in range(M):
        for j in range(N):
            rhs.append(cost[i,j])
    return rhs

def sparse_constraints_first(M,N): #this one produces constraints regarding the first marginal
    constraints = []

    for j in range(M):
        constraints.append( [["{0}".format(j*N+k) for k in range(N)],[1]*N])

    return constraints

def sparse_constraints_second(M,N): #this one produces constraints regarding the second marginal
    constraints = []

    for i in range(N):
        constraints.append( [["{0}".format(k*N+i) for k in range(M)],[1]*M])     
                        
    return constraints

def solve_Kantorovich_with_details(cost, mu, nu, epr=0, JustObjValue=False):
    import gc  # For explicit garbage collection

    hideOutput = True
    M = len(mu)
    N = len(nu)
    
    # Identify active target indices (where nu is non-zero)
    active_targets = [i for i, val in enumerate(nu) if val > 0]
    reduced_N = len(active_targets)

    # Slice the cost matrix to retain only necessary columns
    print("Slicing the cost matrix to include only active target districts...")
    reduced_cost = cost[:, active_targets]
    del cost  # Free up memory used by the original matrix
    gc.collect()  # Explicit garbage collection

    # Adjust nu for the reduced target set
    reduced_nu = nu[active_targets]
    if not np.isclose(np.sum(nu), np.sum(reduced_nu)):
        raise ValueError("Reduced nu does not preserve total mass of original nu.")
    
    nu_top = reduced_nu * (1 + epr * 0.01)
    nu_bottom = reduced_nu * (1 - epr * 0.01)

    OT_prob = cplex.Cplex()

    # Set memory emphasis to prioritize memory savings
    OT_prob.parameters.emphasis.memory.set(1)
    OT_prob.parameters.workmem.set(24000)  # Set working memory to 24 GB
    OT_prob.parameters.workdir.set("/tmp")  # Set work directory for disk usage

    if hideOutput:
        OT_prob.set_log_stream(None)
        OT_prob.set_error_stream(None)
        OT_prob.set_warning_stream(None)
        OT_prob.set_results_stream(None)

    OT_prob.objective.set_sense(OT_prob.objective.sense.minimize)

    # Add variables and constraints
    Var_names = ["{0}".format(i) for i in range(M * reduced_N)]
    OT_prob.variables.add(obj=np.ravel(reduced_cost), names=Var_names)
    OT_prob.linear_constraints.add(lin_expr=sparse_constraints_first(M, reduced_N), rhs=mu, senses=["E"] * M)
    OT_prob.linear_constraints.add(lin_expr=sparse_constraints_second(M, reduced_N), rhs=nu_top, senses=["L"] * reduced_N)
    OT_prob.linear_constraints.add(lin_expr=sparse_constraints_second(M, reduced_N), rhs=nu_bottom, senses=["G"] * reduced_N)

    print("OT problem constraints have been set; now attempting to solve...")

    OT_prob.solve()

    print("OT problem has been solved; now copying results to `sol` matrix...")
    sol = np.array(OT_prob.solution.get_values())

    print("Now extracting the total cost...")
    total_cost = OT_prob.solution.get_objective_value()

    print("Total cost extracted; now reshaping the solution into a transport matrix...")

    # Reshape the solution into a reduced transport matrix
    transport_matrix_reduced = sol.reshape(M, reduced_N)

    # Initialize a full zero matrix with the original dimensions
    transport_matrix = np.zeros((M, N))

    # Map the reduced transport matrix back to the original shape
    for reduced_target_id, target_id in enumerate(active_targets):
        transport_matrix[:, target_id] = transport_matrix_reduced[:, reduced_target_id]

    # Flatten the padded transport matrix back into a 1D array
    sol_padded = transport_matrix.flatten()

    # Map reduced targets back to the original target IDs
    transport_details = {}
    for source_id in range(M):
        transport_details[source_id] = {}
        for target_id in range(N):  # Iterate over all target IDs
            flow = transport_matrix[source_id, target_id]
            if flow > 0:  # Only record non-zero flows
                transport_details[source_id][target_id] = flow

    print('(Kantorovich) Total Cost =', total_cost)

    if JustObjValue:
        return total_cost
    else:
        return total_cost, transport_matrix, transport_details
    
def solve_Kantorovich_with_details_old(cost, mu, nu, epr=0, JustObjValue=False):
    hideOutput = True
    M = len(mu)
    N = len(nu)
    nu_top = nu * (1 + epr * 0.01)
    nu_bottom = nu * (1 - epr * 0.01)

    OT_prob = cplex.Cplex()
    
    # Set memory emphasis to prioritize memory savings
    OT_prob.parameters.emphasis.memory.set(1)  # 1 = Emphasize memory efficiency
    
    # Set working memory limit to 24 GB (24000 MB)
    OT_prob.parameters.workmem.set(24000)  # CPLEX accepts this in MB

    # Enable out-of-core processing
    OT_prob.parameters.workdir.set("/tmp")  # Ensure the directory has sufficient disk space

    if hideOutput:  # Suppress CPLEX output
        OT_prob.set_log_stream(None)
        OT_prob.set_error_stream(None)
        OT_prob.set_warning_stream(None)
        OT_prob.set_results_stream(None)

    OT_prob.objective.set_sense(OT_prob.objective.sense.minimize)

    # Add variables and constraints
    Var_names = ["{0}".format(i) for i in range(M * N)]
    OT_prob.variables.add(obj=np.ravel(cost), names=Var_names)
    OT_prob.linear_constraints.add(lin_expr=sparse_constraints_first(M, N), rhs=mu, senses=["E"] * M)
    OT_prob.linear_constraints.add(lin_expr=sparse_constraints_second(M, N), rhs=nu_top, senses=["L"] * N)
    OT_prob.linear_constraints.add(lin_expr=sparse_constraints_second(M, N), rhs=nu_bottom, senses=["G"] * N)

    print("OT problem constraints have been set; now attempting to solve...")

    # Solve the optimization problem
    OT_prob.solve()

    print("OT problem has been solved; now copying results to `sol` matrix...")
    sol = np.array(OT_prob.solution.get_values())
    
    print("Now extracting the total cost...")
    total_cost = OT_prob.solution.get_objective_value()

    print("Total cost extracted; now reshaping the solution into a transport matrix...")
    # Reshape the solution into a transport matrix
    transport_matrix = sol.reshape(M, N)

    # Extract mass transport details
    transport_details = {}
    for source_id in range(M):
        transport_details[source_id] = {}
        for target_id in range(N):
            flow = transport_matrix[source_id, target_id]
            if flow > 0:  # Only record non-zero flows
                transport_details[source_id][target_id] = flow
    
    print('(Kantorovich) Total Cost =', total_cost)

    if JustObjValue:
        return total_cost
    else:
        return total_cost, transport_matrix, transport_details
    
# Note about the transport_details return variable:
# transport_details is a dictionary that maps each source district (key) 
# to another dictionary of target districts and the corresponding transported mass (values).
# 
# Example structure of transport_details:
# transport_details = {
#     0: {4: 50.0},  # Source district 0 sends 50 units to target district 4
#     1: {4: 100.0, 5: 50.0},  # Source district 1 splits mass between targets 4 and 5
#     2: {5: 150.0}   # Source district 2 sends 150 units to target district 5
# }
# 
# To utilize transport_details:
# 1. Iterate over each source district to access the transported mass:
#    for source, targets in transport_details.items():
#        print(f"Source District {source} transports:")
#        for target, mass in targets.items():
#            print(f"  -> {mass} units to Target District {target}")
# 2. Use this information for reporting, analysis, or visualization.
#    For example:
#    - Calculate total mass transported to each target district.
#    - Identify sources that split mass among multiple targets.
#    - Visualize transport flows on a map.
# 
# 3. Validate Conservation of Mass:
#    - The sum of transported mass from each source (row of transport matrix) should equal the population in mu.
#    - The sum of transported mass to each target (column of transport matrix) should match the population in nu.
# 
# 4. Example: Calculate total mass transported to each target district:
#    total_mass_to_targets = {}
#    for targets in transport_details.values():
#        for target, mass in targets.items():
#            total_mass_to_targets[target] = total_mass_to_targets.get(target, 0) + mass
#    print("Total mass to each target:", total_mass_to_targets)
#
# By processing transport_details, you can gain insights into the optimal transport solution.

def solve_Kantorovich(cost, mu, nu,epr=0,JustObjValue = False):
    hideOutput = True
    M = len(mu)
    N = len(nu)
    #epr = percentage of error allowed when matching district populations
    nu_top = nu * (1+epr*(0.01))
    nu_bottom = nu * (1-epr*(0.01))

    print("")
    print("(solve_Kantorovich)")
    print("")
    print("length of mu", M)
    print("length of nu", N)
    print("cost matrix size:", cost.shape)
          
    #### Initialize the cplex class and optimize settings here    
    OT_prob = cplex.Cplex()
    
    if hideOutput: # Suppress CPLEX output
        OT_prob.set_log_stream(None)
        OT_prob.set_error_stream(None)
        OT_prob.set_warning_stream(None)
        OT_prob.set_results_stream(None)
    
    # Set the MILP gap (relative)
    # Determines the acceptable relative gap between the best integer solution found and the best bound.
    # Value range: 0 to 1 (e.g., 0.01 means 1% gap)
    OT_prob.parameters.mip.tolerances.mipgap.set(0.1)  # Relative MIP gap
    # Set absolute MIP gap
    # Determines the acceptable absolute gap between the best integer solution found and the best bound.
    # Value range: Non-negative real numbers (e.g., 0.1 means a gap of 0.1)
    OT_prob.parameters.mip.tolerances.absmipgap.set(0.1)  # Absolute MIP gap
    OT_prob.parameters.timelimit.set(3600)  # 1 hour time limit
    OT_prob.parameters.threads.set(8)  # Use 8 threads
    # Set the MIP emphasis
    # Changes the emphasis of the MIP optimization (0: balance, 1: feasibility, 2: optimality, 3: hidden feasibility, 4: best bound).
    # Value range: Integers from 0 to 4
    OT_prob.parameters.emphasis.mip.set(0)  # Balance optimality and feasibility
    # Set the node limit
    # Sets the maximum number of nodes to explore in the branch-and-bound algorithm.
    # Value range: Positive integers (e.g., 10000 means exploring up to 10,000 nodes)
    OT_prob.parameters.mip.limits.nodes.set(10000)  # Node limit
    # Set heuristic frequency
    # Controls how frequently to apply heuristics during the branch-and-bound search.
    # Value range: Non-negative integers (e.g., 10 means applying heuristics every 10 nodes, 0 means automatic setting)
    OT_prob.parameters.mip.strategy.heuristicfreq.set(10)  # Heuristic frequency
    # Set MIR cut generation level
    # Controls the level of mixed-integer rounding (MIR) cut generation.
    # Value range: Integers from -1 to 2 (-1: automatic, 0: no cuts, 1: moderate, 2: aggressive)
    OT_prob.parameters.mip.cuts.mircut.set(-1)  # automatic
    # Set Gomory cut generation level
    # Controls the level of Gomory cut generation.
    # Value range: Integers from -1 to 2 (-1: automatic, 0: no cuts, 1: moderate, 2: aggressive)
    OT_prob.parameters.mip.cuts.gomory.set(-1)  # Gomory cuts aggressively
    OT_prob.objective.set_sense(OT_prob.objective.sense.minimize)
    #OT_prob.parameters.read.datacheck.set(OT_prob.parameters.read.datacheck.values.off)
    #OT_prob.parameters.emphasis.memory.set = 1
    
    #### Add variables to model and objective functional
    Var_names = ["{0}".format(i) for i in range(M*N)]
    OT_prob.variables.add(obj = np.ravel(cost), names = Var_names) 

    #### Set up constraints
    OT_prob.linear_constraints.add(lin_expr = sparse_constraints_first(M,N), rhs= mu, senses= ["E"] * M)         
    OT_prob.linear_constraints.add(lin_expr = sparse_constraints_second(M,N), rhs= nu_top, senses= ["L"] * N)
    OT_prob.linear_constraints.add(lin_expr = sparse_constraints_second(M,N), rhs= nu_bottom, senses= ["G"] * N)                                               
    #OT_prob.parameters.lpmethod.set(OT_prob.parameters.lpmethod.values.dual)                            
                            
    #### Find the minimizer            
    #OT_prob.parameters.lpmethod.set(1)
    OT_prob.parameters.barrier.display.set(2)
    OT_prob.parameters.simplex.tolerances.optimality.set(1e-9)
    OT_prob.parameters.emphasis.numerical.set(1) #setting this to 1 might improve numerical accuracy, default is zero
    OT_prob.solve()
    
    ####Reversing the normalization: 
    #sol =np.sum(mu)*np.array(OT_prob.solution.get_values())
    sol = np.array(OT_prob.solution.get_values())
    total_cost = OT_prob.solution.get_objective_value()
    
    print('(Kantorovich) Total Cost = ' + str(OT_prob.solution.get_objective_value()))
    #### Starts interactive python terminal
    #import code
    #code.interact(local=locals())
     
    if JustObjValue == True:
        return OT_prob.solution.get_objective_value()
    else: 
        return total_cost, sol

def solve_Dual(cost,mu,nu):
    
    M = len(mu)
    N = len(nu)
                    
    print("")
    print("(solve_Dual)")
    print("")
    
    #### Initialize the cplex class    
    OT_prob = cplex.Cplex()    
    OT_prob.objective.set_sense(OT_prob.objective.sense.maximize)
    
    #### Add variables to model and objective functional
    Objective = []
    for i in range(M):
        Objective.append(mu[i])
    for j in range(N):
        Objective.append(nu[j])
    Var_names = ["{0}".format(i) for i in range(M+N)]
    OT_prob.variables.add(obj = Objective, names = Var_names)
    
    #### Set up constraints        -
    
    OT_prob.linear_constraints.add(lin_expr = sparse_constraints_dual(M,N), rhs= sparse_constraints_dual_rhs(cost,M,N), senses= ["L"] * (N*M))         

    #OT_prob.parameters.lpmethod.set(1)
    OT_prob.parameters.barrier.display.set(2)
    OT_prob.parameters.emphasis.numerical.set(1)
    OT_prob.parameters.simplex.tolerances.optimality.set(1e-9)

    #import code
    #code.interact(local=locals())    

    OT_prob.solve()
    print('(Dual) Total Cost = '+str(OT_prob.solution.get_objective_value()))
    
    return OT_prob.solution.get_values()

def main():
    pass

if __name__ == '__main__':
    main()
