import code
import sys
import numpy as np

import matplotlib.pyplot as plt
from scipy import sparse

"""
This code sets up the scheme of Caffarelli-Kochengin-Oliker (CKO) to approximate solutions to the optimal transportation problem.

We refer also to work of Abedin-Gutierrez which extends the CKO scheme to Generated Jacobian Equations.
"""

def PreImagePop(i,b,f,c): 
    #This computes the measure of the preimage of i
    #i is an integer, b a 1-d array of length K, 0<=i<K, 
    #and f is a non-negative 1-d array of length N, c is a (N,K) array
    K = len(b)
    N = len(f)
    V_i = set()
    
    for x in range(N):
        Aff = []
        for j in range(K):
            Aff.append(c[x,j]-b[j])
        if Aff[i] == min(Aff):
            V_i.add(x)
            
    Measure = 0.0
    
    for x in V_i:
        Measure = Measure + f[x]
    
    return Measure

def UpdateCoord(i,b,f,g,c,delta):
    # This effectively executes the going from b^i to b^(i+1)
    # i is an integer
    # b a 1-d array of length K, and i<K-1
    
    b_bar = 0 #Initialization
    
    if PreImagePop(i,b,f,c) >= g[i]-delta:  
        b_bar = b[i]
    if PreImagePop(i,b,f,c) < g[i]-delta:
        b_bar = FindBetaBar(i,b,f,g[i],c,delta)
        # You need to understand and quantify the step where you pick b_bar. 
        # This will require thinking more seriously about the nature of the scheme,
        # particularly in the fully discrete setting.  
    b[i] = b_bar
    
    return b
    
def FindBetaBar(i,b,f,g_i,c,delta,b_top = 10000):
    # We start with h_top already preset
    # and known to be such that V(b_i) < g_i-delta and
    # V(h_top) > g_i+delta. We compute V at the average of b_i and h_top,
    # if V is < g_i, we let b[i] = average,
    # if V is > g_i+delta, we let h_top = average,
    # this makes b_i and h_top approach each other geometrically until (possibly)
    # h_top-h_min < L^{-1}delta, where L is the Lipschitz constant of the V_i
    # In this case V_max-V_min < delta. If V_min or V_max is in [g_i,g_i+delta], you are done.
    # There is one thing that can prevent the above from working out: V_i has jumps of size larger than delta. 
    
    V_i = PreImagePop(i,b,f,c)
    b_temp = b
    b_bot = b[i]
    b_temp[i] = b_top
    V_bot = V_i
    V_top = PreImagePop(i,b_temp,f,c)    
                        
    if i==0:
        return b[i]
        
    else:    
        while V_top-V_bot>delta and b_top-b_bot>1:
            b_temp[i]= (0.5)*b_bot + (0.5)*b_top
            V_temp = PreImagePop(i,b_temp,f,c)
            #print('PASS')

            if V_temp < g_i:
                V_bot = V_temp
                b_bot = b_temp[i]
            if V_temp > g_i+delta:
                V_top = V_temp
                b_top = b_temp[i]
            else:
                break
            
        if b_top-b_bot <= 1 and V_top-V_bot>delta:
            print('WARNING: V_{0} has jumps larger than {1}'.format(i,delta))
            b_temp[i] = b_bot
        
        return b_temp[i]

def FindBetaBar2(i,b,f,g_i,c,delta,b_top = 10000):
    
    if b[i]<b[0]-30:
        return FindBetaBar(i,b,f,g_i,c,delta)
    else:
        print('Using FindBetaBar2')
        b[i] = b[i]+1
        if PreImagePop(i,b,f,c)<g_i+delta:
            return b[i]
        else:
            return b[i]-1
    
    
    '''V_i = PreImagePop(i,b,f,c)
    b_temp = b
    b_bot = b[i]
    b_temp[i] = b_top
    V_bot = V_i
    V_top = PreImagePop(i,b_temp,f,c)    
                        
    if i==0:
        return b[i]
        
    else:    
        while V_top-V_bot>delta and b_top-b_bot>1:
            b_temp[i]= (0.5)*b_bot + (0.5)*b_top
            V_temp = PreImagePop(i,b_temp,f,c)
            #print('PASS')

            if V_temp < g_i:
                V_bot = V_temp
                b_bot = b_temp[i]
            if V_temp > g_i+delta:
                V_top = V_temp
                b_top = b_temp[i]
            else:
                break
            
        if b_top-b_bot <= 1 and V_top-V_bot>delta:
            print('WARNING: V_{0} has jumps larger than {1}'.format(i,delta))
            b_temp[i] = b_bot
        
        return b_temp[i]'''
    
    
# Initially, you take b = (beta_0,0,...,0), with beta_0 a large negative number such that H_i(b) = 0 for all i>0.
# W_delta is a nonempty set, defined by the properties:
# b_1 = beta_0,
# H_i(b) <= f_i + delta for each i (i= 1,...,K-1)

# The idea is to have H_i(b) <= f_i + delta and >= f_i - delta for all i
# If for a given i we have H_i(b) < f_i-delta then we must increase b_i somewhat in order to increase H_i(b)
# (all while) the other H_i'(b) possibly decrease

# Here is how we update b_i:
# If we have H_i(b) < f_i-delta then
# we let b_star be the largest* number in (0,b_i) such that  *or as large as possible
# changing the i-th coordinate of b to b_star (and calling the new vector b_bar)
# we have H_i(b_bar) >= f_i
# Now, the key thing that we need to check happens 
# (which may not happen for arbitrary small delta in the fully discrete case)
# is that we also have
# H_i(b_bar) < f_i+delta. But this is ok, for now.

def CKOscheme(b,f,g,c,delta):
    b_seq = []
    CKO_stops = False
    i = 0
    while CKO_stops == False:
        b_seq = np.zeros( (len(f),len(b)))
        
        i = 0
        for j in range(len(b)):
            b = UpdateCoord(j,b,f,g,c,delta)
            b_seq[j] = b       
            sys.stdout.write("\rIteration #{0} ".format(i))
            sys.stdout.flush()
            i = i+1
             
        print('')
        #for j in range(len(b)):
        #    print(b_seq[j])       
        print(b_seq[len(b)-1])
        print('')        
        
        counter = 0
        for j in range(len(b)):
            if (b_seq[j] == b).all():
                counter = counter + 1
                
        #print(counter)
        if counter == len(b):
            CKO_stops = True

    return b
    
def main():
    pass

if __name__ == '__main__':
    main()
