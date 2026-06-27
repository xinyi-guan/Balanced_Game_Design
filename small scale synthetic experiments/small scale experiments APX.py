#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 10:14:40 2025

@author: xinyiguan
"""


import sys
import math
import numpy as np
import pandas as pd


from docplex.mp.model import Model

import cplex

import csv

from contextlib import redirect_stdout


num_heros = 11
num_features = 12

np.random.seed(12345)  
seed_vec = np.unique(np.random.randint(low = 100, high = 9999, size = 10))
#seed_vec = [ 646 2277 3541 3592 4194 4578 4678 6898 7583 7809]


a_sol_vec = []
mixed_prob_sol_vec = []


# Open the CSV file in write mode
with open('small-scale-summary-results-APX_without_callbacks.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['random_seed', 'objective_value', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11'])

    
    for current_seed in np.flip(seed_vec):
        np.random.seed(current_seed)
        beta_0_sum = np.around(np.random.uniform(1.2,2,(num_heros, num_heros)),3)
        beta_k = np.around(np.random.uniform(0.05*2,0.05*4,(num_features, num_heros, num_heros)),3)


        ## piecewise points
        lb_vec = np.array([
                    [beta_0_sum[i,j]- beta_0_sum[j,i] - sum(beta_k[k,j,i] for k in range(num_features)) 
                     for j in range(num_heros)]
                    for i in range(num_heros)
        ])

        ub_vec =np.array([
            [beta_0_sum[i,j] + sum( beta_k[k,i,j] for k in range(num_features)) - beta_0_sum[j,i]
                for j in range(num_heros)
            ]
            for i in range(num_heros)
        ])

        ## piecewise points 
        z_pieces = np.arange(math.floor(np.min(lb_vec)),0,0.5).tolist() + np.arange(0.5,math.ceil(np.max(ub_vec))+0.5,0.5).tolist()
        
        y_pieces = [1/(1+math.exp(-i)) for i in z_pieces]

        y_z_slope = [(y_pieces[p+1] - y_pieces[p])/(z_pieces[p+1] - z_pieces[p]) for p in range(len(z_pieces)-1)]
        
        A = np.zeros((num_heros-1, int(num_heros*(num_heros-1)/2)))
        for q_r in range(num_heros-1):
            #A[i,]
            j = q_r+1+1 #w_entry j-th column
            if j <= num_heros-1:
                for i in range(1,j):
                    q_c = (i-1)*num_heros - sum(s for s in range(i)) + (j-i) - 1 ##correspond to w[i,j]; 
                    A[q_r,q_c] = 1
                for j_2 in range(j+1,num_heros+1):
                    q_c = (j-1)*num_heros - sum(s for s in range(j)) + (j_2 - j) - 1 ##correspond to w[j,j_2]
                    A[q_r,q_c] = -1
            else:
                for i in range(1,j):
                    q_c = (i-1)*num_heros - sum(s for s in range(i)) + (j-i) - 1 ##correspond to w[i,j]; 
                    A[q_r,q_c] = 1


        output_filename = f'small-scale-randomseed_{current_seed}_APX_without_callbacks.txt'

        with open(output_filename, 'w') as f:
            with redirect_stdout(f):
                m = Model()
                m.parameters.mip.display.set(4)
                
                kt = np.zeros((num_heros-1), dtype=object)
                # kt_abs = np.zeros((num_heros-1), dtype=object)
                for i in range(num_heros-1):
                    kt[i] = m.continuous_var(lb=-1, ub=1, name=f"kt_{i}")
                    
                b_vars = np.zeros((num_heros-1), dtype=object)
                b_vars_abs = np.zeros((num_heros-1), dtype=object)
                for i in range(num_heros-1):
                    b_vars[i] = m.continuous_var(lb = -2, name=f"b_vars_{i}")
                    b_vars_abs[i] = m.continuous_var(name=f"b_vars_abs_{i}")
                for i in range(num_heros-1):
                    m.add_constraint(b_vars[i] == kt[i] - (1/num_heros) * m.sum(kt[j] for j in range(num_heros-1) ) )
                    m.add_constraint(b_vars_abs[i] >= b_vars[i])
                    m.add_constraint(b_vars_abs[i] >= -b_vars[i])
                    
                w = np.zeros((num_heros, num_heros), dtype=object)
                for i in range(num_heros):
                    for j in range(num_heros):
                        w[i][j] = m.continuous_var(lb=0,ub=1,name=f"w_{i}_{j}")  

                y = np.zeros((num_heros, num_heros), dtype=object)
                for i in range(num_heros):
                    for j in range(num_heros):
                        #y[i][j] = m.continuous_var(lb=0.35, ub=0.65, name=f"y_{i}_{j}") 
                        y[i][j] = m.continuous_var(lb=0.3, ub=0.7, name=f"y_{i}_{j}") 

                a = np.zeros((num_features, num_heros), dtype=object)
                for k in range(num_features):
                    for i in range(num_heros):
                        a[k][i] = m.binary_var(name=f"a_{k}_{i}")  

                u = np.zeros((num_heros, num_heros), dtype=object)
                for i in range(num_heros):
                    for j in range(num_heros):
                        u[i][j] = m.continuous_var(lb=None, ub=None) 
                    
                z = np.zeros((num_heros, num_heros), dtype=object)
                for i in range(num_heros):
                    for j in range(num_heros):
                        z[i][j] = m.continuous_var(lb = math.floor(np.min(lb_vec)), ub = math.ceil(np.max(ub_vec))) 



                for i in range(num_heros):
                    for j in range(i, num_heros):
                        # 1. y[i][j] + y[j][i] == 1
                        # w[i][j] + w[j][i] == 1
                        m.add_constraint(y[i][j] + y[j][i] == 1, f"mutual_y_exclusive_{i}_{j}")
                        m.add_constraint(w[i][j] + w[j][i] == 1, f"mutual_w_exclusive_{i}_{j}")
                    
                for i in range(num_heros):
                    for j in range(num_heros):         
                        m.add_constraint(u[i][j] == m.dot(a[:,i], beta_k[:,i,j]) + beta_0_sum[i,j], ctname=f"u_def_{i}_{j}")
                    
                for i in range(num_heros):
                    for j in range(i+1,num_heros):
                        m.add_constraint(z[i][j] == u[i][j]-u[j][i])


                ## model piecewise functions as SOS2 constraints
                weights = np.zeros((num_heros, num_heros,len(z_pieces)), dtype=object)
                for i in range(num_heros):
                    for j in range(num_heros):
                        for p in range(len(z_pieces)):
                            weights[i][j][p] = m.continuous_var(lb=0, ub=1, name=f"w_{i}_{j}_{p}")

                # Add constraints for upper triangle (i < j)
                for i in range(num_heros):
                    for j in range(i+1, num_heros):
                        # 1. z[i][j] = sum(z_pieces[p] * weights[i][j][p])
                        m.add_constraint(
                            z[i][j] == m.sum(z_pieces[p] * weights[i][j][p] for p in range(len(z_pieces))),
                            ctname=f"z_piecewise_{i}_{j}"
                        )
                    
                        # 2. y[i][j] = sum(y_pieces[p] * weights[i][j][p])
                        m.add_constraint(
                            y[i][j] == m.sum(y_pieces[p] * weights[i][j][p] for p in range(len(z_pieces))),
                            ctname=f"y_piecewise_{i}_{j}"
                            )
                    
                        # 3. Sum of weights = 1
                        m.add_constraint(
                            m.sum(weights[i][j][p] for p in range(len(z_pieces))) == 1,
                            ctname=f"weight_sum_{i}_{j}"
                            )
                    
                        # 4. SOS2 constraint
                        m.add_sos2([weights[i][j][p] for p in range(len(z_pieces))])
                        
                        
                #Y-W: l2 norm distance
                for i in range(1,num_heros):
                    for j in range(i+1, num_heros+1): #w[i,j]
                        q_c = (i-1)*num_heros - sum(s for s in range(i)) + (j-i) - 1 
                        m.add_constraint(y[i-1][j-1] - w[i-1][j-1] == m.sum(kt[s]*A[s,q_c] for s in range(num_heros-1)) )
                    
                for j in range(num_heros):
                    m.add_constraint(m.sum(w[i,j] for i in range(num_heros))>=0.5 * num_heros)

                #L2 norm
                #m.minimize(m.sum(kt[s]**2 for s in range(num_heros-1)))
                m.minimize(m.sum(b_vars_abs[s] for s in range(num_heros-1)))
                

                #solver_start_time = time.time()        

                m.parameters.mip.strategy.heuristiceffort = 0 #no heuristic
                
                m.parameters.mip.strategy.nodeselect = 2
                
                m.parameters.mip.limits.nodes = 500000
                
                m.solve(log_output=True)
               


        ######record results
        
        obj_val = m.objective_value

        
        a_sol = np.zeros((num_features, num_heros))
        for k in range(num_features):
            for i in range(num_heros):
                a_sol[k][i] = a[k][i].solution_value
                
        df = pd.DataFrame(a_sol)
        df.to_csv(f'small-scale-best_incumbent_game_design-randomseed_{current_seed}_APX_without_callbacks.csv', index=False, header=False)
                
        a_sol_vec.append(a_sol)
        
        ####### calculate zero-sum LP under a_sol
        best_z = np.zeros((num_heros, num_heros))
        best_y = np.zeros((num_heros, num_heros))

        for i in range(num_heros-1):
            for j in range(i+1, num_heros):
                best_z[i][j] = beta_0_sum[i][j] + sum(beta_k[k,i,j] * a_sol[k][i] for k in range(num_features)) - beta_0_sum[j][i] - sum(beta_k[k,j,i] * a_sol[k][j] for k in range(num_features))
                best_y[i][j] = 1/(1 + math.exp(-best_z[i][j]))
                best_y[j][i] = 1 - best_y[i][j]

        for i in range(num_heros):
            best_y[i][i] = 0.5


        m_LP = Model()
        prob = np.zeros((num_heros, ), dtype=object)
        for i in range(num_heros):
            prob[i] = m_LP.continuous_var(lb=0)
        tt = m_LP.continuous_var(lb=0)

        for j in range(num_heros):
            m_LP.add_constraint(m_LP.sum(prob[i]*best_y[i,j] for i in range(num_heros))>=0.5)
            m_LP.add_constraint(prob[j]<=tt)
            
        m_LP.add_constraint(m_LP.sum(prob[i] for i in range(num_heros))==1)
        m_LP.minimize(tt)
        m_LP.solve(log_output=False) 
        #true_max_pi = tt.solution_value 
        true_p1, true_p2, true_p3, true_p4, true_p5, true_p6, true_p7, true_p8, true_p9, true_p10, true_p11 = [prob[i].solution_value for i in range(num_heros)]
        
        writer.writerow([current_seed, obj_val, true_p1, true_p2, true_p3, true_p4, true_p5, true_p6, true_p7, true_p8, true_p9, true_p10, true_p11])

        print(f"********* Current seed is {current_seed}: soling status is {m.get_solve_status()}; obj_val is {obj_val}; true_p1 is {true_p1}; true_p2 is {true_p2}; true_p3 is {true_p3}; true_p4 is {true_p4}; true_p5 is {true_p5}; true_p6 is {true_p6}; true_p7 is {true_p7}.********* ")

        
        
        
        
        
        
        
        

