#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 11:59:08 2025

@author: xinyiguan
"""



import gurobipy as gp
from gurobipy import GRB
from gurobipy import *

import sys
import math
import numpy as np
import pandas as pd

import csv


num_heros = 11
num_features = 12


np.random.seed(12345)  
seed_vec = np.unique(np.random.randint(low = 100, high = 9999, size = 10))
#seed_vec = [ 646 2277 3541 3592 4194 4578 4678 6898 7583 7809]

np.random.seed(2025) 
incum_seed_vec = np.random.random(size = 1000000)


theta_sol_vec = []
a_sol_vec = []
mixed_prob_sol_vec = []


# Open the CSV file in write mode
with open('small-scale-summary-results-solve_APX-RJ_in_Gurobi.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['random_seed', 'model_objective_value', 'current_obj', 'true_p1', 'true_p2', 'true_p3', 'true_p4', 'true_p5', 'true_p6', 'true_p7', 'true_p8','true_p9','true_p10', 'true_p11'])

    
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

        output_filename = f'small-scale-randomseed_{current_seed}_solve_APX-RJ_in_Gurobi.txt'

        incumbents_a=[]
        incumbents_y=[]
        incumbents_w=[]
        incumbents_obj=[]
        
        
        
        
        incumbent_max_delta_pi = []
        incumbent_min_delta_pi = []
        
        def gurobi_incumbent_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                
                
                a_val = model.cbGetSolution(a)
                y_val = model.cbGetSolution(y)
                w_val = model.cbGetSolution(w)

                # 
                modified_w = []
                for i in range(1, num_heros):
                    modified_w_vec = []
                    for j in range(num_heros):
                        modified_w_vec.append(w_val[i,j] - 0.5)
                    modified_w.append(modified_w_vec)
                modified_w.append([1 for j in range(num_heros)])
                modified_w = np.array(modified_w)

                b_vector = np.zeros(num_heros)
                for i in range(1, num_heros):
                    b_vector[i-1] = -(1/num_heros) * sum(y_val[i,j] - w_val[i,j] for j in range(num_heros))

                min_eigen_value = np.min(np.linalg.eig(np.matmul(np.transpose(modified_w), modified_w))[0])
                if min_eigen_value >= 1e-14:
                    delta_p_vec = np.matmul(np.linalg.inv(modified_w), b_vector)
                    min_delta_pi = min(delta_p_vec)
                    max_delta_pi = max(delta_p_vec)
                else:
                    delta_p_vec = np.zeros(num_heros)
                    max_delta_pi = float("inf")
                    min_delta_pi = -float("inf")

                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                
                incumbents_a.append(a_val)
                incumbents_y.append(y_val)
                incumbents_w.append(w_val)
                incumbents_obj.append(obj)
                incumbent_max_delta_pi.append(max_delta_pi)
                incumbent_min_delta_pi.append(min_delta_pi)

                # Reject incumbent if condition is met
                if (len(incumbents_a) > 5) and (max_delta_pi > min(incumbent_max_delta_pi)):
                    print("reject suboptimal incumbent")
                    # Add a lazy constraint that is violated by the current solution
                    model.cbLazy(gp.quicksum(
                        (1 - a[k,i]) if a_val[k,i] > 0.5 else a[k,i]
                        for k in range(num_features) for i in range(num_heros)) >= 1)

                elif (len(incumbents_a) > 5) and (incum_seed_vec[len(incumbents_a)] < 0.99) and (max_delta_pi == min(incumbent_max_delta_pi)):
                    print("reject current best incumbent")
                    model.cbLazy(gp.quicksum(
                        (1 - a[k,i]) if a_val[k,i] > 0.5 else a[k,i]
                        for k in range(num_features) for i in range(num_heros)) >= 1)

                
                
                    
                    
        m = gp.Model()
        
        m.setParam('OutputFlag', 1)         # Ensure logging is on
        m.setParam('DisplayInterval', 1)    # Log every node/iteration
        m.setParam('LogToConsole', 1)       # Log to console (default)
        m.setParam('LogFile', output_filename)  # Also log to a file
        
        

        kt = np.empty((num_heros-1), dtype=object)
        for s in range(num_heros-1):
            kt[s] = m.addVar(lb=-1, ub=1)
            
        b_vars = np.empty((num_heros-1), dtype=object) 
        b_vars_abs = np.empty((num_heros-1), dtype=object)        
        for s in range(num_heros-1):             
            b_vars[s] = m.addVar(lb=-2,ub=GRB.INFINITY)
            b_vars_abs[s] = m.addVar(lb=0, ub=GRB.INFINITY)
        for i in range(num_heros-1):                     
            m.addConstr(b_vars[i] == kt[i] - (1/num_heros) * gp.quicksum(kt[j] for j in range(num_heros-1) ) )                     
            m.addConstr(b_vars_abs[i] >= b_vars[i])                     
            m.addConstr(b_vars_abs[i] >= -b_vars[i])
            
        
        
        w = m.addVars(num_heros, num_heros, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        y = m.addVars(num_heros, num_heros, vtype=GRB.CONTINUOUS, lb=0.3, ub=0.7)

        a = m.addVars(num_features, num_heros, vtype=GRB.BINARY)
        

        u = np.empty((num_heros, num_heros), dtype=object)
        for i in range(num_heros):
            for j in range(num_heros):
                u[i, j] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"u_{i}_{j}")

        z = np.empty((num_heros, num_heros), dtype=object)
        for i in range(num_heros):
            for j in range(num_heros):
                z[i, j] = m.addVar(lb=math.floor(np.min(lb_vec)), ub=math.ceil(np.max(ub_vec)), name=f"z_{i}_{j}")

        # Constraints
        for i in range(num_heros):
            for j in range(i, num_heros):
                # 1. y[i][j] + y[j][i] == 1
                m.addConstr(y[i, j] + y[j, i] == 1)
                m.addConstr(w[i, j] + w[j, i] == 1)

        for i in range(num_heros):
            for j in range(num_heros):
                # m.dot(a[:,i], beta_k[:,i,j]) + beta_0_sum[i,j]
                m.addConstr(
                    u[i, j] == gp.quicksum(a[k, i] * beta_k[k, i, j] for k in range(num_features)) + beta_0_sum[i, j],
                    name=f"u_def_{i}_{j}"
                )

        for i in range(num_heros):
            for j in range(i+1, num_heros):
                m.addConstr(z[i, j] == u[i, j] - u[j, i], name=f"z_def_{i}_{j}")

        # Piecewise/SOS2 constraints
        weights = np.empty((num_heros, num_heros, len(z_pieces)), dtype=object)
        for i in range(num_heros):
            for j in range(num_heros):
                for p in range(len(z_pieces)):
                    weights[i, j, p] = m.addVar(lb=0, ub=1, name=f"w_{i}_{j}_{p}")

        for i in range(num_heros):
            for j in range(i+1, num_heros):
                # 1. z[i][j] = sum(z_pieces[p] * weights[i][j][p])
                m.addConstr(
                    z[i, j] == gp.quicksum(z_pieces[p] * weights[i, j, p] for p in range(len(z_pieces))),
                    name=f"z_piecewise_{i}_{j}"
                )
                # 2. y[i][j] = sum(y_pieces[p] * weights[i][j][p])
                m.addConstr(
                    y[i, j] == gp.quicksum(y_pieces[p] * weights[i, j, p] for p in range(len(z_pieces))))
                # 3. Sum of weights = 1
                m.addConstr(
                    gp.quicksum(weights[i, j, p] for p in range(len(z_pieces))) == 1,
                    name=f"weight_sum_{i}_{j}"
                )
                # 4. SOS2 constraint
                m.addSOS(GRB.SOS_TYPE2, [weights[i, j, p] for p in range(len(z_pieces))])

        # Y-W: l2 norm distance 
        for i in range(1, num_heros):
            for j in range(i+1, num_heros+1):
                q_c = (i-1)*num_heros - sum(s for s in range(i)) + (j-i) - 1 
                m.addConstr(y[i-1, j-1] - w[i-1, j-1] == gp.quicksum(kt[s]*A[s, q_c] for s in range(num_heros-1)))

        for j in range(num_heros):
            m.addConstr(gp.quicksum(w[i, j] for i in range(num_heros)) >= 0.5 * num_heros)
            

        # Objective
        m.setObjective(gp.quicksum(b_vars_abs[s] for s in range(num_heros-1)), GRB.MINIMIZE)

        
        m.Params.LazyConstraints = 1
        
        m.setParam('TimeLimit', 3600)

        # Optimize
        
        m.optimize(gurobi_incumbent_callback)
                
        
        
        approx_m_obj = m.ObjVal
         
        
        current_max_pi = float("inf")
        best_a_design = np.zeros((num_features, num_heros))
        true_p1, true_p2, true_p3, true_p4, true_p5, true_p6, true_p7, true_p8, true_p9, true_p10, true_p11 = [0,0,0,0,0,0,0,0,0,0,0]
        current_cor_obj = float("inf")
        
        for s in range(len(incumbents_a)):
            a_sol = incumbents_a[s]
            best_z = np.zeros((num_heros, num_heros))
            best_y = np.zeros((num_heros, num_heros))

            for i in range(num_heros-1):
                for j in range(i+1, num_heros):
                    best_z[i][j] = beta_0_sum[i][j] + sum(beta_k[k,i,j] * a_sol[k,i] for k in range(num_features)) - beta_0_sum[j][i] - sum(beta_k[k,j,i] * a_sol[k,j] for k in range(num_features))
                    best_y[i][j] = 1/(1 + math.exp(-best_z[i][j]))
                    best_y[j][i] = 1 - best_y[i][j]

            for i in range(num_heros):
                best_y[i][i] = 0.5


            m_LP = gp.Model()

            # Variables
            prob = np.empty(num_heros, dtype=object)
            for i in range(num_heros):
                prob[i] = m_LP.addVar(lb=0, name=f"prob_{i}")
            tt = m_LP.addVar(lb=0, name="tt")

            # Constraints
            for j in range(num_heros):
                m_LP.addConstr(gp.quicksum(prob[i] * best_y[i, j] for i in range(num_heros)) >= 0.5, name=f"y_sum_{j}")
                m_LP.addConstr(prob[j] <= tt, name=f"prob_le_tt_{j}")

            m_LP.addConstr(gp.quicksum(prob[i] for i in range(num_heros)) == 1, name="prob_sum")

            # Objective
            m_LP.setObjective(tt, GRB.MINIMIZE)

            # Solve (suppress log output)
            m_LP.Params.OutputFlag = 0
            m_LP.optimize()

            # Extract solution
            true_max_pi = tt.X
            
            if true_max_pi < current_max_pi:
                best_a_design = a_sol
                current_cor_obj = incumbents_obj[s]
                current_max_pi = true_max_pi
                true_p1, true_p2, true_p3, true_p4, true_p5, true_p6, true_p7, true_p8, true_p9, true_p10, true_p11 = [prob[i].X for i in range(num_heros)]
                
        
        writer.writerow([current_seed, approx_m_obj, current_cor_obj, true_p1, true_p2, true_p3, true_p4, true_p5, true_p6, true_p7, true_p8, true_p9, true_p10, true_p11])

        print(f"********* Current seed is {current_seed}: soling status is {m.Status}; approx_m_obj is {approx_m_obj}; current_cor_obj is {current_cor_obj}; true_p1 is {true_p1}; true_p2 is {true_p2}; true_p3 is {true_p3}; true_p4 is {true_p4}; true_p5 is {true_p5}; true_p6 is {true_p6}; true_p7 is {true_p7}.********* ")

                


                
                
                
                