#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 11:59:08 2025

@author: xinyiguan
"""


# Change the current working directory


import gurobipy as gp
from gurobipy import GRB

import sys
import math
import numpy as np
import pandas as pd
import csv


num_heros = 29
num_features = 29


np.random.seed(23456)  
seed_vec = np.unique(np.random.randint(low = 100, high = 99999, size = 10))
#seed_vec = [6566 12639 14591 53962 60594 81354 88942 93532 98639 98780]


theta_sol_vec = []
a_sol_vec = []
mixed_prob_sol_vec = []


# Open the CSV file in write mode
with open('large-scale-results-restrict_y_ij_0.3-0.7-directly_solve_Gurobi.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['random_seed', 'objective_value', 'true_max_pi', 'true_min_pi'])

    
    for current_seed in seed_vec:
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
        if (math.ceil(np.max(ub_vec)) > 4) and (math.ceil(np.max(ub_vec)) % 2 ==0):
            z_pieces = np.arange(math.floor(np.min(lb_vec)),-4,2).tolist() + np.arange(-4,0,0.5).tolist() + np.arange(0.5,4,0.5).tolist() + np.arange(4,math.ceil(np.max(ub_vec))+0.5,2).tolist()
        elif (math.ceil(np.max(ub_vec)) > 4) and (math.ceil(np.max(ub_vec)) % 2 ==1):
            z_pieces = np.arange(math.floor(np.min(lb_vec))-1,-4,2).tolist() + np.arange(-4,0,0.5).tolist() + np.arange(0.5,4,0.5).tolist() + np.arange(4,math.ceil(np.max(ub_vec))+1.5,2).tolist()
        else:   
            z_pieces = np.arange(math.floor(np.min(lb_vec)),0,0.5).tolist() + np.arange(0.5,math.ceil(np.max(ub_vec))+0.5,0.5).tolist()
        
        y_pieces = [1/(1+math.exp(-i)) for i in z_pieces]

        y_z_slope = [(y_pieces[p+1] - y_pieces[p])/(z_pieces[p+1] - z_pieces[p]) for p in range(len(z_pieces)-1)]


        output_filename = f'large-scale-experiments_{current_seed}_N=29_K=29_solve_non-convex-problem_directly_solve_Gurobi.txt'

        m = gp.Model()
        
        m.setParam('OutputFlag', 1)         # Ensure logging is on
        m.setParam('DisplayInterval', 1)    # Log every node/iteration
        m.setParam('LogToConsole', 1)       # Log to console (default)
        m.setParam('LogFile', output_filename)  # Also log to a file
        
        theta = m.addVar(lb=0, ub=1, name="theta")

        mixed_prob = np.empty(num_heros, dtype=object)
        for i in range(num_heros):
            mixed_prob[i] = m.addVar(lb=0, ub=1, name=f"prob_{i}")

        y = np.empty((num_heros, num_heros), dtype=object)
        for i in range(num_heros):
            for j in range(num_heros):
                y[i, j] = m.addVar(lb=0.3, ub=0.7, name=f"y_{i}_{j}")

        a = np.empty((num_features, num_heros), dtype=object)
        for k in range(num_features):
            for i in range(num_heros):
                a[k, i] = m.addVar(vtype=GRB.BINARY, name=f"a_{k}_{i}")

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
                m.addConstr(y[i, j] + y[j, i] == 1, name=f"mutual_y_exclusive_{i}_{j}")

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
                    y[i, j] == gp.quicksum(y_pieces[p] * weights[i, j, p] for p in range(len(z_pieces))),
                    name=f"y_piecewise_{i}_{j}"
                )
                # 3. Sum of weights = 1
                m.addConstr(
                    gp.quicksum(weights[i, j, p] for p in range(len(z_pieces))) == 1,
                    name=f"weight_sum_{i}_{j}"
                )
                # 4. SOS2 constraint
                m.addSOS(GRB.SOS_TYPE2, [weights[i, j, p] for p in range(len(z_pieces))])

        
        for j in range(num_heros):
            m.addConstr(gp.quicksum(mixed_prob[i] * (y[i, j] - 0.5) for i in range(num_heros)) >= 0, name=f"mixed_prob_y_{j}")
            m.addConstr(mixed_prob[j] <= theta, name=f"mixed_prob_theta_{j}")

        m.addConstr(gp.quicksum(mixed_prob[i] for i in range(num_heros)) == 1, name="mixed_prob_sum")

        # Objective
        m.setObjective(theta, GRB.MINIMIZE)

        # model is non-convex:
        m.Params.NonConvex = 2
        
        m.setParam('TimeLimit', 3600*2)

        # Optimize
        m.optimize()
                
        
        theta_sol = theta.X
        theta_sol_vec.append(theta_sol)
        
        
        a_sol = np.zeros((num_features, num_heros))
        
        for k in range(num_features):
            for i in range(num_heros):
                a_sol[k][i] = a[k, i].X
                
        df = pd.DataFrame(a_sol)
        df.to_csv(f'large_scale-best_incumbent_game_design_y_ij-0.3-0.7_randomseed_{current_seed}_directly_solve_Gurobi.csv', index=False, header=False)
                
        a_sol_vec.append(a_sol)
        
        ####### calculate zero-sum game Eq strategy under a_sol
        best_z = np.zeros((num_heros, num_heros))
        best_y = np.zeros((num_heros, num_heros))

        for i in range(num_heros-1):
            for j in range(i+1, num_heros):
                best_z[i][j] = beta_0_sum[i][j] + sum(beta_k[k,i,j] * a_sol[k][i] for k in range(num_features)) - beta_0_sum[j][i] - sum(beta_k[k,j,i] * a_sol[k][j] for k in range(num_features))
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
        
        true_min_pi = min([prob[i].X for i in range(num_heros)])
        
                
        writer.writerow([current_seed, theta_sol, true_max_pi, true_min_pi])

        
        print(f"********* Current seed is {current_seed}: soling status is {m.Status}; theta_sol is {theta_sol}; true_max_pi is {true_max_pi}; true_min_pi is {true_min_pi}.********* ")

                
                