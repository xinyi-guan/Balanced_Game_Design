#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 20:19:05 2025

@author: xinyiguan
"""


import sys
import math
import numpy as np
import pandas as pd
import time

from docplex.mp.model import Model

import cplex

import csv


###large-scale
num_heros = 29
num_features = 29

np.random.seed(23456)  
seed_vec = np.unique(np.random.randint(low = 100, high = 99999, size = 10))
#seed_vec = [6566 12639 14591 53962 60594 81354 88942 93532 98639 98780]

a_sol_vec = []


def solve_McCormick_relax_MILP(beta_0_sum, beta_k, num_heros, num_features, z_lb_vec, z_ub_vec, mix_prob_lb_vec, mix_prob_ub_vec, y_ub_mat, y_lb_mat):
    m = Model()
    theta = m.continuous_var(lb=0, ub=1, name="theta")
    
    mixed_prob = np.zeros((num_heros), dtype=object)
    for i in range(num_heros):
        mixed_prob[i] = m.continuous_var(lb=0,ub=1,name=f"prob_{i}")
        
    for i in range(num_heros):
        m.add_constraint(mixed_prob[i] <= mix_prob_ub_vec[i])
        m.add_constraint(mixed_prob[i] >= mix_prob_lb_vec[i])
        m.add_constraint(mixed_prob[i] <= theta)
        
    m.add_constraint(m.sum(mixed_prob[i] for i in range(num_heros)) == 1)
    
    y = np.zeros((num_heros, num_heros), dtype=object)
    for i in range(num_heros):
        for j in range(num_heros):
            y[i][j] = m.continuous_var(lb=0.3, ub=0.7, name=f"y_{i}_{j}") 
            #y[i][j] = m.continuous_var(lb=0, ub=1, name=f"y_{i}_{j}")  
            
    prod_p_y = np.zeros((num_heros, num_heros), dtype=object)
    for i in range(num_heros):
        for j in range(num_heros):
            prod_p_y[i][j] = m.continuous_var(lb=0, name=f"prod_p_y_{i}_{j}") 
            #prod_p_y[i][j] = m.continuous_var(lb=0, ub=1, name=f"prod_p_y_{i}_{j}") 
            
    for j in range(num_heros):
        m.add_constraint(m.sum(prod_p_y[i][j] for i in range(num_heros) ) >=0.5)
        
    for i in range(num_heros):
        for j in range(num_heros):
            m.add_constraint( prod_p_y[i][j] <= mix_prob_ub_vec[i]*y[i][j] + mixed_prob[i]*y_lb_mat[i][j] - mix_prob_ub_vec[i]*y_lb_mat[i][j])
            m.add_constraint( prod_p_y[i][j] <= mixed_prob[i]*y_ub_mat[i][j] + mix_prob_lb_vec[i]*y[i][j] - mix_prob_lb_vec[i]*y_ub_mat[i][j])
            m.add_constraint( prod_p_y[i][j] >= mix_prob_lb_vec[i]*y[i][j] + mixed_prob[i]*y_lb_mat[i][j] - mix_prob_lb_vec[i]*y_lb_mat[i][j])
            m.add_constraint( prod_p_y[i][j] >= mix_prob_ub_vec[i]*y[i][j] + mixed_prob[i]*y_ub_mat[i][j] - mix_prob_ub_vec[i]*y_ub_mat[i][j])
            
            
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
            z[i][j] = m.continuous_var(lb = math.floor(np.min(z_lb_vec)), ub = math.ceil(np.max(z_ub_vec))) 



    for i in range(num_heros):
        for j in range(i, num_heros):
            # 1. y[i][j] + y[j][i] == 1
            m.add_constraint(y[i][j] + y[j][i] == 1, f"mutual_y_exclusive_{i}_{j}")
            #m.add_constraint(w[i][j] + w[j][i] == 1, f"mutual_w_exclusive_{i}_{j}")
        
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
            
    m.minimize(theta)
    m.parameters.mip.strategy.heuristiceffort = 0  # no heuristic
    m.parameters.mip.strategy.nodeselect = 2
    m.parameters.timelimit = 60*10
    m.solve(log_output=False)
    
    solve_status = m.get_solve_status()
    if (str(solve_status) == 'JobSolveStatus.FEASIBLE_SOLUTION') | (str(solve_status) == 'JobSolveStatus.OPTIMAL_SOLUTION'):
        current_node_LB = theta.solution_value
        current_node_mix_prob_val = [mixed_prob[i].solution_value for i in range(num_heros)]
        current_node_a_val = np.zeros((num_features, num_heros))
        for k in range(num_features):
            for i in range(num_heros):
                current_node_a_val[k][i] = a[k][i].solution_value
                
        current_node_winrate_val = np.zeros((num_heros, num_heros))
        current_node_prod_p_y = np.zeros((num_heros, num_heros))
        #current_node_z = np.zeros((num_heros, num_heros))
        # for i in range(num_heros-1):
        #     for j in range(i+1, num_heros):
        #         current_node_z[i][j] = beta_0_sum[i][j] + sum(beta_k[k,i,j] * current_node_a_val[k][i] for k in range(num_features)) - beta_0_sum[j][i] - sum(beta_k[k,j,i] * current_node_a_val[k][j] for k in range(num_features))
        #         current_node_winrate_val[i][j] = 1/(1 + math.exp(-current_node_z[i][j]))
        #         current_node_winrate_val[j][i] = 1 - current_node_winrate_val[i][j]

        # for i in range(num_heros):
        #     current_node_winrate_val[i][i] = 0.5
        for i in range(num_heros):
             for j in range(num_heros):
                 current_node_winrate_val[i][j] = y[i][j].solution_value
                 current_node_prod_p_y[i][j] = prod_p_y[i][j].solution_value
        
    else:
        current_node_LB = float('inf')
        current_node_mix_prob_val = [0.0 for i in range(num_heros)]
        current_node_a_val = float('inf') * np.ones((num_features, num_heros))
        current_node_winrate_val = float('inf') * np.ones((num_heros, num_heros))
        current_node_prod_p_y = float('inf') * np.ones((num_heros, num_heros))
        
             
    
        
    return current_node_LB, current_node_mix_prob_val, current_node_winrate_val, current_node_prod_p_y, current_node_a_val, solve_status




def solve_zero_sum_game_LP(current_node_winrate_val):
    m_LP = Model()
    prob = np.zeros((num_heros, ), dtype=object)
    for i in range(num_heros):
        prob[i] = m_LP.continuous_var(lb=0)
    tt = m_LP.continuous_var(lb=0)

    for j in range(num_heros):
        m_LP.add_constraint(m_LP.sum(prob[i]*current_node_winrate_val[i,j] for i in range(num_heros))>=0.5)
        m_LP.add_constraint(prob[j]<=tt)
        
    m_LP.add_constraint(m_LP.sum(prob[i] for i in range(num_heros))==1)
    m_LP.minimize(tt)
    m_LP.solve(log_output=False)
    current_node_UB = tt.solution_value 
    return current_node_UB

    

# Open the CSV file in write mode
with open('large-scale-summary-results-spatial_BB.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    
    writer.writerow(['random_seed', 'prob_vec', 'max_pi', 'min_pi'])
    
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
        
        
        #large-scale
        if (math.ceil(np.max(ub_vec)) > 4) and (math.ceil(np.max(ub_vec)) % 2 ==0):
            z_pieces = np.arange(math.floor(np.min(lb_vec)),-4,2).tolist() + np.arange(-4,0,0.5).tolist() + np.arange(0.5,4,0.5).tolist() + np.arange(4,math.ceil(np.max(ub_vec))+0.5,2).tolist()
        elif (math.ceil(np.max(ub_vec)) > 4) and (math.ceil(np.max(ub_vec)) % 2 ==1):
            z_pieces = np.arange(math.floor(np.min(lb_vec))-1,-4,2).tolist() + np.arange(-4,0,0.5).tolist() + np.arange(0.5,4,0.5).tolist() + np.arange(4,math.ceil(np.max(ub_vec))+1.5,2).tolist()
        else:   
            z_pieces = np.arange(math.floor(np.min(lb_vec)),0,0.5).tolist() + np.arange(0.5,math.ceil(np.max(ub_vec))+0.5,0.5).tolist()

        
        y_pieces = [1/(1+math.exp(-i)) for i in z_pieces]
        y_z_slope = [(y_pieces[p+1] - y_pieces[p])/(z_pieces[p+1] - z_pieces[p]) for p in range(len(z_pieces)-1)]

        
        ### get upper and lower bound of y[i][j]
        y_ub_mat = np.zeros((num_heros, num_heros))
        y_lb_mat = np.zeros((num_heros, num_heros))
        for i in range(num_heros):
            for j in range(num_heros):
                y_ub_mat[i][j] = 1 / (1 + math.exp(-ub_vec[i][j]))

                y_lb_mat[i][j] = 1 / (1 + math.exp(-lb_vec[i][j]))

        
        root_node_mix_prob_lb_vec = np.zeros(num_heros)
        root_node_mix_prob_ub_vec = np.ones(num_heros)

        root_node_LB, root_node_mix_prob_val, root_node_winrate_val, root_node_prod_p_y, root_node_a_val, root_node_solving_status  = solve_McCormick_relax_MILP(beta_0_sum, beta_k, num_heros, num_features, lb_vec, ub_vec, root_node_mix_prob_lb_vec, root_node_mix_prob_ub_vec, y_ub_mat, y_lb_mat)

        a_val_Dict = {}
        a_val_Dict[0] = root_node_a_val
        best_a_val = root_node_a_val
        
        LB_list_modified = []
        LB_list_modified.append(root_node_LB)
        current_LB = root_node_LB
        
        ### get current node's UB
        root_node_UB = solve_zero_sum_game_LP(root_node_winrate_val)
        UB_list=[]
        UB_list.append(root_node_UB)
        current_UB = root_node_UB
        
        
        
        ### initialize best UB; initialize incumbent node index as root
        best_UB = float('inf')
        incumbent_node = 0
        
        pi_lb_ub_Dict = {}
        pi_lb_ub_Dict[0] = np.zeros((3,num_heros))
        #pi_lb_ub_Dict[1][0,:] = root_node_mix_prob_lb_vec # lb
        pi_lb_ub_Dict[0][1,:] = root_node_mix_prob_ub_vec  #UB
        pi_lb_ub_Dict[0][2,:] = np.ones((num_heros))
        
        #branching condition
        diff_prod_p_y_and_pi_times_yij = np.zeros(num_heros)
        for i in range(num_heros):
            diff_prod_p_y_and_pi_times_yij[i] = sum(abs(root_node_prod_p_y[i][j]-root_node_mix_prob_val[i]*root_node_winrate_val[i][j]) for j in range(num_heros) )
            
        branchingvar = np.argmax(diff_prod_p_y_and_pi_times_yij) #np.Int
        pi_lb_ub_Dict[0][2,branchingvar] = 0.25*root_node_mix_prob_val[branchingvar] + 0.75*(root_node_mix_prob_lb_vec[branchingvar]+root_node_mix_prob_ub_vec[branchingvar])/2
        
        tol_UB_LB = 0.001
        itercounter = 0
        
        start_time = time.time()
        current_elapsed_time = 0
        
        
        while (min(LB_list_modified) < float('inf')) & (current_elapsed_time <= 3600*2): #large-scale
            itercounter = itercounter + 1
            split_node_index = np.argmin(LB_list_modified) # parent node
            
            parent_node_LB = LB_list_modified[split_node_index]
            parent_node_UB = UB_list[split_node_index]
            if (best_UB - parent_node_LB <= tol_UB_LB):
                # quit loop and return incumbent_node
                break

            if (parent_node_UB < best_UB):
                incumbent_node = split_node_index
                best_UB = parent_node_UB
                best_a_val = a_val_Dict[incumbent_node]
            
            if (best_UB - parent_node_LB > tol_UB_LB):
                #### partition node into two child nodes:  largest_node_index+1 and largest_node_index+2
                #left child node : largest_node_index+1 ; mix_prob[branchingvar] <= ...
                left_node_mix_prob_lb_vec = pi_lb_ub_Dict[split_node_index][0,:]
                left_node_mix_prob_ub_vec = np.min(pi_lb_ub_Dict[split_node_index][1:3,:],axis=0)
                
                imposed_global_mix_prob_ub_vec = best_UB * np.ones(num_heros)
                left_node_mix_prob_ub_vec = np.min(np.stack([left_node_mix_prob_ub_vec, imposed_global_mix_prob_ub_vec], axis = 0),axis=0)
                left_child_index = len(LB_list_modified)
                
                if np.all(left_node_mix_prob_lb_vec <= left_node_mix_prob_ub_vec):
                    pi_lb_ub_Dict[left_child_index] = np.ones((3,num_heros))
                    pi_lb_ub_Dict[left_child_index][0,:] = left_node_mix_prob_lb_vec
                    pi_lb_ub_Dict[left_child_index][1,:] = left_node_mix_prob_ub_vec
                    
                    left_node_LB, left_node_mix_prob_val, left_node_winrate_val, left_node_prod_p_y, left_node_a_val, left_node_solving_status = solve_McCormick_relax_MILP(beta_0_sum, beta_k, num_heros, num_features, lb_vec, ub_vec, left_node_mix_prob_lb_vec, left_node_mix_prob_ub_vec, y_ub_mat, y_lb_mat)
                    
                    if (str(left_node_solving_status) == 'JobSolveStatus.FEASIBLE_SOLUTION') | (str(left_node_solving_status) == 'JobSolveStatus.OPTIMAL_SOLUTION'):
                        a_val_Dict[left_child_index] = left_node_a_val
                        left_node_UB = solve_zero_sum_game_LP(left_node_winrate_val)
                        #branching condition for left node
                        diff_prod_p_y_and_pi_times_yij = np.zeros(num_heros)
                        for i in range(num_heros):
                            diff_prod_p_y_and_pi_times_yij[i] = sum(abs(left_node_prod_p_y[i,j]-left_node_mix_prob_val[i]*left_node_winrate_val[i][j]) for j in range(num_heros))
                        
                        branchingvar = np.argmax(diff_prod_p_y_and_pi_times_yij)
                        cutoff_val = 0.25*left_node_mix_prob_val[branchingvar] + 0.75*(left_node_mix_prob_lb_vec[branchingvar]+left_node_mix_prob_ub_vec[branchingvar])/2
                        pi_lb_ub_Dict[left_child_index][2,branchingvar] = cutoff_val
                        
                        print(f"**** Iteration {itercounter}: left child node branchingvar is {branchingvar}, cutoffvalue is {cutoff_val}.****")
                        
                    else:
                        left_node_LB = float('inf')
                        left_node_UB = float('inf')
                    
                else:
                    left_node_LB = float('inf')
                    left_node_UB = float('inf')
                    
                ### update UB_List and LB_List
                LB_list_modified.append(left_node_LB)
                UB_list.append(left_node_UB)
                
                right_child_index = left_child_index + 1
                #right child node :
                updated_pi_lb_index = np.where(pi_lb_ub_Dict[split_node_index][2,:] < 1)[0][0]
                right_node_mix_prob_lb_vec = pi_lb_ub_Dict[split_node_index][0,:].copy()
                right_node_mix_prob_lb_vec[updated_pi_lb_index] = max(pi_lb_ub_Dict[split_node_index][2,updated_pi_lb_index],right_node_mix_prob_lb_vec[updated_pi_lb_index])
                right_node_mix_prob_ub_vec = pi_lb_ub_Dict[split_node_index][1,:]
                right_node_mix_prob_ub_vec = np.min(np.stack([right_node_mix_prob_ub_vec, imposed_global_mix_prob_ub_vec], axis = 0),axis=0)
                
                if np.all(right_node_mix_prob_lb_vec <= right_node_mix_prob_ub_vec):
                    pi_lb_ub_Dict[right_child_index] = np.ones((3,num_heros))
                    pi_lb_ub_Dict[right_child_index][0,:] = right_node_mix_prob_lb_vec
                    pi_lb_ub_Dict[right_child_index][1,:] = right_node_mix_prob_ub_vec
                    
                    right_node_LB, right_node_mix_prob_val, right_node_winrate_val, right_node_prod_p_y, right_node_a_val, right_node_solving_status = solve_McCormick_relax_MILP(beta_0_sum, beta_k, num_heros, num_features, lb_vec, ub_vec, right_node_mix_prob_lb_vec, right_node_mix_prob_ub_vec, y_ub_mat, y_lb_mat)
                    
                    if (str(right_node_solving_status) == 'JobSolveStatus.FEASIBLE_SOLUTION') | (str(right_node_solving_status) == 'JobSolveStatus.OPTIMAL_SOLUTION'):
                        a_val_Dict[right_child_index] = right_node_a_val
                        right_node_UB = solve_zero_sum_game_LP(right_node_winrate_val)
                        #branching condition for left node
                        diff_prod_p_y_and_pi_times_yij = np.zeros(num_heros)
                        for i in range(num_heros):
                            diff_prod_p_y_and_pi_times_yij[i] = sum(abs(right_node_prod_p_y[i,j]-right_node_mix_prob_val[i]*right_node_winrate_val[i][j]) for j in range(num_heros) )
                        
                        branchingvar = np.argmax(diff_prod_p_y_and_pi_times_yij)
                        cutoff_val = 0.25*right_node_mix_prob_val[branchingvar] + 0.75*(right_node_mix_prob_lb_vec[branchingvar]+right_node_mix_prob_ub_vec[branchingvar])/2
                        
                        pi_lb_ub_Dict[right_child_index][2,branchingvar] = cutoff_val
                        
                        print(f"**** Iteration {itercounter}: right child node branchingvar is {branchingvar}, cutoffvalue is {cutoff_val}.****")
                        
                    else:
                        right_node_LB = float('inf')
                        right_node_UB = float('inf')
                    
                else:
                    right_node_LB = float('inf')
                    right_node_UB = float('inf')

                ### update UB_List and LB_List
                LB_list_modified.append(right_node_LB)
                UB_list.append(right_node_UB)
                
        
            ##remove parent node from LB_list_modified
            LB_list_modified[split_node_index] = float('inf')
            
            current_gap = best_UB - parent_node_LB
            
            print("*************************************************************")
            print(f"**** Iteration {itercounter}: best UB is {best_UB}, current LB is {parent_node_LB}, gap = {current_gap}****")
            print(f"**** Iteration {itercounter}: parent node is {split_node_index}, parent node LB is {parent_node_LB}, parent node UB is {parent_node_UB}****")
            print(f"**** Iteration {itercounter}: left child node is {left_child_index}, left child node LB is {left_node_LB}, left child node UB is {left_node_UB}****")
            print(f"**** Iteration {itercounter}: right child node is {right_child_index}, right child node LB is {right_node_LB}, right child node UB is {right_node_UB}****")
            
            current_elapsed_time = time.time() - start_time
            



            
        ######record results!!!!!!!!!!!!!!!!!!

                
        a_sol_vec.append(best_a_val)
        
        df = pd.DataFrame(best_a_val)
        df.to_csv(f'large-scale-best_incumbent_game_design_randomseed_{current_seed}_spatial_BB.csv', index=False, header=False)
        
        
        ####### calculate zero-sum LP under a_sol
        best_z = np.zeros((num_heros, num_heros))
        best_y = np.zeros((num_heros, num_heros))

        for i in range(num_heros-1):
            for j in range(i+1, num_heros):
                best_z[i][j] = beta_0_sum[i][j] + sum(beta_k[k,i,j] * best_a_val[k][i] for k in range(num_features)) - beta_0_sum[j][i] - sum(beta_k[k,j,i] * best_a_val[k][j] for k in range(num_features))
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
        
        
        max_pi = tt.solution_value 
        prob_vec = [prob[i].solution_value for i in range(num_heros)]

        min_pi = min(prob_vec)
        
        writer.writerow([current_seed, prob_vec, max_pi, min_pi])
        print(f"********* Current seed is {current_seed}: max_pi is {max_pi}; min_pi is {min_pi}.********* ")



