# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 21:52:07 2025

@author: xinyiguan
"""


import sys
import math
import numpy as np
import pandas as pd
import random
import time

from docplex.mp.model import Model

import copy

import csv


num_heros = 11
num_features = 12

np.random.seed(12345)  
seed_vec = np.unique(np.random.randint(low = 100, high = 9999, size = 10))
#seed_vec = [ 646 2277 3541 3592 4194 4578 4678 6898 7583 7809]

iter_time_limit = 3600

a_sol_vec = []

def indices_by_distance_from_target(lst):
    target = 1 / len(lst)
    unique_values = set(lst)
    # Sort unique values by their distance from target, descending
    sorted_values = sorted(unique_values, key=lambda x: abs(x - target), reverse=True)
    result = []
    for val in sorted_values:
        indices = [i for i, x in enumerate(lst) if x == val]
        result.append(indices)
    return result


def solve_zero_sum_game_LP(beta_0_sum, beta_k, a_design, num_heros, num_features):
    z_mat = np.zeros((num_heros, num_heros))
    y_mat = np.zeros((num_heros, num_heros))

    for i in range(num_heros-1):
        for j in range(i+1, num_heros):
            z_mat[i][j] = beta_0_sum[i][j] + sum(beta_k[k,i,j] * a_design[k][i] for k in range(num_features)) - beta_0_sum[j][i] - sum(beta_k[k,j,i] * a_design[k][j] for k in range(num_features))
            y_mat[i][j] = 1/(1 + math.exp(-z_mat[i][j]))
            y_mat[j][i] = 1 - y_mat[i][j]

    for i in range(num_heros):
        y_mat[i][i] = 0.5


    m_LP = Model()
    prob = np.zeros((num_heros, ), dtype=object)
    for i in range(num_heros):
        prob[i] = m_LP.continuous_var(lb=0)
    tt = m_LP.continuous_var(lb=0)

    for j in range(num_heros):
        m_LP.add_constraint(m_LP.sum(prob[i]*y_mat[i,j] for i in range(num_heros))>=0.5)
        m_LP.add_constraint(prob[j]<=tt)
        
    m_LP.add_constraint(m_LP.sum(prob[i] for i in range(num_heros))==1)
    m_LP.minimize(tt)
    m_LP.solve(log_output=False) 
    
    mixed_prob_vec = [prob[i].solution_value for i in range(num_heros)]
    
    
    largest_prob_indices = [i for i, x in enumerate(mixed_prob_vec) if x == max(mixed_prob_vec)]
    
    smallest_prob_indices = [i for i, x in enumerate(mixed_prob_vec) if x == min(mixed_prob_vec)]
    
    
    return largest_prob_indices, smallest_prob_indices, mixed_prob_vec

    

with open('small-scale-summary-results-local_search.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['random_seed', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11'])   
    
    for current_seed in np.flip(seed_vec):
        np.random.seed(current_seed)
        beta_0_sum = np.around(np.random.uniform(1.2,2,(num_heros, num_heros)),3)
        beta_k = np.around(np.random.uniform(0.05*2,0.05*4,(num_features, num_heros, num_heros)),3)
        
        a_design_list = []
        max_pi_list = []
        start_time = time.time()
        current_elapsed_time = 0
        
        iter_count = 0
        current_a_design = np.zeros((num_features, num_heros))
        a_design_list.append(current_a_design)
        current_largest_prob_indices, current_smallest_prob_indices, current_mixed_prob_vec = solve_zero_sum_game_LP(beta_0_sum, beta_k, current_a_design, num_heros, num_features)
        current_max_pi = max(current_mixed_prob_vec)
        max_pi_list.append(current_max_pi)
        print("*************************************************************")
        print(f"**** Iteration {iter_count}: current_largest_prob_indices is {current_largest_prob_indices}, current_max_pi is {current_max_pi}.****")
        
        while (current_elapsed_time <= iter_time_limit) and (current_max_pi > 1/num_heros):
            iter_count = iter_count + 1
                        
            whether_change = False
            
            visit_indices_list = indices_by_distance_from_target(current_mixed_prob_vec)
            inferior_indices_list = []
            inferior_max_pi_list = []
            
            for i in range(len(visit_indices_list)):
                current_visit_indices_list = visit_indices_list[i]
                while (not whether_change) and (len(current_visit_indices_list) >= 1):
                    #change a_design of character No.current_largest_prob_idx:
                    current_idx = random.choice(current_visit_indices_list)
                    current_visit_indices_list.remove(current_idx)
                    
                    
                    try_change_resulting_max_pi = []
                    try_change_resulting_prob_vec = []
                    #try_change_resulting_largest_prob_indices = []
                    #try_change_resulting_smallest_prob_indices = []
                    for f_idx in range(num_features):
                        try_current_a_design = copy.deepcopy(current_a_design)
                        try_current_a_design[f_idx][current_idx] = 1 - try_current_a_design[f_idx][current_idx]
                        try_largest_prob_indices, try_smallest_prob_indices, try_mixed_prob_vec = solve_zero_sum_game_LP(beta_0_sum, beta_k, try_current_a_design, num_heros, num_features)
                        try_change_resulting_max_pi.append(max(try_mixed_prob_vec))
                        try_change_resulting_prob_vec.append(try_mixed_prob_vec)
                        #try_change_resulting_largest_prob_indices.append(try_largest_prob_indices)
                        #try_change_resulting_smallest_prob_indices.append(try_smallest_prob_indices)
                    
                    #change_f_idx = np.argmin(try_change_resulting_max_pi)
                    change_f_idx_list =[i for i, x in enumerate(try_change_resulting_max_pi) if x == min(try_change_resulting_max_pi)]
                    for i in range(len(change_f_idx_list)):
                        change_f_idx = change_f_idx_list[i]
                        try_current_a_design = copy.deepcopy(current_a_design)
                        try_current_a_design[change_f_idx][current_idx] = 1 - current_a_design[change_f_idx][current_idx]
                        # Check if new_a_design visisted before
                        exists = any(np.array_equal(try_current_a_design, arr) for arr in a_design_list)
                    
                        if (not exists) and (try_change_resulting_max_pi[change_f_idx] <= current_max_pi):
                            whether_change = True
                            current_a_design = copy.deepcopy(try_current_a_design)
                            a_design_list.append(current_a_design)
                            # update
                            current_max_pi = try_change_resulting_max_pi[change_f_idx]
                            max_pi_list.append(current_max_pi)
                            current_mixed_prob_vec = try_change_resulting_prob_vec[change_f_idx]
                            #current_largest_prob_indices = try_change_resulting_largest_prob_indices[change_f_idx]
                            #current_smallest_prob_indices = try_change_resulting_smallest_prob_indices[change_f_idx]
                            current_elapsed_time = time.time() - start_time
                            print("*************************************************************")
                            print(f"**** Iteration {iter_count}: change_f_idx is {change_f_idx}. After changing, current_max_pi is {current_max_pi}.****")
                            break
                        elif (not exists) and (try_change_resulting_max_pi[change_f_idx] > current_max_pi):
                            inferior_indices_list.append([change_f_idx,current_idx])
                            inferior_max_pi_list.append(try_change_resulting_max_pi[change_f_idx])


                if (whether_change):
                    break

        
            if (not whether_change):
                if (len(inferior_max_pi_list)>=1):
                    j = np.argmin(inferior_max_pi_list)
                    row_idx, col_idx = inferior_indices_list[j]
                
                    current_a_design[row_idx][col_idx] = 1 - current_a_design[row_idx][col_idx]               
                    whether_change = True
                    a_design_list.append(current_a_design)
                    #update
                    current_largest_prob_indices, current_smallest_prob_indices, current_mixed_prob_vec = solve_zero_sum_game_LP(beta_0_sum, beta_k, current_a_design, num_heros, num_features)
                    current_max_pi = max(current_mixed_prob_vec)
                    max_pi_list.append(current_max_pi)
                    current_elapsed_time = time.time() - start_time
                    print("*************************************************************")
                    print(f"**** Iteration {iter_count}: select idx is {[row_idx,col_idx]}. After changing, current_max_pi is {current_max_pi}.****")
                else:
                    while (not whether_change):                        
                        ### randomnly change a_design
                        i = np.random.randint(num_features)
                        j = np.random.randint(num_heros)
                        try_current_a_design = copy.deepcopy(current_a_design)
                        try_current_a_design[i][j] = 1 - current_a_design[i][j]
                        # Check if new_a_design visisted before
                        exists = any(np.array_equal(try_current_a_design, arr) for arr in a_design_list)
                        
                        if (not exists):
                            whether_change = True
                            current_a_design = copy.deepcopy(try_current_a_design)
                            a_design_list.append(current_a_design)
                            #update
                            current_largest_prob_indices, current_smallest_prob_indices, current_mixed_prob_vec = solve_zero_sum_game_LP(beta_0_sum, beta_k, current_a_design, num_heros, num_features)
                            current_max_pi = max(current_mixed_prob_vec)
                            max_pi_list.append(current_max_pi)
                            current_elapsed_time = time.time() - start_time
                            print("*************************************************************")
                            print(f"**** Iteration {iter_count}: randomly change idx is {[i,j]}. After changing, current_max_pi is {current_max_pi}.****")

                    
    
                    
        ######record results
        a_sol_vec.append(current_a_design)
        
        #### find the best a_design with smallest max pi
        best_a_idx = np.argmin(np.array(max_pi_list))
        best_a = a_design_list[best_a_idx]
        
        df = pd.DataFrame(current_a_design)
        df.to_csv(f'small-scale-greedy_last_found_game_design_randomseed_{current_seed}_local_search.csv', index=False, header=False)
        
        df = pd.DataFrame(best_a)
        df.to_csv(f'small-scale-best_greedy_game_design_randomseed_{current_seed}_local_search.csv', index=False, header=False)
        
        best_largest_prob_indices, best_smallest_prob_indices, best_mixed_prob_vec = solve_zero_sum_game_LP(beta_0_sum, beta_k, best_a, num_heros, num_features)
        true_p1, true_p2, true_p3, true_p4, true_p5, true_p6, true_p7, true_p8, true_p9, true_p10, true_p11 = best_mixed_prob_vec
        
        writer.writerow([current_seed, true_p1, true_p2, true_p3, true_p4, true_p5, true_p6, true_p7, true_p8, true_p9, true_p10, true_p11])

        print(f"********* Current seed is {current_seed}: true_p1 is {true_p1}; true_p2 is {true_p2}; true_p3 is {true_p3}; true_p4 is {true_p4}; true_p5 is {true_p5}; true_p6 is {true_p6}; true_p7 is {true_p7}.********* ")

    
