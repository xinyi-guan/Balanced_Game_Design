# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 17:02:34 2025

@author: xinyiguan
"""


import sys
import math
import numpy as np
import pandas as pd
import time

from docplex.mp.model import Model

import cplex

#import cplex.callbacks as cpx_cb

#from docplex.mp.callbacks.cb_mixin import *
import csv

from contextlib import redirect_stdout

import threading

candidate_lock = threading.Lock()



num_heros = 11
num_features = 12
num_scenarios = 100


np.random.seed(12345)  
seed_vec = np.unique(np.random.randint(low = 100, high = 9999, size = 10))


np.random.seed(2025) 
incum_seed_vec = np.random.random(size = 1000000)
np.random.seed(1234)
solution_polish_seed_vec= np.random.random(size = 10000000)


num_a_design_kept_each_scenario = 500
        
        
with open('scenario-decomposition-summary-results-dynamicsearch.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['random_seed', 'max_pi_among_multiscs', 'avg_max_pi_among_multiscs', 'min_pi_among_multiscs'])
    
    for current_seed in seed_vec:
        np.random.seed(current_seed)
        beta_0_sum_base = np.around(np.random.uniform(1.2,2,(num_heros, num_heros)),3)
        beta_k_base = np.around(np.random.uniform(0.1,0.2,(num_features, num_heros, num_heros)),3)
        epsilon_0_sum = np.around(np.random.normal(0,0.02,(num_scenarios,num_heros, num_heros)),3)
        epsilon_k= np.around(np.random.normal(0,0.01,(num_scenarios,num_features, num_heros, num_heros)),3)
        
        beta_0_sum_scs = np.zeros((num_scenarios,num_heros, num_heros))
        beta_k_scs = np.zeros((num_scenarios, num_features, num_heros, num_heros))
        
        multi_scenario_top_candidates_list = []
        
        
        for sc in range(num_scenarios):
            beta_0_sum_scs[sc,:,:] = beta_0_sum_base + epsilon_0_sum[sc,:,:]
            beta_k_scs[sc,:,:,:] = beta_k_base + epsilon_k[sc,:,:,:]
            
            beta_0_sum = beta_0_sum_base + epsilon_0_sum[sc,:,:]
            beta_k = beta_k_base + epsilon_k[sc,:,:,:]
            
            
            
            
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
        
        
            ################################# original
            candidate_a=[]
            #candidate_y=[]
            #candidate_w=[]
            candidate_obj=[]

            #candidate_delta_p = []
            candidate_max_delta_pi = []
            candidate_min_delta_pi = []

            real_incumbent_a=[]
            #real_incumbent_obj=[]
            polish_time_vec = []
                                
                    
                            
            
            
            class IncumbentPruningCallback(object):
                def __init__(self, solver_start_time):
                    self.nb_incumbents = 0
                    self.solver_start_time = solver_start_time
                    self.best_corresponding_min_delta_pi = float('inf')
                    self.best_max_delta_pi = float('inf')
                    
                    

                def invoke(self, context):
                    #self.invoke_count += 1
                    
                    try:
                        # 1. Retrieve incumbent solutions in candidate context
                        if context.in_candidate():
                            self.record_delete_candidate(context)
                            
                        
                        # 2. Prune nodes in relaxation context based on condition
                        
                        if context.in_relaxation():
                            self.myprune_nodes(context)
                            
                            
                    except Exception as e:
                        print(f"Error in callback: {str(e)}")
                        raise

                def record_delete_candidate(self, context):
                    """Record all generated feasible integer solutions"""
                    if not context.is_candidate_point():
                        raise Exception('Unbounded solution')
                        
                    obj = context.get_candidate_objective()
                    #print(f"********** found candidate solution {obj}**************")
                    current_incumbent_obj = context.get_incumbent_objective()
                    print(f"at candidate status, current incumbent solution {current_incumbent_obj}**************")
                    
                    ### record candidate solutions
                    a_val = np.zeros((num_features, num_heros))
                    y_val = np.zeros((num_heros, num_heros))
                    w_val = np.zeros((num_heros, num_heros))
                    #kt_val = np.zeros(num_heros-1)
                    
                    for k in range(num_features):
                        for i in range(num_heros):
                            a_val[k][i] = round(context.get_candidate_point(f"a_{k}_{i}"))
                            #print(f"********** found candidate solution a design {a_val}**************")
                            
                    for i in range(num_heros):
                        for j in range(num_heros):
                            y_val[i][j] = context.get_candidate_point(f"y_{i}_{j}")
                            w_val[i][j] = context.get_candidate_point(f"w_{i}_{j}")
                            
                            
                            
                            
                    modified_w=[]
                    for i in range(1,num_heros):
                        modified_w_vec = []
                        for j in range(num_heros):
                            modified_w_vec.append(w_val[i][j] - 0.5)
                        
                        modified_w.append(modified_w_vec)
                
                    modified_w.append([1 for j in range(num_heros)])
                    modified_w = np.array(modified_w)
                    
                    b_vector = np.zeros(num_heros)
                    for i in range(1,num_heros):
                        b_vector[i-1]= -(1/num_heros) * sum(y_val[i][j] - w_val[i][j] for j in range(num_heros))
                    
                    min_eigen_value = np.min(np.linalg.eig(np.matmul(np.transpose(modified_w),modified_w))[0])
                    if min_eigen_value >= 10**(-14):
                        #min_singular_value = math.sqrt(min_eigen_value)
                        delta_p_vec = np.matmul(np.linalg.inv(modified_w),b_vector)
                        min_delta_pi = min(delta_p_vec)
                        
                        max_delta_pi = max(delta_p_vec)
                        
                    else:
                        #min_singular_value = 10**(-14)
                        delta_p_vec = np.zeros(num_heros)
                        max_delta_pi = float("inf")
                        min_delta_pi = -float("inf")
                    print(f"********** found candidate solution {obj};  max delta pi {max_delta_pi}, min delta pi {min_delta_pi}**************")
                        
                    # All shared data access is protected here
                    with candidate_lock:
                        try_candidate_a = candidate_a.copy()
                        try_candidate_a.append(a_val)
                        if np.unique(try_candidate_a,axis=0).shape[0] > np.unique(candidate_a,axis=0).shape[0]: #it is possible that obj same but a design is different
                            candidate_a.append(a_val)
                            #candidate_y.append(y_val)
                            #candidate_w.append(w_val)
                            candidate_obj.append(obj)
                            #incumbent_singular.append(min_singular_value)
                            
                            candidate_max_delta_pi.append(max_delta_pi)
                            candidate_min_delta_pi.append(min_delta_pi)
                            #new_candidate_a_generated = True
                            
                            
                        
                        try_candidate_a = 0
                        if self.best_max_delta_pi > max_delta_pi:
                            self.best_max_delta_pi = max_delta_pi
                            self.best_corresponding_min_delta_pi = min_delta_pi
                            
                        ### record current incumbent solutions
                        if current_incumbent_obj < 1000: #check whether exists incumbent
                        #The returned value may be a huge value (such as 1e75) to indicate that no incumbent was found yet
                            inc_a_val = np.zeros((num_features, num_heros))
                            
                            
                            for k in range(num_features):
                                for i in range(num_heros):
                                    inc_a_val[k][i] = round(context.get_incumbent(f"a_{k}_{i}"))
                                    #print(f"********** found candidate solution a design {a_val}**************")
                                    
                            
                            
                                
                            try_real_incumbent_a = real_incumbent_a.copy()
                            try_real_incumbent_a.append(inc_a_val)
                            if np.unique(try_real_incumbent_a,axis=0).shape[0] > np.unique(real_incumbent_a,axis=0).shape[0]: #it is possible that obj same but a design is different
                                self.nb_incumbents += 1
                                #print(f"at candidate status, new incumbent solution {current_incumbent_obj}**************")
                                real_incumbent_a.append(inc_a_val)
                                
                                
                            try_real_incumbent_a = 0 
                                
                        ##### reject candidate
                        if (self.nb_incumbents >= 5) and (max_delta_pi>min(candidate_max_delta_pi)):
                            context.reject_candidate()
                            #reject_candidate_idx_list.append(self.nb_candidates-1)
                            print(f"at candidate status, rejecting suboptimal candidate solution, current incumbent solution is {context.get_incumbent_objective()}**************")
                        elif (self.nb_incumbents >= 5) and (max_delta_pi==min(candidate_max_delta_pi)) and (incum_seed_vec[len(candidate_a)]<0.99):
                            context.reject_candidate()
                            #reject_candidate_idx_list.append(self.nb_candidates-1)
                            print(f"at candidate status, rejecting equally optimal candidate colution, current incumbent solution is {context.get_incumbent_objective()}**************")

                    

                        

                def myprune_nodes(self, context):
                    """Prune nodes based on custom condition"""
                    # Only protect the check and read of shared variables with the lock
                    with candidate_lock:
                        nb_incumbents = self.nb_incumbents
                        best_corresponding_min_delta_pi = self.best_corresponding_min_delta_pi
                        #solver_start_time = self.solver_start_time
                        # Make a copy of candidate_max_delta_pi to avoid holding the lock too long
                        candidate_max_delta_pi_snapshot = list(candidate_max_delta_pi)

                    if nb_incumbents>0:                        
                        if (best_corresponding_min_delta_pi >= -1/num_heros): 
                        
                            LP_obj = context.get_relaxation_objective()
                            LP_y_val = np.zeros((num_heros, num_heros))
                            for i in range(num_heros):
                                for j in range(num_heros):
                                    LP_y_val[i][j] = context.get_relaxation_point(f"y_{i}_{j}")
                                    
                            LP_w_val = np.zeros((num_heros, num_heros))
                            for i in range(num_heros):
                                for j in range(num_heros):
                                    LP_w_val[i][j] = context.get_relaxation_point(f"w_{i}_{j}")
                    
                            LP_modified_w=[]
                            for i in range(1,num_heros):
                                LP_modified_w_vec = []
                                for j in range(num_heros):
                                    LP_modified_w_vec.append(LP_w_val[i][j] - 0.5)
                        
                                LP_modified_w.append(LP_modified_w_vec)
                
                            LP_modified_w.append([1 for j in range(num_heros)])
                            LP_modified_w = np.array(LP_modified_w)
                            
                            LP_b_vector = np.zeros(num_heros)
                            for i in range(1,num_heros):
                                LP_b_vector[i-1]= -(1/num_heros) * sum(LP_y_val[i][j] - LP_w_val[i][j] for j in range(num_heros))
                                
                            
                            min_eigen_value = np.min(np.linalg.eig(np.matmul(np.transpose(LP_modified_w),LP_modified_w))[0])
                            if min_eigen_value >= 10**(-14):
                                LP_delta_p_vec = np.matmul(np.linalg.inv(LP_modified_w),LP_b_vector)
                                LP_max_delta_pi = max(LP_delta_p_vec)
                                LP_min_delta_pi = min(LP_delta_p_vec)
                                
                            else:
                                LP_max_delta_pi = float("inf")
                                LP_min_delta_pi = - float("inf")
                                
                            
                            
                            LP_b_vars_abs_vector = np.zeros(num_heros-1)
                            for i in range(num_heros-1):
                                LP_b_vars_abs_vector[i] = context.get_relaxation_point(f"b_vars_abs_{i}")
                                
                            l1_norm_obj_vector = sum(s for s in LP_b_vars_abs_vector)
                            
                            try_LP_max_delta_pi_vec = np.zeros(num_heros)
                                
                            if l1_norm_obj_vector >= 10**(-10):
                                normialized_LP_max_delta_pi = LP_max_delta_pi / l1_norm_obj_vector
                                
                            elif min_eigen_value >= 10**(-14):
                                
                                l1_norm_try_b_vector = (1/num_heros) * ((num_heros-1) + (num_heros-2))
                                try_b_vector = -(1/num_heros) * np.ones(num_heros)
                                try_b_vector[-1] = 0
                                for i in range(1,num_heros):
                                    try_b_vector[i-1] = (1/num_heros) * (num_heros-1)
                                    try_LP_delta_p_vec = np.matmul(np.linalg.inv(LP_modified_w),try_b_vector)
                                    #try_LP_max_delta_pi_vec[i-1] = max(try_LP_delta_p_vec) / l2_norm_try_b_vector
                                    try_LP_max_delta_pi_vec[i-1] = max(try_LP_delta_p_vec) / l1_norm_try_b_vector
                                
                                ## kt is uniform vector
                                try_b_vector = (1/num_heros) * np.ones(num_heros)
                                try_b_vector[-1] = 0
                                
                                l1_norm_try_b_vector = (1/num_heros) * (num_heros-1)
                                try_LP_delta_p_vec = np.matmul(np.linalg.inv(LP_modified_w),try_b_vector)
                                try_LP_max_delta_pi_vec[num_heros-1] = max(try_LP_delta_p_vec) / l1_norm_try_b_vector
                                
                                normialized_LP_max_delta_pi = min(try_LP_max_delta_pi_vec)
                            else: #LP_obj =0 and min_eigen_value = 0
                                normialized_LP_max_delta_pi = float("inf")
                    
                            
                            if (LP_max_delta_pi > 1.5*min(min(candidate_max_delta_pi_snapshot),1.0/num_heros)) or (normialized_LP_max_delta_pi > 5):
                                #print(f"incumbent_max_delta_pi: {self.best_max_delta_pi}; incumbent_min_delta_pi: {self.best_corresponding_min_delta_pi}")
                                print(f"Pruning node with LP_relaxation_obj {LP_obj} (LP_max_delta_pi: {LP_max_delta_pi}; LP_min_delta_pi: {LP_min_delta_pi}, normialized_LP_max_delta_pi is {normialized_LP_max_delta_pi}.)")
                                context.prune_current_node()
                                
                
                        
            
            output_filename = f'Scenario-{sc}-randomseed_{current_seed}_N=11_K=12_dynamicsearch.txt'
            
            with open(output_filename, 'w') as f:
                with redirect_stdout(f):
                    m = Model()
                    m.parameters.mip.display.set(4)
                    
                    kt = np.zeros((num_heros-1), dtype=object)
                    
                    for i in range(num_heros-1):
                        kt[i] = m.continuous_var(lb=-1, ub=1, name=f"kt_{i}") 
                        
                    b_vars = np.zeros((num_heros-1), dtype=object)
                    b_vars_abs = np.zeros((num_heros-1), dtype=object)
                    for i in range(num_heros-1):
                        b_vars[i] = m.continuous_var(lb=-2,name=f"b_vars_{i}")
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
                            y[i][j] = m.continuous_var(lb=0.3, ub=0.7, name=f"y_{i}_{j}") 
                            #y[i][j] = m.addVar(vtype="C", lb=0, ub=1) 

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
                    
                    solver_start_time = time.time()
                    prunecb = IncumbentPruningCallback(solver_start_time)
                    
                    contextmask = (cplex.callbacks.Context.id.relaxation |
                                   cplex.callbacks.Context.id.candidate )
                                   #cplex.callbacks.Context.id.branching)


                    if contextmask:
                        m.cplex.set_callback(prunecb, contextmask)
                        
                    m.parameters.mip.strategy.nodeselect = 2  
                    
                    m.parameters.timelimit= 3600
                    m.parameters.randomseed = 2025 ## reproduction of dynamic search
                    m.parameters.parallel = 1  # 1 means deterministic mode
                    
                    m.solve(log_output=True)
                        
            
            
            
            ## record results
            first_pieces_indices = np.where(np.array(candidate_min_delta_pi) > -1/num_heros)[0]
            num_candidates = len(candidate_min_delta_pi)
            num_candidates_in_1st_piece = len(first_pieces_indices)
            if (num_candidates_in_1st_piece > 0):
                sorted_first_pieces_indices = first_pieces_indices[np.argsort(np.array(candidate_max_delta_pi)[first_pieces_indices])]
                best_max_delta_pi = candidate_max_delta_pi[sorted_first_pieces_indices[0]]
            
            if (num_candidates_in_1st_piece > 0) and (num_candidates_in_1st_piece <= num_a_design_kept_each_scenario):
                multi_scenario_top_candidates_list.extend([candidate_a[first_idx] for first_idx in first_pieces_indices] )
            elif (num_candidates_in_1st_piece > num_a_design_kept_each_scenario):
                multi_scenario_top_candidates_list.extend([candidate_a[first_idx] for first_idx in sorted_first_pieces_indices[0:num_a_design_kept_each_scenario] ] )
            else:
                #top_candidates_vec = [np.zeros((num_features, num_heros))]
                best_max_delta_pi = float('inf')
                
            print(f"********* Current seed is {current_seed} and Scenario-{sc} finishes: best_max_delta_pi_in_1st_piece is {best_max_delta_pi}; num_candidates_in_1st_piece is {num_candidates_in_1st_piece}, num_candidates is {num_candidates}.********* ")

            #multi_scenario_top_candidates_list.append(top_candidates_vec)
        
        ## check all possible solustions
        num_feasible_a_design = len(multi_scenario_top_candidates_list)
        best_avg_max_delta_pi_among_scenarios = float('inf')
        max_pi_among_multiscs = float('inf')
        min_pi_among_multiscs = float('inf')
        best_a_val = np.zeros((num_features, num_heros))
        
        for a_idx in range(num_feasible_a_design):
             feasible_a = multi_scenario_top_candidates_list[a_idx]
             feasible_max_pi_list = []
             feasible_min_pi_list = []
             for sc in range(num_scenarios):
                 ####### calculate zero-sum LP under a_sol
                 best_z = np.zeros((num_heros, num_heros))
                 best_y = np.zeros((num_heros, num_heros))

                 for i in range(num_heros-1):
                     for j in range(i+1, num_heros):
                         best_z[i][j] = beta_0_sum_scs[sc,i,j] + sum(beta_k_scs[sc,k,i,j] * feasible_a[k][i] for k in range(num_features)) - beta_0_sum_scs[sc,j,i] - sum(beta_k_scs[sc,k,j,i] * feasible_a[k][j] for k in range(num_features))
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
                 
                 feasible_max_pi_list.append(tt.solution_value)
                 feasible_min_pi_list.append(min([prob[i].solution_value for i in range(num_heros)]))
             
             mean_feasible_max_pi = np.mean(feasible_max_pi_list)
             
             if (mean_feasible_max_pi < best_avg_max_delta_pi_among_scenarios):
                 best_avg_max_delta_pi_among_scenarios = mean_feasible_max_pi
                 best_a_val = feasible_a
                 max_pi_among_multiscs = max(feasible_max_pi_list)
                 min_pi_among_multiscs = min(feasible_min_pi_list)
                 
        writer.writerow([current_seed, max_pi_among_multiscs, best_avg_max_delta_pi_among_scenarios, min_pi_among_multiscs])
        
        print(f"********* Current seed is {current_seed}: max_pi_among_multiscs is {max_pi_among_multiscs}; best_avg_max_delta_pi_among_scenarios is {best_avg_max_delta_pi_among_scenarios}; min_pi_among_multiscs is {min_pi_among_multiscs}.********* ")

        df = pd.DataFrame(best_a_val)
        df.to_csv(f'scenario-decomposition-best_candidate_game_design_randomseed_{current_seed}-dynamicsearch.csv', index=False, header=False)

        
        
        
        
        
    
    
    
