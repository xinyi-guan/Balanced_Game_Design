# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 17:02:34 2025

@author: xinyiguan
"""


#import sys
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

min_candidate_max_delta_pi_vec_org = []
min_candidate_obj_vec_org = []
min_candidate_min_delta_pi_vec_org = []
num_candidate_in_1st_piece_vec_org = []
min_candidate_max_delta_pi_in_1st_piece_vec_org = []
num_unique_candidates_vec_org = []
num_candidates_vec_org = []

#solver_end_time_list = []
best_candidate_vec_org = []


    


with open('multi-scenarios-summary-results-dynamicsearch.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['random_seed', 'avg_max_pi_given_in_solver', 'corresponding_avg_min_pi_given_in_solver', 'max_pi_among_multiscs', 'avg_max_pi_among_multiscs', 'min_pi_among_multiscs'])
    
    for current_seed in seed_vec:
        np.random.seed(current_seed)
        beta_0_sum_base = np.around(np.random.uniform(1.2,2,(num_heros, num_heros)),3)
        beta_k_base = np.around(np.random.uniform(0.1,0.2,(num_features, num_heros, num_heros)),3)
        epsilon_0_sum = np.around(np.random.normal(0,0.02,(num_scenarios,num_heros, num_heros)),3)
        epsilon_k= np.around(np.random.normal(0,0.01,(num_scenarios,num_features, num_heros, num_heros)),3)
        
        beta_0_sum_scs = np.zeros((num_scenarios,num_heros, num_heros))
        beta_k_scs = np.zeros((num_scenarios, num_features, num_heros, num_heros))
        
        for s in range(num_scenarios):
            beta_0_sum_scs[s,:,:] = beta_0_sum_base + epsilon_0_sum[s,:,:]
            beta_k_scs[s,:,:,:] = beta_k_base + epsilon_k[s,:,:,:]
            
        ## piecewise points
        lb_vec_scs = np.array([
                    [[beta_0_sum_scs[s,i,j]- beta_0_sum_scs[s,j,i] - sum(beta_k_scs[s,k,j,i] for k in range(num_features)) 
                     for j in range(num_heros)]
                    for i in range(num_heros)]
                    for s in range(num_scenarios)
        ])

        ub_vec_scs =np.array([
            [[beta_0_sum_scs[s,i,j] + sum( beta_k_scs[s,k,i,j] for k in range(num_features)) - beta_0_sum_scs[s,j,i]
                for j in range(num_heros)]
            for i in range(num_heros)]
            for s in range(num_scenarios)
        ])
        
        z_pieces = np.arange(math.floor(np.min(lb_vec_scs)),0,0.5).tolist() + np.arange(0.5,math.ceil(np.max(ub_vec_scs))+0.5,0.5).tolist()
        
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
        #polish_time_vec = []
        #real_incumbent_obj=[]
        
                
                
                        
        
        
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
                print(f"********** found candidate solution {obj}**************")
                current_incumbent_obj = context.get_incumbent_objective()
                print(f"at candidate status, current incumbent solution {current_incumbent_obj}**************")
                
                ### record candidate solutions
                a_val = np.zeros((num_features, num_heros))
                y_val = np.zeros((num_scenarios,num_heros, num_heros))
                w_val = np.zeros((num_scenarios,num_heros, num_heros))
                #kt_val = np.zeros(num_heros-1)
                
                for k in range(num_features):
                    for i in range(num_heros):
                        a_val[k][i] = round(context.get_candidate_point(f"a_{k}_{i}"))
                        #print(f"********** found candidate solution a design {a_val}**************")
                        
                for s in range(num_scenarios):
                    for i in range(num_heros):
                        for j in range(num_heros):
                            y_val[s][i][j] = context.get_candidate_point(f"y_{s}_{i}_{j}")
                            w_val[s][i][j] = context.get_candidate_point(f"w_{s}_{i}_{j}")
                
                        
                max_delta_pi_list = [] 
                min_delta_pi_list = []
                for s in range(num_scenarios):
                    modified_w=[]
                    for i in range(1,num_heros):
                        modified_w_vec = []
                        for j in range(num_heros):
                            modified_w_vec.append(w_val[s][i][j] - 0.5)
                        
                        modified_w.append(modified_w_vec)
                
                    modified_w.append([1 for j in range(num_heros)])
                    modified_w = np.array(modified_w)
                    
                    b_vector = np.zeros(num_heros)
                    for i in range(1,num_heros):
                        b_vector[i-1]= -(1/num_heros) * sum(y_val[s][i][j] - w_val[s][i][j] for j in range(num_heros))
                    
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
                    print(f"********** found candidate solution {obj}, In scenario {s}:  max delta pi {max_delta_pi}, min delta pi {min_delta_pi}**************")
                    max_delta_pi_list.append(max_delta_pi)
                    min_delta_pi_list.append(min_delta_pi)
                    
                avg_max_delta_pi = np.mean(max_delta_pi_list)
                avg_min_delta_pi = np.mean(min_delta_pi_list)     
                               
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
                        #incumbent_obj_div_singular.append(obj/min_singular_value)
                        #candidate_delta_p.append(delta_p_vec)
                        candidate_max_delta_pi.append(avg_max_delta_pi)
                        candidate_min_delta_pi.append(avg_min_delta_pi)
                        #new_candidate_a_generated = True
                        
                        
                        
                    
                    try_candidate_a = 0
                    if self.best_max_delta_pi > avg_max_delta_pi:
                        self.best_max_delta_pi = avg_max_delta_pi
                        self.best_corresponding_min_delta_pi = avg_min_delta_pi
                        
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
                    if (self.nb_incumbents >= 5) and (avg_max_delta_pi>=min(candidate_max_delta_pi)):
                        context.reject_candidate()
                        #reject_candidate_idx_list.append(self.nb_candidates-1)
                        print(f"at candidate status, rejecting suboptimal candidate solution, current incumbent solution is {context.get_incumbent_objective()}**************")
                    elif (self.nb_incumbents >= 5) and (avg_max_delta_pi==min(candidate_max_delta_pi)) and (incum_seed_vec[len(candidate_a)]<0.99):
                        context.reject_candidate()
                        #reject_candidate_idx_list.append(self.nb_candidates-1)
                        print(f"at candidate status, rejecting equally optimal candidate solution,, current incumbent solution is {context.get_incumbent_objective()}**************")
                    
                    
                    

                    

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
                        
                        LP_y_val = np.zeros((num_scenarios, num_heros, num_heros))
                        for s in range(num_scenarios):
                            for i in range(num_heros):
                                for j in range(num_heros):
                                    LP_y_val[s][i][j] = context.get_relaxation_point(f"y_{s}_{i}_{j}")
                                
                        
                        LP_w_val = np.zeros((num_scenarios, num_heros, num_heros))
                        for s in range(num_scenarios):
                            for i in range(num_heros):
                                for j in range(num_heros):
                                    LP_w_val[s][i][j] = context.get_relaxation_point(f"w_{s}_{i}_{j}")
                
                        normialized_LP_max_delta_pi_list = []
                        LP_max_delta_pi_list = []
                        for s in range(num_scenarios):
                            LP_modified_w=[]
                            for i in range(1,num_heros):
                                LP_modified_w_vec = []
                                for j in range(num_heros):
                                    LP_modified_w_vec.append(LP_w_val[s][i][j] - 0.5)
                        
                                LP_modified_w.append(LP_modified_w_vec)
                
                            LP_modified_w.append([1 for j in range(num_heros)])
                            LP_modified_w = np.array(LP_modified_w)
                            
                            LP_b_vector = np.zeros(num_heros)
                            for i in range(1,num_heros):
                                LP_b_vector[i-1]= -(1/num_heros) * sum(LP_y_val[s][i][j] - LP_w_val[s][i][j] for j in range(num_heros))
                                
                            
                            min_eigen_value = np.min(np.linalg.eig(np.matmul(np.transpose(LP_modified_w),LP_modified_w))[0])
                            if min_eigen_value >= 10**(-14):
                                LP_delta_p_vec = np.matmul(np.linalg.inv(LP_modified_w),LP_b_vector)
                                LP_max_delta_pi = max(LP_delta_p_vec)
                                LP_min_delta_pi = min(LP_delta_p_vec)
                                
                            else:
                                LP_max_delta_pi = float("inf")
                                LP_min_delta_pi = - float("inf")
                                
                            LP_max_delta_pi_list.append(LP_max_delta_pi)
                                
                            
                            
                            LP_b_vars_abs_vector = np.zeros(num_heros-1)
                            for i in range(num_heros-1):
                                LP_b_vars_abs_vector[i] = context.get_relaxation_point(f"b_vars_abs_{s}_{i}")
                                
                            l1_norm_obj_vector = sum(ss for ss in LP_b_vars_abs_vector)
                                
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
                            
                            normialized_LP_max_delta_pi_list.append(normialized_LP_max_delta_pi)
                            
                        avg_LP_max_delta_pi = np.mean(LP_max_delta_pi_list)
                        #avg_LP_min_delta_pi = np.mean(LP_min_delta_pi_list)
                        avg_normialized_LP_max_delta_pi = np.mean(normialized_LP_max_delta_pi_list)
                            
                
                        
                        if (avg_LP_max_delta_pi > 1.5*min(min(candidate_max_delta_pi_snapshot),1.0/num_heros)) or (avg_normialized_LP_max_delta_pi > 5):
                            #print(f"incumbent_max_delta_pi: {self.best_max_delta_pi}; incumbent_min_delta_pi: {self.best_corresponding_min_delta_pi}")
                            print(f"Pruning node with LP_relaxation_obj {LP_obj} (avg_LP_max_delta_pi: {avg_LP_max_delta_pi}; avg_normialized_LP_max_delta_pi is {avg_normialized_LP_max_delta_pi}.)")
                            context.prune_current_node()
                            
            
                    
                    
        
        
        output_filename = f'multi-scenarios-randomseed_{current_seed}_dynamicsearch.txt'
        
        with open(output_filename, 'w') as f:
            with redirect_stdout(f):
                m = Model()
                m.parameters.mip.display.set(4)
                
                m = Model()
                kt = np.zeros((num_scenarios, num_heros-1), dtype=object)
                
                for s in range(num_scenarios):
                    for i in range(num_heros-1):
                        kt[s][i] = m.continuous_var(lb=-1, ub=1, name=f"kt_{s}_{i}")
                        
                b_vars = np.zeros((num_scenarios, num_heros-1), dtype=object)
                b_vars_abs = np.zeros((num_scenarios, num_heros-1), dtype=object)
                for s in range(num_scenarios):
                    for i in range(num_heros-1):
                        b_vars[s][i] = m.continuous_var(lb=-2,name=f"b_vars_{s}_{i}")
                        b_vars_abs[s][i] = m.continuous_var(name=f"b_vars_abs_{s}_{i}")
                
                for s in range(num_scenarios):
                    for i in range(num_heros-1):
                        m.add_constraint(b_vars[s][i] == kt[s][i] - (1/num_heros) * m.sum(kt[s][j] for j in range(num_heros-1) ) )
                        m.add_constraint(b_vars_abs[s][i] >= b_vars[s][i])
                        m.add_constraint(b_vars_abs[s][i] >= -b_vars[s][i])
                
                    
                w = np.zeros((num_scenarios, num_heros, num_heros), dtype=object)
                for s in range(num_scenarios):
                    for i in range(num_heros):
                        for j in range(num_heros):
                            w[s][i][j] = m.continuous_var(lb=0,ub=1,name=f"w_{s}_{i}_{j}")  
                

                y = np.zeros((num_scenarios, num_heros, num_heros), dtype=object)
                for s in range(num_scenarios):
                    for i in range(num_heros):
                        for j in range(num_heros):
                            #y[s][i][j] = m.continuous_var(lb=0.3, ub=0.7, name=f"y_{s}_{i}_{j}") #infeasible
                            y[s][i][j] = m.continuous_var(lb=0.3, ub=0.7, name=f"y_{s}_{i}_{j}") #infeasible
                            #y[s][i][j] = m.continuous_var(lb=0.2, ub=0.8, name=f"y_{s}_{i}_{j}") #infeasible
                

                a = np.zeros((num_features, num_heros), dtype=object)
                for k in range(num_features):
                    for i in range(num_heros):
                        a[k][i] = m.binary_var(name=f"a_{k}_{i}")  

                u = np.zeros((num_scenarios, num_heros, num_heros), dtype=object)
                for s in range(num_scenarios):
                    for i in range(num_heros):
                        for j in range(num_heros):
                            u[s][i][j] = m.continuous_var(lb=None, ub=None) 
                
                    
                z = np.zeros((num_scenarios, num_heros, num_heros), dtype=object)
                for s in range(num_scenarios):
                    for i in range(num_heros):
                        for j in range(num_heros):
                            z[s][i][j] = m.continuous_var(lb = math.floor(np.min(lb_vec_scs)), ub = math.ceil(np.max(ub_vec_scs))) 

                

                for s in range(num_scenarios):
                    for i in range(num_heros):
                        for j in range(i, num_heros):
                            # 1. y[i][j] + y[j][i] == 1
                            # w[i][j] + w[j][i] == 1
                            m.add_constraint(y[s][i][j] + y[s][j][i] == 1, f"mutual_y_exclusive_{s}_{i}_{j}")
                            m.add_constraint(w[s][i][j] + w[s][j][i] == 1, f"mutual_w_exclusive_{s}_{i}_{j}")
                
                for s in range(num_scenarios):   
                    for i in range(num_heros):
                        for j in range(num_heros):         
                            m.add_constraint(u[s][i][j] == m.dot(a[:,i], beta_k_scs[s,:,i,j]) + beta_0_sum_scs[s,i,j], ctname=f"u_def_{s}_{i}_{j}")

                for s in range(num_scenarios):       
                    for i in range(num_heros):
                        for j in range(i+1,num_heros):
                            m.add_constraint(z[s][i][j] == u[s][i][j]-u[s][j][i])
                


                ## model piecewise functions as SOS2 constraints
                weights = np.zeros((num_scenarios, num_heros, num_heros,len(z_pieces)), dtype=object)
                for s in range(num_scenarios):
                    for i in range(num_heros):
                        for j in range(num_heros):
                            for p in range(len(z_pieces)):
                                weights[s][i][j][p] = m.continuous_var(lb=0, ub=1)
                

                # Add constraints for upper triangle (i < j)
                for s in range(num_scenarios):
                    for i in range(num_heros):
                        for j in range(i+1, num_heros):
                            # 1. z[i][j] = sum(z_pieces[p] * weights[i][j][p])
                            m.add_constraint(
                                z[s][i][j] == m.sum(z_pieces[p] * weights[s][i][j][p] for p in range(len(z_pieces))))
                        
                            # 2. y[i][j] = sum(y_pieces[p] * weights[i][j][p])
                            m.add_constraint(
                                y[s][i][j] == m.sum(y_pieces[p] * weights[s][i][j][p] for p in range(len(z_pieces))))
                        
                            # 3. Sum of weights = 1
                            m.add_constraint(
                                m.sum(weights[s][i][j][p] for p in range(len(z_pieces))) == 1)
                        
                            # 4. SOS2 constraint
                            m.add_sos2([weights[s][i][j][p] for p in range(len(z_pieces))])
                
                        
                        
                #Y-W: l2 norm distance
                for s in range(num_scenarios):
                    for i in range(1,num_heros):
                        for j in range(i+1, num_heros+1): #w[i,j]
                            q_c = (i-1)*num_heros - sum(s_2 for s_2 in range(i)) + (j-i) - 1 
                            m.add_constraint(y[s][i-1][j-1] - w[s][i-1][j-1] == m.sum(kt[s][s_2]*A[s_2,q_c] for s_2 in range(num_heros-1)) )

                for s in range(num_scenarios):           
                    for j in range(num_heros):
                        m.add_constraint(m.sum(w[s][i][j] for i in range(num_heros))>=0.5 * num_heros)
                

                #L2 norm
                m.minimize(m.sum((1/num_scenarios) * m.sum(b_vars_abs[s][s_2] for s_2 in range(num_heros-1)) for s in range(num_scenarios)) )
                
                
                solver_start_time = time.time()
                prunecb = IncumbentPruningCallback(solver_start_time)
                
                contextmask = (cplex.callbacks.Context.id.relaxation |
                               cplex.callbacks.Context.id.candidate )
                               


                if contextmask:
                    m.cplex.set_callback(prunecb, contextmask)
                    
                m.parameters.mip.strategy.nodeselect = 2  
                
                m.parameters.timelimit= 3600
                m.parameters.randomseed = 2025 ## reproduction of dynamic search
                m.parameters.parallel = 1  # 1 means deterministic mode
                
                m.solve(log_output=True)
                
                



        
        ######### check results
        min_candidate_max_delta_pi_vec_org.append(min(candidate_max_delta_pi))

        best_idx = np.argmin(np.array(candidate_max_delta_pi))

        

        min_candidate_obj_vec_org.append(candidate_obj[best_idx])

        min_candidate_min_delta_pi_vec_org.append(candidate_min_delta_pi[best_idx])

        first_pieces_indices = np.where(np.array(candidate_min_delta_pi) > -1/num_heros)[0]
        num_candidate_in_1st_piece_vec_org.append(len(np.unique(np.round(np.array(candidate_min_delta_pi)[first_pieces_indices], decimals = 5)) ))


        if num_candidate_in_1st_piece_vec_org[-1] != 0:
            min_candidate_max_delta_pi_in_1st_piece_vec_org.append(min(np.array(candidate_max_delta_pi)[first_pieces_indices]))
            best_first_pieces_idx = first_pieces_indices[np.argmin(np.array(candidate_max_delta_pi)[first_pieces_indices])]
            best_candidate_vec_org.append(candidate_a[best_first_pieces_idx])
        else:
            min_candidate_max_delta_pi_in_1st_piece_vec_org.append(0)
            best_candidate_vec_org.append(np.zeros((num_features, num_heros)))

        


        df = pd.DataFrame(best_candidate_vec_org[-1])
        df.to_csv(f'multi-scenarios-best_candidate_game_design_randomseed_{current_seed}_dynamicsearch.csv', index=False, header=False)

        unique_candidate_a = np.unique(candidate_a, axis=0)
        num_unique_candidates_vec_org.append(len(unique_candidate_a)) 
        num_candidates_vec_org.append(len(candidate_a)) 


        #writer.writerow([current_seed, min_candidate_max_delta_pi_vec_org[-1], min_candidate_min_delta_pi_vec_org[-1], min_candidate_max_delta_pi_in_1st_piece_vec_org[-1], num_unique_candidates_vec_org[-1], num_candidates_vec_org[-1], num_candidate_in_1st_piece_vec_org[-1]])


        print(f"********* Current seed is {current_seed}: best_max_delta_pi is {min_candidate_max_delta_pi_vec_org[-1]}; corresponding min_delta_pi is {min_candidate_min_delta_pi_vec_org[-1]}; best_max_delta_pi_in_1st_piece is {min_candidate_max_delta_pi_in_1st_piece_vec_org[-1]}; num_unique_candidates is {num_unique_candidates_vec_org[-1]}; num_candidates is {num_candidates_vec_org[-1]};num_candidate_in_1st_piece is {num_candidate_in_1st_piece_vec_org[-1]}.********* ")
            
        feasible_a = best_candidate_vec_org[-1]
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
            
            print(f"********* Current seed is {current_seed} and Scenario-{sc}: max pi is {tt.solution_value} , min pi is {feasible_min_pi_list[-1]}.")
        
        mean_feasible_max_pi = np.mean(feasible_max_pi_list)
        max_pi_among_multiscs = max(feasible_max_pi_list)
        min_pi_among_multiscs = min(feasible_min_pi_list)
        print(f"********* Current seed is {current_seed}: best_avg_max_delta_pi is {mean_feasible_max_pi}; max_pi_among_multiscs is {max_pi_among_multiscs}; min_pi_among_multiscs is {min_pi_among_multiscs}.********* ")
        
            
        writer.writerow([current_seed, min_candidate_max_delta_pi_vec_org[-1], min_candidate_min_delta_pi_vec_org[-1], max_pi_among_multiscs, mean_feasible_max_pi, min_pi_among_multiscs])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
