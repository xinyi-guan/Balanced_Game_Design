# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 15:34:00 2025

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


num_heros = 29
num_features = 29

np.random.seed(23456)  
seed_vec = np.unique(np.random.randint(low = 100, high = 99999, size = 10))
#seed_vec = [6566 12639 14591 53962 60594 81354 88942 93532 98639 98780]

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

best_candidate_vec_org = []




# Open the CSV file in write mode
with open('large-scale-summary-results-10-instances-APX-RJ-PN-solution-polishing.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['random_seed', 'best_max_delta_pi', 'correspond_min_delta_pi', 'best_max_delta_pi_1st_piece', 'num_unique_candidates', 'num_candidates', 'num_candidate_1st_piece'])
    
    for current_seed in np.flip(seed_vec)[0:1]:
        np.random.seed(current_seed)
        beta_0_sum = np.around(np.random.uniform(1.2,2,(num_heros, num_heros)),3)
        beta_k = np.around(np.random.uniform(0.1,0.2,(num_features, num_heros, num_heros)),3)


        
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

        ## piecewise points for logistic function    
        if (math.ceil(np.max(ub_vec)) > 4) and (math.ceil(np.max(ub_vec)) % 2 ==0):
            z_pieces = np.arange(math.floor(np.min(lb_vec)),-4,2).tolist() + np.arange(-4,0,0.5).tolist() + np.arange(0.5,4,0.5).tolist() + np.arange(4,math.ceil(np.max(ub_vec))+0.5,2).tolist()
        elif (math.ceil(np.max(ub_vec)) > 4) and (math.ceil(np.max(ub_vec)) % 2 ==1):
            z_pieces = np.arange(math.floor(np.min(lb_vec))-1,-4,2).tolist() + np.arange(-4,0,0.5).tolist() + np.arange(0.5,4,0.5).tolist() + np.arange(4,math.ceil(np.max(ub_vec))+1.5,2).tolist()
        else:   
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
        candidate_a=[] # maintain all candidate solutions identified by record_delete_candidate function       
        candidate_obj=[]
        candidate_max_delta_pi = []
        candidate_min_delta_pi = []

        real_incumbent_a=[]  # maintain all incumbent solutions identified by record_delete_candidate function  
        #real_incumbent_obj=[]
        
        ## build sub-MIP for solution polishing
        def solve_sub_mip(fixed_rows, fixed_cols, current_a):
            sub_m = Model()
            sub_kt = np.zeros((num_heros-1), dtype=object)
            
            for i in range(num_heros-1):
                sub_kt[i] = sub_m.continuous_var(lb=-1, ub=1)
                            
            
            sub_b_vars = np.zeros((num_heros-1), dtype=object)
            sub_b_vars_abs = np.zeros((num_heros-1), dtype=object)
            for i in range(num_heros-1):
                sub_b_vars[i] = sub_m.continuous_var(lb=-2)
                sub_b_vars_abs[i] = sub_m.continuous_var()
            for i in range(num_heros-1):
                sub_m.add_constraint(sub_b_vars[i] == sub_kt[i] - (1/num_heros) * sub_m.sum(sub_kt[j] for j in range(num_heros-1) ) )
                sub_m.add_constraint(sub_b_vars_abs[i] >= sub_b_vars[i])
                sub_m.add_constraint(sub_b_vars_abs[i] >= -sub_b_vars[i])
                

                
            sub_w = np.zeros((num_heros, num_heros), dtype=object)
            for i in range(num_heros):
                for j in range(num_heros):
                    sub_w[i][j] = sub_m.continuous_var(lb=0,ub=1)  

            sub_y = np.zeros((num_heros, num_heros), dtype=object)
            for i in range(num_heros):
                for j in range(num_heros):
                    sub_y[i][j] = sub_m.continuous_var(lb=0.3, ub=0.7) 
                    #y[i][j] = m.addVar(vtype="C", lb=0, ub=1) 

            sub_a = np.zeros((num_features, num_heros), dtype=object)
            for k in range(num_features):
                for i in range(num_heros):
                    sub_a[k][i] = sub_m.binary_var()
                    
            
            if len(fixed_rows) > 0:
                for r, c in zip(fixed_rows, fixed_cols): 
                    sub_m.add_constraint( sub_a[r][c] == current_a[r][c] )

            sub_u = np.zeros((num_heros, num_heros), dtype=object)
            for i in range(num_heros):
                for j in range(num_heros):
                    sub_u[i][j] =sub_m.continuous_var(lb=None, ub=None) 
                
            sub_z = np.zeros((num_heros, num_heros), dtype=object)
            for i in range(num_heros):
                for j in range(num_heros):
                    sub_z[i][j] = sub_m.continuous_var(lb = math.floor(np.min(lb_vec)), ub = math.ceil(np.max(ub_vec))) 



            for i in range(num_heros):
                for j in range(i, num_heros):
                    # 1. y[i][j] + y[j][i] == 1
                    # w[i][j] + w[j][i] == 1
                    sub_m.add_constraint(sub_y[i][j] + sub_y[j][i] == 1)
                    sub_m.add_constraint(sub_w[i][j] + sub_w[j][i] == 1)
                
            for i in range(num_heros):
                for j in range(num_heros):         
                    sub_m.add_constraint(sub_u[i][j] == sub_m.dot(sub_a[:,i], beta_k[:,i,j]) + beta_0_sum[i,j])
                
            for i in range(num_heros):
                for j in range(i+1,num_heros):
                    sub_m.add_constraint(sub_z[i][j] == sub_u[i][j]-sub_u[j][i])


            ## model piecewise functions as SOS2 constraints
            sub_weights = np.zeros((num_heros, num_heros,len(z_pieces)), dtype=object)
            for i in range(num_heros):
                for j in range(num_heros):
                    for p in range(len(z_pieces)):
                        sub_weights[i][j][p] = sub_m.continuous_var(lb=0, ub=1, name=f"w_{i}_{j}_{p}")

            # Add constraints for upper triangle (i < j)
            for i in range(num_heros):
                for j in range(i+1, num_heros):
                    # 1. z[i][j] = sum(z_pieces[p] * weights[i][j][p])
                    sub_m.add_constraint(
                        sub_z[i][j] == sub_m.sum(z_pieces[p] * sub_weights[i][j][p] for p in range(len(z_pieces))) )    

                
                    # 2. y[i][j] = sum(y_pieces[p] * weights[i][j][p])
                    sub_m.add_constraint(
                        sub_y[i][j] == sub_m.sum(y_pieces[p] * sub_weights[i][j][p] for p in range(len(z_pieces))) )
                
                    # 3. Sum of weights = 1
                    sub_m.add_constraint(
                        sub_m.sum(sub_weights[i][j][p] for p in range(len(z_pieces))) == 1 )
                
                    # 4. SOS2 constraint
                    sub_m.add_sos2([sub_weights[i][j][p] for p in range(len(z_pieces))])
                    
                    
            #Y-W: l2 norm distance
            for i in range(1,num_heros):
                for j in range(i+1, num_heros+1): #w[i,j]
                    sub_q_c = (i-1)*num_heros - sum(s for s in range(i)) + (j-i) - 1 
                    sub_m.add_constraint(sub_y[i-1][j-1] - sub_w[i-1][j-1] == sub_m.sum(sub_kt[s]*A[s,sub_q_c] for s in range(num_heros-1)) )
                
            for j in range(num_heros):
                sub_m.add_constraint(sub_m.sum(sub_w[i,j] for i in range(num_heros))>=0.5 * num_heros)

                        
            sub_m.minimize(sub_m.sum(sub_b_vars_abs[s] for s in range(num_heros-1)))
            
            sub_m.parameters.mip.strategy.nodeselect = 2
            sub_m.parameters.mip.limits.nodes = 500
            sub_m.solve(log_output=False)
            #print(f"solving sub_mip status: {sub_m.get_solve_status()}")
            if (str(sub_m.get_solve_status()) == 'JobSolveStatus.FEASIBLE_SOLUTION') or (str(sub_m.get_solve_status()) == 'JobSolveStatus.OPTIMAL_SOLUTION'):
                sub_mip_obj = sub_m.objective_value
                                
                a_val = np.zeros((num_features, num_heros))
                y_val = np.zeros((num_heros, num_heros))
                w_val = np.zeros((num_heros, num_heros))
                
                
                for k in range(num_features):
                    for i in range(num_heros):
                        a_val[k][i] = round(sub_a[k][i].solution_value)
                        #print(f"********** found candidate solution a design {a_val}**************")
                        
                for i in range(num_heros):
                    for j in range(num_heros):
                        y_val[i][j] = sub_y[i][j].solution_value
                        w_val[i][j] = sub_w[i][j].solution_value
                        
    
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
                #print(f"min_eigen_value: {min_eigen_value}")
                if min_eigen_value >= 10**(-14):
                    delta_p_vec = np.matmul(np.linalg.inv(modified_w),b_vector)
                    sub_m_min_delta_pi = min(delta_p_vec)
                    sub_m_max_delta_pi = max(delta_p_vec)
                    
                else:
                    #delta_p_vec = np.zeros(num_heros)
                    sub_m_max_delta_pi = float("inf")
                    sub_m_min_delta_pi = -float("inf")
                
            else:
                sub_mip_obj = float('inf')
                sub_m_max_delta_pi = float('inf')
                sub_m_min_delta_pi = - float('inf')
                a_val = np.zeros((num_features, num_heros))
                
            return sub_m_max_delta_pi, sub_m_min_delta_pi, a_val, sub_mip_obj
                
                
                        
        
        
        class IncumbentPruningCallback(object):
            def __init__(self, solver_start_time):
                self.nb_incumbents = 0
                self.solver_start_time = solver_start_time
                self.best_corresponding_min_delta_pi = float('inf')
                self.best_max_delta_pi = float('inf')
                self.top_P_candidate_a = []
                self.largest_max_delta_pi_in_top_P_candidate_a = 0
                self.idx_largest_max_delta_pi_in_top_P_candidate_a = 0
                self.top_P_candidate_max_delta_pi = []
                self.polish_time = 0
                self.num_new_a_previous_polishing = 0
                self.num_solution_polishing = 1
                

            def invoke(self, context):
                #self.invoke_count += 1
                
                try:
                    # 1. Retrieve incumbent solutions in candidate context
                    if context.in_candidate():
                        self.record_delete_candidate(context)
                        
                    
                    # 2. Prune nodes in relaxation context based on condition
                    
                    if context.in_relaxation():
                        self.myprune_nodes(context)
                        self.my_solution_polishing(context)
                        
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
                #print(f"at candidate status, current incumbent solution {current_incumbent_obj}**************")
                
                ### record candidate solutions
                a_val = np.zeros((num_features, num_heros))
                y_val = np.zeros((num_heros, num_heros))
                w_val = np.zeros((num_heros, num_heros))
                
                
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
                        candidate_obj.append(obj)
                        
                        candidate_max_delta_pi.append(max_delta_pi)
                        candidate_min_delta_pi.append(min_delta_pi)
                        #new_candidate_a_generated = True
                        
                        if (len(self.top_P_candidate_a)<40):
                            self.top_P_candidate_a.append(a_val)
                            self.top_P_candidate_max_delta_pi.append(max_delta_pi)
                            if (self.largest_max_delta_pi_in_top_P_candidate_a < max_delta_pi):
                                self.largest_max_delta_pi_in_top_P_candidate_a = max_delta_pi
                                self.idx_largest_max_delta_pi_in_top_P_candidate_a = len(self.top_P_candidate_a) - 1
                        elif (self.largest_max_delta_pi_in_top_P_candidate_a > max_delta_pi):
                            self.top_P_candidate_a[self.idx_largest_max_delta_pi_in_top_P_candidate_a] = a_val
                            self.top_P_candidate_max_delta_pi[self.idx_largest_max_delta_pi_in_top_P_candidate_a] = max_delta_pi
                            self.largest_max_delta_pi_in_top_P_candidate_a = max(self.top_P_candidate_max_delta_pi)
                            self.idx_largest_max_delta_pi_in_top_P_candidate_a = np.argmax(self.top_P_candidate_max_delta_pi)
                            
                            
                            
                    
                        
                        
                    
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
                                
                        
                            
                        try_real_incumbent_a = real_incumbent_a.copy()
                        try_real_incumbent_a.append(inc_a_val)
                        if np.unique(try_real_incumbent_a,axis=0).shape[0] > np.unique(real_incumbent_a,axis=0).shape[0]: #it is possible that obj same but a design is different
                            self.nb_incumbents += 1
                            #print(f"at candidate status, new incumbent solution {current_incumbent_obj}**************")
                            real_incumbent_a.append(inc_a_val)
                            
                            
                        try_real_incumbent_a = 0 
                            
                    
                    # ##### reject candidate
                    if (self.nb_incumbents >= 20) and (max_delta_pi>min(candidate_max_delta_pi)):
                        context.reject_candidate()
                        
                        print(f"at candidate status, rejecting suboptimal candidate solution, current incumbent solution is {context.get_incumbent_objective()}**************")
                    elif (self.nb_incumbents >= 20) and (max_delta_pi==min(candidate_max_delta_pi)) and (incum_seed_vec[len(candidate_a)]<0.5):
                        context.reject_candidate()
                        
                        print(f"at candidate status, rejecting equally optimal candidate colution, current incumbent solution is {context.get_incumbent_objective()}**************")


                    

            def myprune_nodes(self, context):
                """Prune nodes based on custom condition"""
                # Only protect the check and read of shared variables with the lock
                with candidate_lock:
                    nb_incumbents = self.nb_incumbents
                    best_corresponding_min_delta_pi = self.best_corresponding_min_delta_pi
                    
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
                
                        
                        if (LP_max_delta_pi > 1.5*min(min(candidate_max_delta_pi_snapshot),1.0/num_heros)) or (normialized_LP_max_delta_pi > 50):
                            
                            print(f"Pruning node with LP_relaxation_obj {LP_obj} (LP_max_delta_pi: {LP_max_delta_pi}; LP_min_delta_pi: {LP_min_delta_pi}, normialized_LP_max_delta_pi is {normialized_LP_max_delta_pi}.)")
                            context.prune_current_node()
                            
            
            def my_solution_polishing(self, context):
                new_a_from_polishing_list = []
                
                new_max_delta_pi_from_polishing_list = []
                
                with candidate_lock:
                    solver_start_time = self.solver_start_time
                    #candidate_a_snapshot = list(candidate_a)
                    top_P_candidate_a_snapshot = self.top_P_candidate_a 
                    largest_max_delta_pi_in_top_P_candidate_a_snapshot = self.largest_max_delta_pi_in_top_P_candidate_a 
                    idx_largest_max_delta_pi_in_top_P_candidate_a_snapshot = self.idx_largest_max_delta_pi_in_top_P_candidate_a 
                    top_P_candidate_max_delta_pi_snapshot = self.top_P_candidate_max_delta_pi 
                
                self.polish_time = self.polish_time + 1 
                if (len(top_P_candidate_a_snapshot) >= 40) and (time.time() - solver_start_time >= 60*60) and (solution_polish_seed_vec[self.polish_time]>0.8) and (self.num_solution_polishing <= 50): 
                   self.num_solution_polishing = self.num_solution_polishing + 1
                   print(f"my_solution_polishing starts. current largest_max_delta_pi_in_top_P_candidate_a is {largest_max_delta_pi_in_top_P_candidate_a_snapshot}. ") 
                   # Mutation—repeat 20 times
                   freq = 0.5
                   increment = 0.2
                   for __ in range(20):
                       rand_int = np.random.randint(0, 40)
                       seed_a = top_P_candidate_a_snapshot[rand_int]
                       seed_a_max_delta_pi = top_P_candidate_max_delta_pi_snapshot[rand_int]
                       num_fixed_elements = math.floor(num_heros * num_features * freq)
                       # Flatten the array to work with 1D indices
                       flat_indices = np.arange(seed_a.size)
                       
                       selected_flat_indices = np.random.choice(flat_indices, size=num_fixed_elements, replace=False)
                       # Convert flat indices to 2D indices
                       selected_rows, selected_cols = np.unravel_index(selected_flat_indices, seed_a.shape)
                       sub_m_max_delta_pi, sub_m_min_delta_pi, sub_m_a, sub_m_obj = solve_sub_mip(selected_rows, selected_cols, seed_a)
                       print(f"in mutation phase: sub_m_max_delta_pi is {sub_m_max_delta_pi}; sub_m_obj is {sub_m_obj}")
                       
                       with candidate_lock: 
                           try_candidate_a = candidate_a.copy()
                           try_candidate_a.append(sub_m_a)
                           if np.unique(try_candidate_a,axis=0).shape[0] > np.unique(candidate_a,axis=0).shape[0]: #it is possible that obj same but a design is different
                               candidate_a.append(sub_m_a)
                               candidate_obj.append(sub_m_obj)
                               candidate_max_delta_pi.append(sub_m_max_delta_pi)
                               candidate_min_delta_pi.append(sub_m_min_delta_pi)
                               
                               try_candidate_a = 0
                               
                               if (largest_max_delta_pi_in_top_P_candidate_a_snapshot > sub_m_max_delta_pi):
                                   new_a_from_polishing_list.append(sub_m_a)
                                   new_max_delta_pi_from_polishing_list.append(sub_m_max_delta_pi)
                                   
                                   top_P_candidate_a_snapshot[idx_largest_max_delta_pi_in_top_P_candidate_a_snapshot] = sub_m_a
                                   top_P_candidate_max_delta_pi_snapshot[idx_largest_max_delta_pi_in_top_P_candidate_a_snapshot] = sub_m_max_delta_pi
                                   largest_max_delta_pi_in_top_P_candidate_a_snapshot = max(top_P_candidate_max_delta_pi_snapshot)
                                   idx_largest_max_delta_pi_in_top_P_candidate_a_snapshot = np.argmax(top_P_candidate_max_delta_pi_snapshot)
                                   
                                   
                       #update freq: compared with seed solution
                       if (seed_a_max_delta_pi == sub_m_max_delta_pi):
                           #freq decrease
                           if (freq - increment >0.002):
                               freq = freq - increment
                           print(f"freq is {freq}; freq is decreasing")
                       elif (seed_a_max_delta_pi < sub_m_max_delta_pi):
                           #freq increase
                           if (freq + increment < 0.998):
                               freq = freq + increment
                           print(f"freq is {freq}; freq is increasing")
                           
                       # update increment
                       if (increment * 0.5 >= 0.01):
                           increment = increment * 0.5
                       else:
                           increment = 0.01
                       
                   
                   print(f"my_solution_polishing-Mutation ends. # of new_a is {len(new_a_from_polishing_list)}. current largest_max_delta_pi_in_top_P_candidate_a is {largest_max_delta_pi_in_top_P_candidate_a_snapshot}.") 
                   
                   if (len(new_a_from_polishing_list)>5) or (self.num_new_a_previous_polishing>5): # have enough variation in top_P_candidate_a
                       # Combination—repeat 40 times
                       for iterid in range(40):
                           if (iterid < 39): # pairs of solution
                               rand_int1, rand_int2 = np.random.choice(np.arange(40), size=2, replace=False)
                               
                               seed_a_1 = top_P_candidate_a_snapshot[rand_int1]
                               seed_a_2 = top_P_candidate_a_snapshot[rand_int2]
                               selected_rows, selected_cols = np.where(seed_a_1 == seed_a_2) ##it is possible that selected_rows = array([], dtype=int64)
                               sub_m_max_delta_pi, sub_m_min_delta_pi, sub_m_a, sub_m_obj = solve_sub_mip(selected_rows, selected_cols, seed_a_1)
                               print(f"in combination phase: sub_m_max_delta_pi is {sub_m_max_delta_pi}; sub_m_obj is {sub_m_obj}")
                           else:
                               stacked = np.stack(top_P_candidate_a_snapshot)
                               # Find indices where all elements are equal across arrays
                               selected_rows, selected_cols = np.where(np.all(stacked == stacked[0], axis=0))
                               sub_m_max_delta_pi, sub_m_min_delta_pi, sub_m_a, sub_m_obj = solve_sub_mip(selected_rows, selected_cols, top_P_candidate_a_snapshot[1])
                               print(f"in combination phase: sub_m_max_delta_pi is {sub_m_max_delta_pi}; sub_m_obj is {sub_m_obj}")
                        
                           with candidate_lock:                     
                               try_candidate_a = candidate_a.copy()
                               try_candidate_a.append(sub_m_a)
                               if np.unique(try_candidate_a,axis=0).shape[0] > np.unique(candidate_a,axis=0).shape[0]: #it is possible that obj same but a design is different
                                   candidate_a.append(sub_m_a)
                                   candidate_obj.append(sub_m_obj)
                                   candidate_max_delta_pi.append(sub_m_max_delta_pi)
                                   candidate_min_delta_pi.append(sub_m_min_delta_pi) 
                                   
                                   try_candidate_a = 0
                                   
                                   if (largest_max_delta_pi_in_top_P_candidate_a_snapshot > sub_m_max_delta_pi):
                                       new_a_from_polishing_list.append(sub_m_a)
                                       new_max_delta_pi_from_polishing_list.append(sub_m_max_delta_pi)
                                       
                                       top_P_candidate_a_snapshot[idx_largest_max_delta_pi_in_top_P_candidate_a_snapshot] = sub_m_a
                                       
                                       top_P_candidate_max_delta_pi_snapshot[idx_largest_max_delta_pi_in_top_P_candidate_a_snapshot] = sub_m_max_delta_pi
                                       
                                       largest_max_delta_pi_in_top_P_candidate_a_snapshot = max(top_P_candidate_max_delta_pi_snapshot)
                                       idx_largest_max_delta_pi_in_top_P_candidate_a_snapshot = np.argmax(top_P_candidate_max_delta_pi_snapshot)

                               
                           
                   
                  
                   with candidate_lock:
                       for iidx in range(len(new_a_from_polishing_list)):
                           if (self.largest_max_delta_pi_in_top_P_candidate_a > new_max_delta_pi_from_polishing_list[iidx]):
                               self.top_P_candidate_a[self.idx_largest_max_delta_pi_in_top_P_candidate_a] = new_a_from_polishing_list[iidx]
                               self.top_P_candidate_max_delta_pi[self.idx_largest_max_delta_pi_in_top_P_candidate_a] = new_max_delta_pi_from_polishing_list[iidx]
                               self.largest_max_delta_pi_in_top_P_candidate_a = max(self.top_P_candidate_max_delta_pi)
                               self.idx_largest_max_delta_pi_in_top_P_candidate_a = np.argmax(self.top_P_candidate_max_delta_pi)
                       
                       
                   print(f"my_solution_polishing ends. # of new_a is {len(new_a_from_polishing_list)}. current largest_max_delta_pi_in_top_P_candidate_a is {largest_max_delta_pi_in_top_P_candidate_a_snapshot}.")      
                   #self.polish_time = self.polish_time + 1 
                   self.num_new_a_previous_polishing = len(new_a_from_polishing_list)
                    
                    
                    
                
                
        
        output_filename = f'large-scale-randomseed_{current_seed}_APX-RJ-PN-solution-polishing.txt'
        
        with open(output_filename, 'w') as f:
            with redirect_stdout(f):
                m = Model()
                m.parameters.mip.display.set(4)
                
                kt = np.zeros((num_heros-1), dtype=object)
                #kt_abs = np.zeros((num_heros-1), dtype=object)
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

                
                m.minimize(m.sum(b_vars_abs[s] for s in range(num_heros-1)))
                
                solver_start_time = time.time()
                prunecb = IncumbentPruningCallback(solver_start_time)
                
                contextmask = (cplex.callbacks.Context.id.relaxation |
                               cplex.callbacks.Context.id.candidate )
                               


                if contextmask:
                    m.cplex.set_callback(prunecb, contextmask)
                    
                m.parameters.mip.strategy.nodeselect = 2  
                
                m.parameters.timelimit= 3600*2
                m.parameters.randomseed = 2025 ## reproduction of dynamic search
                m.parameters.parallel = 1  # 1 means deterministic mode
                
                m.solve(log_output=True)
        
            
            
        ######### check results
        min_candidate_max_delta_pi_vec_org.append(min(candidate_max_delta_pi))

        best_idx = np.argmin(np.array(candidate_max_delta_pi))

        

        min_candidate_obj_vec_org.append(candidate_obj[best_idx])

        min_candidate_min_delta_pi_vec_org.append(candidate_min_delta_pi[best_idx])

        first_pieces_indices = np.where(np.array(candidate_min_delta_pi) > -1/num_heros)[0]
        
        num_candidate_in_1st_piece_vec_org.append(len(first_pieces_indices))

        
        if len(first_pieces_indices) != 0:
            min_candidate_max_delta_pi_in_1st_piece_vec_org.append(min(np.array(candidate_max_delta_pi)[first_pieces_indices]))
            best_first_pieces_idx = first_pieces_indices[np.argmin(np.array(candidate_max_delta_pi)[first_pieces_indices])]
            best_candidate_vec_org.append(candidate_a[best_first_pieces_idx])
        else:
            min_candidate_max_delta_pi_in_1st_piece_vec_org.append(0)
            best_candidate_vec_org.append(np.zeros((num_features, num_heros)))

        


        df = pd.DataFrame(best_candidate_vec_org[-1])
        df.to_csv(f'large-scale-best_candidate_game_design_randomseed_{current_seed}-APX-RJ-PN-solution-polishing.csv', index=False, header=False)

        unique_candidate_a = np.unique(candidate_a, axis=0)
        num_unique_candidates_vec_org.append(len(unique_candidate_a)) 
        num_candidates_vec_org.append(len(candidate_a)) 


        writer.writerow([current_seed, min_candidate_max_delta_pi_vec_org[-1], min_candidate_min_delta_pi_vec_org[-1], min_candidate_max_delta_pi_in_1st_piece_vec_org[-1], num_unique_candidates_vec_org[-1], num_candidates_vec_org[-1], num_candidate_in_1st_piece_vec_org[-1]])


        print(f"********* Current seed is {current_seed}: best_max_delta_pi is {min_candidate_max_delta_pi_vec_org[-1]}; corresponding min_delta_pi is {min_candidate_min_delta_pi_vec_org[-1]}; best_max_delta_pi_in_1st_piece is {min_candidate_max_delta_pi_in_1st_piece_vec_org[-1]}; num_unique_candidates is {num_unique_candidates_vec_org[-1]}; num_candidates is {num_candidates_vec_org[-1]};num_candidate_in_1st_piece is {num_candidate_in_1st_piece_vec_org[-1]}.********* ")




