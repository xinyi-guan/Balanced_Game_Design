#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:46:59 2025

@author: xinyiguan
"""


import sys
import math
import numpy as np
import pandas as pd
import random
import time

from docplex.mp.model import Model

import cplex

import cplex.callbacks as cpx_cb

from docplex.mp.callbacks.cb_mixin import *

from threading import Lock

import csv



num_heros = 11
num_features = 12

np.random.seed(12345)  
seed_vec = np.unique(np.random.randint(low = 100, high = 9999, size = 10))
#seed_vec = [ 646 2277 3541 3592 4194 4578 4678 6898 7583 7809]


np.random.seed(2025) 
incum_seed_vec = np.random.random(size = 1000000)


np.random.seed(6789) 
nodeselection_seed_vec = np.random.random(size = 8000000)


min_incumbent_max_delta_pi_vec_org = []
min_incumbent_obj_vec_org = []
min_incumbent_min_delta_pi_vec_org = []
num_incumbent_in_1st_piece_vec_org = []
min_incumbent_max_delta_pi_in_1st_piece_vec_org = []
num_unique_incumbents_vec_org = []
best_incumbent_vec_org = []


node_depth_threshold = 130
node_selection_prob = 0.999



with open('small-scale-summary-results-10-instances_APX-RJ-PN-SL.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['random_seed', 'best_max_delta_pi', 'correspond_min_delta_pi', 'best_max_delta_pi_1st_piece', 'num_unique_incumbents', 'num_incumbent_1st_piece'])
    
    # Loop and write each row
    for current_seed in np.flip(seed_vec):
        np.random.seed(current_seed)
        beta_0_sum = np.around(np.random.uniform(1.2,2,(num_heros, num_heros)),3)
        beta_k = np.around(np.random.uniform(0.1,0.2,(num_features, num_heros, num_heros)),3)


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
                    
        ################################# original

        incumbent_a=[]
        incumbent_y=[]
        incumbent_w=[]
        incumbent_obj=[]
        #incumbent_singular=[]
        #incumbent_time=[]
        incumbent_max_delta_pi = []
        incumbent_min_delta_pi = []
        #incumbent_nodeprocessed = []

        node_depth_vec = []
        
        selection_timing=[-1]
        
        
        lock = Lock()
        modified_w_inv_times_normlaized_b_dict = {}
        
        


        
        class CustomIncumbentCallback(ModelCallbackMixin, cpx_cb.IncumbentCallback):


            def __init__(self, env):
                
                cpx_cb.IncumbentCallback.__init__(self, env)
                ModelCallbackMixin.__init__(self)
                self.nb_incumbents = 0
                    
                    
            def __call__(self):
                self.nb_incumbents += 1
                obj = self.get_objective_value()
                #incum_sol = self.get_values()
                a_val = np.zeros((num_features, num_heros))
                y_val = np.zeros((num_heros, num_heros))
                w_val = np.zeros((num_heros, num_heros))
                for k in range(num_features):
                    for i in range(num_heros):
                        a_val[k][i] = round(self.get_values(f"a_{k}_{i}"))
                
                for i in range(num_heros):
                    for j in range(num_heros):
                        y_val[i][j] = self.get_values(f"y_{i}_{j}")
                        w_val[i][j] = self.get_values(f"w_{i}_{j}")
                        
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
                    
                #generate_incumbent_time = time.time()-solver_start_time
                
                currrent_num_nodes_processed = self.get_num_nodes()
                #incumbent_nodeprocessed.append(currrent_num_nodes_processed)
                
                print(f"********** found incumbent solution {obj} from node ID {self.get_node_ID()} at node depth {self.get_current_node_depth()};  max delta pi {max_delta_pi}, min delta pi {min_delta_pi}**************")
                print(f"the incumbent comes from source {self.get_solution_source()}; and the number of nodes processed is {currrent_num_nodes_processed}")    
                
                    
                    
                try_incumbent_a = incumbent_a.copy()
                try_incumbent_a.append(a_val)
                if np.unique(try_incumbent_a,axis=0).shape[0] > np.unique(incumbent_a,axis=0).shape[0]:
                    incumbent_a.append(a_val)
                    incumbent_y.append(y_val)
                    incumbent_w.append(w_val)
                    incumbent_obj.append(obj)
                    #incumbent_singular.append(min_singular_value)
                    #incumbent_obj_div_singular.append(obj/min_singular_value)
                    #incumbent_delta_p.append(delta_p_vec)
                    incumbent_max_delta_pi.append(max_delta_pi)
                    incumbent_min_delta_pi.append(min_delta_pi)
                    #incumbent_time.append(generate_incumbent_time)
                    #incumbent_nodeprocessed.append(currrent_num_nodes_processed)
                    #new_incumbent_a_generated = True
                    
                # else:
                #     new_incumbent_a_generated = False
                
                try_incumbent_a = 0                

            
                
                ### reject heuristic candidate
                if (len(incumbent_a)>5) and (max_delta_pi>min(incumbent_max_delta_pi)):
                    print(f"reject suboptimal incumbent at node depth {self.get_current_node_depth()}")
                    self.reject()
                elif (len(incumbent_a)>5) and (incum_seed_vec[len(incumbent_a)]<0.99) and (max_delta_pi==min(incumbent_max_delta_pi)):
                    print(f"reject current best incumbent at node depth {self.get_current_node_depth()}")
                    self.reject()
                
                    
                    
                
                
                
        class MyBranch(ModelCallbackMixin, cpx_cb.BranchCallback):
            def __init__(self, env):
                
                cpx_cb.BranchCallback.__init__(self, env)
                ModelCallbackMixin.__init__(self)
                self.nb_called = 0
                


            def __call__(self):
                self.nb_called += 1
                
                LP_obj = self.get_objective_value()
                
                LP_y_val = np.zeros((num_heros, num_heros))
                for i in range(num_heros):
                    for j in range(num_heros):
                        LP_y_val[i][j] = self.get_values(f"y_{i}_{j}")
                
                LP_w_val = np.zeros((num_heros, num_heros))
                for i in range(num_heros):
                    for j in range(num_heros):
                        LP_w_val[i][j] = self.get_values(f"w_{i}_{j}")

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
                if min_eigen_value >= 10**(-10):  ## >= 10**(-14)
                    LP_delta_p_vec = np.matmul(np.linalg.inv(LP_modified_w),LP_b_vector)
                    LP_max_delta_pi = max(LP_delta_p_vec)
                    LP_min_delta_pi = min(LP_delta_p_vec)
                    
                else:
                    LP_max_delta_pi = float("inf")
                    LP_min_delta_pi = - float("inf")
                
                                
                LP_b_vars_abs_vector = np.zeros(num_heros-1)
                for i in range(num_heros-1):
                    LP_b_vars_abs_vector[i] = self.get_values(f"b_vars_abs_{i}")
                    
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
                    
                print(f"********* Current node id is {self.get_node_ID()} with depth {self.get_current_node_depth()}: current_node_LP_obj is{LP_obj}; max delta pi {LP_max_delta_pi}; min delta pi {LP_min_delta_pi}; normalized_LP_max_delta_pi is {normialized_LP_max_delta_pi}********* ")

                  
                ######## record the node depth 
                current_node_depth = self.get_current_node_depth()
                node_depth_vec.append(current_node_depth)
                
                LP_a_val = np.zeros((num_features, num_heros))
                count=0
                for k in range(num_features):
                    for i in range(num_heros):
                        LP_a_val[k][i] = self.get_values(f"a_{k}_{i}")
                        if (LP_a_val[k][i]>=0.01) and (LP_a_val[k][i]<=0.99):
                            count=count+1
                
                    
                
                ### pruning node
                
                idx = self.get_node_ID()
                
                if ((normialized_LP_max_delta_pi > 5) and (current_node_depth >= node_depth_threshold)):#N=11
                    print(f"********** prune node ID {idx} with depth {current_node_depth} and whose normialized_LP_max_delta_pi is {normialized_LP_max_delta_pi}**************")
                    self.prune()
                    
                elif (len(incumbent_a)>=1) and (LP_max_delta_pi>1.5*min(min(incumbent_max_delta_pi),1.0/num_heros)) and (current_node_depth >= node_depth_threshold):  
                    self.prune()
                    print(f"********** prune node ID {idx} with max delta pi {LP_max_delta_pi}**************")
                    
                elif (count > 0):
                    current_num_branches = self.get_num_branches()
                    for i in range(current_num_branches):
                        child_node_seq_id = self.make_cplex_branch(i) ##seq_id of newly generated node
                        with lock:
                            modified_w_inv_times_normlaized_b_dict[child_node_seq_id] = normialized_LP_max_delta_pi #parent node's modified_w_inv_times_normlaized_b
                        print(f"child node sequence id {child_node_seq_id} has the parent node {self.get_node_ID()} with normialized_LP_max_delta_pi as {normialized_LP_max_delta_pi}, there are {count} non-zeros")
                        
                                


        class CustomNodeSelection(ModelCallbackMixin, cpx_cb.NodeCallback):

            
            def __init__(self, env):
                
                cpx_cb.NodeCallback.__init__(self, env)
                ModelCallbackMixin.__init__(self)
                self.node_called = 0
                
                        
            def __call__(self):
                self.node_called += 1
                print(f"the number of nodes processed so far is {self.get_num_nodes()}")    
                if len(node_depth_vec) >= 100:
                    ##### (selection_timing[-1]+1 != len(node_depth_vec)): node selection cannot happen at two consecutive nodes
                    if (nodeselection_seed_vec[len(node_depth_vec)] > node_selection_prob) and (node_depth_vec[-1] != node_depth_vec[-2]+1) and (node_depth_vec[-1] != node_depth_vec[-2]) and (selection_timing[-1]+1 != len(node_depth_vec)):  
                        num_remaining_node = self.get_num_remaining_nodes()
                        #print(f"nodecallback invoked; the number of remaining nodes is {num_remaining_node}")
                        
                        remain_node_seq_id_list = [int(self.get_node_ID(i)[0]) for i in range(num_remaining_node)]
                        
                        
                        remain_node_estimate_obj_list = [math.log(self.get_estimated_objective_value(i)+0.0000001) for i in range(num_remaining_node)]
                        
                        remain_node_modified_w_inv_times_normlaized_b_list = list(map(modified_w_inv_times_normlaized_b_dict.get, remain_node_seq_id_list))
                                            
                        remain_node_modified_w_inv_times_normlaized_b_array = np.nan_to_num(np.log(np.array(remain_node_modified_w_inv_times_normlaized_b_list, dtype=float)),nan=float("inf"))
                        
                        modified_w_inv_times_normlaized_b_array_with_weighted_node_estimate = np.array(remain_node_estimate_obj_list) + remain_node_modified_w_inv_times_normlaized_b_array
                        
                        next_visit_node_id = np.argmin(modified_w_inv_times_normlaized_b_array_with_weighted_node_estimate)
                        self.select_node(int(next_visit_node_id))
                        
                        print(f"*********Select new node id {int(next_visit_node_id)} with its weighted Wb and node estimate is {modified_w_inv_times_normlaized_b_array_with_weighted_node_estimate[int(next_visit_node_id)]}, where node estimate is {self.get_estimated_objective_value(int(next_visit_node_id))} and node_LP_obj is {self.get_objective_value(int(next_visit_node_id))}*********")
                        selection_timing.append(len(node_depth_vec))
        


        
        output_filename = f'small-scale_instance_randomseed_{current_seed}_N=11_K=12_APX-RJ-PN-SL.txt'
        
        original_stdout = sys.stdout
        with open(output_filename, 'w') as f:
            sys.stdout = f
            m = Model()
            m.parameters.mip.display.set(4)
            
            m = Model()
            kt = np.zeros((num_heros-1), dtype=object)
            
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
                    m.add_constraint(y[i][j] + y[j][i] == 1)
                    m.add_constraint(w[i][j] + w[j][i] == 1)
                
            for i in range(num_heros):
                for j in range(num_heros):         
                    m.add_constraint(u[i][j] == m.dot(a[:,i], beta_k[:,i,j]) + beta_0_sum[i,j])
                
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
                        z[i][j] == m.sum(z_pieces[p] * weights[i][j][p] for p in range(len(z_pieces))))
                
                    # 2. y[i][j] = sum(y_pieces[p] * weights[i][j][p])
                    m.add_constraint(
                        y[i][j] == m.sum(y_pieces[p] * weights[i][j][p] for p in range(len(z_pieces))))
                
                    # 3. Sum of weights = 1
                    m.add_constraint(
                        m.sum(weights[i][j][p] for p in range(len(z_pieces))) == 1)
                
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
            m.minimize(m.sum(b_vars_abs[s] for s in range(num_heros-1)))
            

            solver_start_time = time.time()
            m.register_callback(CustomIncumbentCallback)
            m.register_callback(MyBranch)
            m.register_callback(CustomNodeSelection)
            

            m.parameters.mip.strategy.heuristiceffort = 0 #no heuristic
            
            m.parameters.mip.strategy.nodeselect = 2
            
            m.parameters.mip.limits.nodes = 500000
            
            m.solve(log_output=True)
            
            sys.stdout = original_stdout
        

        
        ######### check results
        min_incumbent_max_delta_pi_vec_org.append(min(incumbent_max_delta_pi))

        best_idx = np.argmin(np.array(incumbent_max_delta_pi))
        

        min_incumbent_obj_vec_org.append(incumbent_obj[best_idx])
        
        min_incumbent_min_delta_pi_vec_org.append(incumbent_min_delta_pi[best_idx])
        
        first_pieces_indices = np.where(np.array(incumbent_min_delta_pi) > -1/num_heros)[0]
        num_incumbent_in_1st_piece_vec_org.append(len(np.unique(np.round(np.array(incumbent_min_delta_pi)[first_pieces_indices], decimals = 5)) ))
        
        
        if num_incumbent_in_1st_piece_vec_org[-1] != 0:
            min_incumbent_max_delta_pi_in_1st_piece_vec_org.append(min(np.array(incumbent_max_delta_pi)[first_pieces_indices]))
            best_first_pieces_idx = first_pieces_indices[np.argmin(np.array(incumbent_max_delta_pi)[first_pieces_indices])]
            
            best_incumbent_vec_org.append(incumbent_a[best_first_pieces_idx])
        else:
            min_incumbent_max_delta_pi_in_1st_piece_vec_org.append(0)
            best_incumbent_vec_org.append(np.zeros((num_features, num_heros)))
        
        #print(f"********* Current seed is {current_seed}: best_max_delta_pi is {min_incumbent_max_delta_pi_vec_org[-1]}; corresponding min_delta_pi is {min_incumbent_min_delta_pi_vec_org[-1]}; best_max_delta_pi_in_1st_piece is {min_incumbent_max_delta_pi_in_1st_piece_vec_org[-1]}********* ")

        df = pd.DataFrame(best_incumbent_vec_org[-1])
        df.to_csv(f'small-scale-best_incumbent_game_design_randomseed_{current_seed}_N=11_K=12_APX-RJ-PN-SL.csv', index=False, header=False)


        num_incumbents = len(incumbent_a)
        unique_incumbent_a = np.unique(incumbent_a, axis=0)
        num_unique_incumbents_vec_org.append(len(unique_incumbent_a)) 
        
        writer.writerow([current_seed, min_incumbent_max_delta_pi_vec_org[-1], min_incumbent_min_delta_pi_vec_org[-1], min_incumbent_max_delta_pi_in_1st_piece_vec_org[-1], num_unique_incumbents_vec_org[-1], num_incumbent_in_1st_piece_vec_org[-1]])

        
        print(f"********* Current seed is {current_seed}: best_max_delta_pi is {min_incumbent_max_delta_pi_vec_org[-1]}; corresponding min_delta_pi is {min_incumbent_min_delta_pi_vec_org[-1]}; best_max_delta_pi_in_1st_piece is {min_incumbent_max_delta_pi_in_1st_piece_vec_org[-1]}; num_unique_incumbents is {num_unique_incumbents_vec_org[-1]}; num_incumbents is {num_incumbents}; num_incumbent_in_1st_piece is {num_incumbent_in_1st_piece_vec_org[-1]}.********* ")


                

    


