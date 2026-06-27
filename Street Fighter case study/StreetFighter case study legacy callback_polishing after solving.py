#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 15:24:26 2025

@author: xinyiguan
"""


import sys
import math
import numpy as np
import pandas as pd
import random
import time

from docplex.mp.model import Model

import cplex.callbacks as cpx_cb

from docplex.mp.callbacks.cb_mixin import *

from threading import Lock

np.random.seed(2025)
incum_seed_vec = np.random.random(size = 1000000)


np.random.seed(6789)
nodeselection_seed_vec = np.random.random(size = 8000000)

node_depth_threshold = 40
node_selection_prob1 = 0.001





roles = ['Alex', 'CharlieNash', 'Chun-Li', 'Dan', 'Guile', 'MBison', 'Ryu']
design_len = [7, 8, 13, 8, 8, 7, 6]

num_heroes = len(roles)
num_features = max(design_len)

all_indices = [(k,i) for i in range(num_heroes) for k in range(design_len[i])]

#MNL_models_parameters = pd.read_csv("case_study_MNL_models_parameters.csv")
MNL_models_parameters = pd.read_csv("MNL_models_parameters_streetfighter.csv")


MNL_params_dict = {}
pair_dict = {}
iter_counter = -1

for i in range(num_heroes - 1):
    for j in range(i + 1, num_heroes):
        iter_counter = iter_counter + 1
        ## parameter_row is a pd series
        parameter_row = MNL_models_parameters[(MNL_models_parameters['attacker'] == roles[i]) & (MNL_models_parameters['defender'] == roles[j])].iloc[0].dropna()
        MNL_params_dict[(i, j)] = parameter_row[2:].to_numpy()
        
        pair_dict[(i,j)] = iter_counter

        
ub_list = []
lb_list = []
for i in range(num_heroes - 1):
    for j in range(i + 1, num_heroes):
        ##ub of u_ij-u_ji
        arr = MNL_params_dict[(i, j)][1:]
        ub = MNL_params_dict[(i, j)][0] + arr[arr > 0].sum()
        lb = MNL_params_dict[(i, j)][0] + arr[arr <= 0].sum()
        ub_list.append(ub)
        lb_list.append(lb)
        
        
        
z_pieces = np.arange(math.floor(np.min(lb_list)),0,0.5).tolist() + np.arange(0.5,math.ceil(np.max(ub_list))+0.5,0.5).tolist()
      
y_pieces = [1/(1+math.exp(-i)) for i in z_pieces]

y_z_slope = [(y_pieces[p+1] - y_pieces[p])/(z_pieces[p+1] - z_pieces[p]) for p in range(len(z_pieces)-1)]


A = np.zeros((num_heroes-1, int(num_heroes*(num_heroes-1)/2)))
for q_r in range(num_heroes-1):
    #A[i,]
    j = q_r+1+1 #w_entry j-th column
    if j <= num_heroes-1:
        for i in range(1,j):
            q_c = (i-1)*num_heroes - sum(s for s in range(i)) + (j-i) - 1 ##correspond to w[i,j]; 
            A[q_r,q_c] = 1
        for j_2 in range(j+1,num_heroes+1):
            q_c = (j-1)*num_heroes - sum(s for s in range(j)) + (j_2 - j) - 1 ##correspond to w[j,j_2]
            A[q_r,q_c] = -1
    else:
        for i in range(1,j):
            q_c = (i-1)*num_heroes - sum(s for s in range(i)) + (j-i) - 1 ##correspond to w[i,j]; 
            A[q_r,q_c] = 1

incumbent_a=[]
incumbent_y=[]
incumbent_w=[]
incumbent_obj=[]
incumbent_max_delta_pi = []
incumbent_min_delta_pi = []
incumbent_nodeprocessed = []

node_depth_vec = []
selection_timing=[-1]

lock = Lock()
modified_w_inv_times_normalized_b_dict = {}

## This function generates additional 100 designs from directly solving the approximate model with no-good integer cuts. It iteratively solves the model, retrieves the solution, and adds a cut to exclude that solution in the next iteration. The function returns lists of the generated designs (alist), their corresponding y values (ylist), and their objective values (objlist).
def generate_designs_with_integer_cuts(max_num_designs=100):
    """Generate additional designs from the approximate model using no-good integer cuts."""
    approx_output_filename = "case_study_approx_model_integer_cuts.txt"

    alist = []
    ylist = []
    objlist = []

    original_stdout = sys.stdout
    with open(approx_output_filename, "w") as f:
        sys.stdout = f

        m = Model()

        kt = np.zeros((num_heroes - 1), dtype=object)
        for i in range(num_heroes - 1):
            kt[i] = m.continuous_var(lb=-1, ub=1, name=f"kt_{i}")

        b_vars = np.zeros((num_heroes - 1), dtype=object)
        b_vars_abs = np.zeros((num_heroes - 1), dtype=object)
        for i in range(num_heroes - 1):
            b_vars[i] = m.continuous_var(lb=-2, name=f"b_vars_{i}")
            b_vars_abs[i] = m.continuous_var(name=f"b_vars_abs_{i}")
        for i in range(num_heroes - 1):
            m.add_constraint(b_vars[i] == kt[i] - (1 / num_heroes) * m.sum(kt[j] for j in range(num_heroes - 1)))
            m.add_constraint(b_vars_abs[i] >= b_vars[i])
            m.add_constraint(b_vars_abs[i] >= -b_vars[i])

        w = np.zeros((num_heroes, num_heroes), dtype=object)
        for i in range(num_heroes):
            for j in range(num_heroes):
                w[i][j] = m.continuous_var(lb=0, ub=1, name=f"w_{i}_{j}")

        y = np.zeros((num_heroes, num_heroes), dtype=object)
        for i in range(num_heroes):
            for j in range(num_heroes):
                y[i][j] = m.continuous_var(lb=0.3, ub=0.7, name=f"y_{i}_{j}")

        a = np.zeros((num_features, num_heroes), dtype=object)
        for k in range(num_features):
            for i in range(num_heroes):
                a[k][i] = m.binary_var(name=f"a_{k}_{i}")

        for i in range(num_heroes):
            for k in range(design_len[i], num_features):
                m.add_constraint(a[k][i] == 0)

        z = np.zeros((num_heroes, num_heroes), dtype=object)
        for i in range(num_heroes):
            for j in range(num_heroes):
                z[i][j] = m.continuous_var(lb=math.floor(np.min(lb_list)), ub=math.ceil(np.max(ub_list)))

        for i in range(num_heroes):
            for j in range(i, num_heroes):
                m.add_constraint(y[i][j] + y[j][i] == 1, f"mutual_y_exclusive_{i}_{j}")
                m.add_constraint(w[i][j] + w[j][i] == 1, f"mutual_w_exclusive_{i}_{j}")

        for i in range(num_heroes - 1):
            for j in range(i + 1, num_heroes):
                m.add_constraint(
                    z[i][j]
                    == MNL_params_dict[(i, j)][0]
                    + m.sum(MNL_params_dict[(i, j)][p + 1] * a[p][i] for p in range(design_len[i]))
                    + m.sum(
                        MNL_params_dict[(i, j)][design_len[i] + 1 + p2] * a[p2][j]
                        for p2 in range(design_len[j])
                    )
                )

        weights = np.zeros((num_heroes, num_heroes, len(z_pieces)), dtype=object)
        for i in range(num_heroes):
            for j in range(num_heroes):
                for p in range(len(z_pieces)):
                    weights[i][j][p] = m.continuous_var(lb=0, ub=1, name=f"w_{i}_{j}_{p}")

        for i in range(num_heroes - 1):
            for j in range(i + 1, num_heroes):
                m.add_constraint(
                    z[i][j] == m.sum(z_pieces[p] * weights[i][j][p] for p in range(len(z_pieces))),
                    ctname=f"z_piecewise_{i}_{j}",
                )
                m.add_constraint(
                    y[i][j] == m.sum(y_pieces[p] * weights[i][j][p] for p in range(len(z_pieces))),
                    ctname=f"y_piecewise_{i}_{j}",
                )
                m.add_constraint(
                    m.sum(weights[i][j][p] for p in range(len(z_pieces))) == 1,
                    ctname=f"weight_sum_{i}_{j}",
                )
                m.add_sos2([weights[i][j][p] for p in range(len(z_pieces))])

        for i in range(1, num_heroes):
            for j in range(i + 1, num_heroes + 1):
                q_c = (i - 1) * num_heroes - sum(s for s in range(i)) + (j - i) - 1
                m.add_constraint(y[i - 1][j - 1] - w[i - 1][j - 1] == m.sum(kt[s] * A[s, q_c] for s in range(num_heroes - 1)))

        for j in range(num_heroes):
            m.add_constraint(m.sum(w[i, j] for i in range(num_heroes)) >= 0.5 * num_heroes)

        m.minimize(m.sum(b_vars_abs[s] for s in range(num_heroes - 1)))
        m.parameters.mip.strategy.nodeselect = 2
        m.parameters.timelimit = 3600

        initial_sol = m.solve(log_output=True)
        if initial_sol is not None:
            for cut_id in range(max_num_designs):
                if m.solution is None:
                    break

                objlist.append(m.solution.get_objective_value())

                sol_a = np.zeros((num_features, num_heroes), dtype=object)
                a_zero_ind = []
                a_one_ind = []
                for k in range(num_features):
                    for i in range(num_heroes):
                        a_val = m.solution.get_value(a[k][i])
                        sol_a[k][i] = round(a_val)
                        if a_val >= 0.9:
                            a_one_ind.append((k, i))
                        else:
                            a_zero_ind.append((k, i))

                sol_y = np.zeros((num_heroes, num_heroes), dtype=object)
                for i in range(num_heroes):
                    for j in range(num_heroes):
                        sol_y[i][j] = m.solution.get_value(y[i][j])

                alist.append(sol_a)
                ylist.append(sol_y)

                m.add_constraint(
                    m.sum(a[k, i] for (k, i) in a_zero_ind)
                    + m.sum((1 - a[k, i]) for (k, i) in a_one_ind)
                    >= 1
                )
                next_sol = m.solve(log_output=False)
                print(cut_id)
                if next_sol is None:
                    break

    sys.stdout = original_stdout
    return alist, ylist, objlist


class CustomIncumbentCallback(ModelCallbackMixin, cpx_cb.IncumbentCallback):


    def __init__(self, env):
        
        cpx_cb.IncumbentCallback.__init__(self, env)
        ModelCallbackMixin.__init__(self)
        self.nb_incumbents = 0
            
            
    def __call__(self):
        self.nb_incumbents += 1
        obj = self.get_objective_value()
        #incum_sol = self.get_values()
        a_val = np.zeros((num_features, num_heroes))
        y_val = np.zeros((num_heroes, num_heroes))
        w_val = np.zeros((num_heroes, num_heroes))
        for k in range(num_features):
            for i in range(num_heroes):
                a_val[k][i] = round(self.get_values(f"a_{k}_{i}"))
        
        for i in range(num_heroes):
            for j in range(num_heroes):
                y_val[i][j] = self.get_values(f"y_{i}_{j}")
                w_val[i][j] = self.get_values(f"w_{i}_{j}")
                
        modified_w=[]
        for i in range(1,num_heroes):
            modified_w_vec = []
            for j in range(num_heroes):
                modified_w_vec.append(w_val[i][j] - 0.5)
            
            modified_w.append(modified_w_vec)
    
        modified_w.append([1 for j in range(num_heroes)])
        modified_w = np.array(modified_w)
        
        b_vector = np.zeros(num_heroes)
        for i in range(1,num_heroes):
            b_vector[i-1]= -(1/num_heroes) * sum(y_val[i][j] - w_val[i][j] for j in range(num_heroes))
        
        min_eigen_value = np.min(np.linalg.eig(np.matmul(np.transpose(modified_w),modified_w))[0])
        if min_eigen_value >= 10**(-14):
            delta_p_vec = np.matmul(np.linalg.inv(modified_w),b_vector)
            min_delta_pi = min(delta_p_vec)
            max_delta_pi = max(delta_p_vec)
        else:
            delta_p_vec = np.zeros(num_heroes)
            max_delta_pi = float("inf")
            min_delta_pi = -float("inf")

        try_incumbent_a = incumbent_a.copy()
        try_incumbent_a.append(a_val)
        if np.unique(try_incumbent_a,axis=0).shape[0] > np.unique(incumbent_a,axis=0).shape[0]:
            incumbent_a.append(a_val)
            incumbent_y.append(y_val)
            incumbent_w.append(w_val)
            incumbent_obj.append(obj)
            incumbent_max_delta_pi.append(max_delta_pi)
            incumbent_min_delta_pi.append(min_delta_pi)
        
        try_incumbent_a = 0

        if (len(incumbent_a)>5) and (max_delta_pi>min(incumbent_max_delta_pi)):
            print(f"reject suboptimal incumbent with {obj}")
            self.reject()
        elif (len(incumbent_a)>5) and (incum_seed_vec[len(incumbent_a)]<0.99) and (max_delta_pi==min(incumbent_max_delta_pi)):
            print(f"reject current best incumbent with {obj}")
            self.reject()
            


class MyBranch(ModelCallbackMixin, cpx_cb.BranchCallback):
    def __init__(self, env):
        
        cpx_cb.BranchCallback.__init__(self, env)
        ModelCallbackMixin.__init__(self)
        self.nb_called = 0
        #self.parent_node_seq_id = 0 #parent_node_seq_id of current node
        #self._lock = Lock()


    def __call__(self):
        self.nb_called += 1
        
        LP_obj = self.get_objective_value()
        
        LP_y_val = np.zeros((num_heroes, num_heroes))
        for i in range(num_heroes):
            for j in range(num_heroes):
                LP_y_val[i][j] = self.get_values(f"y_{i}_{j}")
        
        LP_w_val = np.zeros((num_heroes, num_heroes))
        for i in range(num_heroes):
            for j in range(num_heroes):
                LP_w_val[i][j] = self.get_values(f"w_{i}_{j}")

        LP_modified_w=[]
        for i in range(1,num_heroes):
            LP_modified_w_vec = []
            for j in range(num_heroes):
                LP_modified_w_vec.append(LP_w_val[i][j] - 0.5)
    
            LP_modified_w.append(LP_modified_w_vec)

        LP_modified_w.append([1 for j in range(num_heroes)])
        LP_modified_w = np.array(LP_modified_w)
                       
        LP_b_vector = np.zeros(num_heroes)
        for i in range(1,num_heroes):
            LP_b_vector[i-1]= -(1/num_heroes) * sum(LP_y_val[i][j] - LP_w_val[i][j] for j in range(num_heroes))
        
        min_eigen_value = np.min(np.linalg.eig(np.matmul(np.transpose(LP_modified_w),LP_modified_w))[0])
        if min_eigen_value >= 10**(-10):  ## >= 10**(-14)
            LP_delta_p_vec = np.matmul(np.linalg.inv(LP_modified_w),LP_b_vector)
            LP_max_delta_pi = max(LP_delta_p_vec)
            LP_min_delta_pi = min(LP_delta_p_vec)
            
        else:
            LP_max_delta_pi = float("inf")
            LP_min_delta_pi = - float("inf")
        
        LP_b_vars_abs_vector = np.zeros(num_heroes-1)
        for i in range(num_heroes-1):
            LP_b_vars_abs_vector[i] = self.get_values(f"b_vars_abs_{i}")
            
        l1_norm_obj_vector = sum(s for s in LP_b_vars_abs_vector)
            
        try_LP_max_delta_pi_vec = np.zeros(num_heroes)
            
        if l1_norm_obj_vector >= 10**(-10):
            normalized_LP_max_delta_pi = LP_max_delta_pi / l1_norm_obj_vector
        elif min_eigen_value >= 10**(-14):
            l1_norm_try_b_vector = (1/num_heroes) * ((num_heroes-1) + (num_heroes-2))
            try_b_vector = -(1/num_heroes) * np.ones(num_heroes)
            try_b_vector[-1] = 0
            for i in range(1,num_heroes):
                try_b_vector[i-1] = (1/num_heroes) * (num_heroes-1)
                try_LP_delta_p_vec = np.matmul(np.linalg.inv(LP_modified_w),try_b_vector)
                try_LP_max_delta_pi_vec[i-1] = max(try_LP_delta_p_vec) / l1_norm_try_b_vector
            
            try_b_vector = (1/num_heroes) * np.ones(num_heroes)
            try_b_vector[-1] = 0
            l1_norm_try_b_vector = (1/num_heroes) * (num_heroes-1)
            try_LP_delta_p_vec = np.matmul(np.linalg.inv(LP_modified_w),try_b_vector)
            try_LP_max_delta_pi_vec[num_heroes-1] = max(try_LP_delta_p_vec) / l1_norm_try_b_vector
            
            normalized_LP_max_delta_pi = min(try_LP_max_delta_pi_vec)
        else: #LP_obj =0 and min_eigen_value = 0
            normalized_LP_max_delta_pi = float("inf")
            
        print(f"********* Current node id is {self.get_node_ID()} with depth {self.get_current_node_depth()}: current_node_LP_obj is{LP_obj}; max delta pi {LP_max_delta_pi}; min delta pi {LP_min_delta_pi}; normalized_LP_max_delta_pi is {normalized_LP_max_delta_pi}********* ")

          
        current_node_depth = self.get_current_node_depth()
        node_depth_vec.append(current_node_depth)
        LP_a_val = np.zeros((num_features, num_heroes))
        count=0
        for k in range(num_features):
            for i in range(num_heroes):
                LP_a_val[k][i] = self.get_values(f"a_{k}_{i}")
                if (LP_a_val[k][i]>=0.01) and (LP_a_val[k][i]<=0.99):
                    count=count+1

            
        
        ### pruning node
        
        idx = self.get_node_ID()
        
        if ((normalized_LP_max_delta_pi > 40) and (current_node_depth >= node_depth_threshold)):
            print(f"********** prune node ID {idx} with depth {current_node_depth} and whose normalized_LP_max_delta_pi is {normalized_LP_max_delta_pi}**************")
            self.prune()
            
        elif (LP_max_delta_pi>1.5*min(min(incumbent_max_delta_pi),1.0/num_heroes)) and (current_node_depth >= node_depth_threshold):
            self.prune()
            print(f"********** prune node ID {idx} with max delta pi {LP_max_delta_pi}**************")
            
        elif (count > 0):
            current_num_branches = self.get_num_branches()
            for i in range(current_num_branches):
                child_node_seq_id = self.make_cplex_branch(i) ##seq_id of newly generated node
                with lock:
                    modified_w_inv_times_normalized_b_dict[child_node_seq_id] = normalized_LP_max_delta_pi # parent node's normalized score
                print(f"child node sequence id {child_node_seq_id} has the parent node {self.get_node_ID()} with normalized_LP_max_delta_pi as {normalized_LP_max_delta_pi}, there are {count} non-zeros")

            
    


class CustomNodeSelection(ModelCallbackMixin, cpx_cb.NodeCallback):

    
    def __init__(self, env):
        
        cpx_cb.NodeCallback.__init__(self, env)
        ModelCallbackMixin.__init__(self)
        self.node_called = 0
        #self.parent_node_seq_id = 0 #parent_node_seq_id of current node
        #self._lock = Lock()


    def __call__(self):
        self.node_called += 1
        print(f"the number of nodes processed so far is {self.get_num_nodes()}")    
        if len(node_depth_vec) >= 100:
            
            if len(node_depth_vec) >= 100:
                
                if (node_depth_vec[-1] != node_depth_vec[-2]+1) and (node_depth_vec[-1] != node_depth_vec[-2]) and (selection_timing[-1]+1 != len(node_depth_vec)): 
                    num_remaining_node = self.get_num_remaining_nodes()
                    #print(f"nodecallback invoked; the number of remaining nodes is {num_remaining_node}")
                    
                    
                    if (nodeselection_seed_vec[len(node_depth_vec)] < node_selection_prob1):
                        remain_node_seq_id_list = [int(self.get_node_ID(i)[0]) for i in range(num_remaining_node)]
                       
                        remain_node_estimate_obj_list = [math.log(self.get_objective_value(i)+0.0000001) for i in range(num_remaining_node)]
                    
                        remain_node_modified_w_inv_times_normalized_b_list = list(map(modified_w_inv_times_normalized_b_dict.get, remain_node_seq_id_list))

                        remain_node_modified_w_inv_times_normalized_b_array = np.nan_to_num(np.log(np.array(remain_node_modified_w_inv_times_normalized_b_list, dtype=float)),nan=float("inf"))

                        modified_w_inv_times_normalized_b_array_with_weighted_node_estimate = np.array(remain_node_estimate_obj_list) + remain_node_modified_w_inv_times_normalized_b_array

                        next_visit_node_id = np.argmin(modified_w_inv_times_normalized_b_array_with_weighted_node_estimate)
                        self.select_node(int(next_visit_node_id))
                        print(f"*********select new node id {int(next_visit_node_id)} with its weighted Wb and node estimate is {modified_w_inv_times_normalized_b_array_with_weighted_node_estimate[int(next_visit_node_id)]}, where node estimate is {self.get_estimated_objective_value(int(next_visit_node_id))} and node_LP_obj is {self.get_objective_value(int(next_visit_node_id))}*********")
                        selection_timing.append(len(node_depth_vec))
                                            



output_filename = 'case_study_legacy_callbacks_MNL_polishingaftersolving.txt'


#solver_start_time_2 = time.time()

original_stdout = sys.stdout
with open(output_filename, 'w') as f:
    sys.stdout = f
    m = Model()
    m.parameters.mip.display.set(4)
    
    kt = np.zeros((num_heroes-1), dtype=object)
    
    for i in range(num_heroes-1):
        kt[i] = m.continuous_var(lb=-1, ub=1, name=f"kt_{i}")
        
        
    b_vars = np.zeros((num_heroes-1), dtype=object)
    b_vars_abs = np.zeros((num_heroes-1), dtype=object)
    for i in range(num_heroes-1):
        b_vars[i] = m.continuous_var(lb=-2,name=f"b_vars_{i}")
        b_vars_abs[i] = m.continuous_var(name=f"b_vars_abs_{i}")
    for i in range(num_heroes-1):
        m.add_constraint(b_vars[i] == kt[i] - (1/num_heroes) * m.sum(kt[j] for j in range(num_heroes-1) ) )
        m.add_constraint(b_vars_abs[i] >= b_vars[i])
        m.add_constraint(b_vars_abs[i] >= -b_vars[i])
        
        
    w = np.zeros((num_heroes, num_heroes), dtype=object)
    for i in range(num_heroes):
        for j in range(num_heroes):
            w[i][j] = m.continuous_var(lb=0,ub=1,name=f"w_{i}_{j}")  

    y = np.zeros((num_heroes, num_heroes), dtype=object)
    for i in range(num_heroes):
        for j in range(num_heroes):
            y[i][j] = m.continuous_var(lb=0.3, ub=0.7, name=f"y_{i}_{j}") 
            #y[i][j] = m.addVar(vtype="C", lb=0, ub=1) 

    a = np.zeros((num_features, num_heroes), dtype=object)
    for k in range(num_features):
        for i in range(num_heroes):
            a[k][i] = m.binary_var(name=f"a_{k}_{i}") 
            
    for i in range(num_heroes):
        for k in range(design_len[i],num_features):
            m.add_constraint(a[k][i] == 0)

        
    z = np.zeros((num_heroes, num_heroes), dtype=object)
    for i in range(num_heroes):
        for j in range(num_heroes):
            z[i][j] = m.continuous_var(lb = math.floor(np.min(lb_list)), ub = math.ceil(np.max(ub_list))) 


    for i in range(num_heroes):
        for j in range(i, num_heroes):
            # 1. y[i][j] + y[j][i] == 1
            # w[i][j] + w[j][i] == 1
            m.add_constraint(y[i][j] + y[j][i] == 1)
            m.add_constraint(w[i][j] + w[j][i] == 1)
    
        
    for i in range(num_heroes-1):
        for j in range(i+1,num_heroes):
            m.add_constraint(z[i][j] == MNL_params_dict[(i, j)][0] + m.sum(MNL_params_dict[(i, j)][p+1] * a[p][i] for p in range(design_len[i]) ) + m.sum(MNL_params_dict[(i, j)][design_len[i]+1+p2] * a[p2][j] for p2 in range(design_len[j]) ) )
            

    ## model piecewise functions as SOS2 constraints
    weights = np.zeros((num_heroes, num_heroes,len(z_pieces)), dtype=object)
    for i in range(num_heroes):
        for j in range(num_heroes):
            for p in range(len(z_pieces)):
                weights[i][j][p] = m.continuous_var(lb=0, ub=1, name=f"w_{i}_{j}_{p}")
                


    # Add constraints for upper triangle (i < j)
    for i in range(num_heroes-1):
        for j in range(i+1, num_heroes):
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
    for i in range(1,num_heroes):
        for j in range(i+1, num_heroes+1): #w[i,j]
            q_c = (i-1)*num_heroes - sum(s for s in range(i)) + (j-i) - 1 
            m.add_constraint(y[i-1][j-1] - w[i-1][j-1] == m.sum(kt[s]*A[s,q_c] for s in range(num_heroes-1)) )
        
    for j in range(num_heroes):
        m.add_constraint(m.sum(w[i,j] for i in range(num_heroes))>=0.5 * num_heroes)

    
    m.minimize(m.sum(b_vars_abs[s] for s in range(num_heroes-1)))

    #solver_start_time = time.time()
    m.register_callback(CustomIncumbentCallback)
    m.register_callback(MyBranch)
    m.register_callback(CustomNodeSelection)
    

    
    m.parameters.mip.strategy.nodeselect = 2
    
    m.parameters.timelimit= 3600
    
    
    m.solve(log_output=True)
    
    sys.stdout = original_stdout



#solver_end_time_2 = time.time() - solver_start_time_2 

# Generate additional designs from the approximate model with integer cuts.
alist, _, _ = generate_designs_with_integer_cuts(max_num_designs=100)

############## combine incumbent_a and alist
####
incumbent_a_list = [np.round(arr).astype(int) for arr in incumbent_a]

candidate_a = incumbent_a_list + alist

unique_a_list = []
seen = set()
unique_max_delta_pi_list = []
#unique_min_delta_pi_list = []

for arr in candidate_a:
    arr_tuple = tuple(map(tuple,arr))
    if arr_tuple not in seen:
        unique_a_list.append(arr)
        seen.add(arr_tuple)
        ## calculate max_delta_pi
        
        current_y_val = np.zeros((num_heroes, num_heroes))
        current_z_val = np.zeros((num_heroes, num_heroes))
        for i in range(num_heroes-1):
            for j in range(i+1, num_heroes):
                current_z_val[i][j] = MNL_params_dict[(i, j)][0] + sum(MNL_params_dict[(i, j)][p+1] * arr[p][i] for p in range(design_len[i]) ) + sum(MNL_params_dict[(i, j)][design_len[i]+1+p2] * arr[p2][j] for p2 in range(design_len[j]) )  
                current_y_val[i][j] = 1/(1 + math.exp(-current_z_val[i][j]))
                current_y_val[j][i] = 1 - current_y_val[i][j]

        for i in range(num_heroes):
            current_y_val[i][i] = 0.5


        m_LP = Model()
        prob = np.zeros((num_heroes, ), dtype=object)
        for i in range(num_heroes):
            prob[i] = m_LP.continuous_var(lb=0)
        tt = m_LP.continuous_var(lb=0)

        for j in range(num_heroes):
            m_LP.add_constraint(m_LP.sum(prob[i]*current_y_val[i,j] for i in range(num_heroes))>=0.5)
            m_LP.add_constraint(prob[j]<=tt)
            
        m_LP.add_constraint(m_LP.sum(prob[i] for i in range(num_heroes))==1)
        m_LP.minimize(tt)
        m_LP.solve(log_output=False) 
        true_max_delta_pi = max([prob[i].solution_value for i in range(num_heroes)]) - 1/num_heroes
        #true_min_delta_pi = min([prob[i].solution_value for i in range(num_heroes)]) - 1/num_heroes

        unique_max_delta_pi_list.append(true_max_delta_pi)
        #unique_min_delta_pi_list.append(true_min_delta_pi)
        
########
#len(unique_a_list)  

solver_found_a = unique_a_list.copy()
solver_found_max_delta_pi = unique_max_delta_pi_list.copy()
orign_largest_max_delta_pi_in_top_P_20_a = unique_max_delta_pi_list[np.argsort(np.array(unique_max_delta_pi_list))[19]] 
#unique_a_list = solver_found_a.copy()
#unique_max_delta_pi_list = solver_found_max_delta_pi.copy()


################################# solver stop; start solution polishing
#polish_start_time = time.time()

num_new_a_previous_polishing = 0
number_of_polishing = 100


## build sub-MIP for solution polishing
def solve_sub_mip(fixed_rows, fixed_cols, current_a):
    sub_m = Model()
    
    sub_kt = np.zeros((num_heroes-1), dtype=object)
    
    for i in range(num_heroes-1):
        sub_kt[i] = sub_m.continuous_var(lb=-1, ub=1)
        

    
    sub_b_vars = np.zeros((num_heroes-1), dtype=object)
    sub_b_vars_abs = np.zeros((num_heroes-1), dtype=object)
    for i in range(num_heroes-1):
        sub_b_vars[i] = sub_m.continuous_var(lb=-2)
        sub_b_vars_abs[i] = sub_m.continuous_var()
    for i in range(num_heroes-1):
        sub_m.add_constraint(sub_b_vars[i] == sub_kt[i] - (1/num_heroes) * sub_m.sum(sub_kt[j] for j in range(num_heroes-1) ) )
        sub_m.add_constraint(sub_b_vars_abs[i] >= sub_b_vars[i])
        sub_m.add_constraint(sub_b_vars_abs[i] >= -sub_b_vars[i])
        #sub_m.add_constraint(sub_b_vars_abs[i] == sub_m.abs(sub_b_vars[i]))
        
    sub_w = np.zeros((num_heroes, num_heroes), dtype=object)
    for i in range(num_heroes):
        for j in range(num_heroes):
            sub_w[i][j] = sub_m.continuous_var(lb=0,ub=1,name=f"sub_w_{i}_{j}")  

    sub_y = np.zeros((num_heroes, num_heroes), dtype=object)
    for i in range(num_heroes):
        for j in range(num_heroes):
            sub_y[i][j] = sub_m.continuous_var(lb=0.3, ub=0.7,name=f"sub_y_{i}_{j}") 
            #y[i][j] = m.addVar(vtype="C", lb=0, ub=1) 

    sub_a = np.zeros((num_features, num_heroes), dtype=object)
    for k in range(num_features):
        for i in range(num_heroes):
            sub_a[k][i] = sub_m.binary_var(name=f"sub_a_{k}_{i}")
            
    for i in range(num_heroes):
        for k in range(design_len[i],num_features):
            sub_m.add_constraint(sub_a[k][i] == 0)
    
    if len(fixed_rows) > 0:
        for r, c in zip(fixed_rows, fixed_cols): 
            sub_m.add_constraint( sub_a[r][c] == current_a[r][c] )

        
    sub_z = np.zeros((num_heroes, num_heroes), dtype=object)
    for i in range(num_heroes):
        for j in range(num_heroes):
            sub_z[i][j] = sub_m.continuous_var(lb = math.floor(np.min(lb_list)), ub = math.ceil(np.max(ub_list))) 



    for i in range(num_heroes):
        for j in range(i, num_heroes):
            # 1. y[i][j] + y[j][i] == 1
            # w[i][j] + w[j][i] == 1
            sub_m.add_constraint(sub_y[i][j] + sub_y[j][i] == 1)
            sub_m.add_constraint(sub_w[i][j] + sub_w[j][i] == 1)
        
            
    for i in range(num_heroes):
        for j in range(i+1,num_heroes):
            sub_m.add_constraint(sub_z[i][j] == MNL_params_dict[(i, j)][0] + sub_m.sum(MNL_params_dict[(i, j)][p+1] * sub_a[p][i] for p in range(design_len[i]) ) + sub_m.sum(MNL_params_dict[(i, j)][design_len[i]+1+p2] * sub_a[p2][j] for p2 in range(design_len[j]) ) )
            


    ## model piecewise functions as SOS2 constraints
    sub_weights = np.zeros((num_heroes, num_heroes,len(z_pieces)), dtype=object)
    for i in range(num_heroes):
        for j in range(num_heroes):
            for p in range(len(z_pieces)):
                sub_weights[i][j][p] = sub_m.continuous_var(lb=0, ub=1, name=f"w_{i}_{j}_{p}")

    # Add constraints for upper triangle (i < j)
    for i in range(num_heroes):
        for j in range(i+1, num_heroes):
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
    for i in range(1,num_heroes):
        for j in range(i+1, num_heroes+1): #w[i,j]
            sub_q_c = (i-1)*num_heroes - sum(s for s in range(i)) + (j-i) - 1 
            sub_m.add_constraint(sub_y[i-1][j-1] - sub_w[i-1][j-1] == sub_m.sum(sub_kt[s]*A[s,sub_q_c] for s in range(num_heroes-1)) )
        
    for j in range(num_heroes):
        sub_m.add_constraint(sub_m.sum(sub_w[i,j] for i in range(num_heroes))>=0.5 * num_heroes)

    
    sub_m.minimize(sub_m.sum(sub_b_vars_abs[s] for s in range(num_heroes-1)))
    
    
    sub_m.parameters.mip.strategy.nodeselect = 2
    sub_m.parameters.mip.limits.nodes = 500
    
    sub_m.solve(log_output=False)
    #print(f"solving sub_mip status: {sub_m.get_solve_status()}")
    if (str(sub_m.get_solve_status()) == 'JobSolveStatus.FEASIBLE_SOLUTION') or (str(sub_m.get_solve_status()) == 'JobSolveStatus.OPTIMAL_SOLUTION'):
        sub_mip_obj = sub_m.objective_value
        #print(f"sub_mip_obj: {sub_mip_obj}")
        a_sol = np.zeros((num_features, num_heroes))
        for k in range(num_features):
            for i in range(num_heroes):
                a_sol[k][i] = round(sub_a[k][i].solution_value)
        
        z_sol = np.zeros((num_heroes, num_heroes))
        y_sol = np.zeros((num_heroes, num_heroes))

        for i in range(num_heroes-1):
            for j in range(i+1, num_heroes):
                z_sol[i][j] = MNL_params_dict[(i, j)][0] + sum(MNL_params_dict[(i, j)][p+1] * a_sol[p][i] for p in range(design_len[i]) ) + sum(MNL_params_dict[(i, j)][design_len[i]+1+p2] * a_sol[p2][j] for p2 in range(design_len[j]) )  
                y_sol[i][j] = 1/(1 + math.exp(-z_sol[i][j]))
                y_sol[j][i] = 1 - y_sol[i][j]

        for i in range(num_heroes):
            y_sol[i][i] = 0.5
            
        sub_m_LP = Model()
        prob = np.zeros((num_heroes, ), dtype=object)
        for i in range(num_heroes):
            prob[i] = sub_m_LP.continuous_var(lb=0)
        tt = sub_m_LP.continuous_var(lb=0)

        for j in range(num_heroes):
            sub_m_LP.add_constraint(sub_m_LP.sum(prob[i]*y_sol[i,j] for i in range(num_heroes))>=0.5)
            sub_m_LP.add_constraint(prob[j]<=tt)
            
        sub_m_LP.add_constraint(sub_m_LP.sum(prob[i] for i in range(num_heroes))==1)
        sub_m_LP.minimize(tt)
        sub_m_LP.solve(log_output=False) 
        sub_m_max_delta_pi = max([prob[i].solution_value for i in range(num_heroes)]) - 1/num_heroes
        # sub_m_min_delta_pi = min([prob[i].solution_value for i in range(num_heroes)]) - 1/num_heroes
        

        
    else:
        sub_mip_obj = float('inf')
        sub_m_max_delta_pi = float('inf')
        #sub_m_min_delta_pi = - float('inf')
        a_sol = np.zeros((num_features, num_heroes))
        
    return sub_m_max_delta_pi, a_sol, sub_mip_obj, str(sub_m.get_solve_status())
        
        

#solution_polish_pool_update = False
#len_polish_pool = min(40,len(incumbent_a))
len_polish_pool = 40
sort_index = np.argsort(np.array(unique_max_delta_pi_list))
top_P_candidate_a = np.array(unique_a_list)[sort_index[0:len_polish_pool]]
largest_max_delta_pi_in_top_P_candidate_a = unique_max_delta_pi_list[sort_index[len_polish_pool-1]]
top_P_candidate_max_delta_pi = np.array(unique_max_delta_pi_list)[sort_index[0:len_polish_pool]]
idx_largest_max_delta_pi_in_top_P_candidate_a = len_polish_pool-1

num_new_a_previous_polishing = 0


for _ in range(number_of_polishing):
    num_new_a_from_mutation = 0
    num_new_a_from_combination = 0
    #new_obj_from_polishing_list = []
    #new_max_delta_pi_from_polishing_list = []
    freq = 0.5
    increment = 0.2
    
    # Mutation—repeat 20 times
    for __ in range(20):
        rand_int = np.random.randint(0, len_polish_pool)
        #print(f"rand_int is {rand_int}")
        seed_a = top_P_candidate_a[rand_int]
        seed_a_max_delta_pi = top_P_candidate_max_delta_pi[rand_int]
        num_fixed_elements = math.floor(sum(design_len) * freq)
        
        selected_indices = random.sample(all_indices,num_fixed_elements)
        
        selected_rows,selected_cols = list(zip(*selected_indices))
        sub_m_max_delta_pi, sub_m_a, sub_m_obj, sub_m_solve_status = solve_sub_mip(selected_rows, selected_cols, seed_a)
        print(f"in mutation phase: sub_m_max_delta_pi is {sub_m_max_delta_pi}; sub_m_solve_status is {sub_m_solve_status}")        

        #update freq: compared with seed solution
        if (seed_a_max_delta_pi == sub_m_max_delta_pi):
            #freq decrease
            if (freq - increment >0.05):
                freq = freq - increment
            print(f"freq is {freq}; freq is decreasing")
        elif (seed_a_max_delta_pi < sub_m_max_delta_pi):
            #freq increase
            if (freq + increment < 0.95):
                freq = freq + increment
            print(f"freq is {freq}; freq is increasing")
            
        # update increment
        if (increment * 0.75 >= 0.01):
            increment = increment * 0.75
        else:
            increment = 0.01


        
        sub_m_a_tuple = tuple(map(tuple,sub_m_a))        
        if sub_m_a_tuple not in seen:
            unique_a_list.append(sub_m_a)
            seen.add(sub_m_a_tuple)               
            unique_max_delta_pi_list.append(sub_m_max_delta_pi)
            num_new_a_from_mutation = num_new_a_from_mutation + 1
            
            ####update solution pool:
            if (largest_max_delta_pi_in_top_P_candidate_a > sub_m_max_delta_pi):
                top_P_candidate_a[idx_largest_max_delta_pi_in_top_P_candidate_a] = sub_m_a
                top_P_candidate_max_delta_pi[idx_largest_max_delta_pi_in_top_P_candidate_a] = sub_m_max_delta_pi
                largest_max_delta_pi_in_top_P_candidate_a = max(top_P_candidate_max_delta_pi)
                idx_largest_max_delta_pi_in_top_P_candidate_a = np.argmax(top_P_candidate_max_delta_pi)
           
        
    print(f"my_solution_polishing-Mutation ends. # of new_a is {num_new_a_from_mutation}.") 
    
    if (num_new_a_from_mutation>=0) or (num_new_a_previous_polishing>=0): # have enough variation in top_P_candidate_a
        for iterid in range(40):
            if (iterid < 39): # pairs of solution
                rand_int1, rand_int2 = np.random.choice(np.arange(len_polish_pool), size=2, replace=False)
                #print(f"rand_int1 is {rand_int1}, rand_int2 is {rand_int2}. current len of top_P_candidate_a_snapshot is {len(top_P_candidate_a_snapshot)}.")
                seed_a_1 = top_P_candidate_a[rand_int1]
                seed_a_2 = top_P_candidate_a[rand_int2]
                selected_rows, selected_cols = np.where(seed_a_1 == seed_a_2) ##it is possible that selected_rows = array([], dtype=int64)
                sub_m_max_delta_pi, sub_m_a, sub_m_obj, sub_m_solve_status = solve_sub_mip(selected_rows, selected_cols, seed_a_1)
                print(f"in combination phase: sub_m_max_delta_pi is {sub_m_max_delta_pi}; sub_m_solve_status is {sub_m_solve_status}")
            else:
                stacked = np.stack(top_P_candidate_a)
                # Find indices where all elements are equal across arrays
                selected_rows, selected_cols = np.where(np.all(stacked == stacked[0], axis=0))
                sub_m_max_delta_pi, sub_m_a, sub_m_obj, sub_m_solve_status = solve_sub_mip(selected_rows, selected_cols, top_P_candidate_a[1])
                print(f"in combination phase: sub_m_max_delta_pi is {sub_m_max_delta_pi}; sub_m_solve_status is {sub_m_solve_status}")
         
            sub_m_a_tuple = tuple(map(tuple,sub_m_a))        
            if sub_m_a_tuple not in seen:
                unique_a_list.append(sub_m_a)
                seen.add(sub_m_a_tuple)               
                unique_max_delta_pi_list.append(sub_m_max_delta_pi)
                num_new_a_from_combination = num_new_a_from_combination + 1
                
                ####update solution pool:
                if (largest_max_delta_pi_in_top_P_candidate_a > sub_m_max_delta_pi):
                    top_P_candidate_a[idx_largest_max_delta_pi_in_top_P_candidate_a] = sub_m_a
                    top_P_candidate_max_delta_pi[idx_largest_max_delta_pi_in_top_P_candidate_a] = sub_m_max_delta_pi
                    largest_max_delta_pi_in_top_P_candidate_a = max(top_P_candidate_max_delta_pi)
                    idx_largest_max_delta_pi_in_top_P_candidate_a = np.argmax(top_P_candidate_max_delta_pi)
                
                
    print(f"my_solution_polishing ends. # of new_a is {num_new_a_from_combination}.")
    num_new_a_previous_polishing = num_new_a_from_mutation + num_new_a_from_combination




#polish_end_time = time.time() - polish_start_time    


################################ polishing stop
#len(unique_a_list) - len(solver_found_a)  
#min(unique_max_delta_pi_list[len(solver_found_a):]) 



##################### check results
best_idx = np.argmin(np.array(unique_max_delta_pi_list))
#incumbent_obj[best_idx] 
unique_max_delta_pi_list[best_idx] 
#incumbent_min_delta_pi[best_idx] 


#### best_a
best_candidate_a = unique_a_list[best_idx]

best_z = np.zeros((num_heroes, num_heroes))
best_y = np.zeros((num_heroes, num_heroes))

for i in range(num_heroes-1):
    for j in range(i+1, num_heroes):
        best_z[i][j] = MNL_params_dict[(i, j)][0] + sum(MNL_params_dict[(i, j)][p+1] * best_candidate_a[p][i] for p in range(design_len[i]) ) + sum(MNL_params_dict[(i, j)][design_len[i]+1+p2] * best_candidate_a[p2][j] for p2 in range(design_len[j]) )  
        best_y[i][j] = 1/(1 + math.exp(-best_z[i][j]))
        best_y[j][i] = 1 - best_y[i][j]

for i in range(num_heroes):
    best_y[i][i] = 0.5


m_LP = Model()
prob = np.zeros((num_heroes, ), dtype=object)
for i in range(num_heroes):
    prob[i] = m_LP.continuous_var(lb=0)
tt = m_LP.continuous_var(lb=0)

for j in range(num_heroes):
    m_LP.add_constraint(m_LP.sum(prob[i]*best_y[i,j] for i in range(num_heroes))>=0.5)
    m_LP.add_constraint(prob[j]<=tt)
    
m_LP.add_constraint(m_LP.sum(prob[i] for i in range(num_heroes))==1)
m_LP.minimize(tt)
m_LP.solve(log_output=False) 
true_max_pi = tt.solution_value 

Nash_Eq_strategy = [prob[i].solution_value for i in range(num_heroes)]

# [0.1429606336747004,
#  0.14443636970680296,
#  0.14048990623287974,
#  0.14409153278522288,
#  0.14237117613099465,
#  0.14458910757686844,
#  0.14106127389253093]


    
# #### find top 20 design with smallest incumbent_max_delta_pi   
sort_index = np.argsort(np.array(unique_max_delta_pi_list))
np.array(unique_max_delta_pi_list)[sort_index[0:20]]



for i in range(20):
    df = pd.DataFrame(unique_a_list[sort_index[i]])
    df.to_csv(f'top_{i+1}_design.csv', index=False, header = False)


