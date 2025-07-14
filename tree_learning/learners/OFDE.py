import numpy as np
#import pandas as pd
import random

from collections import Counter

from tree_learning.utils.structure import kruskal_algo
from .base import TreeLearner

class OFDE(TreeLearner):

    def __init__(self, data, k):
        super(OFDE, self).__init__(data=data, k=k)

        #initialize vector p for OFDE
        self.p = np.zeros(self.n*(self.n-1)//2)

    def conditional_distributions_set(self, no_parent=False):
        """
        Getting all the occurrences for all the sets of parents-children for all the possible edges in a graph
        """
        columns = range(len(self.data.columns))
        variables = list(columns)
        cond_pmf_values = {}
        for v in variables:
            #added to replace function one_var_conditional_distributions_set called in run_ofde
            if no_parent: 
                joint_states = self.data.groupby([v] , observed=True).size().fillna(0)
                cond_pmf_values[(v)] = joint_states 
            else: 
                parents = list(self.data.columns)
                parents.remove(v) #possible parents
                for p in parents:
                    #calculate joint P(node1 and node2)
                    joint_states = self.data.groupby([v] + [p], observed=True).size().unstack(p).fillna(0)
                    cond_pmf_values[(v, p)] = joint_states # conditional proba P(Xi = xi|Xj = xj)
        return cond_pmf_values

    def precompute_conditional_distributions(self):
        cond_probas = self.conditional_distributions_set(no_parent=False)
        cond_proba_one_var = self.conditional_distributions_set(no_parent=True)
        return {
            "cond_probas": cond_probas,
            "cond_proba_one_var": cond_proba_one_var
        }

    def edge_cond_proba_dict(self, edge, Ne, one_var=False):
        """
        To retrieve the occurrence of a certain edge from the total previous dict containing all edges
        """

        if one_var: 
            proba_dict = {key:1/self.k for key in [i for i in range(self.k)]} # 1 variable 
        else: 
            keys = [(i, j) for i in range(self.k) for j in range(self.k)]
            proba_dict = {key: 1 / self.k**2 for key in keys} # 2 vars 

        for x, val in Ne.items():
            if x==edge:
                df = Ne[x]
                for row in df.index:
                    if one_var: 
                        proba_dict[row]= ((df.at[row]+1/2)/(self.T+self.k/2))
                    else:
                        for col in df.columns: 
                            proba_dict[(row, col)]= ((df.at[row, col]+1/2)/(self.T+(self.k**2)/2)) 
        return proba_dict


    def get_weight_phi(self, p, c, precomputed_cond_probas, precomputed_cond_proba_1_var):
        """
        Compute weight phi 
        """
        t = self.current_time
        value_i = self.data.iloc[t-1,p]
        value_j = self.data.iloc[t-1,c]

        two_d_marginal = self.edge_cond_proba_dict((p,c), precomputed_cond_probas) 
        one_d_p = self.edge_cond_proba_dict(p, precomputed_cond_proba_1_var, one_var=True) 
        one_d_c = self.edge_cond_proba_dict(c, precomputed_cond_proba_1_var, one_var=True)

        #Filter out unnecessary values and only keep those in dataset : phi calculation are only done for relevant values 
        
        theta_i = one_d_p[value_i]
        theta_j = one_d_c[value_j]
        theta_ij = two_d_marginal[(value_i,value_j)]
        return theta_i * theta_j / theta_ij if theta_ij > 0 else 0

    def learn_weights(self, precomputed):

        #from function precompute_phi 
        precomputed_phi = {}
        precomputed_cond_probas = precomputed['cond_probas']
        precomputed_cond_proba_1_var = precomputed['cond_proba_one_var']

        for i in range(self.n):
            for j in range(i+1, self.n):
                #for each (parent,child) we have a list 
                dpt_list = []
                for t in range(1, self.T+1):
                    self.current_time = t
                    phi_time_t = self.get_weight_phi(i, j, precomputed_cond_probas, precomputed_cond_proba_1_var)
                    dpt_list.append(phi_time_t)
                precomputed_phi[(i,j)] = dpt_list
        return precomputed_phi

    def update_weight_matrix(self, w, structure, precomputed_phi, **kwargs):
        """
        Follow-the-Perturbed-Leader algorithm 
            t is the current time step 
            w is the weight matrix to update with FPL 
        """
        t = self.current_time
        w = w.copy()
        #horizon independent case
        beta = (1/self.n)*np.sqrt(2/t) 

        for i in range(self.n):
            for j in range(i+1,self.n):
                #different perturbations are added to different edges 
                perturbation = random.uniform(0, 1/beta)
                print('precomputed_phi',precomputed_phi)
                phi_i_j = precomputed_phi[(i,j)][t-1]   
                w[i][j] = perturbation + phi_i_j
                w[j][i] = w[i][j] 

        return w

    def learn_structure(self, w, **kwargs):
        """
        Run Kruskal's algorithm and update p via swap rounding.
        """
        structure = kruskal_algo(w)
        p_intermediate = self.generate_p_vector(structure)
        self.p = self.rounding(p_intermediate)
        return structure 


    def generate_p_vector(self, f):
        """
        Generate p vector for Swap rounding method to project on matroid 
            p: a numpy array with unique non-zero edges
            obtained from the previous time step  
            f: the structure edges outputed by Kruskal's algorithm 
            alpha: the mixing variable
        """
        alpha = 1/(4*np.sqrt(2*self.current_time))
        # f is the list of edges so convert it to a numpy array with 0 and 1 
        f_array = []
        for i in range(self.n):
            for j in range(i+1,self.n):
                if f[i][j]!=0:
                    f_array.append(1)  #when edge exists
                else:
                    f_array.append(0)  #no edge
                    
        f_array = np.array(f_array)
        return alpha*self.p +(1-alpha)*f_array

    @staticmethod
    def rounding(x):
        """
        Swap rounding method to project on matroid 
        Input: x: this will be the fractional point aka the weight that FPL gave as output 
        """
        x = x.copy()
        length = len(x)-1
        
        # Arbitrarely select two components in the convex combination i and j 
        # with condition: i should be different from j 
        while True:
            i = random.randint(0, length)
            j = random.randint(0, length)
            if i != j:
                break
        
        if x[i] == 0 and x[j] == 0: #if so select a new index i 
            i = random.randint(0,t)
        elif x[i] + x[j] <= 1:
            if np.random.rand() < x[i] / (x[i] + x[j]):
                x[i], x[j] = x[i] + x[j], 0
            else:
                x[i], x[j] = 0, x[i] + x[j]
        else:
            if np.random.rand() < (1 - x[j]) / (2 - x[i] - x[j]):
                x[i], x[j] = 1, x[i] + x[j] - 1
            else:
                x[i], x[j] = x[i] + x[j] - 1, 1         
        return x 


