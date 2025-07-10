import numpy as np
import pandas as pd
import networkx as nx
import random

from sklearn.metrics import mutual_info_score

from utils import *

## TO DO : import Kruskal directly from utils 

class Chow_Liu(TreeLearner):

    def __init__(self, data):
        super(Chow_Liu, self).__init__(data=data)
        self.data = data 
        self.n = data.shape[1]
        self.T = data.shape[0]
    
    def mi(self):
        """
        Compute mutual information (mi) values: mi values for all the data steps 
        directly taken using sklearn package 
        """
        mi = {}
        for i in range(self.n):
            for j in range(i+1,self.n):
                labels_i = self.data.iloc[:,i].tolist()
                labels_j = self.data.iloc[:,j].tolist()
                mi[(i,j)] = mutual_info_score(labels_i, labels_j)
        return mi 

    def learn_weights(self, precomputed_mi):
        w = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(i+1,self.n):
                w[i][j] = precomputed_mi[(i,j)]
                w[j][i] = w[i][j] 
        return w

    def learn_structure(self):
        # call Kruskal from utils.py 
        # weights need to be learnt first 
        
        w = learn_weights()
        structure = kruskal_algo(w)
        
        return w, structure 
    




#############################
# Chow-Liu code 
############################# 

def chowliu(data):
    
    T = len(data)
    n = data.shape[1]
    precomputed_mi = mi(data)

    w = get_weights(data, precomputed_mi)
    structure = kruskal_algo(w)
                
    return w, structure

##############################
# running code on datat example 
##############################
def run_chow_liu(n,T,k):
    
        dataset = pd.read_csv(f'/home/fadwa/Desktop/datasets_tree_distribution/16mar/k{k}/{n}_n_{T}_ts.csv',index_col=0)
        w, f = chowliu(dataset)
        np.savetxt(f"/home/fadwa/Desktop/datasets_tree_distribution/16mar/k{k}/ChowLiu/{n}_n_{T}_ts_w_{n}_n_{T}_ts.csv", w, delimiter=",")    
        np.savetxt(f"/home/fadwa/Desktop/datasets_tree_distribution/16mar/k{k}/ChowLiu/{n}_n_{T}_ts_adj_{n}_n_{T}_ts.csv", f, delimiter=",")    

        filename = f"/home/fadwa/Desktop/datasets_tree_distribution/16mar/k{k}/ChowLiu/{n}_n_{T}_ts_{n}_n_{T}_ts.txt"
        with open(filename, "w") as file:
            file.write("w (weight matrix), f (adj matrix from Kruskal)\n")
            file.write(f"{w}\n")
            file.write(f"{f}\n")
           
#synthetic data    
n_list = [5,5,10,10,20,20,50,50,100,100]
t_list = [20,50,20,50,50,100,500,1000,500,1000]
k = 5
for i in range(len(n_list)):
    run_chow_liu(n_list[i],t_list[i],k)

