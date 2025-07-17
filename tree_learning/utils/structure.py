import numpy as np
from numba import jit

#########################################
# using numba jit for faster running time 
#########################################

@jit(nopython=True)
def indegree_matrix(w):
    n = w.shape[0]
    D_in = np.zeros((n, n))
    # create a list of neighbors and compute vertex weights
    weight_vertex = np.sum(w, axis=1)
    for i in range(n):
        D_in[i][i] = weight_vertex[i]
    return D_in

@jit(nopython=True)
def Lapl_1(w):
    D = indegree_matrix(w)
    L_1 = D - w
    return L_1


@jit(nopython=True)
def Lapl_1_r(w, r):
    l_1 = Lapl_1(w)
    return remove_r(l_1, r)

@jit(nopython=True)
def remove_r(arr, r):
    return arr[np.arange(arr.shape[0]) != r][:, np.arange(arr.shape[1]) != r]

@jit(nopython=True)
def delete(w, e):
    n = w.shape[0]
    i, j = e
    delete_e_w = np.copy(w)
    delete_e_w[i, j] = 0
    delete_e_w[j, i] = 0
    return delete_e_w

@jit(nopython=True)
def determinant(matrix):
    det = np.linalg.det(matrix)
    return det

@jit(nopython=True)
def contraction(adj_matrix, e):
    """
    Returns the new contracted adjacency matrix.
    i, j = e; i is always the root (0) and j is merged into it.
    The other nodes k are moved to k-1 in the new matrix.
    """
    i, j = e
    n = len(adj_matrix)
    # compute new row/column by summing the weights of node j into node 0
    new_weights = np.copy(adj_matrix[0])
    new_weights += adj_matrix[j]
    # remove self-loops
    new_weights[0] = 0
    new_weights[j] = 0
    # create new reduced matrix with size n-1, n-1
    reduced_size = n-1
    new_matrix = np.zeros((reduced_size, reduced_size))
    # fill new matrix from old one 
    row_idx = 0
    for old_row in range(n):
        if old_row in (0, j):  # skip row 0 (already updated) and row j (merged)
            continue
        col_idx = 0
        for old_col in range(n):
            if old_col in (0, j):  
                continue
            new_matrix[row_idx, col_idx] = adj_matrix[old_row, old_col]
            col_idx += 1
        row_idx += 1
    # insert the new merged row/column
    new_matrix[0, 1:] = new_weights[2:]
    new_matrix[1:, 0] = new_weights[2:]
    return new_matrix

@jit(nopython=True)
def insert_good_edge(u,v):
    """
    Helper funtion: all edges must have form (u,v) with u<v. 
    """
    if u<v:
        return (u,v)
    else:
        return (v,u)

@jit(nopython=True)
def get_cm_orig_vertices(w_after_contraction,w_before_contraction,e,previous_cm):
    """
    Returns the contraction mapping (CM) with the correct values from the previous cm. 
    Takes :  
    e : contracted edge
    previous_cm : the contraction_step-1 CM from which we take the original values (name of the edge in the original graph)
    """
    cm = {}
    i,j = e 
    
    for u in range(w_after_contraction.shape[0]):  
        for v in range(w_after_contraction.shape[1]):
            if u<v:
               if u==j-1: #contracted edge name in the new adj
                first, second = insert_good_edge(i, v+1) 
                w_1 = w_before_contraction[first, second]
                first, second = insert_good_edge(j, v+1)
                w_2 = w_before_contraction[first, second]
                rand = np.random.uniform(0,1)
                if rand < w_1/(w_1+w_2):
                    cm[(u,v)] = insert_good_edge(i,v+1)
                else:
                    cm[(u,v)] = insert_good_edge(j,v+1)
               elif v==j-1:
                first, second = insert_good_edge(i, u+1)
                w_1 = w_before_contraction[first, second]
                first, second = insert_good_edge(j, u+1)
                w_2 = w_before_contraction[first, second]
                rand = np.random.uniform(0,1)
                if rand < w_1/(w_1+w_2):
                    cm[(u,v)] = insert_good_edge(i,u+1)
                else:
                    cm[(u,v)] = insert_good_edge(j,u+1)
               elif u!=j-1 and v!=j-1:
                cm[(u,v)] = insert_good_edge(u+1,v+1)
    #now we iterate over the constructed CM and we change the values into the values of that value (which will be key in the previous CM)
    for key,val in cm.items():
        for key_prev,val_prev in previous_cm.items():
            if val == key_prev:
                cm[key] = val_prev
    return cm 

@jit(nopython=True)
def initialize_cm(w):
    """
    Initial contraction map has same edges in keys and values.
    """
    initial_cm = {}
    n = w.shape[0]
    for u in range(n): 
        for v in range(n):
            if u<v: 
                e = (u,v)
                initial_cm[e]=e 
    return initial_cm 

@jit(nopython=True)
def choose_edge(CM_current,contracted_edge,edges):
    """
    Selects the edge CM_current[e]
    """
    k=0
    k, j_0 = contracted_edge # the first index will always be zero but we keep it nonetheless to avoid confusion
    if edges.size==0:#when edges list is empty: this is the first step where we add the first contracted edge to the list of edges in the spanning tree 
        edge = contracted_edge
    else:
        edge = CM_current[contracted_edge]
    return edge 

#############################
# Sampling a spanning tree 
#############################
@jit(nopython=True)
def sampler(w, r=0):
    w = w.astype(np.float64)
    n=w.shape[0]
    edges = np.zeros((n-1, 2), dtype=np.int64)
    edge_count = 0
    CM_previous = initialize_cm(w) 
    
    i = 1
    while True: 
        it = 0
        while edge_count != n-1:
            e = (0, i)
            deleted_adj = delete(w, e)
            arr1 = Lapl_1_r(deleted_adj,r=r)
            arr2 = Lapl_1_r(w,r=r)
            p_e = determinant(arr1-arr2) #matrix tree theorem 
           
            u = (np.random.uniform(0,1))
            if p_e <= u:
                adj_before = w  
                w = contraction(w, e) # contract the edge {0,i}
                CM_current = get_cm_orig_vertices(w,adj_before,e,CM_previous)
                edges[edge_count] = choose_edge(CM_previous, e, edges[:edge_count])
                edge_count += 1
                CM_previous = CM_current 
                #del CM_current
                #del adj_before
                #gc.collect() # empty memory (needed for big experiments) 
                i = 1 #reset i to 1 again because we restart from the root 0 everytime    
            else:
                w=delete(w,e)
                i+=1  
            it+=1
            if edge_count == n-1:
                return edges



# Maximum-weight spanning tree (MWST) algorithm : kruskal's algorithm 
## SOURCE : https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
## Citation : This code is contributed by Neelam Yadav
## Citation : Improved by James Graça-Jones

# Class for union and find operations in kruskal's algorithm 
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    
    def union(self, u, v):
    # Merge Sets: If the roots are different, the subsets are merged. The subset with the smaller rank is attached to the root of the subset with the larger rank. If both ranks are equal, one root becomes the parent of the other, and its rank is incremented.
    
        root_u = self.find(u)
        root_v = self.find(v)
        
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
                
def kruskal_algo(w): 
    n = w.shape[0]
    list_edges = []
    edges = []
    adjacency_matrix = np.zeros((n,n))
    e = 0 #to index the edges in the result
    
    for i in range(n):
        for j in range(i+1,n):
            if w[i][j]!=0:
                edges.append((w[i][j] , i, j ))
    edges.sort() #sorting edges by non decreasing weight
    
    uf = UnionFind(n)
    
    for weight, i, j in edges:
        if uf.find(i) != uf.find(j):
            uf.union(i, j)
            list_edges.append((min(i,j), max(i,j)))
            adjacency_matrix[i][j]=1
            adjacency_matrix[j][i]=1
    
    return list_edges



