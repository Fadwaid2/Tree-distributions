import numpy as np 

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
            list_edges.append((i, j))
            adjacency_matrix[i][j]=1
            adjacency_matrix[j][i]=1
    return adjacency_matrix



