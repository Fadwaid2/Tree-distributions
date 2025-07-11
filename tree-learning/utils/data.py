import pandas as pd
import numpy as np 
import networkx as nx 

def node_neighbors(G):
    """
    Returns a dictionary where each node is mapped to a set of its neighbors in an undirected graph.
    Parameters:
    - G: NetworkX Graph (undirected)
   
    Returns:
    - neighbors_dict: Dictionary {node: set(neighbors)}
    """
    neighbors_dict = {node: set(G.neighbors(node)) for node in G.nodes}
    return neighbors_dict

def sample_from_cpt(prob, parent_values, k):
    if not parent_values:
        return np.random.choice(k) 
    neighbor_counts = np.bincount(parent_values, minlength=k)
    influence_probs = neighbor_counts / sum(neighbor_counts) 
    final_probs = prob * influence_probs + (1 - prob) * (1 / k)
    final_probs /= final_probs.sum()
    return np.random.choice(k, p=final_probs)

def generate_samples(G, k, num_samples=100, noise_prob=0.5):
    samples = []
    neighbors_map = node_neighbors(G)  # get neighbors for each node
    for _ in range(num_samples):
        sample = {}
        root = np.random.choice(list(G.nodes))
        sample[root] = int(np.random.rand() > 0.2)
        for node in G.nodes:
            if node == root:
                continue  
            neighbors = neighbors_map[node]
            neighbor_values = [sample[neighbor] for neighbor in neighbors if neighbor in sample]
            # sample based on neighbors' values
            sample[node] = sample_from_cpt(0.7, neighbor_values, k)

        # adding noise
        noisy_sample = {key: (value if np.random.rand() > noise_prob else k-1-value) 
                        for key, value in sample.items()}
        samples.append(list(noisy_sample.values()))  
    return samples

def create_synthetic_data(n, T, k, e = 0.6, noise = 0.5, tree = True): 
    """
        Generate synthetic data from random graph or tree (depending on bool velue of 'tree' parameter)
        Parameters : 
            n: number of variables
            T: number of samples
            k: alphabet size 
            e (erdos renyi parameter): probability for edge creation. 
            tree (bool): generate tree or random non tree otherwise 
    """
    if tree: 
        # generate tree graph 
        G = nx.generators.trees.random_tree(n)
    else: 
        # generate a random non tree graph  
        G= nx.erdos_renyi_graph(n,e)
    #store the true edges in a text file 
    filename = f"k{k}/{n}_n_{T}_ts.txt"
    with open(filename, "w") as file:
        file.write(f"{list(G.edges())}\n")
    data = pd.DataFrame(generate_samples(G, k=k, num_samples=T, noise_prob=noise))
    data.to_csv(f'/k{k}/{n}_n_{T}_ts.csv')
    return data

def conditional_distributions_set(data, k):
    """
    Computes the conditional probability tables (CPTs) for all variable pairs (used for the log likelihood on the test dataset).

    Parameters:
        data (pd.DataFrame)
        k: alphabet size  
    Returns:
        dict: CPTs in the form { (parents, node) -> pd.DataFrame }
    """
    data.columns = range(len(data.columns)) 
    all_obs = len(data)
    variables = list(data.columns)
    cond_pmf_values = {}
    for v in variables:
        possible_parents = [p for p in variables if p != v]  
        for p in possible_parents:
            alphabet = list(range(k)) 
            joint_states = data.groupby([v, p], observed=True).size().unstack(p).fillna(0)
            joint_states = joint_states.reindex(index=alphabet, columns=alphabet, fill_value=0).fillna(0)
            cond_pmf_values[(p, v)] = joint_states.div(joint_states.sum(axis=0), axis=1).fillna(0)
    return cond_pmf_values
