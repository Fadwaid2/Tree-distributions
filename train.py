from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import json
import numpy as np

def main(args): 
    # Either generate synthetic data or load dataset
    if args.synthetic:
        print("Generating synthetic data...")
        train_data, test_data = generate_synthetic_data(args.n, args.seed)
    else:
        print(f"Reading train data from {args.train_data}")
        train_data = pd.read_csv(args.train_data, index_col=0)
        print(f"Reading test data from {args.test_data}")
        test_data = pd.read_csv(args.test_data, index_col=0)
    
    # Initialize weight matrix 
    w = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(w, 0)

    # Initialize the learning algorithm 
    if args.method == 'RWM':
        learner = RWM(epsilon=args.epsilon)
    elif args.method == 'OFDE':
        learner = OFDE()
    elif args.method == 'Chow-Liu':
        learner = 
        # TO DO : do something here because it is not online learning so run some function there 

    # Precompute conditional distributions 
    precomputed = learner.precompute_conditional_distributions()
    # Compute weights 
    precompute_weights = learner.learn_weights(precomputed) 

    # Online Learning loop over T samples  
    for t in range(1, args.T+1):
        # track current time step in algorithms 
        algo.current_time = t 
        structure = learner.learn_structure(w)
        w = learner.update_weight_matrix(w, structure, precomputed_weights)

    # evaluate 
    results = {'log-likelihood':,
               'shd ':, 
               'bayesian-test': 
                }


    # TO DO : Save results
    args.output_folder.mkdir(exist_ok=True)
    with open(args.output_folder / 'arguments.json', 'w') as f:
        json.dump(vars(args), f, default=str)
   # to edit here 
    with open(args.output_folder / 'results.json', 'w') as f:
        json.dump(results, f, default=list)


if __name__ == '__main__':
     
    parser = ArgumentParser(description='Learning Tree-structured distributions')

    parser.add_argument('--train_data', type=Path, help='Path to the training dataset (CSV)')
    parser.add_argument('--test_data', type=Path, help='Path to the testing dataset (CSV)')
    parser.add_argument('--synthetic', type=bool, help='Flag to generate synthetic data instead of loading')
    parser.add_argument('--output_folder', type=Path, required=True, help='Directory to save results and arguments')
    parser.add_argument('--n', type=int, required=True, help='Number of nodes (variables) in the distribution')
    parser.add_argument('--T', type=int, default=10, help='Number of time steps for online learning')
    parser.add_argument('--seed', type=int, default=22, help='Random seed for synthetic data')

    # Algorithms 
    algorithm = parser.add_argument_group('Method')
    algorithm.add_argument('--method', type=str, choices=['Chow-Liu', 'RWM', 'OFDE'], required=True, help='Algorithm to learn the tree distribution')
    algorithm.add_argument('--epsilon', type=float, default=0.9, help='Epsilon value for RWM algorithm')

    args = parser.parse_args()

    main(args)
