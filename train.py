from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np


def main(args): 

    #initialize weight matrix 
    w = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(w, 0)

    # precompute conditional distributions 
    precomputed = algo.precompute_conditional_distributions()
    
    #compute weights 
    precompute_weights = algo.learn_weights(precomputed) 

    # online learning loop in range T 
    for t in range(1, T+1):
        # track current time step in algorithms 
        algo.current_time = t 

        structure = algo.learn_structure(w)
        w = algo.update_weight_matrix(w, structure, precomputed_weights)

    # evaluate 
    results = {'to do':to do,
               ' ': }


    # TO DO : Save results
    args.output_folder.mkdir(exist_ok=True)
    with open(args.output_folder / 'arguments.json', 'w') as f:
        json.dump(vars(args), f, default=str)
   # to edit here 
    with open(args.output_folder / 'results.json', 'w') as f:
        json.dump(results, f, default=list)


if __name__ == '__main__':
     
    parser = ArgumentParser(description='Learn Tree-structured distribution')

    # To do : add arguments in parser here 

    algorithm = parser.add_argument_group('Method')
    algorithm.add_argument('--method', type=str, choices = ['Chow-Liu','RWM','OFDE'], help='select an algorithm to learn the tree distribution.')
    algorithm.add_argument('--epsilon', type=float, default=0.9, help='choose an epsilon value for RWM algorithm.')

    args = parser.parse_args()

    main(args)
