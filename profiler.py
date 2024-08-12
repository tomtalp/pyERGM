from utils import *
from ergm import ERGM
from metrics import *

import numpy as np

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--type", type=str)
args = argparser.parse_args()

sampson_matrix = np.array(
    [[0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], 
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0], 
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1], 
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], 
    [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
    [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
    [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], 
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], 
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], 
    [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], 
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], 
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 
    [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]]
)

n = sampson_matrix.shape[0]

if args.type == "p1":
    print("### PROFILING A p1 model""")
    
    metrics = [NumberOfEdgesDirected(), OutDegree(), InDegree(), TotalReciprocity()]
    is_directed = True
    estimated_p_seed = np.sum(sampson_matrix) / (n*(n-1))
    fitted_model = ERGM(n, metrics, is_directed=is_directed, n_networks_for_grad_estimation=500, n_mcmc_steps=100, seed_MCMC_proba=estimated_p_seed)
    grads, _ = fitted_model.fit(sampson_matrix, lr=0.01, opt_steps=400, sliding_grad_window_k=30, sample_pct_growth=0.05, lr_decay_pct=0.1, steps_for_decay=50, optimization_method="newton_raphson")

elif args.type == "ER":
    print("### PROFILING AN ER model""")
    
    number_of_edges_metric = NumberOfEdgesDirected()
    fitted_model = ERGM(n, [number_of_edges_metric], is_directed=True, n_networks_for_grad_estimation=500, n_mcmc_steps=50)
    grads, _ = fitted_model.fit(sampson_matrix, lr=1, opt_steps=100, sliding_grad_window_k=20, sample_pct_growth=0.05, optimization_method="newton_raphson")

else:
    raise ValueError("Invalid `type` argument. Choose between 'p1' and 'ER'.")