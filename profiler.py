from utils import *
from ergm import ERGM
from metrics import *

import numpy as np
import pandas as pd
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
    fitted_model = ERGM(n, metrics, is_directed=is_directed, sample_size=1000, n_mcmc_steps=n, seed_MCMC_proba=estimated_p_seed)
    grads, _ = fitted_model.fit(sampson_matrix, lr=0.005, opt_steps=300, sliding_grad_window_k=30, sample_pct_growth=0.05, lr_decay_pct=0.1, steps_for_decay=25, optimization_method="newton_raphson")

elif args.type == "ER":
    print("### PROFILING AN ER model""")
    
    number_of_edges_metric = NumberOfEdgesDirected()
    fitted_model = ERGM(n, [number_of_edges_metric], is_directed=True, n_networks_for_grad_estimation=500, n_mcmc_steps=50)
    grads, _ = fitted_model.fit(sampson_matrix, lr=1, opt_steps=100, sliding_grad_window_k=20, sample_pct_growth=0.05, optimization_method="newton_raphson")
elif args.type == "big_network":
    big_network_df = pd.read_csv("./connectome_data.csv")
    big_network_df = big_network_df.drop("Unnamed: 0", axis=1)

    W = big_network_df.values

    W = W[:50, :50]

    n = W.shape[0]
    is_directed = True

    estimated_p_seed = np.sum(W) / (n*(n-1))
    print(f"estimated_p_seed = {estimated_p_seed}")
    metrics = [NumberOfEdgesDirected(), TotalReciprocity()]

    fitted_model = ERGM(n, metrics, is_directed=is_directed, sample_size=2000, n_mcmc_steps=n, seed_MCMC_proba=estimated_p_seed)

    # # convergence_criterion = "zero_grad_norm"
    convergence_criterion = "hotelling"

    hotelling_conf = 0.99
    grads, hotelling = fitted_model.fit(W, lr=0.5, opt_steps=40, 
                            lr_decay_pct=0.05, steps_for_decay=1,
                            sliding_grad_window_k=20, sample_pct_growth=0.05, 
                            convergence_criterion=convergence_criterion, 
                            optimization_method="newton_raphson",
                            hotelling_confidence=hotelling_conf,
                            mcmc_burn_in=1000, mcmc_steps_per_sample=1000
                            )
elif args.type == "arxiv_network":
    W = np.loadtxt("./datasets/arxiv/csNE_adjacency_matrix.csv", delimiter=",")
    n_nodes = W.shape[0]
    is_directed = False

    metrics = [NumberOfEdgesUndirected(), UndirectedDegree()]

    mcmle_model = ERGM(n_nodes, metrics, is_directed=is_directed, fix_collinearity=True, sample_size=20000)
    mcmle_opt_results = mcmle_model.fit(W, no_mple=True, 
                                    steps_for_decay=1, theta_init_method="uniform", lr=1, 
                                    opt_steps=10, convergence_criterion="zero_grad_norm", mcmc_burn_in=1000,
                                    cov_matrix_estimation_method="multivariate_initial_sequence")

    print(f"Model parameters - ")
    print(mcmle_model.get_model_parameters())
else:
    raise ValueError(f"Type {args.type} is invalid")