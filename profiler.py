from pyERGM.utils import *
from pyERGM.ergm import ERGM
from pyERGM.metrics import *

import numpy as np
import pandas as pd
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--type", type=str)
args = argparser.parse_args()

# sampson_matrix = np.array(
#     [[0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], 
#     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0], 
#     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1], 
#     [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
#     [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], 
#     [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#     [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
#     [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
#     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
#     [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], 
#     [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], 
#     [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
#     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
#     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], 
#     [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], 
#     [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], 
#     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 
#     [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]]
# )

# n = sampson_matrix.shape[0]
# is_directed = True

# metrics = [NumberOfEdgesDirected(), InDegree(), OutDegree(), TotalReciprocity()]

# fitted_model = ERGM(n, metrics, is_directed=is_directed)
# nets = fitted_model.generate_networks_for_sample(sample_size=500, sampling_method="metropolis_hastings", burn_in=500000, mcmc_steps_per_sample=50000)



import pandas as pd
from pyERGM.ergm import ERGM
from pyERGM.metrics import *
import pickle
from sklearn.metrics import roc_auc_score
import os
import sys

def edit_connectivity(full_connectivity_df):
    connectivity_array = full_connectivity_df.to_numpy(copy=True)
    connectivity_array[connectivity_array>1] = 1
    np.fill_diagonal(connectivity_array, 0)
    return connectivity_array

path = "datasets/gal/"
attributes = pd.read_csv(path + "cellinfo_cook_Adam.csv", index_col = 0).reset_index(drop = True)
full_connectivity = pd.read_csv(path + "connectome_cook_Adam.csv", index_col = 0)
binary_connectivity = edit_connectivity(full_connectivity)

########################################################################################################################
# train model on the full data
########################################################################################################################

n_nodes = len(attributes)

# define metrics
num_edges = NumberOfEdgesDirected()
total_reciprocity = TotalReciprocity()
cell_types_metric = NumberOfEdgesTypesDirected(list(attributes["subtype"]))
rich_club_metric = NumberOfEdgesTypesDirected(list(attributes["rich_club"]))
distances_metric = SumDistancesConnectedNeurons(list(attributes["pos"]), is_directed=True)

# train the model
model = ERGM(n_nodes, [num_edges, total_reciprocity, cell_types_metric, rich_club_metric, distances_metric], is_directed=True)
# res_model = model.fit(binary_connectivity, convergence_criterion = "model_bootstrap", num_model_sub_samples = 50, model_subsample_size=2000, mcmc_burn_in=5*10**5, mcmc_sample_size=7500, mcmc_steps_per_sample=5*10**4, steps_for_decay=1)
model_parameters = model.get_model_parameters()

# ########################################################################################################################
# # generate new networks
# ########################################################################################################################
Nsim = 500
generated_networks = model.generate_networks_for_sample(sampling_method="metropolis_hastings", sample_size= Nsim, burn_in=5*10**5, mcmc_steps_per_sample=5*10**4)