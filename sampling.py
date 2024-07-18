import numpy as np
from utils import *

class Sampler():
    def __init__(self):
        pass

    def sample(self, n_iter):
        pass

class Gibbs(Sampler):
    def __init__(self, feature_functions):
        super().__init__()

        self.feature_functions = feature_functions
    
    def _calculate_diff_statistics(self, W1, W2):
        G1 = connectivity_matrix_to_G(W1)
        G2 = connectivity_matrix_to_G(W2)

        stats1 = [f(G1) for f in self.feature_functions]
        stats2 = [f(G2) for f in self.feature_functions]

        diff_stats = np.array(stats2) - np.array(stats1)

        return diff_stats

    def sample(self, seed_network, n_iter=500):
        n = seed_network.shape[0]

        network = seed_network.copy()

        for i in range(n_iter):
            random_entry = get_random_nondiagonal_matrix_entry(n)

            perturbed_net = network.copy()
            perturbed_net[random_entry] = 1 - perturbed_net[random_entry]

            


        
        


# class MCMC():
#     def __init__(self, G, model):
#         self.G = G
#         self.model = model

#     def sample(self, n_iter):
#         for i in range(n_iter):
#             # Propose a new graph
#             G_proposed = self.propose()
#             # Calculate the acceptance probability
#             acceptance_prob = self.calculate_acceptance_prob(G_proposed)
#             # Accept or reject the proposal
#             if acceptance_prob > np.random.rand():
#                 self.G = G_proposed