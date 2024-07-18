import numpy as np
from utils import *

class Sampler():
    def __init__(self):
        pass

    def sample(self, n_iter):
        pass

class Gibbs(Sampler):
    def __init__(self, network_stats_calculator):
        super().__init__()

        self.network_stats_calculator = network_stats_calculator
    
    def _calculate_diff_statistics(self, y_plus, y_minus):
        stats1 = self.network_stats_calculator.calculate_statistics(y_plus)
        stats2 = self.network_stats_calculator.calculate_statistics(y_minus)
        print(f"stats1: {stats1}, stats2: {stats2}")
        diff_stats = stats1 - stats2

        return diff_stats

    def sample(self, seed_network, n_iter=500):
        n = seed_network.shape[0]

        network = seed_network.copy()

        for i in range(n_iter):
            print(f"iter {i}")
            random_entry = get_random_nondiagonal_matrix_entry(n)
            
            network[random_entry] = 1
            network[random_entry[::-1]] = 1

            perturbed_net = network.copy()
            perturbed_net[random_entry] = 0
            perturbed_net[random_entry[::-1]] = 0

            print("network")
            print(network)
            print("perturbed_net")
            print(perturbed_net)

            diff_stats = self._calculate_diff_statistics(network, perturbed_net)
            print("DIFF STATS - ")
            print(diff_stats)
            


        
        


# # class MCMC():
# #     def __init__(self, G, model):
# #         self.G = G
# #         self.model = model

# #     def sample(self, n_iter):
# #         for i in range(n_iter):
# #             # Propose a new graph
# #             G_proposed = self.propose()
# #             # Calculate the acceptance probability
# #             acceptance_prob = self.calculate_acceptance_prob(G_proposed)
# #             # Accept or reject the proposal
# #             if acceptance_prob > np.random.rand():
# #                 self.G = G_proposed