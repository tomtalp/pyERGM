import numpy as np
from utils import *

class Sampler():
    def __init__(self):
        pass

    def sample(self, n_iter):
        pass

class Gibbs(Sampler):
    def __init__(self, network_stats_calculator, is_directed):
        super().__init__()

        self.network_stats_calculator = network_stats_calculator
        self.is_directed = is_directed
    
    def _calculate_diff_statistics(self, y_plus, y_minus):
        stats1 = self.network_stats_calculator.calculate_statistics(y_plus)
        stats2 = self.network_stats_calculator.calculate_statistics(y_minus)
        
        diff_stats = stats1 - stats2

        return diff_stats
    
    def sample(self, seed_network, parameters, n_iter=500):
        n = seed_network.shape[0]

        current_network = seed_network.copy()

        for i in range(n_iter):
            # print(f"iter {i}")

            random_entry = get_random_nondiagonal_matrix_entry(n)
            
            candidate_a = perturb_network_by_overriding_edge(current_network, 1, random_entry[0], random_entry[1], self.is_directed)
            candidate_b = perturb_network_by_overriding_edge(current_network, 0, random_entry[0], random_entry[1], self.is_directed)

            # print("candidate_a")
            # print(candidate_a)
            # print("candidate_b")
            # print(candidate_b)

            diff_stats = self._calculate_diff_statistics(candidate_a, candidate_b)
            # print("DIFF STATS - ")
            # print(diff_stats)

            weighted_diff = np.dot(diff_stats, parameters)
            # print(f"weighted_diff: {weighted_diff}")

            acceptante_proba = min(1, np.exp(weighted_diff))
            # print(f"acceptance proba: {acceptante_proba}")

            if np.random.rand() < acceptante_proba:
                current_network = candidate_a.copy()
            else:
                current_network = candidate_b.copy()
        
        return current_network