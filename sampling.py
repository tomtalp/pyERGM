import numpy as np
from utils import *

class Sampler():
    def sample(self, initial_state, n_iterations):
        pass

class NaiveMetropolisHastings(Sampler):
    def __init__(self, thetas, network_stats_calculator, is_directed=False):
        """
        An implementation for the symmetric proposal Metropolis-Hastings algorithm for ERGMS, using the logit
        of the acceptance rate. See docs for more details.
        Throughout this implementation, networks are represented as adjacency matrices.

        Parameters
        ----------
        thetas : np.ndarray
            Coefficients of the ERGM
        
        network_stats_calculator : NetworkStatistics
            A NetworkStatistics object that can calculate statistics of a network.
        
        """
        super().__init__()
        self.thetas = thetas
        self.network_stats_calculator = network_stats_calculator
        self.is_directed = is_directed

    def override_network_edge(self, network, i, j, value):
        """
        Override the edge between nodes i and j with the value `value` in the network.
        """
        if value not in [0, 1]:
            raise ValueError("Naive MH sampling only has dyads as edges. Value must be 0 or 1.")
        
        perturbed_net = network.copy()
        perturbed_net[i, j] = value
        if not self.is_directed:
            perturbed_net[j, i] = value

        return perturbed_net
    
    def _calculate_weighted_change_score(self, y_plus, y_minus):
        """
        Calculate g(y_plus)-g(y_minus) and then inner product with thetas.
        """
        g_plus = self.network_stats_calculator.calculate_statistics(y_plus)
        g_minus = self.network_stats_calculator.calculate_statistics(y_minus)
        change_score = g_plus - g_minus

        return np.dot(self.thetas, change_score)

    def sample(self, initial_state, n_iterations):
        current_network = initial_state.copy()

        for i in range(n_iterations):
            random_entry = get_random_nondiagonal_matrix_entry(current_network.shape[0])

            y_plus = self.override_network_edge(current_network, random_entry[0], random_entry[1], 1)
            y_minus = self.override_network_edge(current_network, random_entry[0], random_entry[1], 0)

            change_score = self._calculate_weighted_change_score(y_plus, y_minus)
            acceptance_proba = min(1, np.exp(change_score))

            if np.random.rand() < acceptance_proba:
                current_network = y_plus.copy()
            else:
                current_network = y_minus.copy()
            
        return current_network