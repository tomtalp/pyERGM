from copy import deepcopy
import numpy as np
from utils import *

class Sampler():
    def __init__(self, thetas, network_stats_calculator, is_directed=False):
        self.thetas = deepcopy(thetas)
        self.network_stats_calculator = deepcopy(network_stats_calculator)
        self.is_directed = is_directed

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
        
        is_directed : bool
            A boolean flag indicating whether the network is directed or not.
        
        """
        super().__init__(thetas, network_stats_calculator, is_directed)

    def flip_network_edge(self, current_network, i, j):
        """
        Flip the edge between nodes i, j. If it's an undirected network, we flip entries W_i,j and W_j,i.
        """
        proposed_network = current_network.copy()
        proposed_network[i, j] = 1 - proposed_network[i, j]
        
        if not self.is_directed:
            proposed_network[j, i] = 1 - proposed_network[j, i]
        
        return proposed_network
    
    def _calculate_weighted_change_score(self, proposed_network, current_network):
        """
        Calculate g(proposed_network)-g(current_network) and then inner product with thetas.
        """
        g_proposed = self.network_stats_calculator.calculate_statistics(proposed_network)
        g_current = self.network_stats_calculator.calculate_statistics(current_network)
        change_score = g_proposed - g_current

        return np.dot(self.thetas, change_score)

    def sample(self, initial_state, n_iterations):
        current_network = initial_state.copy()

        for i in range(n_iterations):
            random_entry = get_random_nondiagonal_matrix_entry(current_network.shape[0])
            
            proposed_network = self.flip_network_edge(current_network, random_entry[0], random_entry[1])

            change_score = self._calculate_weighted_change_score(proposed_network, current_network)
            acceptance_proba = min(1, np.exp(change_score))

            if np.random.rand() <= acceptance_proba:
                current_network = proposed_network.copy()
            else:
                current_network = current_network.copy()
            
        return current_network