from copy import deepcopy
import numpy as np
from utils import *
from metrics import MetricsCollection


class Sampler():
    def __init__(self, thetas, network_stats_calculator):
        self.thetas = deepcopy(thetas)
        self.network_stats_calculator = deepcopy(network_stats_calculator)

    def sample(self, initial_state, n_iterations):
        pass


class NaiveMetropolisHastings(Sampler):
    def __init__(self, thetas, network_stats_calculator: MetricsCollection, burn_in=1000, steps_per_sample=10):
        """
        An implementation for the symmetric proposal Metropolis-Hastings algorithm for ERGMS, using the logit
        of the acceptance rate. See docs for more details.
        Throughout this implementation, networks are represented as adjacency matrices.

        Parameters
        ----------
        thetas : np.ndarray
            Coefficients of the ERGM
        
        network_stats_calculator : MetricsCollection
            A MetricsCollection object that can calculate statistics of a network.
        """
        super().__init__(thetas, network_stats_calculator)

        ## TODO - these two params need to be dependent on the network size
        self.burn_in = burn_in
        self.steps_per_sample = steps_per_sample


    def _calculate_weighted_change_score(self, current_network, indices: tuple):
        """
        Calculate g(proposed_network)-g(current_network) and then inner product with thetas.
        """
        change_score = self.network_stats_calculator.calc_change_scores(current_network, indices)
        return np.dot(self.thetas, change_score)

    def _flip_network_edge(self, current_network, i, j):
        """
        Flip the edge between nodes i, j. If it's an undirected network, we flip entries W_i,j and W_j,i.
        NOTE! This function changes the network that is passed by reference
        """
        current_network[i, j] = 1 - current_network[i, j]
        if not self.network_stats_calculator.is_directed:
            current_network[j, i] = 1 - current_network[j, i]


    # def _flip_network_edge(self, current_network, i, j):
    #     """
    #     Flip the edge between nodes i, j. If it's an undirected network, we flip entries W_i,j and W_j,i.
    #     NOTE! This function changes the network that is passed by reference
    #     """
    #     _is_directed = self.network_stats_calculator.is_directed
    #     self._static_flip_edge(current_network, i, j, _is_directed)
    
    # @staticmethod
    # @njit
    # def _static_flip_edge(current_network, i, j, _is_directed):
    #     current_network[i, j] = 1 - current_network[i, j]

    #     if not _is_directed:
    #         current_network[j, i] = 1 - current_network[j, i]

    def sample(self, initial_state, num_of_nets, replace=True):
        """
        Sample networks using the Metropolis-Hastings algorithm.
        
        Parameters
        ----------
        initial_state : np.ndarray
            The initial network to start the Markov Chain from
        
        num_of_nets : int
            The number of networks to sample
        
        replace : bool
            A boolean flag indicating whether we sample with our without replacement. replace=True means networks can be duplicated.
        """
        current_network = initial_state.copy()

        net_size = current_network.shape[0]

        sampled_networks = np.zeros((net_size, net_size, num_of_nets))

        
        edges_to_flip = get_random_edges_to_flip(net_size, self.burn_in + (num_of_nets * self.steps_per_sample)) 
        random_nums_for_change_acceptance = np.random.rand(self.burn_in + (num_of_nets * self.steps_per_sample))


        networks_count = 0
        mcmc_iter_count = 0

        while networks_count != num_of_nets:
            random_entry = edges_to_flip[:, mcmc_iter_count % edges_to_flip.shape[1]]

            change_score = self._calculate_weighted_change_score(current_network, random_entry)
            
            rand_num = random_nums_for_change_acceptance[mcmc_iter_count % edges_to_flip.shape[1]]
            perform_change = change_score >= 1 or rand_num <= min(1, np.exp(change_score))
            if perform_change:
                self._flip_network_edge(current_network, random_entry[0], random_entry[1])

            if (mcmc_iter_count - self.burn_in) % self.steps_per_sample == 0:
                sampled_networks[:, :, networks_count] = current_network

                if not replace: 
                    ## TODO - Since we're flipping coins in the beginning, we're reusing them in case of the non-replacement (since we might need more coin flips that num of networks).
                    ## Consider reflipping coins to get better "pseudorandom properties", in the case that we run out of pre-flipped coins
                    if np.unique(sampled_networks[:, :, :networks_count + 1], axis=2).shape[2] == networks_count + 1:
                        networks_count += 1
                    else:
                        sampled_networks[:, :, networks_count] = np.zeros((net_size, net_size))
                else:
                    networks_count += 1

            mcmc_iter_count += 1

        return sampled_networks
