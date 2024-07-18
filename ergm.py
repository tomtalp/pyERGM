import numpy as np
import networkx as nx
import sampling

from utils import *


class ERGM():
    """
    A representation of an ERGM model. 

    The model is initialized by registering `feature functions` of a graph.
    
    ## TODO - Thetas are currently set to 1 while in dev. We need to fit them!
    """
    def __init__(self, n_nodes, network_statistics: NetworkStatistics, is_directed=False):
        self._n_nodes = n_nodes
        self._thetas = None
        self._feature_functions = []
        self._normalization_factor = None ## TODO - Partition function is first calculated in fit. What should we initialize it to be?
        self._network_statistics = network_statistics
        self._is_directed = is_directed
        self._n_samples_for_normalization = 100


    def calculate_weight(self, W: np.ndarray):
        features = self._network_statistics.calculate_statistics(W)
        weight = np.exp(np.dot(self._thetas, features))

        return weight
    
    def _generate_networks_for_sample(self):
        networks = []
        for _ in range(self._n_samples_for_normalization):
            net = self.sample_network(steps=50, sampling_type="gibbs")
            networks.append(net)
            # print(f"Sampled network: ")
            # print(net)
        
        return networks

    def _calculate_normalization_factor(self):
        networks_for_sample = self._generate_networks_for_sample()
        
        self._normalization_factor = 1
        # print(f"Initial normalization factor: {self._normalization_factor}")

        for network in networks_for_sample:
            weight = self.calculate_weight(network)
            self._normalization_factor += weight
        
        # print(f"Sampled normalization factor: {self._normalization_factor}")

    def fit(self, precalculated_normalization_factor=None, precalculated_thetas=None):
        """
        TODO - This is just a mock implementation. 
        Currently just calculating the normalization factor. 
        """
        if precalculated_thetas is not None:
            self._thetas = precalculated_thetas
        else:
            self._thetas = np.random.uniform(-1, 1, self._n_nodes)

        if precalculated_normalization_factor is not None:
            self._normalization_factor = precalculated_normalization_factor
        else:
            self._calculate_normalization_factor()
    

    def calculate_probability(self, W: np.ndarray):
        """
        Calculate the probability of a graph under the ERGM model.
        
        Parameters
        ----------
        W : np.ndarray
            A connectivity matrix.
        
        Returns
        -------
        prob : float
            The probability of the graph under the ERGM model.
        """

        if self._normalization_factor is None or self._thetas is None:
            raise ValueError("Normalization factor and thetas not set, fit the model before calculating probability.")

        weight = self.calculate_weight(W)
        prob = weight / self._normalization_factor
        
        return prob

    def sample_network(self, sampling_type="gibbs", seed_network=None, steps=500):
        """
        Sample a network from the ERGM model.
        
        Parameters
        ----------
        sampling_type : str
            The type of sampling to use. Currently only `gibbs` is supported.
        seed_network : np.ndarray
            A seed connectivity matrix to start the MCMC sampler from.
            ## TODO - Maybe this should just be a boolean flag/string and not the actual object. and sample_network() will decide with some logic
        steps : int
            The number of steps to run the MCMC sampler.
        
        Returns
        -------
        W : np.ndarray
            The sampled connectivity matrix.
        """
        # if self._normalization_factor is None: # Not sure 
            # raise ValueError("Normalization factor not set, fit the model before sampling.")

        if sampling_type == "gibbs":
            sampler = sampling.Gibbs(self._network_statistics, is_directed=self._is_directed)
        else:
            raise ValueError(f"Sampling type {sampling_type} not supported.")
        
        if seed_network is None:
            G = nx.erdos_renyi_graph(self._n_nodes, 0.2, directed=self._is_directed)
            seed_network = nx.to_numpy_array(G)
        network = sampler.sample(seed_network, parameters=self._thetas ,n_iter=steps)

        return network