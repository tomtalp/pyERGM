import numpy as np
import networkx as nx
import sampling

from utils import *


class ERGM():
    def __init__(self, 
                 n_nodes, 
                 network_statistics: NetworkStatistics, 
                 is_directed=False, 
                 initial_thetas=None, 
                 initial_normalization_factor=None,
                 seed_MCMC_proba=0.25):
        """
        An ERGM model object. 
        
        Parameters
        ----------
        n_nodes : int
            The number of nodes in the network.
        
        network_statistics : NetworkStatistics
            A NetworkStatistics object that can calculate statistics of a network.
        
        is_directed : bool
            Whether the network is directed or not.
        
        initial_thetas : np.ndarray
            The initial values of the coefficients of the ERGM. If not provided, they are randomly initialized.
        
        initial_normalization_factor : float
            The initial value of the normalization factor. If not provided, it is randomly initialized.
        
        seed_MCMC_proba : float
            The probability of a connection in the seed network for MCMC sampling, in case no seed network is provided.
        """
        self._n_nodes = n_nodes
        
        if initial_thetas is not None:
            self._thetas = initial_thetas
        else:
            self._thetas = np.random.uniform(-1, 1, n_nodes)
        
        if initial_normalization_factor is not None:
            self._normalization_factor = initial_normalization_factor
        else:
            self._normalization_factor = np.random.normal(50, 10)

        self._network_statistics = network_statistics
        self._is_directed = is_directed
        self._seed_MCMC_proba = seed_MCMC_proba
        self._n_samples_for_normalization = self._n_nodes**2 # TODO just a random polynomial pick for now...

    def print_model_parameters(self):
        print(f"Number of nodes: {self._n_nodes}")
        print(f"Thetas: {self._thetas}")
        print(f"Normalization factor approx: {self._normalization_factor}")
        print(f"Is directed: {self._is_directed}")
        # print(f"Network statistics: {self._network_statistics}")
        


    def calculate_weight(self, W: np.ndarray):
        features = self._network_statistics.calculate_statistics(W)
        weight = np.exp(np.dot(self._thetas, features))

        return weight
    
    def _generate_networks_for_sample(self):
        networks = []
        for _ in range(self._n_samples_for_normalization):
            net = self.sample_network(steps=50, sampling_method="NaiveMetropolisHastings")
            networks.append(net)
        
        return networks

    def _approximate_normalization_factor(self):
        networks_for_sample = self._generate_networks_for_sample()
        
        self._normalization_factor = 0

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
            self._approximate_normalization_factor()
    

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

    def sample_network(self, sampling_method="NaiveMetropolisHastings", seed_network=None, steps=500):
        """
        Sample a network from the ERGM model using MCMC methods
        
        Parameters
        ----------
        sampling_method : str
            The method of sampling to use. Currently only `NaiveMetropolisHastings` is supported.
        seed_network : np.ndarray
            A seed connectivity matrix to start the MCMC sampler from.
        steps : int
            The number of steps to run the MCMC sampler.
        
        Returns
        -------
        W : np.ndarray
            The sampled connectivity matrix.
        """
        if sampling_method == "NaiveMetropolisHastings":
            sampler = sampling.NaiveMetropolisHastings(self._thetas, self._network_statistics, is_directed=self._is_directed)
        else:
            raise ValueError(f"Sampling method {sampling_method} not supported. See docs for supported samplers.")
        
        if seed_network is None:
            G = nx.erdos_renyi_graph(self._n_nodes, self._seed_MCMC_proba, directed=self._is_directed)
            seed_network = nx.to_numpy_array(G)

        network = sampler.sample(seed_network, steps)

        return network