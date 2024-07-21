import numpy as np
import networkx as nx
from scipy.optimize import minimize

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
        # OR: this is passed by reference, not by value. Consider deepcopy so noone can mess up things from outside the
        # ERGM object.
        self._network_statistics = network_statistics

        if initial_thetas is not None:
            self._thetas = initial_thetas
        else:
            self._thetas = self._get_random_thetas(sampling_method="uniform")

        if initial_normalization_factor is not None:
            self._normalization_factor = initial_normalization_factor
        else:
            self._normalization_factor = np.random.normal(50, 10)

        self._is_directed = is_directed
        self._seed_MCMC_proba = seed_MCMC_proba

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

    def _get_random_thetas(self, sampling_method="uniform"):
        if sampling_method == "uniform":
            return np.random.uniform(-1, 1, self._network_statistics.get_num_of_statistics())
        else:
            raise ValueError(f"Sampling method {sampling_method} not supported. See docs for supported samplers.")

    def _generate_networks_for_sample(self, n_networks, n_mcmc_steps):
        networks = []
        for _ in range(n_networks):
            net = self.sample_network(steps=n_mcmc_steps, sampling_method="NaiveMetropolisHastings")
            networks.append(net)

        return networks

    def _approximate_normalization_factor(self, n_networks, n_mcmc_steps):
        networks_for_sample = self._generate_networks_for_sample(n_networks, n_mcmc_steps)

        self._normalization_factor = 0

        for network in networks_for_sample:
            weight = self.calculate_weight(network)
            self._normalization_factor += weight

    def fit(self, observed_network, n_networks_for_norm=100, n_mcmc_steps=500, verbose=True, optimization_options={}):
        """
        Initial version, a simple MLE calculated by minimizing negative log likelihood.
        Normalizaiton factor is approximated via MCMC.

        Function is then solved via scipy.
        """
        self._thetas = self._get_random_thetas(sampling_method="uniform")

        if verbose:
            print(f"\tStarting fit with initial normalization factor: {self._normalization_factor}")

            if optimization_options != {}:
                print(f"\tOptimization options: {optimization_options}")

        def negative_log_likelihood(thetas):
            """
            Receive a list of thetas and return the negative log likelihood of the model.
            This is done according to - 
                L(theta | y_obs) = log theta^T g(y_obs) - log Z(theta)
            """
            # print(f"""Calculating negative log likelihood for thetas: {thetas}""")
            self._thetas = thetas

            self._approximate_normalization_factor(n_networks_for_norm, n_mcmc_steps)
            Z = self._normalization_factor
            # print(f"Approximated Z - {Z}")

            y_observed_weight = self.calculate_weight(observed_network)

            log_likelihood = np.log(y_observed_weight) - np.log(Z)

            return -log_likelihood

        # OR: Providing the Scipy optimizer with a gradient function can streamline the optimization substantially. The
        # partial derivative of np.log(y_observed_weight) with relation to each entry of theta is trivial (it is the
        # observed statistic), and maybe this is enough (sometimes it is OK to treat the normalization function as a
        # constant). If not, the derivative of the normalization factor is the expected value of the corresponding
        # statistic with relation to the current ERGM distribution. This can be approximated by averaging over a sample
        # (CLT is a good friend in this case).
        result = minimize(negative_log_likelihood, self._thetas, method='Nelder-Mead', options=optimization_options)
        self._thetas = result.x
        print("\tOptimization result:")
        print(f"\tTheta: {self._thetas}")
        print(f"\tNormalization factor: {self._normalization_factor}")
        print(result)

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
            sampler = sampling.NaiveMetropolisHastings(self._thetas, self._network_statistics,
                                                       is_directed=self._is_directed)
        else:
            raise ValueError(f"Sampling method {sampling_method} not supported. See docs for supported samplers.")

        if seed_network is None:
            G = nx.erdos_renyi_graph(self._n_nodes, self._seed_MCMC_proba, directed=self._is_directed)
            seed_network = nx.to_numpy_array(G)

        network = sampler.sample(seed_network, steps)

        return network

# n_nodes = 5
# stats_calculator = NetworkStatistics(metric_names=["num_edges"])
# ergm = ERGM(n_nodes, stats_calculator, is_directed=False)

# ergm.print_model_parameters()

# W = np.array([[0., 0., 0., 0., 1.],
#        [0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 0.],
#        [0., 0., 0., 0., 1.],
#        [1., 0., 0., 1., 0.]])

# ergm.fit(W, verbose=True)

# n_nodes = 4
# stats_calculator = NetworkStatistics(metric_names=["num_edges"])

# theta = np.log(3)
# ergm = ERGM(n_nodes, stats_calculator, is_directed=False, initial_thetas=[theta])
# print("Baseline ERGM parameters - ")
# ergm.print_model_parameters()

# W = ergm.sample_network(sampling_method="NaiveMetropolisHastings", steps=1000)
# G = connectivity_matrix_to_G(W, directed=False)
# real_edge_count = len(G.edges())
# real_triangle_count = sum(nx.triangles(G).values()) // 3
# print(f"Sampled a random network from a model with theta = {theta}")
# print(W)
# print(f"Network has statistics - edge count: {real_edge_count}, triangle count: {real_triangle_count}")

# print("")

# print(f"Now, fit a random ERGM to create networks with similar statistics")
# fitted_ergm = ERGM(n_nodes, stats_calculator, is_directed=False)
# print(f"Initial ERGM parameters:")
# fitted_ergm.print_model_parameters()

# samples_before_fit = []
# for i in range(10):
#     sampled_W = fitted_ergm.sample_network(sampling_method="NaiveMetropolisHastings", steps=1000)
#     G = connectivity_matrix_to_G(sampled_W, directed=False)
#     edge_count = len(G.edges())
#     triangle_count = sum(nx.triangles(G).values()) // 3

#     samples_before_fit.append({"edge_count": edge_count, "triangle_count": triangle_count})

# print(f"Fitting ERGM...")
# fitted_ergm.fit(W)
# print("Done fitting!")

# samples_after_fit = []
# for i in range(10):
#     sampled_W = fitted_ergm.sample_network(sampling_method="NaiveMetropolisHastings", steps=1000)
#     G = connectivity_matrix_to_G(sampled_W, directed=False)
#     edge_count = len(G.edges())
#     triangle_count = sum(nx.triangles(G).values()) // 3

#     samples_after_fit.append({"edge_count": edge_count, "triangle_count": triangle_count})

# print("Samples before fit:")
# print(samples_before_fit)
# print("Samples after fit:")
# print(samples_after_fit)
