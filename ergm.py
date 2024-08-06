import numpy as np
import networkx as nx
from scipy.optimize import minimize, OptimizeResult
import time
from typing import Collection
import sampling

from utils import *
from metrics import *


class ERGM():
    def __init__(self,
                 n_nodes,
                 network_statistics: Collection[Metric],
                 is_directed=False,
                 initial_thetas=None,
                 initial_normalization_factor=None,
                 seed_MCMC_proba=0.25,
                 n_networks_for_grad_estimation=100,
                 n_mcmc_steps=500,
                 verbose=True,
                 optimization_options={}):
        """
        An ERGM model object. 
        
        Parameters
        ----------
        n_nodes : int
            The number of nodes in the network.
        
        network_statistics : list
            A list of Metric objects for calculating statistics of a network.
        
        is_directed : bool
            Whether the network is directed or not.
        
        initial_thetas : np.ndarray
            The initial values of the coefficients of the ERGM. If not provided, they are randomly initialized.
        
        initial_normalization_factor : float
            The initial value of the normalization factor. If not provided, it is randomly initialized.
        
        seed_MCMC_proba : float
            The probability of a connection in the seed network for MCMC sampling, in case no seed network is provided.
        
        n_networks_for_grad_estimation : int
            The number of networks to sample for approximating the normalization factor.
        
        n_mcmc_steps : int
            The number of steps to run the MCMC sampler when sampling a network
        """
        self._n_nodes = n_nodes
        self._is_directed = is_directed
        self._network_statistics = MetricsCollection(network_statistics, self._is_directed)

        if initial_thetas is not None:
            self._thetas = initial_thetas
        else:
            self._thetas = self._get_random_thetas(sampling_method="uniform")

        if initial_normalization_factor is not None:
            self._normalization_factor = initial_normalization_factor
        else:
            self._normalization_factor = np.random.normal(50, 10)

        self._seed_MCMC_proba = seed_MCMC_proba

        self.optimization_iter = 0
        self.optimization_start_time = None

        self.n_networks_for_grad_estimation = n_networks_for_grad_estimation
        self.n_mcmc_steps = n_mcmc_steps
        self.verbose = verbose
        self.optimization_options = optimization_options

    def print_model_parameters(self):
        print(f"Number of nodes: {self._n_nodes}")
        print(f"Thetas: {self._thetas}")
        print(f"Normalization factor approx: {self._normalization_factor}")
        print(f"Is directed: {self._is_directed}")

    def calculate_weight(self, W: np.ndarray):
        if len(W.shape) != 2 or W.shape[0] != self._n_nodes or W.shape[1] != self._n_nodes:
            raise ValueError(f"The dimensions of the given adjacency matrix, {W.shape}, don't comply with the number of"
                             f" nodes in the network: {self._n_nodes}")
        features = self._network_statistics.calculate_statistics(W)
        weight = np.exp(np.dot(self._thetas, features))

        return weight

    def _get_random_thetas(self, sampling_method="uniform"):
        if sampling_method == "uniform":
            return np.random.uniform(-1, 1, self._network_statistics.num_of_metrics)
        else:
            raise ValueError(f"Sampling method {sampling_method} not supported. See docs for supported samplers.")

    def generate_networks_for_sample(self, replace=True):
        sampler = sampling.NaiveMetropolisHastings(self._thetas, self._network_statistics)
        G = nx.erdos_renyi_graph(self._n_nodes, self._seed_MCMC_proba, directed=self._is_directed)
        seed_network = nx.to_numpy_array(G)

        return sampler.sample(seed_network, self.n_networks_for_grad_estimation, replace=replace)

    def _approximate_normalization_factor(self):
        networks_for_sample = self.generate_networks_for_sample(replace=False)

        self._normalization_factor = 0

        for network_idx in range(self.n_networks_for_grad_estimation):
            network = networks_for_sample[:, :, network_idx]
            weight = self.calculate_weight(network)
            self._normalization_factor += weight

        # print(f"Finished generating networks for Z, which is estimated at {self._normalization_factor}")

    def approximate_auto_correlation_function(self, networks_sample: np.ndarray) -> np.ndarray:
        """
        This is gamma hat from Geyer's handbook of mcmc.
        """
        features_of_net_samples = self._calc_sample_statistics(networks_sample)
        features_mean_diff = features_of_net_samples - features_of_net_samples.mean(axis=1)[:, None]
        num_features = features_of_net_samples.shape[0]
        sample_size = networks_sample.shape[2]
        auto_correlation_func = np.zeros((sample_size, num_features, num_features))
        for k in range(sample_size):
            auto_correlation_func[k] = 1 / sample_size * (
                    features_mean_diff[:, :sample_size-k].T.reshape(sample_size - k, num_features, 1) @
                    features_mean_diff[:, k:].T.reshape(sample_size - k, 1, num_features)
            ).sum(axis=0)

        # cross_feature_mean_diff_across_samples = (
        #         features_mean_diff.T.reshape(networks_sample.shape[1], features_of_net_samples.shape[0], 1) @
        #         features_of_net_samples.T.reshape(networks_sample.shape[1], 1, features_of_net_samples.shape[0]))
        # # Padding the cross features mean diffs to eliminate the effect of circular boundaries
        # padded_cross_feature_mean_diff_across_samples = np.concatenate(
        #     (cross_feature_mean_diff_across_samples, np.zeros(cross_feature_mean_diff_across_samples.shape)), axis=0)
        # # Transform to the frequency domain
        # padded_features_mean_diff_fft = np.fft.fft(padded_cross_feature_mean_diff_across_samples, axis=0)
        # # Use the theorem of convolution - it is a product with the conjugate in the frequency domain. Then take the
        # # first half of the resulting array (to take the sums without padding, only lags within the original array).
        # auto_correlation_func = np.fft.ifft(padded_features_mean_diff_fft * np.conjugate(padded_features_mean_diff_fft),
        #                                     axis=0).real[:networks_sample.shape[1]]
        # auto_correlation_func /= networks_sample.shape[1]
        return auto_correlation_func

    def _calc_sample_statistics(self, networks_sample: np.ndarray) -> np.ndarray:
        """
        Calculate the statistics over a sample of networks
        # TODO: there are many Metrics for which this can be calculated more efficiently (without looping). E.g. number
            of edges is just summing up along the 2 first axes of the sample array. Maybe we should export this to
            MetricsCollection and perform it more efficiently when possible (like with the calculation of change_score).
        Parameters
        ----------
        networks_sample
            The networks sample - an array of n X n X sample_size
        Returns
        -------
        an array of the statistics vector per sample (num_features X sample_size)
        """
        features_of_net_samples = np.zeros(
            (self._network_statistics.get_num_of_features(self._n_nodes), networks_sample.shape[2]))
        for i in range(networks_sample.shape[2]):
            features_of_net_samples[:, i] = self._network_statistics.calculate_statistics(
                networks_sample[:, :, i])
        return features_of_net_samples

    def fit(self, observed_network,
            lr=0.001,
            opt_steps=1000,
            steps_for_decay=100,
            lr_decay_pct=0.01,
            l2_grad_thresh=0.001,
            sliding_grad_window_k=10,
            max_sliding_window_size=100,
            max_nets_for_sample=1000,
            sample_pct_growth=0.02):
        """
        Fit an ERGM model to a given network.

        Parameters
        ----------
        observed_network : np.ndarray
            The adjacency matrix of the observed network.
        
        lr : float
            The learning rate for the optimization.
        
        opt_steps : int
            The number of optimization steps to run.
        
        steps_for_decay : int
            The number of steps after which to decay the optimization params. (## TODO - Pick a different step value for different params? Right now all params are decayed with the same interval)
        
        lr_decay_pct : float
            The decay factor for the learning rate
        
        l2_grad_thresh : float
            The threshold for the L2 norm of the gradient to stop the optimization.
        
        sliding_grad_window_k : int
            The size of the sliding window for the gradient, for which we use to calculate the mean gradient norm. This value is then tested against l2_grad_thresh to decide whether optimization halts.
        
        max_sliding_window_size : int
            The maximum size of the sliding window for the gradient.
        
        max_nets_for_sample : int
            The maximum number of networks to sample when approximating the expected network statistics (i.e. E[g(y)])
        
        sample_pct_growth : float
            The percentage growth of the number of networks to sample, which we want to increase over time
    
            
        """

        def nll_grad_hessian(thetas):
            model = ERGM(self._n_nodes, self._network_statistics.metrics, initial_thetas=thetas,
                         is_directed=self._is_directed)

            observed_features = model._network_statistics.calculate_statistics(observed_network)

            networks_for_sample = self.generate_networks_for_sample()
            num_of_features = model._network_statistics.get_num_of_features(model._n_nodes)

            features_of_net_samples = model._calc_sample_statistics(networks_for_sample)

            mean_features = np.mean(features_of_net_samples, axis=1)

            nll_grad = mean_features - observed_features

            # An outer product of the means (E[gi]E[gj])
            cross_prod_mean_features = (mean_features.reshape(num_of_features, 1) @
                                        mean_features.T.reshape(1, num_of_features))
            # A mean of the outer products of the sample (E[gi*gj])
            mean_features_cross_prod = np.mean(
                features_of_net_samples.T.reshape(self.n_networks_for_grad_estimation, num_of_features, 1) @
                features_of_net_samples.T.reshape(self.n_networks_for_grad_estimation, 1, num_of_features), axis=0)

            nll_hessian = mean_features_cross_prod - cross_prod_mean_features

            return nll_grad, nll_hessian

        def true_nll_grad(model):
            """
            ## TODO - THIS IS FOR DEBUG, REMOVE LATER
            """
            observed_features = model._network_statistics.calculate_statistics(observed_network)
            all_probs = model._all_weights / model._normalization_factor
            num_features = model._network_statistics.num_of_metrics
            num_nets = all_probs.size
            all_features_by_all_nets = np.zeros((num_features, num_nets))
            for i in range(num_nets):
                all_features_by_all_nets[:, i] = model._network_statistics.calculate_statistics(
                    construct_adj_mat_from_int(i, self._n_nodes, self._is_directed))
            expected_features = all_features_by_all_nets @ all_probs
            return expected_features - observed_features

        self._thetas = self._get_random_thetas(sampling_method="uniform")
        self.optimization_iter = 0

        print("optimization started")

        self.optimization_start_time = time.time()

        grads = np.zeros((opt_steps, self._network_statistics.num_of_metrics))
        true_grads = np.zeros((opt_steps, self._network_statistics.num_of_metrics))

        for i in range(opt_steps):
            if ((i + 1) % steps_for_decay) == 0:
                lr *= (1 - lr_decay_pct)

                if self.n_networks_for_grad_estimation < max_nets_for_sample:
                    self.n_networks_for_grad_estimation *= (1 + sample_pct_growth)
                    self.n_networks_for_grad_estimation = np.min(
                        [int(self.n_networks_for_grad_estimation), max_nets_for_sample])

                if sliding_grad_window_k < max_sliding_window_size:
                    sliding_grad_window_k *= (1 + sample_pct_growth)
                    sliding_grad_window_k = np.min(
                        [np.ceil(sliding_grad_window_k).astype(int), max_sliding_window_size])

            grad, hessian = nll_grad_hessian(self._thetas)
            try:
                inv_hessian = np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                print("The hessian is not invertible")
                inv_hessian = np.linalg.pinv(hessian)
            self._thetas = self._thetas - lr * inv_hessian @ grad

            grads[i] = grad

            idx_for_sliding_grad = np.max([0, i - sliding_grad_window_k + 1])
            sliding_window_grads = grads[idx_for_sliding_grad:i + 1].mean()

            if i % 100 == 0:
                ## TODO - THIS IS FOR DEBUG.
                # bruteforce = BruteForceERGM(self._n_nodes, self._network_statistics, initial_thetas=self._thetas, is_directed=self._is_directed)
                # true_grad = true_nll_grad(bruteforce)
                # true_grads[i] = true_grad
                true_grad = 0

                delta_t = time.time() - self.optimization_start_time
                print(
                    f"Step {i} - true_grad: {true_grad}, grad: {grads[i - 1]}, window_grad: {sliding_window_grads:.2f} lr: {lr:.10f}, thetas: {self._thetas}, time from start: {delta_t:.2f}, n_networks_for_grad_estimation: {self.n_networks_for_grad_estimation}, sliding_grad_window_k: {sliding_grad_window_k}")

            if np.linalg.norm(sliding_window_grads) <= l2_grad_thresh:
                print(f"Reached threshold of {l2_grad_thresh} after {i} steps. DONE!")
                grads = grads[:i]
                break

        return grads, true_grads

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
            sampler = sampling.NaiveMetropolisHastings(self._thetas, self._network_statistics)
        else:
            raise ValueError(f"Sampling method {sampling_method} not supported. See docs for supported samplers.")

        if seed_network is None:
            G = nx.erdos_renyi_graph(self._n_nodes, self._seed_MCMC_proba, directed=self._is_directed)
            seed_network = nx.to_numpy_array(G)

        network = sampler.sample(seed_network, num_of_nets=1)

        return network


class BruteForceERGM(ERGM):
    """
    A class that implements ERGM by iterating over the entire space of networks and calculating stuff exactly (rather
    than using statistical methods for approximating and sampling).
    This is mainly for tests.
    """
    # The maximum number of nodes that is allowed for carrying brute force calculations (i.e. iterating the whole space
    # of networks and calculating stuff exactly). This becomes not tractable above this limit.
    MAX_NODES_BRUTE_FORCE_DIRECTED = 5
    MAX_NODES_BRUTE_FORCE_NOT_DIRECTED = 7

    def __init__(self,
                 n_nodes,
                 network_statistics: Collection[Metric],
                 is_directed=False,
                 initial_thetas=None):
        super().__init__(n_nodes,
                         network_statistics,
                         is_directed,
                         initial_thetas)
        self._all_weights = self._calc_all_weights()
        self._normalization_factor = self._all_weights.sum()

    def _validate_net_size(self):
        return (
                (self._is_directed and self._n_nodes <= BruteForceERGM.MAX_NODES_BRUTE_FORCE_DIRECTED)
                or
                (not self._is_directed and self._n_nodes <= BruteForceERGM.MAX_NODES_BRUTE_FORCE_NOT_DIRECTED)
        )

    def _calc_all_weights(self):
        if not self._validate_net_size():
            directed_str = 'directed' if self._is_directed else 'not directed'
            size_limit = BruteForceERGM.MAX_NODES_BRUTE_FORCE_DIRECTED if self._is_directed else (
                BruteForceERGM.MAX_NODES_BRUTE_FORCE_NOT_DIRECTED)
            raise ValueError(
                f"The number of nodes {self._n_nodes} is larger than the maximum allowed for brute force of "
                f"{directed_str} graphs calculations {size_limit}")
        num_pos_connects = self._n_nodes * (self._n_nodes - 1)
        if not self._is_directed:
            num_pos_connects //= 2
        space_size = 2 ** num_pos_connects
        all_weights = np.zeros(space_size)
        for i in range(space_size):
            cur_adj_mat = construct_adj_mat_from_int(i, self._n_nodes, self._is_directed)
            all_weights[i] = super().calculate_weight(cur_adj_mat)
        return all_weights

    def calculate_weight(self, W: np.ndarray):
        adj_mat_idx = construct_int_from_adj_mat(W, self._is_directed)
        return self._all_weights[adj_mat_idx]

    def sample_network(self, sampling_method="Exact", seed_network=None, steps=0):
        if sampling_method != "Exact":
            raise ValueError("BruteForceERGM supports only exact sampling (this is its whole purpose)")

        all_nets_probs = self._all_weights / self._normalization_factor
        sampled_idx = np.random.choice(all_nets_probs.size, p=all_nets_probs)
        return construct_adj_mat_from_int(sampled_idx, self._n_nodes, self._is_directed)

    def fit(self, observed_network):
        def nll(thetas):
            model = BruteForceERGM(self._n_nodes, list(self._network_statistics.metrics), initial_thetas=thetas,
                                   is_directed=self._is_directed)
            return np.log(model._normalization_factor) - np.log(model.calculate_weight(observed_network))

        def nll_grad(thetas):
            model = BruteForceERGM(self._n_nodes, list(self._network_statistics.metrics), initial_thetas=thetas,
                                   is_directed=self._is_directed)
            observed_features = model._network_statistics.calculate_statistics(observed_network)
            all_probs = model._all_weights / model._normalization_factor
            num_features = model._network_statistics.num_of_metrics
            num_nets = all_probs.size
            all_features_by_all_nets = np.zeros((num_features, num_nets))
            for i in range(num_nets):
                all_features_by_all_nets[:, i] = model._network_statistics.calculate_statistics(
                    construct_adj_mat_from_int(i, self._n_nodes, self._is_directed))
            expected_features = all_features_by_all_nets @ all_probs
            return expected_features - observed_features

        def after_iteration_callback(intermediate_result: OptimizeResult):
            self.optimization_iter += 1
            cur_time = time.time()
            print(f'iteration: {self.optimization_iter}, time from start '
                  f'training: {cur_time - self.optimization_start_time} '
                  f'log likelihood: {-intermediate_result.fun}')

        self.optimization_iter = 0
        print("optimization started")
        self.optimization_start_time = time.time()
        res = minimize(nll, self._thetas, jac=nll_grad, callback=after_iteration_callback)
        self._thetas = res.x
        # print(res)
