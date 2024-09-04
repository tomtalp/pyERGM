import numpy as np
import networkx as nx
from scipy.optimize import minimize, OptimizeResult
from scipy.spatial.distance import mahalanobis
from scipy.stats import f
# from scipy.sparse.linalg import eigsh
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
                 sample_size=100,
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
        
        sample_size : int
            The number of networks to sample for approximating the normalization factor.
        
        n_mcmc_steps : int
            The number of steps to run the MCMC sampler when sampling a network
        """
        self._n_nodes = n_nodes
        self._is_directed = is_directed
        self._network_statistics = MetricsCollection(network_statistics, self._is_directed, self._n_nodes)

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

        # This is because we assume the sample size is even when estimating the covariance matrix (in
        # calc_capital_gammas).
        if sample_size % 2 != 0:
            sample_size += 1
        self.sample_size = sample_size
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
            return np.random.uniform(-1, 1, self._network_statistics.num_of_features)
        else:
            raise ValueError(f"Sampling method {sampling_method} not supported. See docs for supported samplers.")

    def generate_networks_for_sample(self, seed_network=None, replace=True, burn_in=10000, mcmc_steps_per_sample=1000):
        sampler = sampling.NaiveMetropolisHastings(self._thetas, self._network_statistics, burn_in, mcmc_steps_per_sample)

        if seed_network is None:
            G = nx.erdos_renyi_graph(self._n_nodes, self._seed_MCMC_proba, directed=self._is_directed)
            seed_network = nx.to_numpy_array(G)

        return sampler.sample(seed_network, self.sample_size, replace=replace)

    def _approximate_normalization_factor(self):
        networks_for_sample = self.generate_networks_for_sample(replace=False)

        self._normalization_factor = 0

        for network_idx in range(self.sample_size):
            network = networks_for_sample[:, :, network_idx]
            weight = self.calculate_weight(network)
            self._normalization_factor += weight

        # print(f"Finished generating networks for Z, which is estimated at {self._normalization_factor}")

    @staticmethod
    # @njit
    def approximate_auto_correlation_function(features_of_net_samples: np.ndarray) -> np.ndarray:
        """
        This is gamma hat from Geyer's handbook of mcmc (1D) and Dai and Jones 2017 (multi-D).
        """
        # TODO: it must be possible to vectorize this calculation and spare the for loop. Maybe somehow use the
        #  convolution theorem and go back and forth to the frequency domain using FFT for calculating correlations.
        features_mean_diff = features_of_net_samples - features_of_net_samples.mean(axis=1)[:, None]
        num_features = features_of_net_samples.shape[0]
        sample_size = features_of_net_samples.shape[1]
        auto_correlation_func = np.zeros((sample_size, num_features, num_features))
        for k in range(sample_size):
            auto_correlation_func[k] = 1 / sample_size * (
                    features_mean_diff[:, :sample_size - k].T.reshape(sample_size - k, num_features, 1) @
                    features_mean_diff[:, k:].T.reshape(sample_size - k, 1, num_features)
            ).sum(axis=0)
        return auto_correlation_func

    @staticmethod
    # @njit
    def calc_capital_gammas(auto_corr_funcs: np.ndarray) -> np.ndarray:
        """
        This is the capital gammas hat from Geyer's handbook of mcmc (1D) and Dai and Jones 2017 (multi-D).
        They are simply summations over pairs of consecutive even and odd indices of the auto correlation function (gammas).
        """
        # From Dai and Jones 2017 - a mean of gamma with its transpose (which corresponds to the negative index with the
        # same abs value).
        gamma_tilde = (auto_corr_funcs + np.transpose(auto_corr_funcs, [0, 2, 1])) / 2

        # Note - we assume here an even sample_size, it is forced elsewhere (everytime the sample size is updated).
        sample_size = gamma_tilde.shape[0]
        return (gamma_tilde[np.arange(0, sample_size - 1, 2, dtype=int)] +
                gamma_tilde[np.arange(1, sample_size, 2, dtype=int)])

    @staticmethod
    # TODO: all the static methods in this class should be eorted to utils to avoid calling them with ERGM, and then
    #  maybe they can be precompiled using numba. Tried, and numba doesn't support in axis parameter to np.mean. There
    #  are workarounds, think if it's worth the headache.
    # @njit
    def covariance_matrix_estimation(features_of_net_samples: np.ndarray, method='batch',
                                     num_batches=25) -> np.ndarray:
        """
        Approximate the covariance matrix of the model's features
        Parameters
        ----------
        features_of_net_samples
            The calculated features of the networks that are used for the approximation. Of dimensions
            (num_features X sample_size)
        method
            the method to use for approximating the covariance matrix
            currently supported options are:
                naive
                    A naive estimation from the sample: E[gi*gj] - E[gi]E[gj]
                batch
                    based on difference of means of sample batches from the total mean, as in Geyer's handbook of
                    MCMC (there it is stated for the univariate case, but the generalization is straight forward).
                multivariate_initial_sequence
                    Following Dai and Jones 2017 - the first estimator in section 3.1 (denoted mIS).
        TODO: implement a mechanism that allows to pass arguments that are customized for each method

        Returns
        -------
        The covariance matrix estimation (num_features X num_features).
        """
        if method == 'naive':
            num_features = features_of_net_samples.shape[0]
            sample_size = features_of_net_samples.shape[1]
            mean_features = np.mean(features_of_net_samples, axis=1)
            # An outer product of the means (E[gi]E[gj])
            cross_prod_mean_features = (mean_features.reshape(num_features, 1) @
                                        mean_features.T.reshape(1, num_features))
            # A mean of the outer products of the sample (E[gi*gj])
            mean_features_cross_prod = np.mean(
                features_of_net_samples.T.reshape(sample_size, num_features, 1) @
                features_of_net_samples.T.reshape(sample_size, 1, num_features), axis=0)

            return mean_features_cross_prod - cross_prod_mean_features

        elif method == 'batch':
            num_features = features_of_net_samples.shape[0]
            sample_size = features_of_net_samples.shape[1]
            # Verify that the sample is nicely divided into non-overlapping batches.
            while sample_size % num_batches != 0:
                num_batches += 1
            sample_mean = features_of_net_samples.mean(axis=1)
            batch_size = sample_size // num_batches

            # Divide the sample into batches, and calculate the mean of each one of them
            batches_means = np.zeros((num_features, num_batches))
            for i in range(num_batches):
                batches_means[:, i] = features_of_net_samples[:, i * batch_size:(i + 1) * batch_size].mean(axis=1)

            diff_of_global_mean = batches_means - sample_mean.reshape(num_features, 1)

            # Average the outer products of the differences between batch means and the global mean
            batches_cov_mat_est = np.mean(
                diff_of_global_mean.T.reshape(num_batches, num_features, 1) @
                diff_of_global_mean.T.reshape(num_batches, 1, num_features), axis=0)
            # Multiply by the batch size to compensate for the aggregation into batches.
            return batch_size * batches_cov_mat_est

        elif method == "multivariate_initial_sequence":
            auto_corr_funcs = ERGM.approximate_auto_correlation_function(features_of_net_samples)
            capital_gammas = ERGM.calc_capital_gammas(auto_corr_funcs)

            # In this method, we sum up capital gammas, and choose where to cut the tail (which corresponds to estimates
            # of auto-correlations with large lags within the chain. Naturally, as the lag increases the estimation
            # becomes worse, so the magic here is to determine where to cut). So we first calculate all possible
            # estimators using np.cumsum, and then perform the calculations they specified to determine where to cut.
            possible_cov_mat_ests = -auto_corr_funcs[0] + 2 * np.cumsum(capital_gammas, axis=0)

            # The first condition is to have an index where the estimation is positive-definite, namely all eigen-values
            # are positive. As the both gamma_0 (which is auto_corr_funcs[0]) and the capital gammas are symmetric, all
            # the sum of them is allways symmetric, which ensures real eigen values, and we can simply calculate the
            # eigen value with the smallest algebraic value to determine whether all of them are positive.
            is_positive = False
            first_pos_def_idx = 0
            while not is_positive:
                if first_pos_def_idx == possible_cov_mat_ests.shape[0]:
                    # TODO: ValueError? probably should throw something else. And maybe it is better to try alone some
                    #  of the remediations suggested here and just notify the user...
                    raise ValueError("Got a sample with no valid multivariate_initial_sequence covariance matrix "
                                     "estimation (no possibility is positive-definite). Consider increasing sample size"
                                     " or using a different covariance matrix estimation method.")
                cur_smallest_eigen_val = eigsh(possible_cov_mat_ests[first_pos_def_idx], k=1, which='SA',
                                               return_eigenvectors=False)[0]
                if cur_smallest_eigen_val > 0:
                    is_positive = True
                else:
                    first_pos_def_idx += 1

            # Now we find the farthest idx after first_pos_def_idx fir which the sequence of determinants is strictly
            # monotonically increasing.
            # TODO: this try-catch intends to catch a bug that hasn't reproduced, consider removing
            try:
                # TODO: it might be faster to run in a loop and each time calculate a single determinant, and stop when it
                #  is smaller than the previous one, rather than computing all of them upfront.
                determinants = np.linalg.det(possible_cov_mat_ests[first_pos_def_idx:])
            except IndexError:
                print(f"shape of features_of_net_samples: {features_of_net_samples.shape}\n"
                      f"shape of auto_corr_funcs: {auto_corr_funcs.shape}\n"
                      f"shape of capital_gammas: {capital_gammas.shape}\n"
                      f"shape of possible_cov_mat_ests: {possible_cov_mat_ests.shape}\n"
                      f"first_post_def_idx: {first_pos_def_idx}")
                raise
            tail_cut_idx = first_pos_def_idx + np.where(np.diff(determinants) < 0)[0][0]

            return possible_cov_mat_ests[tail_cut_idx]

        else:
            raise ValueError(f"{method} is an unsupported method for covariance matrix estimation")

    @staticmethod
    # @njit
    def _calculate_optimization_step(observed_features, features_of_net_samples, optimization_method):
        mean_features = np.mean(features_of_net_samples, axis=1)

        nll_grad = mean_features - observed_features

        if optimization_method == "gradient_descent":
            nll_hessian = None

        elif optimization_method == "newton_raphson":
            # TODO: mean_features is calculated again inside this method, so it can be spared, but it makes more sense
            #  to use this design. And in general, maybe it is better to have only one method for estimating the
            #  covariance matrix and use the same estimation everywhere (if a single method can work for all needs).
            nll_hessian = ERGM.covariance_matrix_estimation(features_of_net_samples, method='naive')
        else:
            raise ValueError(
                f"Optimization method {optimization_method} not defined")  # TODO - throw this error in fit()

        return nll_grad, nll_hessian

    @staticmethod
    def do_estimate_covariance_matrix(optimization_method, convergence_criterion):
        if optimization_method == "newton_raphson" or convergence_criterion == "hotelling":
            return True
        return False

    def fit(self, observed_network,
            lr=0.001,
            opt_steps=1000,
            steps_for_decay=100,
            lr_decay_pct=0.01,
            l2_grad_thresh=0.001,
            sliding_grad_window_k=10,
            max_sliding_window_size=100,
            max_nets_for_sample=1000,
            sample_pct_growth=0.02,
            optimization_method="gradient_descent",
            convergence_criterion="hotelling",
            cov_matrix_estimation_method="batch",
            cov_matrix_num_batches=25,
            hotelling_confidence=0.99,
            mcmc_burn_in=1000, 
            mcmc_steps_per_sample=10):
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
        # As in the constructor, the sample size must be even.
        if max_nets_for_sample % 2 != 0:
            max_nets_for_sample += 1

        self._thetas = self._get_random_thetas(sampling_method="uniform")
        self.optimization_iter = 0

        print(f"Initial thetas - {self._thetas}")
        print("optimization started")

        self.optimization_start_time = time.time()
        num_of_features = self._network_statistics.num_of_features

        grads = np.zeros((opt_steps, num_of_features))
        hotelling_statistics = []

        if convergence_criterion == "hotelling":
            hotelling_critical_value = f.ppf(1 - hotelling_confidence, num_of_features,
                                            self.sample_size - num_of_features)  # F(p, n-p) TODO - doc this better

        seed_network = observed_network
        burn_in = mcmc_burn_in
        for i in range(opt_steps):
            if i > 0:
                burn_in = 0
            networks_for_sample = self.generate_networks_for_sample(seed_network=seed_network, burn_in=burn_in, mcmc_steps_per_sample=mcmc_steps_per_sample)
            seed_network = networks_for_sample[:, :, -1]

            features_of_net_samples = self._network_statistics.calculate_sample_statistics(networks_for_sample)
            observed_features = self._network_statistics.calculate_statistics(observed_network)

            grad, hessian = self._calculate_optimization_step(observed_features, features_of_net_samples,
                                                              optimization_method)
            # if ERGM.do_estimate_covariance_matrix(optimization_method, convergence_criterion):
            #     estimated_cov_matrix = ERGM.covariance_matrix_estimation(features_of_net_samples,
            #                                                              method=cov_matrix_estimation_method,
            #                                                              num_batches=cov_matrix_num_batches)
            if optimization_method == "newton_raphson":
                # hessian = estimated_cov_matrix
                inv_hessian = np.linalg.pinv(hessian)
                self._thetas = self._thetas - lr * inv_hessian @ grad

            elif optimization_method == "gradient_descent":
                self._thetas = self._thetas - lr * grad

            grads[i] = grad

            idx_for_sliding_grad = np.max([0, i - sliding_grad_window_k + 1])
            sliding_window_grads = grads[idx_for_sliding_grad:i + 1].mean()
            
            if i % steps_for_decay == 0:
                delta_t = time.time() - self.optimization_start_time
                # print(f"Step {i+1} - grad: {grads[i - 1]}, window_grad: {sliding_window_grads:.2f} lr: {lr:.10f}, thetas: {self._thetas}, time from start: {delta_t:.2f}, sample_size: {self.sample_size}, sliding_grad_window_k: {sliding_grad_window_k}")
                print(
                    f"Step {i + 1} - lr: {lr:.7f}, time from start: {delta_t:.2f}, window_grad: {sliding_window_grads:.2f}")

            if convergence_criterion == "hotelling":
                estimated_cov_matrix = ERGM.covariance_matrix_estimation(features_of_net_samples,
                                                                         method=cov_matrix_estimation_method,
                                                                         num_batches=cov_matrix_num_batches)
                inv_estimated_cov_matrix = np.linalg.pinv(estimated_cov_matrix)
                mean_features = np.mean(features_of_net_samples,
                                        axis=1)  # TODO - this is calculated in `_calculate_optimization_step()` and covariance estimation, consider sharing the two

                dist = mahalanobis(observed_features, mean_features, inv_estimated_cov_matrix)
                # dist = mahalanobis(observed_features, mean_features, inv_hessian)

                hotelling_t_statistic = self.sample_size * dist * dist

                # (n-p / p(n-1))* t^2 ~ F_p, n-p  (#TODO - give reference for this)
                hotelling_as_f_statistic = ((self.sample_size - num_of_features) / (
                        num_of_features * (self.sample_size - 1))) * hotelling_t_statistic

                hotelling_statistics.append({
                    "dist": dist,
                    # "hotelling_t": hotelling_t_statistic,
                    "hotelling_F": hotelling_as_f_statistic,
                    "critical_val": hotelling_critical_value,
                    "inv_cov_norm": np.linalg.norm(inv_estimated_cov_matrix),
                    # "inv_hessian_norm": np.linalg.norm(inv_hessian)
                })

                if ((i + 1) % steps_for_decay) == 0:
                    lr *= (1 - lr_decay_pct)

                    if self.sample_size < max_nets_for_sample:
                        self.sample_size *= (1 + sample_pct_growth)
                        self.sample_size = np.min([int(self.sample_size), max_nets_for_sample])
                        # As in the constructor, the sample size must be even.
                        if self.sample_size % 2 != 0:
                            self.sample_size += 1
                        print(f"\t Sample size increased at step {i + 1} to {self.sample_size}")

                    if sliding_grad_window_k < max_sliding_window_size:
                        sliding_grad_window_k *= (1 + sample_pct_growth)
                        sliding_grad_window_k = np.min(
                            [np.ceil(sliding_grad_window_k).astype(int), max_sliding_window_size])

                if hotelling_as_f_statistic <= hotelling_critical_value:
                    print(f"Reached a confidence of {hotelling_confidence} with the hotelling convergence test! DONE! ")
                    print(
                        f"hotelling - {hotelling_as_f_statistic}, hotelling_critical_value={hotelling_critical_value}")
                    grads = grads[:i]
                    break


            elif convergence_criterion == "zero_grad_norm":
                if np.linalg.norm(sliding_window_grads) <= l2_grad_thresh:
                    print(f"Reached threshold of {l2_grad_thresh} after {i} steps. DONE!")
                    grads = grads[:i]
                    break
            else:
                raise ValueError(f"Convergence criterion {convergence_criterion} not defined")
            
        return grads, hotelling_statistics

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
