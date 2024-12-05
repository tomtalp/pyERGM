import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import f
from pyERGM.sampling import NaiveMetropolisHastings

from pyERGM.metrics import *


class ERGM():
    def __init__(self,
                 n_nodes,
                 metrics_collection: Collection[Metric],
                 is_directed,
                 initial_thetas=None,
                 initial_normalization_factor=None,
                 seed_MCMC_proba=0.25,
                 verbose=True,
                 use_sparse_matrix=False,
                 fix_collinearity=True,
                 collinearity_fixer_sample_size=1000,
                 is_distributed_optimization=False,
                 optimization_options={}):
        """
        An ERGM model object. 
        
        Parameters
        ----------
        n_nodes : int 
            Number of nodes in the network.

        metrics_collection : Collection[Metric] 
            A list of Metric objects for calculating statistics of a network.

        is_directed : bool 
            Whether the graph is directed or not.

        initial_thetas : npdarray 
            Optional. The initial values of the coefficients of the ERGM. If not provided, they are randomly initialized.

        initial_normalization_factor : float 
            Optional. The initial value of the normalization factor. If not provided, it is randomly initialized.

        seed_MCMC_proba : float 
            Optional. The probability of a connection in the seed network for MCMC sampling, in case no seed network is provided. *Defaults to 0.25*

        sample_size : int 
            Optional. The number of networks to sample via MCMC. If number of samples is odd, it is increased by 1. This is because downstream algorithms assume the sample size is even (e.g. the Covariance matrix estimation). *Defaults to 1000*

        use_sparse_matrix : bool
            Optional. Whether to use sparse matrices for the adjacency matrix. 
            Sparse matrices are implemented via PyTorch's Sparse Tensor's, which are still in beta.  *Defaults to False*
        
        fix_collinearity : bool
            Optional. Whether to fix collinearity in the metrics. *Defaults to True*

        collinearity_fixer_sample_size : int
            Optional. The number of networks to sample for fixing collinearity. *Defaults to 1000*
        
        """
        self._n_nodes = n_nodes
        self._is_directed = is_directed
        self._is_distributed_optimization = is_distributed_optimization
        self._metrics_collection = MetricsCollection(metrics_collection, self._is_directed, self._n_nodes,
                                                     use_sparse_matrix=use_sparse_matrix,
                                                     fix_collinearity=fix_collinearity,
                                                     collinearity_fixer_sample_size=collinearity_fixer_sample_size,
                                                     is_collinearity_distributed=self._is_distributed_optimization)

        if initial_thetas is not None:
            self._thetas = initial_thetas
        else:
            self._thetas = self._get_random_thetas(sampling_method="uniform")

        if initial_normalization_factor is not None:
            self._normalization_factor = initial_normalization_factor
        else:
            self._normalization_factor = np.random.normal(50, 10)

        self._seed_MCMC_proba = seed_MCMC_proba

        self.optimization_start_time = None

        self.verbose = verbose
        self.optimization_options = optimization_options

        self._exact_average_mat = None

        self.mh_sampler = NaiveMetropolisHastings(self._thetas, self._metrics_collection)

    def print_model_parameters(self):
        """
        Prints the parameters of the ERGM model.
        """
        print(f"Number of nodes: {self._n_nodes}")
        print(f"Thetas: {self._thetas}")
        print(f"Normalization factor approx: {self._normalization_factor}")
        print(f"Is directed: {self._is_directed}")

    def calculate_weight(self, W: np.ndarray):
        if len(W.shape) != 2 or W.shape[0] != self._n_nodes or W.shape[1] != self._n_nodes:
            raise ValueError(f"The dimensions of the given adjacency matrix, {W.shape}, don't comply with the number of"
                             f" nodes in the network: {self._n_nodes}")
        features = self._metrics_collection.calculate_statistics(W)
        weight = np.exp(np.dot(self._thetas, features))

        return weight

    def _get_random_thetas(self, sampling_method="uniform"):
        if sampling_method == "uniform":
            return np.random.uniform(-1, 1, self._metrics_collection.num_of_features)
        else:
            raise ValueError(f"Sampling method {sampling_method} not supported. See docs for supported samplers.")

    def generate_networks_for_sample(self,
                                     sample_size,
                                     seed_network=None,
                                     replace=True,
                                     burn_in=10000,
                                     mcmc_steps_per_sample=1000,
                                     sampling_method="metropolis_hastings",
                                     edge_proposal_method='uniform'
                                     ):
        if sampling_method == "metropolis_hastings":
            if seed_network is None:
                G = nx.erdos_renyi_graph(self._n_nodes, self._seed_MCMC_proba, directed=self._is_directed)
                seed_network = nx.to_numpy_array(G)
            self.mh_sampler.set_thetas(self._thetas)
            return self.mh_sampler.sample(seed_network, sample_size, replace=replace, burn_in=burn_in,
                                          steps_per_sample=mcmc_steps_per_sample,
                                          edge_proposal_method=edge_proposal_method)
        elif sampling_method == "exact":
            return self._generate_exact_sample(sample_size)
        else:
            raise ValueError(f"Unrecognized sampling method {sampling_method}")

    @staticmethod
    def do_estimate_covariance_matrix(optimization_method, convergence_criterion):
        if optimization_method == "newton_raphson" or convergence_criterion == "hotelling":
            return True
        return False

    def _mple_fit(self, observed_network, lr=0.001, stopping_thr: float = 1e-6, logistic_reg_max_iter=1000):
        """
        Perform MPLE estimation of the ERGM parameters.
        This is done by fitting a logistic regression model, where the X values are the change statistics
        calculated for every edge in the observed network, and the predicted values are the presence of the edge.

        More precisely, we create a train dataset with (X_{i, j}, y_{i,j}) where -
            X_{i, j} = g(y_{i,j}^+) - g(y_{i, j}^-)
            y_{i, j} = 1 if there is an edge between i and j, 0 otherwise

        Parameters
        ----------
        observed_network : np.ndarray
            The adjacency matrix of the observed network.
        
        Returns
        -------
        thetas: np.ndarray
            The estimated coefficients of the ERGM.
        """
        print(f"MPLE with lr {lr}")
        trained_thetas, prediction = mple_logistic_regression_optimization(self._metrics_collection, observed_network,
                                                                           is_distributed=self._is_distributed_optimization,
                                                                           lr=lr,
                                                                           stopping_thr=stopping_thr,
                                                                           max_iter=logistic_reg_max_iter)
        self._exact_average_mat = np.zeros((self._n_nodes, self._n_nodes))

        if self._is_directed:
            self._exact_average_mat[~np.eye(self._n_nodes, dtype=bool)] = prediction
        else:
            upper_triangle_indices = np.triu_indices(self._n_nodes, k=1)
            self._exact_average_mat[upper_triangle_indices] = prediction
            lower_triangle_indices_aligned = (upper_triangle_indices[1], upper_triangle_indices[0])
            self._exact_average_mat[lower_triangle_indices_aligned] = prediction

        return trained_thetas

    def _do_MPLE(self, theta_init_method):
        if not self._metrics_collection._has_dyadic_dependent_metrics or theta_init_method == "mple":
            return True
        return False

    def _generate_exact_sample(self, sample_size: int = 1):
        # TODO: support getting a flag of `replace` which will enable sampling with no replacements (generating samples
        #  of different networks).
        if self._metrics_collection._has_dyadic_dependent_metrics:
            raise ValueError("Cannot sample exactly from a model that is dyadic dependent!")
        if self._exact_average_mat is None:
            raise ValueError("Cannot sample exactly from a model that is not trained! Call `model.fit()` and pass an "
                             "observed network!")

        return sample_from_independent_probabilities_matrix(self._exact_average_mat, sample_size)

    def fit(self, observed_network,
            lr=0.1,
            opt_steps=1000,
            steps_for_decay=100,
            lr_decay_pct=0.01,
            l2_grad_thresh=0.001,
            sliding_grad_window_k=10,
            max_sliding_window_size=100,
            max_nets_for_sample=1000,
            sample_pct_growth=0.02,
            optimization_method="newton_raphson",
            convergence_criterion="hotelling",
            cov_matrix_estimation_method="naive",
            cov_matrix_num_batches=25,
            hotelling_confidence=0.99,
            theta_init_method="mple",
            no_mple=False,
            mcmc_burn_in=1000,
            mcmc_seed_network=None,
            mcmc_steps_per_sample=10,
            mcmc_sample_size=100,
            mple_lr=1,
            mple_stopping_thr=1e-6,
            mple_max_iter=1000,
            edge_proposal_method='uniform',
            **kwargs
            ):
        """
        Fit an ERGM model to a given network with one of the two fitting methods - MPLE or MCMLE.

        Parameters
        ----------
        observed_network : np.ndarray
            The adjacency matrix of the observed network. #TODO - how do we support nx.Graph?

        lr : float
            Optional. The learning rate for the optimization. *Defaults to 0.1*

        opt_steps : int
            Optional. The number of optimization steps to run. *Defaults to 1000*

        steps_for_decay : int
            Optional. The number of steps after which to decay the optimization params. 
            *Defaults to 100* # TODO - redundant parameter?

        lr_decay_pct : float
            Optional. The decay factor for the learning rate. *Defaults to 0.01*

        l2_grad_thresh : float
            Optional. The threshold for the L2 norm of the gradient to stop the optimization. 
            Relevant only for convergence criterion "zero_grad_norm". *Defaults to 0.001*

        sliding_grad_window_k : int
            Optional. The size of the sliding window for the gradient, for which we use to calculate the mean gradient norm. 
            This value is then tested against l2_grad_thresh to decide whether optimization halts.
            Relevant only for convergence criterion "zero_grad_norm". *Defaults to 10*

        max_sliding_window_size : int
            Optional. The maximum size of the sliding window for the gradient. 
            Relevant only for convergence criterion "zero_grad_norm". *Defaults to 100*

        max_nets_for_sample : int
            Optional. The maximum number of networks to sample with MCMC. *Defaults to 1000*
            #TODO - Do we still need this? Seems like increasing the sample size isn't necessary (we'll gonna pick large sample sizes anyway)
            
        sample_pct_growth : float
            Optional. The percentage growth of the number of networks to sample, which we want to increase over time.
            *Defaults to 0.02*. #TODO - Same as `max_nets_for_sample`. Do we still need this?

        optimization_method : str
            Optional. The optimization method to use. Can be either "newton_raphson" or "gradient_descent".
            *Defaults to "newton_raphson"*.

        convergence_criterion : str
            Optional. The criterion for convergence. Can be either "hotelling" or "zero_grad_norm".
            *Defaults to "zero_grad_norm"*.
            # TODO - Revisit this when we fix convergence criterion.
            
        cov_matrix_estimation_method : str
            Optional. The method to estimate the covariance matrix. 
            Supported methods - `naive`, `batch`, `multivariate_initial_sequence`. *Defaults to "batch"*.
        
        cov_matrix_num_batches : int
            Optional. The number of batches to use for estimating the covariance matrix.
            Relevant only for `cov_matrix_estimation_method="batch"`. *Defaults to 25*.

        hotelling_confidence : float
            Optional. The confidence level for the Hotelling's T-squared test. *Defaults to 0.99*.
        
        theta_init_method : str
            Optional. The method to initialize the theta values. Can be either "uniform" or "mple".
            The MPLE method can be used even for dyadic dependent models, since it serves as a good starting point for the MCMLE.
            *Defaults to "mple"*.

        no_mple : bool
            Optional. Whether to skip the MPLE step and go directly to MCMLE. *Defaults to False*.
        
        mcmc_burn_in : int
            Optional. The number of burn-in steps for the MCMC sampler. *Defaults to 1000*.
        
        mcmc_steps_per_sample : int
            Optional. The number of steps to run the MCMC sampler for each sample. *Defaults to 10*.
        
        mcmc_seed_network : np.ndarray
            Optional. The seed network for the MCMC sampler. If not provided, the thetas that are currently set are used to 
            calculate the MPLE prediction, from which the sample is drawn. *Defaults to None*.
        
        mcmc_sample_size : int
            Optional. The number of networks to sample with the MCMC sampler. *Defaults to 100*.
        
        mple_lr : float
            Optional. The learning rate for the logistic regression model in the MPLE step. *Defaults to 0.001*.
        
        mple_stopping_thr : float
            Optional. The stopping threshold for the logistic regression model in the MPLE step. *Defaults to 1e-6*.
        
        mple_max_iter : int
            Optional. The maximum number of iterations for the logistic regression model in the MPLE step. *Defaults to 1000*.

        edge_proposal_method : str
            Optional. The method for the MCMC proposal distribution. This is defined as a distribution over the edges
            of the network, which implies how to choose a proposed graph out of all graphs that are 1-edge-away from the
            current graph. *Defaults to "uniform"*.
        
        Returns 
        -------
        (grads, hotelling_statistics) : (np.ndarray, list)
        # TODO - what do we want to return?
        """
        # This is because we assume the sample size is even when estimating the covariance matrix (in
        # calc_capital_gammas).
        if mcmc_sample_size % 2 != 0:
            mcmc_sample_size += 1

        if theta_init_method == "use_existing":
            print(f"Using existing thetas")
            pass
        elif not no_mple and self._do_MPLE(theta_init_method):
            self._thetas = self._mple_fit(observed_network, lr=mple_lr, stopping_thr=mple_stopping_thr,
                                          logistic_reg_max_iter=mple_max_iter)

            if not self._metrics_collection._has_dyadic_dependent_metrics:
                print(f"Model is dyadic independent - using only MPLE instead of MCMLE")
                return None, None  # TODO - Remove this in the future. Grads are returned only for debug

        elif theta_init_method == "uniform":
            self._thetas = self._get_random_thetas(sampling_method="uniform")
        else:
            raise ValueError(f"Theta initialization method {theta_init_method} not supported")

        if optimization_method not in ["newton_raphson", "gradient_descent"]:
            raise ValueError(f"Optimization method {optimization_method} not supported.")

        # As in the constructor, the sample size must be even.
        if max_nets_for_sample % 2 != 0:
            max_nets_for_sample += 1

        if convergence_criterion == "observed_bootstrap":
            for metric in self._metrics_collection.metrics:
                if not hasattr(metric, "calculate_bootstrapped_features"):
                    raise ValueError(
                        f"metric {metric.name} does not have a calculate_bootstrapped_features method, the "
                        f"model doesn't support observed_bootstrap as a convergence criterion.")
            num_subsamples_data = kwargs.get("num_subsamples_data", 1000)
            data_splitting_method = kwargs.get("data_splitting_method", "uniform")
            bootstrapped_features = self._metrics_collection.bootstrap_observed_features(observed_network,
                                                                                         num_subsamples=num_subsamples_data,
                                                                                         splitting_method=data_splitting_method)
            observed_covariance = covariance_matrix_estimation(bootstrapped_features,
                                                               bootstrapped_features.mean(axis=1), method='naive')
            inv_observed_covariance = np.linalg.inv(observed_covariance)

        print(f"Initial thetas - {self._thetas}")
        print("optimization started")

        self.optimization_start_time = time.time()
        num_of_features = self._metrics_collection.num_of_features

        grads = np.zeros((opt_steps, num_of_features))
        hotelling_statistics = []

        if convergence_criterion == "hotelling":
            hotelling_critical_value = f.ppf(1 - hotelling_confidence, num_of_features,
                                             mcmc_sample_size - num_of_features)  # F(p, n-p) TODO - doc this better

        if mcmc_seed_network is None:
            probabilities_matrix = self.get_mple_prediction(observed_network)
            mcmc_seed_network = sample_from_independent_probabilities_matrix(probabilities_matrix, 1)
            mcmc_seed_network = mcmc_seed_network[:, :, 0]
        burn_in = mcmc_burn_in
        for i in range(opt_steps):
            if i > 0:
                burn_in = 0
            networks_for_sample = self.generate_networks_for_sample(sample_size=mcmc_sample_size,
                                                                    seed_network=mcmc_seed_network, burn_in=burn_in,
                                                                    mcmc_steps_per_sample=mcmc_steps_per_sample,
                                                                    edge_proposal_method=edge_proposal_method)
            mcmc_seed_network = networks_for_sample[:, :, -1]

            features_of_net_samples = self._metrics_collection.calculate_sample_statistics(networks_for_sample)
            mean_features = np.mean(features_of_net_samples, axis=1)
            observed_features = self._metrics_collection.calculate_statistics(observed_network)

            grad = calc_nll_gradient(observed_features, mean_features)
            if ERGM.do_estimate_covariance_matrix(optimization_method, convergence_criterion):
                # This is for allowing numba to compile and pickle the large function
                sys.setrecursionlimit(2000)
                print(f"Started estimating cov matrix ")
                estimated_cov_matrix = covariance_matrix_estimation(features_of_net_samples,
                                                                    mean_features,
                                                                    method=cov_matrix_estimation_method,
                                                                    num_batches=cov_matrix_num_batches)
                print(f"Done estimating cov matrix ")
                inv_estimated_cov_matrix = np.linalg.pinv(estimated_cov_matrix)
            if optimization_method == "newton_raphson":
                self._thetas = self._thetas - lr * inv_estimated_cov_matrix @ grad

            elif optimization_method == "gradient_descent":
                self._thetas = self._thetas - lr * grad

            grads[i] = grad

            idx_for_sliding_grad = np.max([0, i - sliding_grad_window_k + 1])
            sliding_window_grads = grads[idx_for_sliding_grad:i + 1].mean()

            if (i + 1) % steps_for_decay == 0:
                delta_t = time.time() - self.optimization_start_time
                print(
                    f"Step {i + 1} - lr: {lr:.7f}, time from start: {delta_t:.2f}, window_grad: {sliding_window_grads:.2f}")

                lr *= (1 - lr_decay_pct)

                print(self._thetas)

                if mcmc_sample_size < max_nets_for_sample:
                    mcmc_sample_size *= (1 + sample_pct_growth)
                    mcmc_sample_size = np.min([int(mcmc_sample_size), max_nets_for_sample])
                    # As in the constructor, the sample size must be even.
                    if mcmc_sample_size % 2 != 0:
                        mcmc_sample_size += 1
                    print(f"\t Sample size increased at step {i + 1} to {mcmc_sample_size}")

                if sliding_grad_window_k < max_sliding_window_size:
                    sliding_grad_window_k *= (1 + sample_pct_growth)
                    sliding_grad_window_k = np.min(
                        [np.ceil(sliding_grad_window_k).astype(int), max_sliding_window_size])

            if convergence_criterion == "hotelling":
                dist = mahalanobis(observed_features, mean_features, inv_estimated_cov_matrix)

                hotelling_t_statistic = mcmc_sample_size * dist * dist

                # (n-p / p(n-1))* t^2 ~ F_p, n-p  (#TODO - give reference for this)
                hotelling_as_f_statistic = ((mcmc_sample_size - num_of_features) / (
                        num_of_features * (mcmc_sample_size - 1))) * hotelling_t_statistic

                hotelling_statistics.append({
                    "dist": dist,
                    # "hotelling_t": hotelling_t_statistic,
                    "hotelling_F": hotelling_as_f_statistic,
                    "critical_val": hotelling_critical_value,
                    "inv_cov_norm": np.linalg.norm(inv_estimated_cov_matrix),
                    # "inv_hessian_norm": np.linalg.norm(inv_hessian)
                })

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
            elif convergence_criterion == "observed_bootstrap":
                t1 = time.time()
                # TODO: maybe we don't want to test convergence each iteration, but only after some heuristic criteria
                #  have been met (e.g. small gradient, once in a while also after starting the test etc.).
                num_model_sub_samples = kwargs.get("num_model_sub_samples", 100)
                model_subsample_size = kwargs.get("model_subsample_size", 1000)
                mahalanobis_dists = np.zeros(num_model_sub_samples)
                
                sub_sample_indices = np.random.choice(np.arange(mcmc_sample_size), size=num_model_sub_samples * model_subsample_size)

                sub_samples = networks_for_sample[:, :, sub_sample_indices]
                sub_samples_features = self._metrics_collection.calculate_sample_statistics(sub_samples)
                mean_per_subsample = sub_samples_features.reshape(num_of_features, num_model_sub_samples, model_subsample_size).mean(axis=2)

                for cur_subsam_idx in range(num_model_sub_samples):
                    sub_sample_mean = mean_per_subsample[:, cur_subsam_idx]
                    mahalanobis_dists[cur_subsam_idx] = mahalanobis(observed_features, sub_sample_mean, inv_observed_covariance)

                bootstrap_convergence_confidence = kwargs.get("bootstrap_convergence_confidence", 0.95)
                bootstrap_convergence_num_stds_away_thr = kwargs.get("bootstrap_convergence_num_stds_away_thr", 1)
                empirical_threshold = np.quantile(mahalanobis_dists, bootstrap_convergence_confidence)
                t2 = time.time()
                print(f"Bootstrap convergence test = {empirical_threshold},  took {t2 - t1} seconds")

                if bootstrap_convergence_num_stds_away_thr > empirical_threshold:
                    print(f"Reached a confidence of {bootstrap_convergence_confidence} with the bootstrap convergence "
                          f"test! The model is likely to be up to {bootstrap_convergence_num_stds_away_thr} stds from "
                          f"the data, according to the estimated data variability DONE! ")
                    grads = grads[:i]
                    break
            else:
                raise ValueError(f"Convergence criterion {convergence_criterion} not defined")

        self._last_mcmc_chain_features = features_of_net_samples

        return grads, hotelling_statistics

    def get_mcmc_diagnostics(self, sampled_networks=None, observed_network=None):
        """
        Get MCMC diagnostics for the last chain or a sample of networks.

        Parameters
        ----------
        sampled_networks : np.ndarray
            Optional. A (n, n, k) tensor of sampled networks. If not provided, the last chain is used.
        observed_network : np.ndarray
            Optional. The adjacency matrix of the observed network. If provided, the traces of the features across the
            chain are normalized based on the observed network features.
        """
        if sampled_networks is not None:
            print(f"Calculating MCMC diagnostics for a sample of {sampled_networks.shape[2]} networks")
            features = self._metrics_collection.calculate_sample_statistics(sampled_networks)
        elif self._last_mcmc_chain_features is not None:
            print(
                f"Calculating MCMC diagnostics for the last chain, with {self._last_mcmc_chain_features.shape[1]} networks")
            features = self._last_mcmc_chain_features.copy()
        else:
            # TODO: decide what to do in this case. Maybe give the user an option to generate a new chain.
            raise ValueError(
                "No sampled networks provided and no last chain found. Either rerun with a `sampled_networks` parameter, or run the `fit` function first.")

        features_mean = np.mean(features, axis=1)
        features_std = np.std(features, axis=1)

        # Normalize the features based on the observed network
        if observed_network is not None:
            observed_features = self._metrics_collection.calculate_statistics(observed_network)
            features = features - observed_features[:, None]

        trace_per_features = {}
        parameter_names = self._metrics_collection.get_parameter_names()
        for i, feature in enumerate(parameter_names):
            trace = features[i, :]
            trace_per_features[str(feature)] = trace

        quantlies_to_calculate = [0.025, 0.25, 0.5, 0.75, 0.975]
        quantiles = np.quantile(features, quantlies_to_calculate, axis=1)

        quantile_values = {k: q for k, q in zip(quantlies_to_calculate, quantiles)}

        return {
            "trace": trace_per_features,
            "quantiles": quantile_values,
            "mean": features_mean,
            "std": features_std
        }

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

    def get_model_parameters(self):
        parameter_names = self._metrics_collection.get_parameter_names()
        return dict(zip(parameter_names, self._thetas))

    def get_ignored_features(self):
        return self._metrics_collection.get_ignored_features()

    def get_mple_prediction(self, input_graph):
        Xs, ys = self._metrics_collection.prepare_mple_data(input_graph)

        prediction = calc_logistic_regression_predictions(Xs, self._thetas)
        exact_average_mat = np.zeros((self._n_nodes, self._n_nodes))

        if self._is_directed:
            exact_average_mat[~np.eye(self._n_nodes, dtype=bool)] = prediction

        return exact_average_mat


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
                 metrics_collection: Collection[Metric],
                 is_directed=False,
                 initial_thetas=None):
        super().__init__(n_nodes,
                         metrics_collection,
                         is_directed,
                         initial_thetas)

        if is_directed:
            num_connections = (n_nodes ** 2 - n_nodes)
        else:
            num_connections = (n_nodes ** 2 - n_nodes) // 2

        self._num_nets = 2 ** num_connections

        self._num_features = self._metrics_collection.num_of_features

        all_networks = np.zeros((self._n_nodes, self._n_nodes, self._num_nets))
        for i in range(self._num_nets):
            all_networks[:, :, i] = construct_adj_mat_from_int(i, self._n_nodes, is_directed=is_directed)

        self.all_features_by_all_nets = self._metrics_collection.calculate_sample_statistics(all_networks)

        self._all_weights = np.exp(np.sum(self._thetas[:, None] * self.all_features_by_all_nets, axis=0))
        self._normalization_factor = self._all_weights.sum()

    def _validate_net_size(self):
        return (
                (self._is_directed and self._n_nodes <= BruteForceERGM.MAX_NODES_BRUTE_FORCE_DIRECTED)
                or
                (not self._is_directed and self._n_nodes <= BruteForceERGM.MAX_NODES_BRUTE_FORCE_NOT_DIRECTED)
        )

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
            model = BruteForceERGM(self._n_nodes, list(self._metrics_collection.metrics), initial_thetas=thetas,
                                   is_directed=self._is_directed)
            return np.log(model._normalization_factor) - np.log(model.calculate_weight(observed_network))

        def nll_grad(thetas):
            model = BruteForceERGM(self._n_nodes, list(self._metrics_collection.metrics), initial_thetas=thetas,
                                   is_directed=self._is_directed)
            observed_features = model._metrics_collection.calculate_statistics(observed_network)
            all_probs = model._all_weights / model._normalization_factor

            expected_features = self.all_features_by_all_nets @ all_probs
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
        return res
