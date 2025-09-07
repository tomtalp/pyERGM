import datetime
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import f
from pyERGM.sampling import NaiveMetropolisHastings

from pyERGM.metrics import *


class ERGM():
    def __init__(self,
                 n_nodes,
                 metrics_collection: Collection[Metric],
                 is_directed: bool,
                 initial_thetas: dict = None,
                 initial_normalization_factor=None,
                 seed_MCMC_proba=0.25,
                 verbose=True,
                 use_sparse_matrix=False,
                 fix_collinearity=True,
                 collinearity_fixer_sample_size=1000,
                 is_distributed_optimization=False,
                 optimization_options={},
                 **kwargs):
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
                                                     fix_collinearity=fix_collinearity and (initial_thetas is None),
                                                     collinearity_fixer_sample_size=collinearity_fixer_sample_size,
                                                     is_collinearity_distributed=self._is_distributed_optimization,
                                                     num_samples_per_job_collinearity_fixer=kwargs.get(
                                                         'num_samples_per_job_collinearity_fixer', 5),
                                                     ratio_threshold_collinearity_fixer=kwargs.get(
                                                         'ratio_threshold_collinearity_fixer', 5e-6))

        self.n_node_features = self._metrics_collection.n_node_features
        self.node_feature_names = self._metrics_collection.node_feature_names.copy()
        self.node_features_n_categories = self._metrics_collection.node_features_n_categories.copy()

        if initial_thetas is not None:
            if type(initial_thetas) != dict:
                raise ValueError("Initial thetas must be a dictionary keyed by feature names, as returned by "
                                 "`ERGM.get_model_parameters`")
            self._thetas = np.zeros(self._metrics_collection.calc_num_of_features())
            current_model_params = self.get_model_parameters()
            if len(set(initial_thetas.keys()).difference(set(current_model_params.keys()))) > 0:
                raise ValueError("Got initial thetas that do not match the collection of Metrics!")

            total_num_features = len(current_model_params.keys())
            # Iterating the reversed list of keys for not interfering with indexing: always remove the last feature to
            # be removed, thus not changing the indices of features needed to be removed with smaller indices.
            for rev_feat_idx, feat_name in enumerate(list(current_model_params.keys())[::-1]):
                if feat_name not in initial_thetas.keys():
                    self._metrics_collection.remove_feature_by_idx(total_num_features - rev_feat_idx - 1)

            current_model_params = self.get_model_parameters()
            for feat_name in current_model_params.keys():
                current_model_params[feat_name] = initial_thetas[feat_name]
            self._thetas = np.array(list(current_model_params.values()))
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

        self._exact_dyadic_distributions = None

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
        if len(W.shape) != 2 or W.shape[0] != self._n_nodes or W.shape[1] < self._n_nodes:
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
                                     edge_proposal_method='uniform',
                                     edge_node_flip_ratio=None
                                     ):
        if sampling_method == "metropolis_hastings":
            if seed_network is None:
                G = nx.erdos_renyi_graph(self._n_nodes, self._seed_MCMC_proba, directed=self._is_directed)
                seed_connectivity_matrix = nx.to_numpy_array(G)
                seed_neuron_features = np.zeros((self._n_nodes, self.n_node_features))
                for feature_name, feature_indices in self.node_feature_names.items():
                    for feature_index in feature_indices:
                        seed_neuron_features[:, feature_index] = np.random.choice(
                            self.node_features_n_categories[feature_name], size=self._n_nodes)
                seed_network = np.concatenate((seed_connectivity_matrix, seed_neuron_features), axis=1)

            self.mh_sampler.set_thetas(self._thetas)
            return self.mh_sampler.sample(seed_network, sample_size, replace=replace, burn_in=burn_in,
                                          steps_per_sample=mcmc_steps_per_sample,
                                          edge_proposal_method=edge_proposal_method,
                                          node_feature_names=self.node_feature_names,
                                          node_features_n_categories=self.node_features_n_categories,
                                          edge_node_flip_ratio=edge_node_flip_ratio)
        elif sampling_method == "exact":
            return self._generate_exact_sample(sample_size)
        else:
            raise ValueError(f"Unrecognized sampling method {sampling_method}")

    @staticmethod
    def do_estimate_covariance_matrix(optimization_method, convergence_criterion):
        if optimization_method == "newton_raphson" or convergence_criterion == "hotelling":
            return True
        return False

    def _mple_fit(self, observed_networks, optimization_method: str = 'L-BFGS-B', **kwargs):
        """
        Perform MPLE estimation of the ERGM parameters.
        This is done by fitting a logistic regression model, where the X values are the change statistics
        calculated for every edge in the observed network, and the predicted values are the presence of the edge.

        More precisely, we create a train dataset with (X_{i, j}, y_{i,j}) where -
            X_{i, j} = g(y_{i,j}^+) - g(y_{i, j}^-)
            y_{i, j} = 1 if there is an edge between i and j, 0 otherwise

        Parameters
        ----------
        observed_networks : np.ndarray
            The adjacency matrix of the observed network, or an array of adjacency matrices.
        
        Returns
        -------
        thetas: np.ndarray
            The estimated coefficients of the ERGM.
        """
        print("MPLE")
        trained_thetas, prediction, success = mple_logistic_regression_optimization(self._metrics_collection,
                                                                                    observed_networks,
                                                                                    is_distributed=self._is_distributed_optimization,
                                                                                    optimization_method=optimization_method,
                                                                                    num_edges_per_job=kwargs.get(
                                                                                        "num_edges_per_job", 100000))

        self._exact_average_mat = self._rearrange_prediction_to_av_mat(prediction)

        return trained_thetas, success

    def _rearrange_prediction_to_av_mat(self, prediction):
        av_mat = np.zeros((self._n_nodes, self._n_nodes))
        if self._is_directed:
            av_mat[~np.eye(self._n_nodes, dtype=bool)] = prediction
        else:
            upper_triangle_indices = np.triu_indices(self._n_nodes, k=1)
            av_mat[upper_triangle_indices] = prediction
            lower_triangle_indices_aligned = (upper_triangle_indices[1], upper_triangle_indices[0])
            av_mat[lower_triangle_indices_aligned] = prediction
        return av_mat

    def _mple_reciprocity_fit(self, observed_networks, optimization_method: str = 'L-BFGS-B'):
        print("MPLE_RECIPROCITY")
        trained_thetas, prediction, success = mple_reciprocity_logistic_regression_optimization(
            self._metrics_collection,
            observed_networks,
            optimization_method=optimization_method)

        self._exact_dyadic_distributions = prediction
        return trained_thetas, success

    # TODO: decide how a getter for self._exact_average_mat fits in now - if the model is dyadic
    #  independent, the observed_network doesn't matter - predictions will be always the same, and this is the exact
    #  average matrix of the model, so should be computed once and stored. If the model is dyadic dependent, this is an
    #  approximation, and the degree to which changes in the observed_network will change the prediction depend on the
    #  metrics and the specific networks, it can not be pre-determined.
    def get_mple_prediction(self, observed_networks: np.ndarray):
        if observed_networks.ndim == 3:
            observed_networks = observed_networks[..., 0]
        is_dyadic_independent = not self._metrics_collection._has_dyadic_dependent_metrics
        if is_dyadic_independent and self._exact_average_mat is not None:
            return self._exact_average_mat.copy()

        # TODO: handle this case
        if self._is_distributed_optimization:
            raise NotImplementedError(
                "calculating mple predictions distributed without fitting is yet to be implemented")
        Xs = self._metrics_collection.prepare_mple_regressors(observed_networks)
        pred = calc_logistic_regression_predictions(Xs, self._thetas).flatten()
        if is_dyadic_independent:
            self._exact_average_mat = self._rearrange_prediction_to_av_mat(pred)
            return self._exact_average_mat.copy()
        return self._rearrange_prediction_to_av_mat(pred)

    def _do_MPLE(self, theta_init_method):
        if not self._metrics_collection._has_dyadic_dependent_metrics or theta_init_method == "mple":
            return True
        return False

    def _generate_exact_sample(self, sample_size: int = 1):
        # TODO: support getting a flag of `replace` which will enable sampling with no replacements (generating samples
        #  of different networks).
        auto_optimization_scheme = self._metrics_collection.choose_optimization_scheme()
        if auto_optimization_scheme == 'MCMLE':
            # TODO: Actually we can sample for node metrics that are independent, but this is anyhow not implemented
            #  right now, deal with that in the future.
            raise ValueError(
                "Cannot sample exactly from a model that has dependence that not comes from reciprocity or "
                "has node features!")
        if self._exact_average_mat is None and self._exact_dyadic_distributions is None:
            raise ValueError("Cannot sample exactly from a model that is not trained! Call `model.fit()` and pass an "
                             "observed network!")

        if auto_optimization_scheme == 'MPLE':
            return sample_from_independent_probabilities_matrix(self._exact_average_mat, sample_size, self._is_directed)
        elif auto_optimization_scheme == 'MPLE_RECIPROCITY':
            return sample_from_dyads_distribution(self._exact_dyadic_distributions, sample_size)
        else:
            raise ValueError(f"Received an unrecognized optimization scheme from "
                             f"MetricsCollection.choose_optimization_scheme(): {auto_optimization_scheme}. "
                             f"Options are supposed to be MPLE, MPLE_RECIPROCITY, MCMLE.")

    def get_mple_reciprocity_prediction(self):
        if self._metrics_collection.choose_optimization_scheme() == 'MPLE_RECIPROCITY':
            Xs = self._metrics_collection.prepare_mple_reciprocity_regressors()
            mple_reciprocity_prediction = predict_multi_class_logistic_regression(Xs, self._thetas)
            return mple_reciprocity_prediction
        else:
            raise NotImplementedError(
                "get_mple_reciprocity_prediction can only be used for models containing reciprocity, and are otherwise dyadic independent.")

    def calc_model_log_likelihood(self, observed_network: np.ndarray, reduction: str = 'sum',
                                  log_base: float = np.exp(1)):
        if observed_network.ndim != 2 or observed_network.shape[0] != observed_network.shape[1] or \
                observed_network.shape[0] != self._n_nodes:
            raise ValueError(f"Got a connectivity data of dimensions {observed_network.shape}")
        if not np.all(np.unique(observed_network) == np.array([0, 1])):
            raise ValueError("Got a non binary connectivity data! Should contain only zeros or ones")
        model_type = self._metrics_collection.choose_optimization_scheme()
        if model_type == 'MPLE':
            if self._exact_average_mat is None:
                raise RuntimeError("Cannot calculate the likelihood of data before properly fitting the model. Call "
                                   "`model.fit()` and try again.")
            log_like = calc_logistic_regression_predictions_log_likelihood(
                remove_main_diagonal_flatten(self._exact_average_mat).reshape(-1, 1),
                remove_main_diagonal_flatten(observed_network).reshape(-1, 1),
                reduction=reduction,
                log_base=log_base)

            # In case the user asks for individual likelihoods, return them in a format of a matrix.
            # TODO: what should be the values on the main diagonal of the likelihoods matrix? 0's make sense only when
            #  summing up, but the user could have set `reduction='sum'` in the first place... Maybe nans? Or return a
            #  flattened array with out reshaping it into a matrix?
            if reduction == 'none':
                log_like_mat = np.zeros((self._n_nodes, self._n_nodes))
                set_off_diagonal_elements_from_array(log_like_mat, log_like)
                return log_like_mat

            return log_like

        elif model_type == 'MPLE_RECIPROCITY':
            if self._exact_dyadic_distributions is None:
                raise RuntimeError("Cannot calculate the likelihood of data before properly fitting the model. Call "
                                   "`model.fit()` and try again.")
            return log_likelihood_multi_class_logistic_regression(
                convert_connectivity_to_dyad_states(observed_network),
                self._exact_dyadic_distributions,
                reduction=reduction,
                log_base=log_base)
        else:
            raise NotImplementedError("Currently supporting likelihood calculations for models that are synaptic "
                                      "independent or with reciprocal synapses dependent")

    def fit(self,
            observed_networks,
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
            convergence_criterion="model_bootstrap",
            cov_matrix_estimation_method="naive",
            cov_matrix_num_batches=25,
            hotelling_confidence=0.99,
            theta_init_method="mple",
            mcmc_burn_in=1000,
            mcmc_seed_network=None,
            mcmc_steps_per_sample=10,
            mcmc_sample_size=100,
            edge_proposal_method='uniform',
            edge_node_flip_ratio=None,
            observed_node_features=None,
            **kwargs
            ):
        """
        Fit an ERGM model to a given network with one of the two fitting methods - MPLE or MCMLE.

        Parameters
        ----------
        observed_network : np.ndarray
            The observed network connectivity matrix, with shape (n, n).
        
        observed_node_features: dict
            Optional. A dictionary of node features. Each key is the name of the feature, and the value is a list of
            `n` numbers representing the feature of every node. *Defaults to None*.
            
            e.g. - `observed_node_features = {"E_I": [[0, 1, 1, 1]]}`
            
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
            Optional. The method to initialize the theta values. Can be either "uniform", "mple" or "use_existing" (which uses the current thetas).
            The MPLE method can be used even for dyadic dependent models, since it serves as a good starting point for the MCMLE.
            *Defaults to "mple"*.
        
        mcmc_burn_in : int
            Optional. The number of burn-in steps for the MCMC sampler. *Defaults to 1000*.
        
        mcmc_steps_per_sample : int
            Optional. The number of steps to run the MCMC sampler for each sample. *Defaults to 10*.
        
        mcmc_seed_network : np.ndarray
            Optional. The seed network for the MCMC sampler. If not provided, the thetas that are currently set are used to 
            calculate the MPLE prediction, from which the sample is drawn. *Defaults to None*.
        
        mcmc_sample_size : int
            Optional. The number of networks to sample with the MCMC sampler. *Defaults to 100*.

        edge_proposal_method : str
            Optional. The method for the MCMC proposal distribution. This is defined as a distribution over the edges
            of the network, which implies how to choose a proposed graph out of all graphs that are 1-edge-away from the
            current graph. *Defaults to "uniform"*.

        optimization_scheme: str
            Optional. The optimization scheme to use. This can be:
             "AUTO" - Automatically choose the scheme, according to the specification bellow of the conditions by which
                each scheme holds.
             "MPLE" - Maximum-Pseudo-Likelihood - formalizing the problem as a binary logistic regression.
                This holds only when all edges are independent ("dyadic independent model").
             "MPLE_RECIPROCITY" - Maximum-Pseudo-Likelihood w/ Reciprocity - formalizing the problem as a multy-class
                logistic regression. This holds only when the only statistical dependence is between reciprocal pairs
                of edges in a directed graph ((i,j) depends only on (j,i) for all 1<i<j<n).
             "MCMLE" - Markov-Chain-Monte-Carlo-Maximum-Likelihood-Estimation - maximize the likelihood by estimating
                the gradient and hessian using MCMC sampling algorithms. This always holds in theory.
        
        Returns 
        -------
        (grads, hotelling_statistics) : (np.ndarray, list)
        # TODO - what do we want to return?
        """
        # Create the full observed network from adjacency matrix and node features:
        if observed_node_features is not None:
            if len(observed_networks.shape) != 2:
                raise ValueError("Multiple networks are not supported with observed_node_features")
            ordered_observed_node_features = [observed_node_features[fname] for fname in self.node_feature_names.keys()]
            ordered_observed_node_features = [one_d_f for f in ordered_observed_node_features for one_d_f in f]
            ordered_observed_node_features = np.array(ordered_observed_node_features).T
            observed_networks = np.concatenate([observed_networks, ordered_observed_node_features], axis=1)

        # This is because we assume the sample size is even when estimating the covariance matrix (in
        # calc_capital_gammas).
        if mcmc_sample_size % 2 != 0:
            mcmc_sample_size += 1

        # TODO: this is ugly
        is_theta_init = False
        if theta_init_method == "use_existing":
            print(f"Using existing thetas")
            is_theta_init = True
        elif theta_init_method == "uniform":
            self._thetas = self._get_random_thetas(sampling_method="uniform")
            is_theta_init = True

        optimization_scheme = kwargs.get("optimization_scheme", "AUTO")
        if optimization_scheme == "AUTO":
            optimization_scheme = self._metrics_collection.choose_optimization_scheme()
        if optimization_scheme == "MPLE" or (theta_init_method == 'mple' and optimization_scheme == 'MCMLE'):
            self._thetas, success = self._mple_fit(observed_networks,
                                                   optimization_method=kwargs.get('mple_optimization_method',
                                                                                  'L-BFGS-B'),
                                                   num_edges_per_job=kwargs.get('num_edges_per_job', 100000))
            if optimization_scheme == "MPLE":
                print(f"Done training model using MPLE")
                return {"success": success}
            is_theta_init = True
        elif optimization_scheme == "MPLE_RECIPROCITY":
            if not self._is_directed:
                raise ValueError("There is not meaning for reciprocity in undirected graphs, "
                                 "can't perform MPLE_RECIPROCITY optimization.")
            self._thetas, success = self._mple_reciprocity_fit(observed_networks,
                                                               optimization_method=kwargs.get(
                                                                   'mple_optimization_method',
                                                                   'L-BFGS-B'))
            print(f"Done training model using MPLE_RECIPROCITY")
            return {"success": success}
        elif optimization_scheme != "MCMLE":
            raise ValueError(f"Optimization scheme not supported: {optimization_scheme}. "
                             f"Options are: AUTO, MPLE, MPLE_RECIPROCITY, MCMLE")

        if not is_theta_init:
            raise ValueError(f"Theta initialization method {theta_init_method} not supported")

        if optimization_method not in ["newton_raphson", "gradient_descent"]:
            raise ValueError(f"Optimization method {optimization_method} not supported.")

        # As in the constructor, the sample size must be even.
        if max_nets_for_sample % 2 != 0:
            max_nets_for_sample += 1

        if convergence_criterion == "observed_bootstrap":
            if observed_networks.ndim == 3 and observed_networks.shape[-1] > 1:
                raise ValueError("observed_bootstrap doesn't support multiple networks!")
            for metric in self._metrics_collection.metrics:
                if not hasattr(metric, "calculate_bootstrapped_features"):
                    raise ValueError(
                        f"metric {metric.name} does not have a calculate_bootstrapped_features method, the "
                        f"model doesn't support observed_bootstrap as a convergence criterion.")
            num_subsamples_data = kwargs.get("num_subsamples_data", 1000)
            data_splitting_method = kwargs.get("data_splitting_method", "uniform")
            bootstrapped_features = self._metrics_collection.bootstrap_observed_features(observed_networks,
                                                                                         num_subsamples=num_subsamples_data,
                                                                                         splitting_method=data_splitting_method)
            observed_covariance = covariance_matrix_estimation(bootstrapped_features,
                                                               bootstrapped_features.mean(axis=1), method='naive')
            inv_observed_covariance = np.linalg.inv(observed_covariance)

        print(f"Initial thetas - {self._thetas}")
        print(f"{datetime.datetime.now()} optimization started")

        self.optimization_start_time = time.time()
        num_of_features = self._metrics_collection.num_of_features

        grads = np.zeros((opt_steps, num_of_features))

        if mcmc_seed_network is None and self._exact_average_mat is not None:
            probabilities_matrix = self.get_mple_prediction(observed_networks)
            mcmc_seed_network = sample_from_independent_probabilities_matrix(probabilities_matrix, 1, self._is_directed)
            mcmc_seed_network = mcmc_seed_network[:, :, 0]
        burn_in = mcmc_burn_in
        for i in range(opt_steps):
            if i > 0:
                burn_in = 0
            networks_for_sample = self.generate_networks_for_sample(sample_size=mcmc_sample_size,
                                                                    seed_network=mcmc_seed_network, burn_in=burn_in,
                                                                    mcmc_steps_per_sample=mcmc_steps_per_sample,
                                                                    edge_proposal_method=edge_proposal_method,
                                                                    edge_node_flip_ratio=edge_node_flip_ratio)
            mcmc_seed_network = networks_for_sample[:, :, -1]

            features_of_net_samples = self._metrics_collection.calculate_sample_statistics(networks_for_sample)
            mean_features = np.mean(features_of_net_samples, axis=1)
            observed_features = self._metrics_collection.calculate_sample_statistics(
                expand_net_dims(observed_networks)).mean(axis=-1)

            grad = calc_nll_gradient(observed_features, mean_features)
            if ERGM.do_estimate_covariance_matrix(optimization_method, convergence_criterion):
                # This is for allowing numba to compile and pickle the large function
                sys.setrecursionlimit(2000)
                print(f"{datetime.datetime.now()} Started estimating cov matrix ")
                estimated_cov_matrix = covariance_matrix_estimation(features_of_net_samples,
                                                                    mean_features,
                                                                    method=cov_matrix_estimation_method,
                                                                    num_batches=cov_matrix_num_batches)
                print(f"{datetime.datetime.now()} Done estimating cov matrix ")
                inv_estimated_cov_matrix = np.linalg.pinv(estimated_cov_matrix)
                print(f"{datetime.datetime.now()} Done inverting cov matrix ")
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

            print(f"{datetime.datetime.now()} Starting to test for convergence")
            convergence_tester = ConvergenceTester()

            if convergence_criterion == "hotelling":
                print(f"{datetime.datetime.now()} \t Starting a `hotelling` test")
                convergence_results = convergence_tester.hotelling(observed_features, mean_features,
                                                                   inv_estimated_cov_matrix, mcmc_sample_size,
                                                                   hotelling_confidence)
                print(f"{datetime.datetime.now()} \t DONE with `hotelling` test")
                if convergence_results["success"]:
                    print(f"Reached a confidence of {hotelling_confidence} with the hotelling convergence test! DONE! ")
                    grads = grads[:i]
                    break

            elif convergence_criterion == "zero_grad_norm":
                if np.linalg.norm(sliding_window_grads) <= l2_grad_thresh:
                    print(f"Reached threshold of {l2_grad_thresh} after {i} steps. DONE!")
                    grads = grads[:i]

                    # TODO - implement `convergence_results` for this kind of convergence
                    break

            elif convergence_criterion == "observed_bootstrap":
                confidence = kwargs.get("bootstrap_convergence_confidence", 0.95)
                stds_away_thr = kwargs.get("bootstrap_convergence_stds_away_thr", 1)

                convergence_results = convergence_tester.bootstrapped_mahalanobis_from_observed(
                    observed_features,
                    features_of_net_samples,
                    inv_observed_covariance,
                    num_subsamples=kwargs.get("num_model_sub_samples", 100),
                    subsample_size=kwargs.get("model_subsample_size", 1000),
                    confidence=confidence,
                    stds_away_thr=stds_away_thr,
                )

                if convergence_results["success"]:
                    print(f"Reached a confidence of {confidence} with the bootstrap convergence "
                          f"test! The model is likely to be up to {stds_away_thr} stds from "
                          f"the data, according to the estimated data variability DONE! ")
                    grads = grads[:i]
                    break
            elif convergence_criterion == "model_bootstrap":
                print(f"{datetime.datetime.now()} \t Starting a `model_bootstrap` test")
                convergence_results = convergence_tester.bootstrapped_mahalanobis_from_model(
                    observed_features,
                    features_of_net_samples,
                    num_subsamples=kwargs.get("num_model_sub_samples", 100),
                    subsample_size=kwargs.get("model_subsample_size", 1000),
                    confidence=kwargs.get("bootstrap_convergence_confidence", 0.95),
                    stds_away_thr=kwargs.get("bootstrap_convergence_stds_away_thr", 1),
                )
                print(f"{datetime.datetime.now()} \t DONE with `model_bootstrap` test")
                if convergence_results["success"]:
                    print(
                        f"Reached a confidence of {kwargs.get('bootstrap_convergence_confidence', 0.95)} with the bootstrap convergence "
                        f"test! The model is likely to be up to {kwargs.get('bootstrap_convergence_stds_away_thr', 1)} stds from "
                        f"the data, according to the estimated data variability DONE! ")
                    grads = grads[:i]
                    break
            else:
                raise ValueError(f"Convergence criterion {convergence_criterion} not defined")

            sys.stdout.flush()

        self._last_mcmc_chain_features = features_of_net_samples

        return convergence_results

    def get_mcmc_diagnostics(self, sampled_networks=None, observed_network=None):
        """
        Get MCMC diagnostics for the last chain or a sample of networks.

        Parameters
        ----------
        sampled_networks : np.ndarray
            Optional. A (n, n+k, sample_size) tensor of sampled networks. If not provided, the last chain is used.
        observed_network : np.ndarray
            Optional. The network matrix of the observed network. If provided, the traces of the features across the
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
        self._all_weights, self._normalization_factor = self._calc_all_weights()

    def _calc_all_weights(self):
        all_weights = np.exp(np.sum(self._thetas[:, None] * self.all_features_by_all_nets, axis=0))
        normalization_factor = all_weights.sum()
        return all_weights, normalization_factor

    def _validate_net_size(self):
        return (
                (self._is_directed and self._n_nodes <= BruteForceERGM.MAX_NODES_BRUTE_FORCE_DIRECTED)
                or
                (not self._is_directed and self._n_nodes <= BruteForceERGM.MAX_NODES_BRUTE_FORCE_NOT_DIRECTED)
        )

    def calculate_weight(self, W: np.ndarray):
        adj_mat_idx = construct_int_from_adj_mat(W, self._is_directed)
        return self._all_weights[adj_mat_idx]

    def generate_networks_for_sample(self, sample_size, sampling_method="Exact"):
        if sampling_method != "Exact":
            raise ValueError("BruteForceERGM supports only exact sampling (this is its whole purpose)")

        all_nets_probs = self._all_weights / self._normalization_factor
        sampled_idx = np.random.choice(all_nets_probs.size, p=all_nets_probs, size=sample_size)
        return np.stack([construct_adj_mat_from_int(i, self._n_nodes, self._is_directed) for i in sampled_idx], axis=-1)

    def calc_expected_features(self):
        all_probs = self._all_weights / self._normalization_factor

        expected_features = self.all_features_by_all_nets @ all_probs
        return expected_features

    def fit(self, observed_networks):
        def nll(thetas, observed_networks):
            model = BruteForceERGM(self._n_nodes, list(self._metrics_collection.metrics),
                                   initial_thetas={feat_name: thetas[i] for i, feat_name in
                                                   enumerate(self._metrics_collection.get_parameter_names())},
                                   is_directed=self._is_directed)
            log_z = np.log(model._normalization_factor)
            observed_networks_log_weights = np.log(np.array(
                [model.calculate_weight(observed_networks[..., i]) for i in range(observed_networks.shape[2])]))
            return (log_z - observed_networks_log_weights).sum()

        def nll_grad(thetas, observed_networks):
            model = BruteForceERGM(self._n_nodes, list(self._metrics_collection.metrics),
                                   initial_thetas={feat_name: thetas[i] for i, feat_name in
                                                   enumerate(self._metrics_collection.get_parameter_names())},
                                   is_directed=self._is_directed)
            mean_observed_features = model._metrics_collection.calculate_sample_statistics(observed_networks).mean(
                axis=1)
            expected_features = model.calc_expected_features()
            return expected_features - mean_observed_features

        def after_iteration_callback(intermediate_result: OptimizeResult):
            self.optimization_iter += 1
            cur_time = time.time()
            print(f'iteration: {self.optimization_iter}, time from start '
                  f'training: {cur_time - self.optimization_start_time} '
                  f'log likelihood: {-intermediate_result.fun}')

        if observed_networks.ndim == 2:
            observed_networks = observed_networks[..., np.newaxis]

        self.optimization_iter = 0
        print("optimization started")
        self.optimization_start_time = time.time()
        res = minimize(nll, self._thetas, args=(observed_networks,), jac=nll_grad, callback=after_iteration_callback)
        self._thetas = res.x
        self._all_weights, self._normalization_factor = self._calc_all_weights()
        return res


class ConvergenceTester:
    def __init__(self):
        pass

    @staticmethod
    def _get_subsample_features(sampled_networks_features, num_subsamples, subsample_size):
        """
        Receives a sample of networks, and subsample for `num_subsamples` times, each time with `subsample_size` networks.
        For each subsample, calculates the sample statistics and reshapes the result to a tensor of shape (num_of_features, num_subsamples, subsample_size).

        Parameters
        ----------
        sampled_networks_features : np.ndarray
            Features of a sample of networks that will be used for subsampling
        
        num_subsamples : int
            The number of subsamples to draw from the sample.
        
        subsample_size : int
            The size of each subsample.
    
        Returns
        -------
        sub_samples_features : np.ndarray
            A tensor of shape (num_of_features, num_subsamples, subsample_size) containing the features of all subsamples.
        """
        sample_size = sampled_networks_features.shape[1]

        sub_sample_indices = np.random.choice(np.arange(sample_size), size=num_subsamples * subsample_size)
        sub_samples_features = sampled_networks_features[:, sub_sample_indices]
        sub_samples_features = sub_samples_features.reshape(-1, num_subsamples, subsample_size)

        return sub_samples_features

    @staticmethod
    def hotelling(observed_features, mean_features, inverted_sample_cov_matrix, sample_size, confidence=0.99):
        """
        Run the Hotelling's T-squared test for convergence.

        The T-Squarted statistic is calculated as - 
            t^2 = n * dist^2
        where dist is the Mahalanobis distance between the observed and the mean features, used the given
        covariance matrix.

        The T^2 statistic can be transformed into an F statistic - 
            F = (n-p / p(n-1)) * t^2
        where p is the number of features and n is the sample size.
        Finally the F statistic is compared to the critical value of the F distribution with p and n-p degrees of freedom.

        Parameters
        ----------
        observed_features : np.ndarray
            The observed features of the network.
        
        mean_features : np.ndarray
            The mean features of the networks sampled from the model.
        
        inverted_sample_cov_matrix : np.ndarray
            The inverted covariance matrix of the features that were calculated from the model sample.
        
        sample_size : int  
            The number of networks sampled from the model.
        
        confidence : float
            The confidence level for the test. *Defaults to 0.99*.
        """
        dist = mahalanobis(observed_features, mean_features, inverted_sample_cov_matrix)
        hotelling_t_statistic = sample_size * dist * dist

        num_of_features = observed_features.shape[0]

        hotelling_as_f_statistic = ((sample_size - num_of_features) / (
                num_of_features * (sample_size - 1))) * hotelling_t_statistic

        hotelling_critical_value = f.ppf(1 - confidence, num_of_features, sample_size - num_of_features)

        return {
            "success": hotelling_as_f_statistic <= hotelling_critical_value,
            "statistic": hotelling_as_f_statistic,
            "threshold": hotelling_critical_value
        }

    @staticmethod
    def bootstrapped_mahalanobis_from_observed(
            observed_features,
            sampled_networks_features,
            inverted_observed_cov_matrix,
            num_subsamples=100,
            subsample_size=1000,
            confidence=0.95,
            stds_away_thr=1):
        """
        Repeatedly subsample from a sample of networks, and calculate the distance between the subsample mean
        and the observed features. The distance is calculated using the Mahalanobis distance, with the covariance matrix
        being an estimation of the observed covariance. The observed covariance can either be the real covariance of the data,
        or an estimation of the covariance matrix (using methods like `Network splitting augmentation` or `Noise augmentation`).
        """
        mahalanobis_dists = np.zeros(num_subsamples)

        sub_samples_features = ConvergenceTester._get_subsample_features(sampled_networks_features, num_subsamples,
                                                                         subsample_size)
        mean_per_subsample = sub_samples_features.mean(axis=2)

        for cur_subsam_idx in range(num_subsamples):
            cur_subsample_mean = mean_per_subsample[:, cur_subsam_idx]
            mahalanobis_dists[cur_subsam_idx] = mahalanobis(observed_features, cur_subsample_mean,
                                                            inverted_observed_cov_matrix)

        empirical_threshold = np.quantile(mahalanobis_dists, confidence)

        return {
            "success": empirical_threshold < stds_away_thr,
            "statistic": empirical_threshold,
            "threshold": stds_away_thr
        }

    @staticmethod
    def bootstrapped_mahalanobis_from_model(
            observed_features,
            sampled_networks_features,
            num_subsamples=100,
            subsample_size=1000,
            confidence=0.95,
            stds_away_thr=1):
        """
        Repeatedly subsample from a collection of networks sampled from the model (`sampled_networks`), and calculate the Mahalanobis distance 
        between each subsample mean and the observed network. This is equivalent to generating multiple estimations of the model mean & covariance.
        We calculate the cutoff threshold for the Mahalanobis distance, according to the provided `confidence` level, and then verify whether
        the empirical threshold is below `stds_away_thr` (which is standard deviations away from the observed data).

        Parameters
        ----------
        observed_features : np.ndarray
            The observed features of the network.
        
        sampled_networks_features : np.ndarray
            Features of networks sampled from the model.
        
        num_subsamples : int
            The number of subsamples to draw. *Defaults to 100*.

        subsample_size : int
            The size of each subsample. *Defaults to 1000*.
        
        confidence : float
            The confidence level for the test. *Defaults to 0.95*.

        stds_away_thr : float
            The desired threshold for the Mahalanobis distance, in units of std *Defaults to 1*.
        """
        mahalanobis_dists = np.zeros(num_subsamples)

        sub_samples_features = ConvergenceTester._get_subsample_features(sampled_networks_features, num_subsamples,
                                                                         subsample_size)

        for cur_subsam_idx in range(num_subsamples):
            # print(
            #     f"{datetime.datetime.now()} [model_bootstrap] \t\t Working on subsample {cur_subsam_idx}/{num_subsamples}")
            sub_sample = sub_samples_features[:, cur_subsam_idx, :]
            sub_sample_mean = sub_sample.mean(axis=1)
            model_covariance_matrix = covariance_matrix_estimation(sub_sample, sub_sample_mean, method="naive")

            if np.all(model_covariance_matrix == 0):
                mahalanobis_dists[cur_subsam_idx] = np.inf
                continue

            inv_model_cov_matrix = np.linalg.pinv(model_covariance_matrix)

            mahalanobis_dists[cur_subsam_idx] = mahalanobis(observed_features, sub_sample_mean, inv_model_cov_matrix)

        empirical_threshold = np.quantile(mahalanobis_dists, confidence)

        return {
            "success": empirical_threshold < stds_away_thr,
            "statistic": empirical_threshold,
            "threshold": stds_away_thr
        }
