import numpy as np

from pyERGM.sampling import NaiveMetropolisHastings
from pyERGM.mple_optimization import *
from pyERGM.utils import generate_erdos_renyi_matrix
from pyERGM.convergence import ConvergenceTester
from pyERGM.constants import *


class ERGM():
    def __init__(
            self,
            n_nodes,
            metrics_collection: Sequence[Metric],
            is_directed: bool,
            *,
            initial_thetas: dict | None = None,
            verbose=True,
            fix_collinearity=True,
            collinearity_fixer_sample_size=1000,
            is_distributed_optimization=False,
            mask: npt.NDArray[bool] | None = None,
            num_samples_per_job_collinearity_fixer: int = 5,
            ratio_threshold_collinearity_fixer: float = 5e-6,
            nonzero_threshold_collinearity_fixer: float = 0.1,
    ):
        """
        An ERGM model object. 
        
        Parameters
        ----------
        n_nodes : int 
            Number of nodes in the network.

        metrics_collection : Sequence[Metric]
            A list of Metric objects for calculating statistics of a network.

        is_directed : bool 
            Whether the graph is directed or not.

        initial_thetas : npdarray 
            Optional. The initial values of the coefficients of the ERGM. If not provided, they are randomly initialized.

        verbose : bool
            Optional. Whether to print progress information. *Defaults to True*

        fix_collinearity : bool
            Optional. Whether to fix collinearity in the metrics. *Defaults to True*

        collinearity_fixer_sample_size : int
            Optional. The number of networks to sample for fixing collinearity. *Defaults to 1000*

        is_distributed_optimization : bool
            Optional. Whether to use distributed computing for optimization (requires LSF cluster). *Defaults to False*

        mask : npt.NDArray[bool] | None
            Optional. Designating which entries should be taken into account for optimization and metric calculations.
            The shape can be either (n, n) or (n**2 - n, 1). The latter is the flattened version with no main diagonal
            of square mask.
        """
        self._n_nodes = n_nodes
        self._is_directed = is_directed
        self._is_distributed_optimization = is_distributed_optimization
        self._mask = self._init_mask(mask)

        self._metrics_collection = MetricsCollection(
            metrics_collection,
            self._is_directed,
            self._n_nodes,
            fix_collinearity=fix_collinearity and (initial_thetas is None),
            collinearity_fixer_sample_size=collinearity_fixer_sample_size,
            is_collinearity_distributed=self._is_distributed_optimization,
            num_samples_per_job_collinearity_fixer=num_samples_per_job_collinearity_fixer,
            ratio_threshold_collinearity_fixer=ratio_threshold_collinearity_fixer,
            nonzero_threshold_collinearity_fixer=nonzero_threshold_collinearity_fixer,
            mask=self._mask,
        )

        if initial_thetas is not None:
            self.set_parameters_from_dict(initial_thetas)
        else:
            self._thetas = np.random.uniform(-1, 1, self._metrics_collection.num_of_features)

        self._optimization_start_time = None

        self.verbose = verbose

        self._exact_average_mat = None

        self._exact_dyadic_distributions = None

        self.mh_sampler = NaiveMetropolisHastings(self._thetas, self._metrics_collection)

        self._last_mcmc_chain_features = None

    def _init_mask(self, mask: npt.NDArray[bool] | None) -> npt.NDArray[bool]:
        if mask is None:
            _mask = None
        else:
            if mask.shape == (self._n_nodes, self._n_nodes):
                _mask = flatten_square_matrix_to_edge_list(mask, self._is_directed)
            elif ((mask.shape == (self._n_nodes ** 2 - self._n_nodes, 1) and self._is_directed) or
                  (mask.shape == (self._n_nodes ** 2 - self._n_nodes // 2, 1) and not self._is_directed)
            ):
                _mask = mask.copy()
            else:
                raise ValueError(
                    f"Invalid mask shape. Expected: ({self._n_nodes}, {self._n_nodes}) or "
                    f"[({self._n_nodes ** 2 - self._n_nodes}, 1) for "
                    f"directed models or ({self._n_nodes ** 2 - self._n_nodes // 2}, 1) for undirected models]. "
                    f"Received: {mask.shape}, the model is {'' if self._is_directed else 'un'}directed."
                )
        return _mask

    def set_parameters_from_dict(self, params_dict: dict[str, float]):
        # Check for unnamed duplicate metrics - these can't be matched with a provided dict of parameters
        unnamed_duplicates = self._metrics_collection.find_unnamed_duplicate_metrics()
        if unnamed_duplicates:
            class_names = sorted(set(cls for _, cls in unnamed_duplicates))
            raise ValueError(
                f"Cannot set parameters from a dictionary when the model contains unnamed duplicate metrics. "
                f"The following metric classes have duplicates without explicit names: {class_names}. "
                f"Please provide a unique 'name' parameter for each duplicate metric instance."
            )

        self._thetas = np.zeros(self._metrics_collection.calc_num_of_features())
        current_model_params = self.get_model_parameters()
        if len(set(params_dict.keys()).difference(set(current_model_params.keys()))) > 0:
            raise ValueError("Got a dict of parameters that does not match the collection of Metrics!")

        total_num_features = len(current_model_params.keys())
        # Iterating the reversed list of keys for not interfering with indexing: always remove the last feature to
        # be removed, thus not changing the indices of features needed to be removed with smaller indices.
        for rev_feat_idx, feat_name in enumerate(list(current_model_params.keys())[::-1]):
            if feat_name not in params_dict.keys():
                self._metrics_collection.remove_feature_by_idx(total_num_features - rev_feat_idx - 1)

        current_model_params = self.get_model_parameters()
        for feat_name in current_model_params.keys():
            current_model_params[feat_name] = params_dict[feat_name]
        self._thetas = np.array(list(current_model_params.values()))

    def print_model_parameters(self):
        """
        Prints the parameters of the ERGM model.
        """
        logger.info(f"Number of nodes: {self._n_nodes}")
        logger.info(f"Thetas: {self._thetas}")
        logger.info(f"Is directed: {self._is_directed}")

    def calculate_statistics(self, W: np.ndarray) -> np.ndarray:
        """
        Calculate sufficient statistics g(y) for a single network.

        Parameters
        ----------
        W : np.ndarray
            Network adjacency matrix of shape (n, n).

        Returns
        -------
        np.ndarray
            Statistics array of shape (num_features,).
        """
        return self._metrics_collection.calculate_statistics(W)

    def calculate_sample_statistics(self, networks_sample: np.ndarray) -> np.ndarray:
        """
        Calculate sufficient statistics for a sample of networks.

        Parameters
        ----------
        networks_sample : np.ndarray
            Array of networks with shape (n, n, sample_size).

        Returns
        -------
        np.ndarray
            Statistics array of shape (num_features, sample_size).
        """
        return self._metrics_collection.calculate_sample_statistics(networks_sample)

    def generate_networks_for_sample(
            self,
            sample_size,
            seed_network=None,
            replace=True,
            burn_in=10000,
            mcmc_steps_per_sample=1000,
            sampling_method: SamplingMethod | None = None,
            edge_proposal_method=EdgeProposalMethod.UNIFORM,
    ):
        """
        Generate a sample of networks from the current ERGM model.

        Parameters
        ----------
        sample_size : int
            Number of networks to generate.
        seed_network : np.ndarray, optional
            Initial network for MCMC sampling. If None, a network is generated
            from MPLE predictions.
        replace : bool, optional
            Whether to sample with replacement. Default is True.
        burn_in : int, optional
            Number of MCMC steps to discard before sampling. Default is 100 * (n**2).
        mcmc_steps_per_sample : int, optional
            Number of MCMC steps between samples. Default is n**2.
        sampling_method : SamplingMethod, optional
            Sampling method to use. Options: "metropolis_hastings", "exact".
            If None (default), auto-detects based on model type:
            MPLE/MPLE_RECIPROCITY models use "exact", MCMLE models use "metropolis_hastings".
        edge_proposal_method : str, optional
            Edge proposal distribution for MCMC. Default is "uniform".

        Returns
        -------
        np.ndarray
            Array of sampled networks with shape (n, n, sample_size).
        """
        if burn_in is None:
            burn_in = 100 * (self._n_nodes ** 2)

        if mcmc_steps_per_sample is None:
            mcmc_steps_per_sample = self._n_nodes ** 2

        # Auto-detect sampling method if not specified
        if sampling_method is None:
            opt_scheme = self._metrics_collection.choose_optimization_scheme()
            if opt_scheme in (OptimizationScheme.MPLE, OptimizationScheme.MPLE_RECIPROCITY):
                sampling_method = SamplingMethod.EXACT
            else:
                sampling_method = SamplingMethod.METROPOLIS_HASTINGS

        if sampling_method == SamplingMethod.METROPOLIS_HASTINGS:
            if seed_network is None:
                seed_network = self._generate_mple_based_seed()

            self.mh_sampler.set_thetas(self._thetas)
            return self.mh_sampler.sample(seed_network, sample_size, replace=replace, burn_in=burn_in,
                                          steps_per_sample=mcmc_steps_per_sample,
                                          edge_proposal_method=edge_proposal_method)
        elif sampling_method == SamplingMethod.EXACT:
            return self._generate_exact_sample(sample_size, replace=replace)
        else:
            raise ValueError(f"Unrecognized sampling method {sampling_method}")

    @staticmethod
    def do_estimate_covariance_matrix(
            optimization_method: OptimizationMethod,
            convergence_tester: ConvergenceTester,
    ) -> bool:
        return (
                optimization_method == OptimizationMethod.NEWTON_RAPHSON or
                convergence_tester.requires_covariance_estimation
        )

    @staticmethod
    def do_mple(optimization_scheme: OptimizationScheme, theta_init_method: ThetaInitMethod) -> bool:
        return (
                optimization_scheme == OptimizationScheme.MPLE or
                (theta_init_method == ThetaInitMethod.MPLE and optimization_scheme == OptimizationScheme.MCMLE)
        )

    def _mple_fit(self, observed_networks,
                  optimization_method: MPLEOptimizationMethod = MPLEOptimizationMethod.L_BFGS_B,
                  edge_weights: np.ndarray | None = None,
                  num_edges_per_job: int = 100000):
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
        edge_weights : np.ndarray or None, optional
            Non-negative edge weights as a flattened array. If provided, each edge's contribution
            to the log-likelihood is scaled by its weight. Default is None (unweighted).
        num_edges_per_job : int, optional
            Number of edges per job for distributed MPLE. Default is 100000.

        Returns
        -------
        thetas: np.ndarray
            The estimated coefficients of the ERGM.
        """
        logger.info("Using MPLE optimization")
        trained_thetas, prediction, success = mple_logistic_regression_optimization(
            self._metrics_collection,
            observed_networks,
            is_distributed=self._is_distributed_optimization,
            optimization_method=optimization_method,
            sample_weights=edge_weights,
            num_edges_per_job=num_edges_per_job,
        )

        self._exact_average_mat = self._rearrange_prediction_to_av_mat(prediction)

        return trained_thetas, success

    def _rearrange_prediction_to_av_mat(self, prediction):
        return reshape_flattened_off_diagonal_elements_to_square(
            flattened_array=prediction,
            is_directed=self._is_directed,
            flat_mask=self._mask,
        )

    def _mple_reciprocity_fit(self, observed_networks, optimization_method: str = 'L-BFGS-B'):
        logger.info("Using MPLE_RECIPROCITY optimization")
        trained_thetas, prediction, success = mple_reciprocity_logistic_regression_optimization(
            self._metrics_collection,
            observed_networks,
            optimization_method=optimization_method)

        self._exact_dyadic_distributions = prediction
        return trained_thetas, success

    def get_mple_prediction(self, observed_networks: np.ndarray | None = None, num_edges_per_job: int = 100000):
        """
        Get the MPLE-based edge probability predictions.

        For dyadic independent models, returns the exact probability matrix.
        This is cached after first computation.

        Parameters
        ----------
        observed_networks : np.ndarray, optional
            Observed network(s). Required for models with dyadic-dependent metrics
            (e.g., NumberOfTriangles, Reciprocity). Optional for dyadic-independent
            models (e.g., NumberOfEdges, InDegree, OutDegree).
            If provided as 3D array, uses the first network (observed_networks[..., 0]).
        num_edges_per_job : int, optional
            Number of edges per job for distributed computation. Default is 100000.

        Returns
        -------
        np.ndarray
            Matrix of edge probabilities of shape (n, n).

        Raises
        ------
        ValueError
            If observed_networks is None and the model contains dyadic-dependent metrics.
        ValueError
            If distributed optimization is enabled for a model with dyadic-dependent metrics.

        Notes
        -----
        - For dyadic-independent models, observed_networks can be None as edge probabilities
          don't depend on network structure.
        - Distributed computation (via _is_distributed_optimization) requires dyadic independence.
        """
        logger.debug("Calculating MPLE prediction")

        # Check for dyadic dependence requirements
        is_dyadic_independent = not self._metrics_collection._has_dyadic_dependent_metrics

        if observed_networks is None and not is_dyadic_independent:
            raise ValueError(
                "observed_networks is required for models with dyadic-dependent metrics. "
                "This model contains dyadic-dependent metrics: "
                f"{[str(m) for m in self._metrics_collection.metrics if not m._is_dyadic_independent]}"
            )

        # Safe dimension check
        if observed_networks is not None and observed_networks.ndim == 3:
            observed_networks = observed_networks[..., 0]

        # Check cache for dyadic-independent models
        if is_dyadic_independent and self._exact_average_mat is not None:
            return self._exact_average_mat.copy()

        if self._is_distributed_optimization:
            logger.debug("Using distributed optimization for MPLE prediction")
            data_path = distributed_mple_data_chunks_calculations(
                self._metrics_collection,
                observed_networks,
                num_edges_per_job=num_edges_per_job,
            )
            pred = analytical_logistic_regression_predictions_distributed(self._thetas, data_path).flatten()
        else:
            Xs = self._metrics_collection.prepare_mple_regressors(observed_networks)
            pred = calc_logistic_regression_predictions(Xs, self._thetas).flatten()
        if is_dyadic_independent:
            self._exact_average_mat = self._rearrange_prediction_to_av_mat(pred)
            return self._exact_average_mat.copy()
        return self._rearrange_prediction_to_av_mat(pred)

    def _do_MPLE(self, theta_init_method):
        if not self._metrics_collection._has_dyadic_dependent_metrics or theta_init_method == ThetaInitMethod.MPLE:
            return True
        return False

    def _generate_exact_sample(self, sample_size: int = 1, replace: bool = True):
        """
        Generate exact samples from MPLE-based models.

        Parameters
        ----------
        sample_size : int
            Number of networks to sample.
        replace : bool, optional
            If True (default), sample with replacement (networks may repeat).
            If False, sample without replacement (all networks unique).
            Uses adaptive batch sampling with uniqueness checking.

        Returns
        -------
        np.ndarray
            Sampled networks of shape (n, n, sample_size).

        Raises
        ------
        RuntimeError
            If replace=False and unable to generate enough unique networks
            (typically due to low model entropy).

        Notes
        -----
        When replace=False is used with large sample sizes or low-entropy models
        (e.g., very high or very low edge probabilities), the method may struggle
        to find enough unique networks and raise a RuntimeError.
        """
        auto_optimization_scheme = self._metrics_collection.choose_optimization_scheme()

        if auto_optimization_scheme == OptimizationScheme.MPLE:
            return sample_from_independent_probabilities_matrix(
                self.get_mple_prediction(),
                sample_size,
                self._is_directed,
                replace=replace
            )
        elif auto_optimization_scheme == OptimizationScheme.MPLE_RECIPROCITY:
            return sample_from_dyads_distribution(
                self.get_mple_reciprocity_prediction(), sample_size, replace=replace
            )
        else:
            raise ValueError(
                "Cannot sample exactly from a model that has dependence that not comes from reciprocity"
            )

    def _generate_mple_based_seed(self) -> np.ndarray:
        """
        Generate a seed network for MCMC sampling using MPLE predictions.

        For dyadic-independent models: uses get_mple_prediction() directly.
        For reciprocity models: uses get_mple_reciprocity_prediction().
        For other dyadic-dependent models (MCMLE): uses get_mple_prediction()
        with a pseudo-observed ER(0.5) network to compute change statistics.

        Returns
        -------
        np.ndarray
            A single network of shape (n, n) to use as MCMC seed.
        """
        reference_network = None
        match self._metrics_collection.choose_optimization_scheme():
            case OptimizationScheme.MPLE_RECIPROCITY:
                dyad_dists = self.get_mple_reciprocity_prediction()
                sample = sample_from_dyads_distribution(dyad_dists, 1)
                return sample[..., 0]

            case OptimizationScheme.MCMLE:
                # For MCMLE: use ER(0.5) as a reference network. This is meant to allow edge-dependent statistics 
                # to change as a result of a single edge flip, so that we won't get a degenerate matrix for 
                # the MPLE regressors. For example, if we pass None, which is equivalent to passing the empty network, 
                # no edge flip will change the reciprocity statistic, and the corresponding column in the regressors 
                # matrix would be all-0s.
                reference_network = generate_erdos_renyi_matrix(self._n_nodes, 0.5, self._is_directed)

        prob_matrix = self.get_mple_prediction(observed_networks=reference_network)
        sample = sample_from_independent_probabilities_matrix(prob_matrix, 1, self._is_directed)
        return sample[:, :, 0]

    def get_mple_reciprocity_prediction(self):
        """
        Get the dyadic state probability distributions for reciprocity models.

        Returns the probability distribution over the four possible dyadic states
        (no edges, i->j only, j->i only, or reciprocal) for each node pair.

        Returns
        -------
        np.ndarray
            Array of shape (n_choose_2, 4) with probability distributions.

        Raises
        ------
        NotImplementedError
            If the model is independent or has dependencies between non-reciprocal edges.
        """
        if self._metrics_collection.choose_optimization_scheme() == OptimizationScheme.MPLE_RECIPROCITY:
            if self._exact_dyadic_distributions is None:
                Xs = self._metrics_collection.prepare_mple_reciprocity_regressors()
                self._exact_dyadic_distributions = predict_multi_class_logistic_regression(Xs, self._thetas)
            return self._exact_dyadic_distributions
        else:
            raise NotImplementedError(
                "get_mple_reciprocity_prediction can only be used for models containing reciprocity, and are otherwise "
                "dyadic independent."
            )

    def calc_model_log_likelihood(
            self,
            observed_network: np.ndarray,
            reduction: Reduction = Reduction.SUM,
            log_base: float = np.exp(1),
    ):
        """
        Calculate the log-likelihood of observed network(s) under the fitted model.

        This method computes the log-likelihood for models fitted with MPLE or MPLE_RECIPROCITY.
        For dyadic independent models (MPLE), it uses the exact probability predictions.
        For reciprocity models, it uses the dyadic state distributions.

        Parameters
        ----------
        observed_network : np.ndarray
            The observed network adjacency matrix of shape (n, n).
        reduction : Reduction, optional
            How to aggregate likelihoods. Default is Reduction.SUM.
            If Reduction.NONE, returns individual edge/dyad likelihoods.
        log_base : float, optional
            Base for logarithm. Default is e (natural log).

        Returns
        -------
        float or np.ndarray
            Log-likelihood value(s). If reduction='none', returns array of individual likelihoods.

        Raises
        ------
        ValueError
            If network dimensions are incorrect or network is non-binary.
        NotImplementedError
            If model has dependencies other than reciprocity.
        """
        if observed_network.ndim != 2 or observed_network.shape[0] != observed_network.shape[1] or \
                observed_network.shape[0] != self._n_nodes:
            raise ValueError(f"Got a connectivity data of dimensions {observed_network.shape}, "
                             f"expected: {(self._n_nodes, self._n_nodes)}")
        if not np.all(np.unique(observed_network) == np.array([0, 1])):
            raise ValueError("Got a non binary connectivity data! Should contain only zeros or ones")
        model_type = self._metrics_collection.choose_optimization_scheme()
        if model_type == OptimizationScheme.MPLE:
            mask = self._mask if self._mask is not None else ...
            if self._exact_average_mat is not None:
                preds = flatten_square_matrix_to_edge_list(
                    self._exact_average_mat, self._is_directed
                )[mask].reshape(-1, 1)
            else:
                Xs = self._metrics_collection.prepare_mple_regressors()
                preds = calc_logistic_regression_predictions(Xs, self._thetas).flatten()
                self._exact_average_mat = self._rearrange_prediction_to_av_mat(preds)
            log_like = calc_logistic_regression_predictions_log_likelihood(
                preds,
                flatten_square_matrix_to_edge_list(observed_network, self._is_directed)[mask].reshape(-1, 1),
                reduction=reduction,
                log_base=log_base,
            )

            # In case the user asks for individual likelihoods, return them in a format of a matrix.
            if reduction == Reduction.NONE:
                return self._rearrange_prediction_to_av_mat(log_like)
            return log_like

        elif model_type == OptimizationScheme.MPLE_RECIPROCITY:
            if self._exact_dyadic_distributions is None:
                Xs = self._metrics_collection.prepare_mple_reciprocity_regressors()
                self._exact_dyadic_distributions = predict_multi_class_logistic_regression(Xs, self._thetas)
            return log_likelihood_multi_class_logistic_regression(
                convert_connectivity_to_dyad_states(observed_network),
                self._exact_dyadic_distributions,
                reduction=reduction,
                log_base=log_base,
            )
        else:
            raise NotImplementedError("Currently supporting likelihood calculations for models that are synaptic "
                                      "independent or with reciprocal synapses dependent")

    def calc_model_entropy(self, reduction: Reduction = Reduction.SUM, eps: float = 1e-10):
        """
        Calculate the entropy of the fitted ERGM model.

        Entropy measures the uncertainty in the model's probability distribution.
        For dyadic independent models, computes entropy from edge probabilities.
        For reciprocity models, computes entropy from dyadic state distributions.

        Parameters
        ----------
        reduction : Reduction, optional
            How to aggregate entropy. Default is Reduction.SUM.
        eps : float, optional
            Small constant to avoid log(0). Default is 1e-10.

        Returns
        -------
        float
            The entropy value in bits.

        Raises
        ------
        NotImplementedError
            If model has dependencies other than reciprocity.
        """
        model_type = self._metrics_collection.choose_optimization_scheme()
        if model_type == OptimizationScheme.MPLE:
            exact_av_mat = self.get_mple_prediction()
            return calc_entropy_independent_probability_matrix(
                prob_mat=exact_av_mat, is_directed=self._is_directed, reduction=reduction, eps=eps
            )
        elif model_type == OptimizationScheme.MPLE_RECIPROCITY:
            exact_dyads_dist = self.get_mple_reciprocity_prediction()
            return calc_entropy_dyads_dists(exact_dyads_dist, reduction=reduction, eps=eps)
        else:
            raise NotImplementedError(
                "Currently supporting entropy calculations for models that are synaptic independent or with reciprocal "
                "synapses dependent"
            )

    def _validate_reshape_edge_weights(
            self,
            edge_weights: np.ndarray | None,
            optimization_scheme: OptimizationScheme,
    ) -> np.ndarray | None:
        if edge_weights is not None:
            directionality_edge_count_factor = 1 if self._is_directed else 2
            flat_size = (self._n_nodes ** 2 - self._n_nodes) // directionality_edge_count_factor
            if edge_weights.shape == (self._n_nodes, self._n_nodes):
                edge_weights = flatten_square_matrix_to_edge_list(edge_weights, self._is_directed)
            elif edge_weights.shape != (flat_size,):
                raise ValueError(
                    f"edge_weights must have shape ({self._n_nodes}, {self._n_nodes}) or ({flat_size},), "
                    f"got {edge_weights.shape}")
            if np.any(edge_weights < 0):
                raise ValueError("edge_weights must be non-negative")
        if edge_weights is not None and optimization_scheme != OptimizationScheme.MPLE:
            raise NotImplementedError("edge_weights are only supported for MPLE optimization.")
        return edge_weights

    def _init_mcmc_hyperparams(
            self,
            mcmc_sample_size: int | None,
            mcmc_burn_in: int | None,
            mcmc_steps_per_sample: int | None,
    ):
        if mcmc_sample_size is None:
            mcmc_sample_size = 10 * self._n_nodes
        # This is because we assume the sample size is even when estimating the covariance matrix (in
        # calc_capital_gammas).
        elif mcmc_sample_size % 2 != 0:
            mcmc_sample_size += 1

        if mcmc_burn_in is None:
            mcmc_burn_in = 100 * (self._n_nodes ** 2)

        if mcmc_steps_per_sample is None:
            mcmc_steps_per_sample = self._n_nodes ** 2
        return mcmc_sample_size, mcmc_burn_in, mcmc_steps_per_sample

    def fit(
            self,
            observed_networks,
            *,
            lr: float = 0.1,
            opt_steps: int = 1000,
            l2_grad_thresh: float = 0.001,
            sliding_grad_window_k: int = 10,
            optimization_method: OptimizationMethod = OptimizationMethod.NEWTON_RAPHSON,
            convergence_criterion: ConvergenceCriterion = ConvergenceCriterion.MODEL_BOOTSTRAP,
            cov_matrix_estimation_method: CovMatrixEstimationMethod = CovMatrixEstimationMethod.NAIVE,
            cov_matrix_num_batches: int = 25,
            hotelling_confidence: float = 0.5,
            theta_init_method: ThetaInitMethod = ThetaInitMethod.MPLE,
            mcmc_burn_in: int | None = None,
            mcmc_seed_network=None,
            mcmc_steps_per_sample: int | None = None,
            mcmc_sample_size: int | None = None,
            edge_proposal_method: EdgeProposalMethod = EdgeProposalMethod.UNIFORM,
            edge_weights: np.ndarray | None = None,
            optimization_scheme: OptimizationScheme = OptimizationScheme.AUTO,
            mple_optimization_method: MPLEOptimizationMethod = MPLEOptimizationMethod.L_BFGS_B,
            num_edges_per_job: int = 100000,
            num_subsamples_data: int = 1000,
            data_splitting_method: DataBootstrapSplittingMethod = DataBootstrapSplittingMethod.UNIFORM,
            observed_cov_mat_est_method: CovMatrixEstimationMethod = CovMatrixEstimationMethod.NAIVE,
            bootstrap_convergence_confidence: float = 0.95,
            bootstrap_convergence_num_stds_away_thr: float = 1.0,
            num_model_sub_samples: int = 100,
            model_subsample_size: int = 1000,
            model_boot_cov_mat_est_method: CovMatrixEstimationMethod = CovMatrixEstimationMethod.NAIVE,
            mcmle_log_every: int = 50,
    ) -> OptimizationResult:
        """
        Fit an ERGM model to a given network with one of the two fitting methods - MPLE or MCMLE.

        Parameters
        ----------
        observed_networks : np.ndarray
            The observed network connectivity matrix, with shape (n, n) or (n, n, num_networks).

        lr : float
            Optional. The learning rate for the optimization. *Defaults to 0.1*

        opt_steps : int
            Optional. The number of optimization steps to run. *Defaults to 1000*

        l2_grad_thresh : float
            Optional. The threshold for the L2 norm of the gradient to stop the optimization. 
            Relevant only for convergence criterion "zero_grad_norm". *Defaults to 0.001*

        sliding_grad_window_k : int
            Optional. The size of the sliding window for the gradient, for which we use to calculate the mean gradient norm. 
            This value is then tested against l2_grad_thresh to decide whether optimization halts.
            Relevant only for convergence criterion "zero_grad_norm". *Defaults to 10*

        optimization_method : OptimizationMethod or str
            The optimization method for MCMLE. Options: "newton_raphson", "gradient_descent".
            *Defaults to OptimizationMethod.NEWTON_RAPHSON*.

        convergence_criterion : ConvergenceCriterion or str
            The criterion for convergence. Options: "hotelling", "zero_grad_norm", "observed_bootstrap", "model_bootstrap".
            *Defaults to ConvergenceCriterion.MODEL_BOOTSTRAP*.

        cov_matrix_estimation_method : CovMatrixEstimationMethod or str
            The method to estimate the covariance matrix.
            Options: "naive", "batch", "multivariate_initial_sequence". *Defaults to CovMatrixEstimationMethod.NAIVE*.

        cov_matrix_num_batches : int
            The number of batches to use for estimating the covariance matrix.
            Relevant only for `cov_matrix_estimation_method="batch"`. *Defaults to 25*.

        hotelling_confidence : float
            The confidence level for the Hotelling's T-squared test. *Defaults to 0.99*.

        theta_init_method : ThetaInitMethod or str
            The method to initialize the theta values. Options: "uniform", "mple", "use_existing".
            The MPLE method can be used even for dyadic dependent models, since it serves as a good starting point for the MCMLE.
            *Defaults to ThetaInitMethod.MPLE*.

        mcmc_burn_in : int
            The number of burn-in steps for the MCMC sampler. *Defaults to 100*n^2*.

        mcmc_steps_per_sample : int
            The number of steps to run the MCMC sampler for each sample. *Defaults to n^2*.

        mcmc_seed_network : np.ndarray
            The seed network for the MCMC sampler. If not provided, the thetas that are currently set are used to
            calculate the MPLE prediction, from which the sample is drawn. *Defaults to None*.

        mcmc_sample_size : int
            The number of networks to sample with the MCMC sampler. *Defaults to 10*n*.
            Note - if odd, 1 is added to make even (this is necessary for some covariance matrix estimation methods,
            see `pyERGM.utils.calc_capital_gammas`).

        edge_proposal_method : EdgeProposalMethod or str
            The method for the MCMC proposal distribution. Options: "uniform", "features_influence__sum", "features_influence__softmax".
            *Defaults to EdgeProposalMethod.UNIFORM*.

        edge_weights : np.ndarray or None
            Non-negative edge weights for MPLE optimization. *Defaults to None*.

        optimization_scheme : OptimizationScheme or str
            The optimization scheme to use. Options: "AUTO", "MPLE", "MPLE_RECIPROCITY", "MCMLE".
            *Defaults to OptimizationScheme.AUTO*.

        mple_optimization_method : MPLEOptimizationMethod or str
            Optimization method for MPLE (scipy.optimize). Options: "L-BFGS-B", "Newton-CG".
            *Defaults to MPLEOptimizationMethod.L_BFGS_B*.

        num_edges_per_job : int
            Number of edges per job for distributed MPLE. *Defaults to 100000*.

        num_subsamples_data : int
            Number of subsamples for observed bootstrap. *Defaults to 1000*.

        data_splitting_method : DataBootstrapSplittingMethod
            Method for data splitting in bootstrap. *Defaults to "DataBootstrapSplittingMethod.UNIFORM"*.

        observed_cov_mat_est_method: ObservedCovMatEstMethod
            Method for estimating the observed covariance matrix used in ConvergenceCriterion.OBSERVED_BOOTSTRAP.

        bootstrap_convergence_confidence : float
            Confidence level for bootstrap convergence tests. *Defaults to 0.95*.

        bootstrap_convergence_num_stds_away_thr : float
            Number of standard deviations threshold for bootstrap convergence. *Defaults to 1.0*.

        num_model_sub_samples : int
            Number of model subsamples for bootstrap convergence. *Defaults to 100*.

        model_subsample_size : int
            Size of each model subsample for bootstrap convergence. *Defaults to 1000*.

        model_boot_cov_mat_est_method: CovMatEstMethod
            Method for estimating the covariance matrix used in ConvergenceCriterion.MODEL_BOOTSTRAP.

        mcmle_log_every: int
            The gap (int optimization steps) between logs when optimizing with MCMLE. *Defaults to 50*.

        Returns 
        -------
        OptimizationResult
        """

        # Validate that the input matrix is binary
        unique_values = np.unique(observed_networks)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(
                f"observed_networks must be a binary matrix containing only 0s and 1s. "
                f"Found values: {unique_values}"
            )

        if optimization_scheme == OptimizationScheme.AUTO:
            optimization_scheme = self._metrics_collection.choose_optimization_scheme()

        edge_weights = self._validate_reshape_edge_weights(edge_weights, optimization_scheme)

        if ERGM.do_mple(optimization_scheme, theta_init_method):
            self._thetas, success = self._mple_fit(
                observed_networks,
                optimization_method=mple_optimization_method,
                edge_weights=edge_weights,
                num_edges_per_job=num_edges_per_job,
            )
            if optimization_scheme == OptimizationScheme.MPLE:
                logger.info("Done training model using MPLE")
                return OptimizationResult(success=success)
        elif optimization_scheme == OptimizationScheme.MPLE_RECIPROCITY:
            if not self._is_directed:
                raise ValueError("There is not meaning for reciprocity in undirected graphs, "
                                 "can't perform MPLE_RECIPROCITY optimization.")
            self._thetas, success = self._mple_reciprocity_fit(observed_networks,
                                                               optimization_method=mple_optimization_method)
            logger.info("Done training model using MPLE_RECIPROCITY")
            return OptimizationResult(success=success)

        mcmc_sample_size, mcmc_burn_in, mcmc_steps_per_sample = self._init_mcmc_hyperparams(
            mcmc_sample_size, mcmc_burn_in, mcmc_steps_per_sample
        )

        observed_features = self._metrics_collection.calculate_sample_statistics(
            expand_net_dims(observed_networks)).mean(axis=-1)

        convergence_tester = ConvergenceTester.create(
            convergence_criterion,
            observed_features=observed_features,
            l2_grad_thresh=l2_grad_thresh,
            sliding_grad_window_k=sliding_grad_window_k,
            opt_steps=opt_steps,
            num_of_features=self._metrics_collection.num_of_features,
            hotelling_confidence=hotelling_confidence,
            observed_networks=observed_networks,
            metrics_collection=self._metrics_collection,
            data_splitting_method=data_splitting_method,
            num_subsamples_data=num_subsamples_data,
            num_model_sub_samples=num_model_sub_samples,
            model_subsample_size=model_subsample_size,
            bootstrap_convergence_confidence=bootstrap_convergence_confidence,
            bootstrap_convergence_num_stds_away_thr=bootstrap_convergence_num_stds_away_thr,
            observed_cov_mat_est_method=observed_cov_mat_est_method,
            model_boot_cov_mat_est_method=model_boot_cov_mat_est_method,
        )

        logger.info(f"Initial thetas: {self._thetas}")
        logger.info("MCMLE optimization started")
        convergence_results = OptimizationResult(success=False)

        self._optimization_start_time = time.time()
        inv_estimated_cov_matrix = None

        if mcmc_seed_network is None:
            probabilities_matrix = self.get_mple_prediction(observed_networks)
            mcmc_seed_network = sample_from_independent_probabilities_matrix(probabilities_matrix, 1, self._is_directed)
            mcmc_seed_network = mcmc_seed_network[..., 0]
        for i in range(opt_steps):
            networks_for_sample = self.generate_networks_for_sample(
                sample_size=mcmc_sample_size,
                seed_network=mcmc_seed_network,
                burn_in=mcmc_burn_in if i == 0 else 0,
                mcmc_steps_per_sample=mcmc_steps_per_sample,
                sampling_method=SamplingMethod.METROPOLIS_HASTINGS,
                edge_proposal_method=edge_proposal_method,
            )
            mcmc_seed_network = networks_for_sample[..., -1]

            features_of_net_samples = self._metrics_collection.calculate_sample_statistics(networks_for_sample)
            mean_features = np.mean(features_of_net_samples, axis=1)

            grad = calc_nll_gradient(observed_features, mean_features)
            if ERGM.do_estimate_covariance_matrix(optimization_method, convergence_tester):
                # This is for allowing numba to compile and pickle the large function
                sys.setrecursionlimit(2000)
                logger.debug("Started estimating covariance matrix")
                estimated_cov_matrix = covariance_matrix_estimation(
                    features_of_net_samples,
                    mean_features,
                    method=cov_matrix_estimation_method,
                    num_batches=cov_matrix_num_batches,
                )
                logger.debug("Done estimating covariance matrix")
                inv_estimated_cov_matrix = np.linalg.pinv(estimated_cov_matrix)
                logger.debug("Done inverting covariance matrix")
            if optimization_method == OptimizationMethod.NEWTON_RAPHSON:
                self._thetas = self._thetas - lr * inv_estimated_cov_matrix @ grad

            elif optimization_method == OptimizationMethod.GRADIENT_DESCENT:
                self._thetas = self._thetas - lr * grad

            if (i + 1) % mcmle_log_every == 0:
                delta_t = time.time() - self._optimization_start_time
                logger.info(f"Step {i + 1}, time from start: {delta_t:.2f}")
                logger.debug(f"Current thetas: {self._thetas}")

            logger.debug("Starting to test for convergence")
            convergence_tester.update(
                mean_features=mean_features,
                grad=grad,
                step=i,
                features_of_net_samples=features_of_net_samples,
                inv_estimated_cov_matrix=inv_estimated_cov_matrix,
                sample_size=mcmc_sample_size,
            )
            convergence_results = convergence_tester.test()
            self._last_mcmc_chain_features = features_of_net_samples
            if convergence_results.success:
                break

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
            logger.debug(f"Calculating MCMC diagnostics for a sample of {sampled_networks.shape[2]} networks")
            features = self._metrics_collection.calculate_sample_statistics(sampled_networks)
        elif self._last_mcmc_chain_features is not None:
            logger.debug(
                f"Calculating MCMC diagnostics for the last chain, with {self._last_mcmc_chain_features.shape[1]} networks")
            features = self._last_mcmc_chain_features.copy()
        else:
            raise ValueError(
                "No sampled networks provided and no last chain found. Either rerun with a `sampled_networks` "
                "parameter, or run the `fit` function first."
            )

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

    def get_model_parameters(self):
        """
        Get the fitted model parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their fitted values (theta coefficients).
        """
        parameter_names = self._metrics_collection.get_parameter_names()
        return dict(zip(parameter_names, self._thetas))

    def get_ignored_features(self):
        """
        Get the names of features that were ignored due to collinearity.

        Returns
        -------
        tuple
            Names of features excluded from the model to avoid multicollinearity.
        """
        return self._metrics_collection.get_ignored_features()
