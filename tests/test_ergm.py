import unittest

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from pyERGM.utils import *
from pyERGM.metrics import *
from pyERGM.ergm import ERGM
from pyERGM.datasets import sampson_matrix
import sys
from scipy.linalg import eigh

class BruteForceERGM(ERGM):
    """
    Exact ERGM implementation via exhaustive enumeration of all networks.

    This class computes ERGM quantities exactly by enumerating all possible networks
    and calculating statistics, weights, and normalization constants. This is only
    tractable for very small networks (≤5 nodes directed, ≤7 nodes undirected).

    Primarily used for testing and validation of approximate methods.

    Parameters
    ----------
    n_nodes : int
        Number of nodes. Must be ≤5 for directed or ≤7 for undirected networks.
    metrics_collection : Collection[Metric]
        Collection of metrics to compute.
    is_directed : bool, optional
        Whether the network is directed. Default is False.
    initial_thetas : dict, optional
        Initial parameter values. If None, randomly initialized.

    Attributes
    ----------
    MAX_NODES_BRUTE_FORCE_DIRECTED : int
        Maximum nodes for directed networks (5).
    MAX_NODES_BRUTE_FORCE_NOT_DIRECTED : int
        Maximum nodes for undirected networks (7).
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
        """
        Calculate the expected values of all features under the model.

        Returns
        -------
        np.ndarray
            Expected feature values (sufficient statistics) under the current model.
        """
        all_probs = self._all_weights / self._normalization_factor

        expected_features = self.all_features_by_all_nets @ all_probs
        return expected_features

    def fit(self, observed_networks):
        """
        Fit the model parameters to observed network(s) using exact likelihood.

        Uses scipy.optimize.minimize to maximize the exact log-likelihood by
        enumerating all possible networks.

        Parameters
        ----------
        observed_networks : np.ndarray
            Observed network(s) of shape (n, n) or (n, n, num_networks).

        Returns
        -------
        OptimizeResult
            Optimization result object from scipy.optimize.minimize.
        """
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
            logger.info(f'iteration: {self.optimization_iter}, time from start '
                  f'training: {cur_time - self.optimization_start_time} '
                  f'log likelihood: {-intermediate_result.fun}')

        if observed_networks.ndim == 2:
            observed_networks = observed_networks[..., np.newaxis]

        self.optimization_iter = 0
        logger.info("BruteForceERGM optimization started")
        self.optimization_start_time = time.time()
        res = minimize(nll, self._thetas, args=(observed_networks,), jac=nll_grad, callback=after_iteration_callback)
        self._thetas = res.x
        self._all_weights, self._normalization_factor = self._calc_all_weights()
        return res

def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Concordance-Correlation-Coefficient
    Measures the goodness of fit of data and predictions (closeness to the identity line)
    Can be defined as 1 - mean_distance_from_identity/mean_distance_from_identity_assuming_independence
    """
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)


class TestERGM(unittest.TestCase):
    def setUp(self):
        self.metrics = [NumberOfEdgesUndirected(), NumberOfTriangles()]
        self.n_nodes = 3

        self.K = 100
        self.thetas = {str(m): 1 for m in self.metrics}

    def test_calculate_weight(self):
        ergm = ERGM(self.n_nodes, self.metrics, is_directed=False, initial_thetas=self.thetas,
                    initial_normalization_factor=self.K)

        W = np.array([[0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0]])
        weight = ergm.calculate_weight(W)

        expected_num_edges = 3
        expected_num_triangles = 1
        expected_weight = np.exp(expected_num_edges * 1 + expected_num_triangles * 1)

        self.assertEqual(weight, expected_weight)

        W = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])
        weight = ergm.calculate_weight(W)

        expected_num_edges = 0
        expected_num_triangles = 0
        expected_weight = np.exp(expected_num_edges * 1 + expected_num_triangles * 1)

        self.assertEqual(weight, expected_weight)

    def test_calculate_probability(self):
        ergm = ERGM(self.n_nodes, self.metrics, is_directed=False, initial_thetas=self.thetas,
                    initial_normalization_factor=self.K)

        W = np.array([[0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0]])
        probability = ergm.calculate_probability(W)

        expected_num_edges = 3
        expected_num_triangles = 1
        expected_weight = np.exp(expected_num_edges * 1 + expected_num_triangles * 1)
        expected_probability = expected_weight / self.K

        self.assertEqual(probability, expected_probability)

    def test_calculate_probability_wiki_example(self):
        thetas = [-np.log(2), np.log(3)]
        K = 29 / 8

        ergm = ERGM(self.n_nodes, self.metrics, is_directed=False,
                    initial_thetas={str(m): thetas[i] for i, m in enumerate(self.metrics)},
                    initial_normalization_factor=K)

        W_0_edges = np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]])

        probability = ergm.calculate_probability(W_0_edges)
        expected_probability = 1 / K
        self.assertEqual(probability, expected_probability)

        W_1_edges = np.array([[0, 0, 1],
                              [0, 0, 0],
                              [1, 0, 0]])

        probability = ergm.calculate_probability(W_1_edges)
        expected_probability = 0.5 / K
        self.assertEqual(probability, expected_probability)

        W_2_edges = np.array([[0, 1, 1],
                              [1, 0, 0],
                              [1, 0, 0]])

        probability = ergm.calculate_probability(W_2_edges)
        expected_probability = 0.25 / K
        self.assertEqual(probability, expected_probability)

        W_3_edges = np.array([[0, 1, 1],
                              [1, 0, 1],
                              [1, 1, 0]])

        probability = round(ergm.calculate_probability(W_3_edges), 6)
        expected_probability = round(0.375 / K, 6)
        self.assertEqual(probability, expected_probability)

    def test_benchmark_er_convergence(self, n=5, p=0.1, is_directed=False):
        set_seed(9873645)
        print(f"Running an ERGM bruteforce fit with {n} nodes, p={p}, directed={is_directed}")
        num_pos_connect = n * (n - 1)

        if not is_directed:
            num_pos_connect //= 2

        ground_truth_num_edges = round(num_pos_connect * p)
        ground_truth_p = ground_truth_num_edges / num_pos_connect
        ground_truth_theta = np.array([np.log(ground_truth_p / (1 - ground_truth_p))])

        adj_mat_no_diag = np.zeros(num_pos_connect)
        on_indices = np.random.choice(num_pos_connect, size=ground_truth_num_edges, replace=False).astype(int)
        adj_mat_no_diag[on_indices] = 1
        adj_mat = np.zeros((n, n))

        if not is_directed:
            upper_triangle_indices = np.triu_indices(n, k=1)
            adj_mat[upper_triangle_indices] = adj_mat_no_diag
            lower_triangle_indices_aligned = (upper_triangle_indices[1], upper_triangle_indices[0])
            adj_mat[lower_triangle_indices_aligned] = adj_mat_no_diag
        else:
            adj_mat[~np.eye(n, dtype=bool)] = adj_mat_no_diag

        number_of_edges_metric = NumberOfEdgesDirected() if is_directed else NumberOfEdgesUndirected()
        model = BruteForceERGM(n, [number_of_edges_metric], is_directed=is_directed)
        model.fit(adj_mat)

        print(f"ground truth theta: {ground_truth_theta}")
        print(f"fit theta: {model._thetas}")

        for t_model, t_ground_truth in zip(model._thetas, ground_truth_theta):
            self.assertAlmostEqual(t_model, t_ground_truth, places=5)

        non_synapses_indices = np.where(adj_mat_no_diag == 0)[0]
        prediction = ground_truth_p * np.ones(adj_mat_no_diag.size)
        prediction[non_synapses_indices] = 1 - ground_truth_p
        true_log_like = np.log(prediction).sum()
        print(f"true log likelihood: {true_log_like}")

        number_of_edges_metric = NumberOfEdgesDirected() if is_directed else NumberOfEdgesUndirected()
        model_with_true_theta = BruteForceERGM(n, [number_of_edges_metric],
                                               initial_thetas={str(number_of_edges_metric): ground_truth_theta[0]},
                                               is_directed=is_directed)

        ground_truth_model_log_like = np.log(model_with_true_theta.calculate_weight(adj_mat)) - np.log(
            model_with_true_theta._normalization_factor)

        print(f"model with true theta log like: {ground_truth_model_log_like}")
        print(f"normalization factor: {model_with_true_theta._normalization_factor}")

    def test_fit_small_ER_network(self):
        n = 4
        p = 0.25
        is_directed = False

        num_pos_connect = n * (n - 1)
        if not is_directed:
            num_pos_connect //= 2

        ground_truth_num_edges = round(num_pos_connect * p)
        ground_truth_p = ground_truth_num_edges / num_pos_connect
        ground_truth_theta = np.array([np.log(ground_truth_p / (1 - ground_truth_p))])

        adj_mat_no_diag = np.zeros(num_pos_connect)
        on_indices = np.random.choice(num_pos_connect, size=ground_truth_num_edges, replace=False).astype(int)
        adj_mat_no_diag[on_indices] = 1
        adj_mat = np.zeros((n, n))

        if not is_directed:
            upper_triangle_indices = np.triu_indices(n, k=1)
            adj_mat[upper_triangle_indices] = adj_mat_no_diag
            lower_triangle_indices_aligned = (upper_triangle_indices[1], upper_triangle_indices[0])
            adj_mat[lower_triangle_indices_aligned] = adj_mat_no_diag
        else:
            adj_mat[~np.eye(n, dtype=bool)] = adj_mat_no_diag

        W = adj_mat
        print("Fitting matrix - ")
        print(W)

        ergm = ERGM(n,
                    [NumberOfEdgesUndirected()],
                    is_directed=is_directed,
                    seed_MCMC_proba=0.25
                    )

        ergm.fit(W, lr=0.01, opt_steps=300, sliding_grad_window_k=10, sample_pct_growth=0.05, steps_for_decay=20,
                 lr_decay_pct=0.05, mcmc_sample_size=200, mcmc_steps_per_sample=10)

        fit_theta = ergm._thetas[0]
        print(f"ground truth theta: {ground_truth_theta}")
        print(f"fit theta: {fit_theta}")

        # TODO - what criteria to use for testing convergence? From manual tests, it doesn't seem to perfectly converge on the true thetas...
        # Nevertheless, even without an assert, this will catch errors - fit won't work if something breaks.

    def test_MPLE(self):
        n = 4

        M1 = np.array([
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 0]
        ])

        types = ["A", "A", "B", "B"]
        metrics = [NumberOfEdgesTypesDirected(types)]
        model = ERGM(n, metrics, is_directed=True)
        result = model.fit(M1, mple_lr=1)

        self.assertTrue(result["success"])

        inferred_probas_per_type_pairs = list(np.exp(model._thetas) / (1 + np.exp(model._thetas)))

        real_densities_per_type = get_edge_density_per_type_pairs(M1, types)
        real_densities_per_type = list(real_densities_per_type.values())

        for inferred_proba, real_density in zip(inferred_probas_per_type_pairs, real_densities_per_type):
            self.assertAlmostEqual(inferred_proba, real_density, places=4)

    def test_MPLE_convergence_with_auto_LR_decay(self):
        n = 4

        M1 = np.array([
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 0]
        ])

        types = ["A", "A", "B", "B"]
        metrics = [NumberOfEdgesTypesDirected(types)]
        model = ERGM(n, metrics, is_directed=True)
        model.fit(M1, mple_max_iter=1000, mple_lr=1000, mple_stopping_thr=1e-10)

        inferred_probas_per_type_pairs = list(np.exp(model._thetas) / (1 + np.exp(model._thetas)))

        real_densities_per_type = get_edge_density_per_type_pairs(M1, types)
        real_densities_per_type = list(real_densities_per_type.values())

        for inferred_proba, real_density in zip(inferred_probas_per_type_pairs, real_densities_per_type):
            self.assertAlmostEqual(inferred_proba, real_density, places=4)

    def test_sampson_MCMLE(self):
        set_seed(1234)
        metrics = [NumberOfEdgesDirected(), OutDegree(), InDegree(), TotalReciprocity()]
        n_nodes = sampson_matrix.shape[0]

        mcmle_model = ERGM(n_nodes, metrics, is_directed=True)

        convergence_result = mcmle_model.fit(sampson_matrix,
                                             opt_steps=10,
                                             steps_for_decay=1,
                                             lr=1,
                                             mple_lr=0.5,
                                             convergence_criterion="model_bootstrap",
                                             mcmc_burn_in=0,
                                             mcmc_steps_per_sample=n_nodes ** 2,
                                             mcmc_sample_size=1000,
                                             num_model_sub_samples=10,
                                             model_subsample_size=1000,
                                             bootstrap_convergence_confidence=0.95,
                                             bootstrap_convergence_num_stds_away_thr=1,
                                             optimization_scheme='MCMLE'
                                             )

        model_thethas = mcmle_model._thetas

        expected_values = {"edges": -1.1761, "sender2": -0.2945, "sender3": 1.4141, "sender4": 0.3662,
                           "sender5": 0.1315,
                           "sender6": 1.2148, "sender7": 0.6055,
                           "sender8": 1.3609, "sender9": 0.6402, "sender10": 2.0639, "sender11": 1.4355,
                           "sender12": -0.1681,
                           "sender13": -0.2322, "sender14": 0.5841, "sender15": 1.8600,
                           "sender16": 1.4317, "sender17": 1.2211, "sender18": 1.8724, "receiver2": -0.1522,
                           "receiver3": -3.0453,
                           "receiver4": -1.7596, "receiver5": -0.8198, "receiver6": -3.3922,
                           "receiver7": -1.6074, "receiver8": -2.2656, "receiver9": -2.2069, "receiver10": -3.9189,
                           "receiver11": -3.0257, "receiver12": -0.9457, "receiver13": -1.4749, "receiver14": -1.5950,
                           "receiver15": -3.3147, "receiver16": -3.0567, "receiver17": -3.4436, "receiver18": -3.3239,
                           "mutual": 3.6918
                           }
        expected_thetas = np.array(list(expected_values.values()))

        thetas_ccc = ccc(model_thethas, expected_thetas)
        self.assertTrue(thetas_ccc > 0.99)
        self.assertTrue(convergence_result["success"])

    def test_assigning_model_initial_thetas(self):
        # TODO: seems like convergence of the model in this test depends on the seed...
        set_seed(8765)
        n_nodes = 5
        W = np.array([[0, 0, 1, 1, 0],
                      [1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1],
                      [0, 1, 0, 0, 0],
                      [0, 1, 0, 1, 0]])
        metrics_1 = [NumberOfEdgesDirected(),
                     NumberOfEdgesTypesDirected(['A', 'B', 'A', 'A', 'B'])]
        model_1 = ERGM(n_nodes=n_nodes, metrics_collection=metrics_1, is_directed=True)

        model_1.fit(W)

        model_1_params = model_1.get_model_parameters()

        metrics_2 = [NumberOfEdgesDirected(),
                     NumberOfEdgesTypesDirected(['B', 'B', 'B', 'A', 'A'])]
        model_2 = ERGM(n_nodes=n_nodes, metrics_collection=metrics_2, is_directed=True, initial_thetas=model_1_params)

        self.assertTrue(model_2.get_model_parameters() == model_1_params)

    def test_calculate_prediction(self):
        n_nodes = 4
        W = np.array([[0, 1, 0, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0],
                      [0, 0, 1, 0]])

        metrics = [NumberOfEdgesDirected()]
        model = ERGM(n_nodes=n_nodes, metrics_collection=metrics, is_directed=True)
        model.fit(W)

        model_2 = ERGM(n_nodes=n_nodes, metrics_collection=metrics, is_directed=True,
                       initial_thetas=model.get_model_parameters())
        model_2_av_mat = model_2.get_mple_prediction(W)
        expected_model_2_av_mat = 0.5 * np.ones((n_nodes, n_nodes))
        expected_model_2_av_mat[np.diag_indices(n_nodes)] = 0
        self.assertTrue(np.abs(model_2_av_mat - expected_model_2_av_mat).max() < 1e-10)

    def test_sampson_MPLE_RECIPROCITY(self):
        set_seed(8765)

        metrics = [NumberOfEdgesDirected(), OutDegree(), InDegree(), TotalReciprocity()]
        n_nodes = sampson_matrix.shape[0]

        mcmle_model = ERGM(n_nodes, metrics, is_directed=True)

        convergence_result = mcmle_model.fit(sampson_matrix,
                                             opt_steps=10,
                                             steps_for_decay=1,
                                             lr=1,
                                             mple_lr=0.5,
                                             convergence_criterion="model_bootstrap",
                                             mcmc_burn_in=0,
                                             mcmc_steps_per_sample=n_nodes ** 2,
                                             mcmc_sample_size=1000,
                                             num_model_sub_samples=10,
                                             model_subsample_size=1000,
                                             bootstrap_convergence_confidence=0.95,
                                             bootstrap_convergence_num_stds_away_thr=1,
                                             optimization_scheme='MPLE_RECIPROCITY'
                                             )

        model_thethas = mcmle_model._thetas

        expected_values = {"edges": -1.1761, "sender2": -0.2945, "sender3": 1.4141, "sender4": 0.3662,
                           "sender5": 0.1315,
                           "sender6": 1.2148, "sender7": 0.6055,
                           "sender8": 1.3609, "sender9": 0.6402, "sender10": 2.0639, "sender11": 1.4355,
                           "sender12": -0.1681,
                           "sender13": -0.2322, "sender14": 0.5841, "sender15": 1.8600,
                           "sender16": 1.4317, "sender17": 1.2211, "sender18": 1.8724, "receiver2": -0.1522,
                           "receiver3": -3.0453,
                           "receiver4": -1.7596, "receiver5": -0.8198, "receiver6": -3.3922,
                           "receiver7": -1.6074, "receiver8": -2.2656, "receiver9": -2.2069, "receiver10": -3.9189,
                           "receiver11": -3.0257, "receiver12": -0.9457, "receiver13": -1.4749, "receiver14": -1.5950,
                           "receiver15": -3.3147, "receiver16": -3.0567, "receiver17": -3.4436, "receiver18": -3.3239,
                           "mutual": 3.6918
                           }
        expected_thetas = np.array(list(expected_values.values()))

        thetas_ccc = ccc(model_thethas, expected_thetas)
        self.assertTrue(thetas_ccc > 0.99)
        self.assertTrue(convergence_result["success"])

    def test_mple_reciprocity_sampling(self):
        set_seed(8765)
        metrics = [NumberOfEdgesDirected(), OutDegree(), InDegree(), TotalReciprocity()]
        n_nodes = sampson_matrix.shape[0]

        mcmle_model = ERGM(n_nodes, metrics, is_directed=True)

        convergence_result = mcmle_model.fit(sampson_matrix,
                                             optimization_scheme='MPLE_RECIPROCITY'
                                             )

        sample_size = 10
        sampled_networks = mcmle_model.generate_networks_for_sample(sampling_method="exact", sample_size=sample_size)

        self.assertEqual(sampled_networks.shape, (n_nodes, n_nodes, sample_size))
        self.assertEqual(convergence_result["success"], True)

    def test_model_initialization_from_existing_params(self):
        set_seed(1234)
        metrics = [NumberOfEdgesDirected(), OutDegree(), InDegree(), TotalReciprocity()]
        n_nodes = sampson_matrix.shape[0]

        mcmle_model = ERGM(n_nodes, metrics, is_directed=True)

        model_params = mcmle_model.get_model_parameters()

        new_model = ERGM(n_nodes, metrics, is_directed=True, initial_thetas=model_params)

        # If there's a problem with copying the parameters, this will throw an error.
        new_model.generate_networks_for_sample(sample_size=10)

    def test_likelihood_calculations(self):
        set_seed(1234)

        # Independent model
        metrics = [NumberOfEdgesDirected()]
        observed_net = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        observed_p = observed_net.sum() / (observed_net.size - observed_net.shape[0])

        expected_theta = np.log(observed_p / (1 - observed_p))  # log odds ratio for p

        model = ERGM(n_nodes=4, metrics_collection=metrics, is_directed=True)
        result = model.fit(observed_net)
        self.assertTrue(result["success"])

        self.assertTrue(np.abs(model._thetas - expected_theta) < 1e-5)

        network_for_likelihood = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        expected_individual_likes = np.zeros(network_for_likelihood.shape)
        expected_individual_likes[network_for_likelihood == 1] = observed_p
        expected_individual_likes[network_for_likelihood == 0] = 1 - observed_p
        expected_individual_likes[np.diag_indices(network_for_likelihood.shape[0])] = 1

        all_edges_likes = model.calc_model_log_likelihood(network_for_likelihood, reduction='none')
        self.assertTrue(np.all(np.abs(all_edges_likes - np.log(expected_individual_likes))) < 1e-5)

        log_like_sum = model.calc_model_log_likelihood(network_for_likelihood, reduction='sum', log_base=10)
        self.assertTrue(np.abs(log_like_sum - np.log10(expected_individual_likes).sum()) < 1e-5)

        # Model with reciprocity
        metrics = [TotalReciprocity()]
        observed_net = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        observed_reciprocity_p = (observed_net * observed_net.T).sum() / (observed_net.size - observed_net.shape[0])
        expected_theta = -0.5108286293551415  # mean over 4 optimizations of a network with a single reciprocal dyad

        model = ERGM(n_nodes=4, metrics_collection=metrics, is_directed=True)

        result = model.fit(observed_net)
        self.assertTrue(result["success"])

        self.assertTrue(np.abs(model._thetas - expected_theta) < 1e-5)

        network_for_likelihood = np.array([
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])

        dyad_states_for_likelihood = convert_connectivity_to_dyad_states(network_for_likelihood)
        reciprocal_dyads_indices = np.where(dyad_states_for_likelihood.sum(axis=0) == 1)[0]

        all_dyad_likes = model.calc_model_log_likelihood(network_for_likelihood, reduction='none')
        # TODO: we assert here only the likelihood of having a reciprocal dyad. Should find a way (may be numerical) to
        #  evaluate the expected probabilities for the other 3 dyadic states and validate all of them.
        self.assertTrue(
            np.all(np.abs(all_dyad_likes[reciprocal_dyads_indices] - np.log(observed_reciprocity_p)) < 1e-5))

        brute_force_ergm = BruteForceERGM(n_nodes=4, metrics_collection=metrics, is_directed=True)
        brute_force_ergm._thetas = model._thetas
        true_likelihood_net_for_like = brute_force_ergm.calculate_probability(network_for_likelihood)
        calculated_likelihood = model.calc_model_log_likelihood(network_for_likelihood, reduction='sum', log_base=10)
        # TODO: the diff is larger than expected. Not probable that it's a problem, but maybe we should dig into this.
        self.assertTrue(np.abs(np.log10(true_likelihood_net_for_like) - calculated_likelihood) < 0.1)

    def test_mple_multiple_observed_networks(self):
        set_seed(9876)
        metrics = [NumberOfEdgesDirected(), OutDegree()]
        n_nodes = 4
        base_brute_force_model = BruteForceERGM(n_nodes=n_nodes, metrics_collection=metrics, is_directed=True)
        sample_size = 100
        sample = base_brute_force_model.generate_networks_for_sample(sample_size=sample_size)
        reference_brute_force_model = BruteForceERGM(n_nodes=n_nodes, metrics_collection=metrics, is_directed=True)
        reference_brute_force_model.fit(sample)

        tested_model = ERGM(n_nodes=n_nodes, metrics_collection=metrics, is_directed=True)
        tested_model.fit(sample)

        self.assertTrue(np.all(np.abs(tested_model._thetas - reference_brute_force_model._thetas) < 1e-5))

    def test_mple_reciprocity_multiple_observed_networks(self):
        set_seed(347865)
        metrics = [NumberOfEdgesDirected(), TotalReciprocity()]
        n_nodes = 4
        sample_size = 100
        base_brute_force_model = BruteForceERGM(n_nodes=n_nodes, metrics_collection=metrics, is_directed=True)
        sample = base_brute_force_model.generate_networks_for_sample(sample_size=sample_size)
        reference_brute_force_model = BruteForceERGM(n_nodes=n_nodes, metrics_collection=metrics, is_directed=True)
        reference_brute_force_model.fit(sample)
        tested_model = ERGM(n_nodes=n_nodes, metrics_collection=metrics, is_directed=True)
        tested_model.fit(sample)

        self.assertTrue(np.all(np.abs(tested_model._thetas - reference_brute_force_model._thetas) < 1e-5))

    def test_get_mple_reciprocity_prediction(self):
        set_seed(348976)
        metrics = [NumberOfEdgesDirected(), TotalReciprocity()]
        n_nodes = 15
        net = generate_binomial_tensor(net_size=n_nodes, num_samples=1)
        train_model = ERGM(n_nodes=n_nodes, metrics_collection=metrics, is_directed=True)
        train_model.fit(net)
        test_model = ERGM(n_nodes=n_nodes, metrics_collection=metrics, is_directed=True,
                          initial_thetas=train_model.get_model_parameters())
        self.assertTrue(np.all(
            np.abs(test_model.get_mple_reciprocity_prediction() - train_model._exact_dyadic_distributions) < 1e-10))

    def test_mcmle_multiple_observed_networks(self):
        set_seed(7653467)
        metrics = [NumberOfEdgesDirected(), InDegree(), TotalReciprocity()]
        n_nodes = 10
        sample_size = 1000
        base_model = ERGM(n_nodes=n_nodes, metrics_collection=metrics, is_directed=True)
        sample = base_model.generate_networks_for_sample(sample_size=sample_size, burn_in=10000)
        tested_model = ERGM(n_nodes=n_nodes, metrics_collection=metrics, is_directed=True)
        tested_model.fit(
            sample,
            optimization_scheme='MCMLE',
            theta_init_method='uniform',
            lr=0.1,
            mcmc_sample_size=n_nodes ** 3,
            mcmc_steps_per_sample=n_nodes ** 2,
            bootstrap_convergence_confidence=0.99,
            bootstrap_convergence_stds_away_thr=0.75,
        )

        thetas_ccc = ccc(base_model._thetas, tested_model._thetas)
        self.assertTrue(thetas_ccc > 0.96)

    def test_directed_undirected_sample_from_probas(self):
        set_seed(123)
        probability_matrix = np.array([
            [0, 0.5, 0.5, 0.5],
            [0.5, 0, 0.5, 0.5],
            [0.5, 0.5, 0, 0.5],
            [0.5, 0.5, 0.5, 0]
        ])
        n_nodes = probability_matrix.shape[0]

        sample_size = 5
        is_directed = True
        sample = sample_from_independent_probabilities_matrix(probability_matrix, sample_size, is_directed)
        self.assertEqual(sample.shape, (n_nodes, n_nodes, sample_size))
        self.assertTrue(np.all(np.diagonal(sample, axis1=0, axis2=1) == 0))

        sample_size = 1
        is_directed = False
        sample = sample_from_independent_probabilities_matrix(probability_matrix, sample_size, is_directed)

        W = sample[:, :, 0]
        self.assertTrue(np.all(W == W.T))
        self.assertTrue(np.all(W[np.diag_indices(n_nodes)] == 0))

    def test_e_2_e_training_with_mask_directed(self):
        set_seed(348976)
        n = 20
        data = generate_binomial_tensor(net_size=n, num_samples=3, p=0.1)
        types = np.random.choice([1, 2, 3], size=n)
        positions = np.random.rand(n * 3).reshape(n, 3)
        metrics_to_mask = [
            NumberOfEdgesDirected(),
            NumberOfEdgesTypesDirected(types),
            SumDistancesConnectedNeurons(positions, is_directed=True),
        ]

        num_nodes_to_mask = 5
        mask = np.zeros((n, n))
        mask[:-num_nodes_to_mask, :-num_nodes_to_mask] = 1
        mask[np.diag_indices(n)] = 0
        mask = mask.astype(bool)

        masked_model = ERGM(n_nodes=n, metrics_collection=metrics_to_mask, is_directed=True, mask=mask)
        result = masked_model.fit(data)
        self.assertTrue(result["success"])

        metrics_no_mask = [
            NumberOfEdgesDirected(),
            NumberOfEdgesTypesDirected(types[:-num_nodes_to_mask]),
            SumDistancesConnectedNeurons(positions[:-num_nodes_to_mask], is_directed=True),
        ]
        normal_model = ERGM(n_nodes=n - num_nodes_to_mask, metrics_collection=metrics_no_mask, is_directed=True)
        result = normal_model.fit(data[:-num_nodes_to_mask, :-num_nodes_to_mask, ...])
        self.assertTrue(result["success"])

        thetas_ccc = ccc(normal_model._thetas, masked_model._thetas)
        self.assertTrue(thetas_ccc > 0.99)

    def test_e_2_e_training_with_mask_undirected(self):
        set_seed(389476)
        n = 20
        data = generate_binomial_tensor(net_size=n, num_samples=3, p=0.1)
        # symmetrize
        data = np.round((data + data.transpose(1, 0, 2)) / 2)
        types = np.random.choice([1, 2, 3], size=n)
        positions = np.random.rand(n * 3).reshape(n, 3)
        metrics_to_mask = [
            NumberOfEdgesUndirected(),
            NumberOfEdgesTypesUndirected(types),
            SumDistancesConnectedNeurons(positions, is_directed=False),
        ]

        num_nodes_to_mask = 5
        mask = np.zeros((n, n))
        mask[:-num_nodes_to_mask, :-num_nodes_to_mask] = 1
        mask[np.diag_indices(n)] = 0
        mask = mask.astype(bool)

        masked_model = ERGM(n_nodes=n, metrics_collection=metrics_to_mask, is_directed=False, mask=mask)
        result = masked_model.fit(data)
        self.assertTrue(result["success"])

        metrics_no_mask = [
            NumberOfEdgesUndirected(),
            NumberOfEdgesTypesUndirected(types[:-num_nodes_to_mask]),
            SumDistancesConnectedNeurons(positions[:-num_nodes_to_mask], is_directed=False),
        ]
        normal_model = ERGM(n_nodes=n - num_nodes_to_mask, metrics_collection=metrics_no_mask, is_directed=False)
        result = normal_model.fit(data[:-num_nodes_to_mask, :-num_nodes_to_mask, ...])
        self.assertTrue(result["success"])

        thetas_ccc = ccc(normal_model._thetas, masked_model._thetas)
        self.assertTrue(thetas_ccc > 0.99)

    def test_edge_weights_mask_equivalence_directed(self):
        """
        A masked model with metric NumberOfEdgesTypesDirected should produce the same thetas as an unmasked model
        with weights=0 for masked edges and weights=1 for unmasked edges.
        This works because NumberOfEdgesTypesDirected change scores are purely determined by node types (not network
        state), so the Xs rows are identical whether computed over the masked or full network.
        """
        set_seed(348976)
        n = 20
        data = generate_binomial_tensor(net_size=n, num_samples=3, p=0.3)
        types = np.array([1, 2] * 10)

        num_nodes_to_mask = 5
        mask = np.zeros((n, n))
        mask[:-num_nodes_to_mask, :-num_nodes_to_mask] = 1
        mask[np.diag_indices(n)] = 0
        mask = mask.astype(bool)

        # Masked model
        metrics_masked = [NumberOfEdgesTypesDirected(types)]
        masked_model = ERGM(n_nodes=n, metrics_collection=metrics_masked, is_directed=True, mask=mask)
        result_masked = masked_model.fit(data)
        self.assertTrue(result_masked["success"])

        # Weighted model (no mask, but weights=0 where mask=False, weights=1 where mask=True)
        weights = mask.astype(float)
        metrics_weighted = [NumberOfEdgesTypesDirected(types)]
        weighted_model = ERGM(n_nodes=n, metrics_collection=metrics_weighted, is_directed=True)
        result_weighted = weighted_model.fit(data, edge_weights=weights)
        self.assertTrue(result_weighted["success"])

        np.testing.assert_allclose(masked_model._thetas, weighted_model._thetas, rtol=1e-3)

    def test_edge_weights_mask_equivalence_undirected(self):
        """
        Same as directed test but for undirected networks.
        """
        set_seed(389476)
        n = 20
        data = generate_binomial_tensor(net_size=n, num_samples=3, p=0.3)
        data = np.round((data + data.transpose(1, 0, 2)) / 2)
        types = np.array([1, 2] * 10)

        num_nodes_to_mask = 5
        mask = np.zeros((n, n))
        mask[:-num_nodes_to_mask, :-num_nodes_to_mask] = 1
        mask[np.diag_indices(n)] = 0
        mask = mask.astype(bool)

        # Masked model
        metrics_masked = [NumberOfEdgesTypesUndirected(types)]
        masked_model = ERGM(n_nodes=n, metrics_collection=metrics_masked, is_directed=False, mask=mask)
        result_masked = masked_model.fit(data)
        self.assertTrue(result_masked["success"])

        # Weighted model
        weights = mask.astype(float)
        metrics_weighted = [NumberOfEdgesTypesUndirected(types)]
        weighted_model = ERGM(n_nodes=n, metrics_collection=metrics_weighted, is_directed=False)
        result_weighted = weighted_model.fit(data, edge_weights=weights)
        self.assertTrue(result_weighted["success"])

        np.testing.assert_allclose(masked_model._thetas, weighted_model._thetas, rtol=1e-3)

    def test_edge_weights_constant_weight_equivalence(self):
        """Model with all weights = c should produce the same thetas as unweighted model."""
        set_seed(67890)
        n = 15
        data = generate_binomial_tensor(net_size=n, num_samples=3, p=0.2)
        types = np.random.choice([1, 2], size=n)

        # Unweighted model
        metrics_unweighted = [NumberOfEdgesDirected(), NumberOfEdgesTypesDirected(types)]
        unweighted_model = ERGM(n_nodes=n, metrics_collection=metrics_unweighted, is_directed=True)
        result_unw = unweighted_model.fit(data)
        self.assertTrue(result_unw["success"])

        # Weighted model with constant weight = 5
        weights = 5.0 * np.ones((n, n))
        np.fill_diagonal(weights, 0)
        metrics_weighted = [NumberOfEdgesDirected(), NumberOfEdgesTypesDirected(types)]
        weighted_model = ERGM(n_nodes=n, metrics_collection=metrics_weighted, is_directed=True)
        result_w = weighted_model.fit(data, edge_weights=weights)
        self.assertTrue(result_w["success"])

        np.testing.assert_allclose(unweighted_model._thetas, weighted_model._thetas, rtol=1e-3)

    def test_edge_weights_ones_equals_no_weights(self):
        """Model with all weights = 1 should produce exactly the same thetas as unweighted model."""
        set_seed(112233)
        n = 15
        data = generate_binomial_tensor(net_size=n, num_samples=3, p=0.2)
        types = np.random.choice([1, 2], size=n)

        # Unweighted model
        metrics_unweighted = [NumberOfEdgesTypesDirected(types)]
        unweighted_model = ERGM(n_nodes=n, metrics_collection=metrics_unweighted, is_directed=True)
        result_unw = unweighted_model.fit(data)
        self.assertTrue(result_unw["success"])

        # Weighted model with all weights = 1
        weights = np.ones((n, n))
        np.fill_diagonal(weights, 0)
        metrics_weighted = [NumberOfEdgesTypesDirected(types)]
        weighted_model = ERGM(n_nodes=n, metrics_collection=metrics_weighted, is_directed=True)
        result_w = weighted_model.fit(data, edge_weights=weights)
        self.assertTrue(result_w["success"])

        np.testing.assert_allclose(unweighted_model._thetas, weighted_model._thetas, rtol=3e-6)

    def test_edge_weights_upweighting_shifts_theta(self):
        """
        Upweighting edges of a specific type pair should shift the fitted theta for that type pair
        towards the density of edges in that type pair relative to unweighted.

        With NumberOfEdgesTypesDirected, the MPLE solution for each type pair theta is:
            theta = log(p / (1-p)) where p is the (weighted) density.

        When we upweight existing A->B edges (y=1) more than absent ones (y=0),
        the effective density increases, so the A->B theta should increase.
        """
        set_seed(445566)
        n = 20
        types = np.array(["A"] * 10 + ["B"] * 10)
        type_A_indices = np.where(types == "A")[0]
        type_B_indices = np.where(types == "B")[0]

        data = generate_binomial_tensor(net_size=n, num_samples=1, p=0.3)[..., 0]
        np.fill_diagonal(data, 0)

        # Unweighted model
        metrics1 = [NumberOfEdgesTypesDirected(types)]
        model1 = ERGM(n_nodes=n, metrics_collection=metrics1, is_directed=True)
        result1 = model1.fit(data)
        self.assertTrue(result1["success"])

        # Weighted model: upweight present A->B edges by 3x
        weights = np.ones((n, n))
        for i in type_A_indices:
            for j in type_B_indices:
                if data[i, j] == 1:
                    weights[i, j] = 3.0
        np.fill_diagonal(weights, 0)

        metrics2 = [NumberOfEdgesTypesDirected(types)]
        model2 = ERGM(n_nodes=n, metrics_collection=metrics2, is_directed=True)
        result2 = model2.fit(data, edge_weights=weights)
        self.assertTrue(result2["success"])

        # Find the index corresponding to A->B type pair
        ab_idx = metrics2[0]._sorted_type_pairs_indices[("A", "B")]

        # The A->B theta should increase when we upweight present A->B edges
        self.assertGreater(model2._thetas[ab_idx], model1._thetas[ab_idx])

    def test_edge_weights_validation(self):
        """Test that invalid edge weights raise appropriate errors."""
        n = 10
        data = np.random.randint(0, 2, (n, n))
        np.fill_diagonal(data, 0)

        metrics = [NumberOfEdgesDirected()]
        model = ERGM(n_nodes=n, metrics_collection=metrics, is_directed=True)

        # Wrong shape
        with self.assertRaises(ValueError):
            model.fit(data, edge_weights=np.ones((5, 5)))

        # Negative weights
        with self.assertRaises(ValueError):
            model.fit(data, edge_weights=-np.ones((n, n)))