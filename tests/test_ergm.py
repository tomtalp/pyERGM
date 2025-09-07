import unittest

import numpy as np

from pyERGM.utils import *
from pyERGM.metrics import *
from pyERGM.ergm import ERGM, BruteForceERGM
from pyERGM.datasets import sampson_matrix
import sys
from scipy.linalg import eigh


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

    def test_fit_small_ER_network_with_e_i_node_feature(self):
        n = 4
        p = 0.25
        is_directed = False

        num_pos_connect = n * (n - 1)
        if not is_directed:
            num_pos_connect //= 2

        ground_truth_num_edges = round(num_pos_connect * p)
        ground_truth_p = ground_truth_num_edges / num_pos_connect

        ground_truth_theta = np.log(ground_truth_p / (1 - ground_truth_p))

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

        node_features = {"E_I": [[0, 1, 1, 1]]}
        ground_truth_e_nodes = 0.25
        ground_truth_theta_e = np.log(ground_truth_e_nodes / (1 - ground_truth_e_nodes))

        W = adj_mat  # np.concatenate([adj_mat, node_types], axis=1)
        print(f"Fitting matrix ({W.shape}) - ")
        print(W)

        ergm = ERGM(n,
                    [NumberOfEdgesUndirected(), NumberOfNodesPerType(metric_node_feature='E_I', n_node_categories=2)],
                    is_directed=is_directed,
                    seed_MCMC_proba=0.25
                    )

        ergm.fit(W, observed_node_features=node_features, lr=0.1, opt_steps=500, sliding_grad_window_k=10,
                 sample_pct_growth=0.05,
                 steps_for_decay=20, lr_decay_pct=0.05, mcmc_sample_size=300, mcmc_steps_per_sample=10,
                 theta_init_method='uniform')

        fit_theta = ergm._thetas[0]
        fit_theta_e = ergm._thetas[1]
        print(f"ground truth theta: {ground_truth_theta, ground_truth_theta_e}")
        print(f"fit theta: {fit_theta, fit_theta_e}")

        # ergm._thetas = np.array([ground_truth_theta, ground_truth_theta_e])
        sampled_networks = ergm.generate_networks_for_sample(sample_size=100)
        fit_p = sampled_networks[:, :-1, :].mean()
        fit_e_i_fraction = sampled_networks[:, -1, :].mean()
        print(f"ground truth p: {p}")
        print(f"fit p: {fit_p}")
        print(f"ground truth excitatory fraction: {np.array(node_features['E_I']).mean()}")
        print(f"fit excitatory fraction: {fit_e_i_fraction}")

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

        thetas_R_2 = 1 - np.sum((model_thethas - expected_thetas) ** 2) / np.sum(
            (expected_thetas - np.mean(expected_thetas)) ** 2)
        self.assertTrue(thetas_R_2 > 0.99)
        self.assertTrue(convergence_result["success"])

    # def test_sampson_MCMLE_with_node_features(self):
    #     set_seed(1234)
    #     metrics = [NumberOfEdgesDirected(), OutDegree(), InDegree(), TotalReciprocity()]
    #     n_nodes = sampson_matrix.shape[0]
    #
    #     mcmle_model = ERGM(n_nodes, metrics, is_directed=True)
    #
    #     node_features = np.random.choice(3, size=(n_nodes, 2))
    #     # sampson_matrix_with_node_features = np.concatenate([sampson_matrix, node_features], axis=1)
    #     convergence_result = mcmle_model.fit(sampson_matrix,
    #                                          observed_node_features=node_features,
    #                                          opt_steps=10,
    #                                          steps_for_decay=1,
    #                                          lr=1,
    #                                          mple_lr=0.5,
    #                                          convergence_criterion="model_bootstrap",
    #                                          mcmc_burn_in=0,
    #                                          mcmc_steps_per_sample=n_nodes ** 2,
    #                                          mcmc_sample_size=1000,
    #                                          num_model_sub_samples=10,
    #                                          model_subsample_size=1000,
    #                                          bootstrap_convergence_confidence=0.95,
    #                                          bootstrap_convergence_num_stds_away_thr=1,
    #                                          optimization_scheme='MCMLE'
    #                                          )
    #
    #     model_thethas = mcmle_model._thetas
    #
    #     expected_values = {"edges": -1.1761, "sender2": -0.2945, "sender3": 1.4141, "sender4": 0.3662,
    #                        "sender5": 0.1315,
    #                        "sender6": 1.2148, "sender7": 0.6055,
    #                        "sender8": 1.3609, "sender9": 0.6402, "sender10": 2.0639, "sender11": 1.4355,
    #                        "sender12": -0.1681,
    #                        "sender13": -0.2322, "sender14": 0.5841, "sender15": 1.8600,
    #                        "sender16": 1.4317, "sender17": 1.2211, "sender18": 1.8724, "receiver2": -0.1522,
    #                        "receiver3": -3.0453,
    #                        "receiver4": -1.7596, "receiver5": -0.8198, "receiver6": -3.3922,
    #                        "receiver7": -1.6074, "receiver8": -2.2656, "receiver9": -2.2069, "receiver10": -3.9189,
    #                        "receiver11": -3.0257, "receiver12": -0.9457, "receiver13": -1.4749, "receiver14": -1.5950,
    #                        "receiver15": -3.3147, "receiver16": -3.0567, "receiver17": -3.4436, "receiver18": -3.3239,
    #                        "mutual": 3.6918
    #                        }
    #     expected_thetas = np.array(list(expected_values.values()))
    #
    #     thetas_R_2 = 1 - np.sum((model_thethas - expected_thetas) ** 2) / np.sum(
    #         (expected_thetas - np.mean(expected_thetas)) ** 2)
    #     self.assertTrue(thetas_R_2 > 0.99)
    #     self.assertTrue(convergence_result["success"])

    def test_MPLE_regressors_of_different_scales(self):
        # TODO: currently this is a smoke test - we validate nothing: neither convergence nor the thetas/predictions.
        #  Somehow sklearn still finds a slightly better solution than ours.
        set_seed(42)

        W = np.random.randint(0, 2, size=(10, 10))
        W[np.diag_indices(10)] = 0

        metrics = [NumberOfEdgesDirected(),
                   NodeAttrSum([np.random.randint(1, 5) ** 10 for x in range(10)], is_directed=True)]
        model = ERGM(n_nodes=10, metrics_collection=metrics, is_directed=True)

        model.fit(W)

        # sklearn_thetas = np.array([2.82974701e-01, -2.34383474e-07])

        # sklearn_probas = np.array([0.56682151,
        # 0.56682151,
        # 0.50584116,
        # 0.56682151,
        # 0.56682151,
        # 0.56682151,
        # 0.50584116,
        # 0.56682151,
        # 0.56347922,
        # 0.56682151,
        # 0.57015772,
        # 0.5092404 ,
        # 0.57015772,
        # 0.57015772,
        # 0.57015772,
        # 0.5092404 ,
        # 0.57015772,
        # 0.56682151,
        # 0.56682151,
        # 0.57015772,
        # 0.5092404 ,
        # 0.57015772,
        # 0.57015772,
        # 0.57015772,
        # 0.5092404 ,
        # 0.57015772,
        # 0.56682151,
        # 0.50584116,
        # 0.5092404 ,
        # 0.5092404 ,
        # 0.5092404 ,
        # 0.5092404 ,
        # 0.5092404 ,
        # 0.44804742,
        # 0.5092404 ,
        # 0.50584116,
        # 0.56682151,
        # 0.57015772,
        # 0.57015772,
        # 0.5092404 ,
        # 0.57015772,
        # 0.57015772,
        # 0.5092404 ,
        # 0.57015772,
        # 0.56682151,
        # 0.56682151,
        # 0.57015772,
        # 0.57015772,
        # 0.5092404 ,
        # 0.57015772,
        # 0.57015772,
        # 0.5092404 ,
        # 0.57015772,
        # 0.56682151,
        # 0.56682151,
        # 0.57015772,
        # 0.57015772,
        # 0.5092404 ,
        # 0.57015772,
        # 0.57015772,
        # 0.5092404 ,
        # 0.57015772,
        # 0.56682151,
        # 0.50584116,
        # 0.5092404 ,
        # 0.5092404 ,
        # 0.44804742,
        # 0.5092404 ,
        # 0.5092404 ,
        # 0.5092404 ,
        # 0.5092404 ,
        # 0.50584116,
        # 0.56682151,
        # 0.57015772,
        # 0.57015772,
        # 0.5092404 ,
        # 0.57015772,
        # 0.57015772,
        # 0.57015772,
        # 0.5092404 ,
        # 0.56682151,
        # 0.56347922,
        # 0.56682151,
        # 0.56682151,
        # 0.50584116,
        # 0.56682151,
        # 0.56682151,
        # 0.56682151,
        # 0.50584116,
        # 0.56682151])

        # self.assertAlmostEqual()

    def test_assigning_model_initial_thetas(self):
        # TODO: seems like convergence of the model in this test depends on the seed...
        set_seed(8765)
        n_nodes = 5
        W = np.array([[0, 0, 1, 1, 0],
                      [1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1],
                      [0, 1, 0, 0, 0],
                      [0, 1, 0, 1, 0]])
        metrics_1 = [NumberOfEdgesDirected(), NodeAttrSum(np.arange(1, n_nodes + 1), is_directed=True),
                     NumberOfEdgesTypesDirected(['A', 'B', 'A', 'A', 'B'])]
        model_1 = ERGM(n_nodes=n_nodes, metrics_collection=metrics_1, is_directed=True)

        model_1.fit(W)

        model_1_params = model_1.get_model_parameters()

        metrics_2 = [NumberOfEdgesDirected(), NodeAttrSum(np.arange(n_nodes + 1, 1, -1), is_directed=True),
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

        thetas_R_2 = 1 - np.sum((model_thethas - expected_thetas) ** 2) / np.sum(
            (expected_thetas - np.mean(expected_thetas)) ** 2)
        self.assertTrue(thetas_R_2 > 0.99)
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
        model.fit(observed_net)

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

        model.fit(observed_net)

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
        net = generate_binomial_tensor(net_size=n_nodes, node_features_size=0, num_samples=1)
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

        thetas_R_2 = 1 - np.sum((base_model._thetas - tested_model._thetas) ** 2) / np.sum(
            (base_model._thetas - np.mean(base_model._thetas)) ** 2)
        self.assertTrue(thetas_R_2 > 0.96)
    
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