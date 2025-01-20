import unittest
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
        np.random.seed(9873645)
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

        W = adj_mat # np.concatenate([adj_mat, node_types], axis=1)
        print(f"Fitting matrix ({W.shape}) - ")
        print(W)

        ergm = ERGM(n,
                    [NumberOfEdgesUndirected(), NumberOfNodesPerType(metric_node_feature='E_I', n_node_categories=2)],
                    is_directed=is_directed,
                    seed_MCMC_proba=0.25
                    )

        ergm.fit(W, observed_node_features=node_features, lr=0.1, opt_steps=500, sliding_grad_window_k=10, sample_pct_growth=0.05,
                 steps_for_decay=20, lr_decay_pct=0.05, mcmc_sample_size=300, mcmc_steps_per_sample=10, no_mple=True,
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
        np.random.seed(1234)
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
                mcmc_steps_per_sample=n_nodes**2,
                mcmc_sample_size=1000,
                num_model_sub_samples=10,
                model_subsample_size=1000,
                bootstrap_convergence_confidence=0.95,
                bootstrap_convergence_num_stds_away_thr=1,
            )
    
        model_thethas = mcmle_model._thetas

        expected_values = {"edges": -1.1761, "sender2": -0.2945, "sender3": 1.4141, "sender4": 0.3662, "sender5": 0.1315,
               "sender6": 1.2148, "sender7": 0.6055,
               "sender8": 1.3609, "sender9": 0.6402, "sender10": 2.0639, "sender11": 1.4355, "sender12": -0.1681,
               "sender13": -0.2322, "sender14": 0.5841, "sender15": 1.8600,
               "sender16": 1.4317, "sender17": 1.2211, "sender18": 1.8724, "receiver2": -0.1522, "receiver3": -3.0453,
               "receiver4": -1.7596, "receiver5": -0.8198, "receiver6": -3.3922,
               "receiver7": -1.6074, "receiver8": -2.2656, "receiver9": -2.2069, "receiver10": -3.9189,
               "receiver11": -3.0257, "receiver12": -0.9457, "receiver13": -1.4749, "receiver14": -1.5950,
               "receiver15": -3.3147, "receiver16": -3.0567, "receiver17": -3.4436, "receiver18": -3.3239,
               "mutual": 3.6918
               }
        expected_thetas = np.array(list(expected_values.values()))
        
        thetas_R_2 = 1 - np.sum((model_thethas - expected_thetas)**2) / np.sum((expected_thetas - np.mean(expected_thetas))**2)
        self.assertTrue(thetas_R_2 > 0.99)
        self.assertTrue(convergence_result["success"])

    def test_sampson_MCMLE_with_node_features(self):
        np.random.seed(1234)
        metrics = [NumberOfEdgesDirected(), OutDegree(), InDegree(), TotalReciprocity()]
        n_nodes = sampson_matrix.shape[0]

        mcmle_model = ERGM(n_nodes, metrics, is_directed=True)

        node_features = np.random.choice(3, size=(n_nodes, 2))
        sampson_matrix_with_node_features = np.concatenate([sampson_matrix, node_features], axis=1)
        convergence_result = mcmle_model.fit(sampson_matrix_with_node_features,
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

    def test_MPLE_regressors_of_different_scales(self):
        # TODO: currently this is a smoke test - we validate nothing: neither convergence nor the thetas/predictions.
        #  Somehow sklearn still finds a slightly better solution than ours.
        np.random.seed(42)

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
        np.random.seed(8765)
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
