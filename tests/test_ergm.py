import unittest
from pyERGM.utils import *
from pyERGM.metrics import *
from pyERGM.ergm import ERGM, BruteForceERGM
import sys
from scipy.linalg import eigh


class TestERGM(unittest.TestCase):
    def setUp(self):
        self.metrics = [NumberOfEdgesUndirected(), NumberOfTriangles()]
        self.n_nodes = 3

        self.K = 100
        self.thetas = np.ones(MetricsCollection(self.metrics, is_directed=False, n_nodes=self.n_nodes).num_of_features)

    def test_calculate_weight(self):
        ergm = ERGM(self.n_nodes, self.metrics, is_directed=False, initial_thetas=self.thetas, initial_normalization_factor=self.K)

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
        ergm = ERGM(self.n_nodes, self.metrics, is_directed=False, initial_thetas=self.thetas, initial_normalization_factor=self.K)

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

        ergm = ERGM(self.n_nodes, self.metrics, is_directed=False, initial_thetas=thetas, initial_normalization_factor=K)

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
                                               initial_thetas=np.array(ground_truth_theta), is_directed=is_directed)

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
        model.fit(M1, mple_lr=1)

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
