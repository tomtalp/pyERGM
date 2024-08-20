import unittest
from utils import *
from metrics import *
from ergm import ERGM, BruteForceERGM
import sys


class TestERGM(unittest.TestCase):
    def setUp(self):
        self.metrics = [NumberOfEdgesUndirected(), NumberOfTriangles()]
        self.n_nodes = 3

        self.K = 100
        self.thetas = np.ones(MetricsCollection(self.metrics, is_directed=False, n_nodes=self.n_nodes).num_of_features)

    def test_calculate_weight(self):
        ergm = ERGM(self.n_nodes, self.metrics, initial_thetas=self.thetas, initial_normalization_factor=self.K)

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
        ergm = ERGM(self.n_nodes, self.metrics, initial_thetas=self.thetas, initial_normalization_factor=self.K)

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

        ergm = ERGM(self.n_nodes, self.metrics, initial_thetas=thetas, initial_normalization_factor=K)

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
                    sample_size=200,
                    n_mcmc_steps=10,
                    seed_MCMC_proba=0.25
                    )

        ergm.fit(W, lr=0.01, opt_steps=300, sliding_grad_window_k=10, sample_pct_growth=0.05, steps_for_decay=20,
                 lr_decay_pct=0.05)

        fit_theta = ergm._thetas[0]
        print(f"ground truth theta: {ground_truth_theta}")
        print(f"fit theta: {fit_theta}")

        # TODO - what criteria to use for testing convergence? From manual tests, it doesn't seem to perfectly converge on the true thetas...
        # Nevertheless, even without an assert, this will catch errors - fit won't work if something breaks.

    def test_auto_correlation_function(self):
        n = 3
        sample_size = 3
        W1 = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 0]])

        W2 = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [0, 1, 0]])

        W3 = np.array([[0, 1, 1],
                       [0, 0, 1],
                       [1, 1, 0]])

        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2
        sample[:, :, 2] = W3

        expected_gamma_hat_0 = 1 / 9 * np.array([[6, 3], [3, 2]])
        expected_gamma_hat_1 = 1 / 27 * np.array([[-9, -6], [0, -1]])
        expected_gamma_hat_2 = 1 / 27 * np.array([[0, 0], [-3, -2]])

        ergm = ERGM(n, [NumberOfEdgesDirected(), TotalReciprocity()], True)

        gammas = ergm.approximate_auto_correlation_function(sample)

        self.assertTrue(np.abs(expected_gamma_hat_0 - gammas[0]).max() < 10 ** -15)
        self.assertTrue(np.abs(expected_gamma_hat_1 - gammas[1]).max() < 10 ** -15)
        self.assertTrue(np.abs(expected_gamma_hat_2 - gammas[2]).max() < 10 ** -15)

    def test_batches_covariance_estimation(self):
        n = 3
        sample_size = 6
        W1 = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 0]])

        W2 = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 1, 0]])

        W3 = np.array([[0, 0, 1],
                       [0, 0, 1],
                       [1, 1, 0]])

        W4 = np.array([[0, 0, 1],
                       [1, 0, 1],
                       [1, 1, 0]])
        
        W5 = np.array([[0, 0, 1],
                       [1, 0, 0],
                       [1, 0, 0]])
        
        W6 = np.array([[0, 0, 1],
                       [1, 0, 1],
                       [1, 0, 0]])

        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2
        sample[:, :, 2] = W3
        sample[:, :, 3] = W4
        sample[:, :, 4] = W5
        sample[:, :, 5] = W6

        # expected_covariance_batch_estimation = 2 * 0.25 * np.ones((2, 2))
        expected_covariance_batch_estimation = np.ones((2,2)) * 4/9

        ergm = ERGM(n, [NumberOfEdgesDirected(), TotalReciprocity()], True)
        features_of_sample = ergm._network_statistics.calculate_sample_statistics(sample)

        batch_estimation = ergm.covariance_matrix_estimation(features_of_sample, method='batch', num_batches=3)
        print(batch_estimation)
        print(expected_covariance_batch_estimation)
        self.assertTrue(np.abs(expected_covariance_batch_estimation - batch_estimation).max() < 10**-15)
