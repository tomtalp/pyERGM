import unittest
from utils import *
from metrics import *
from ergm import ERGM, BruteForceERGM
import sys


class TestERGM(unittest.TestCase):
    def setUp(self):
        metrics = [NumberOfEdges(), NumberOfTriangles()]
        self.net_stats = MetricsCollection(metrics, is_directed=False)
        self.n_nodes = 3

        self.K = 100
        self.thetas = np.ones(self.net_stats.get_num_of_statistics())

    def test_calculate_weight(self):
        ergm = ERGM(self.n_nodes, self.net_stats, initial_thetas=self.thetas, initial_normalization_factor=self.K)

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
        ergm = ERGM(self.n_nodes, self.net_stats, initial_thetas=self.thetas, initial_normalization_factor=self.K)

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

        ergm = ERGM(self.n_nodes, self.net_stats, initial_thetas=thetas, initial_normalization_factor=K)

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

    def test_benchmark_er_convergence(self, n=4, p=0.25, is_directed=False):
        np.random.seed(9873645)
        num_pos_connect = n * (n - 1)

        if not is_directed:
            num_pos_connect //= 2

        ground_truth_num_edges = round(num_pos_connect * p)
        ground_truth_p = ground_truth_num_edges / num_pos_connect
        # TODO @oren - is this true also for directed graphs? See ergm 2008 paper Sec. 4.1
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

        model = BruteForceERGM(n, MetricsCollection([NumberOfEdges()], is_directed=is_directed), is_directed=is_directed)
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

        model_with_true_theta = BruteForceERGM(n, MetricsCollection([NumberOfEdges()], is_directed=is_directed),
                                               initial_thetas=np.array(ground_truth_theta), is_directed=is_directed)
        
        ground_truth_model_log_like = np.log(model_with_true_theta.calculate_weight(adj_mat)) - np.log(
            model_with_true_theta._normalization_factor)
        
        print(f"model with true theta log like: {ground_truth_model_log_like}")
        print(f"normalization factor: {model_with_true_theta._normalization_factor}")