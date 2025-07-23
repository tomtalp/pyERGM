import unittest

from scipy.stats import pearsonr

from pyERGM.utils import *
from pyERGM.metrics import MetricsCollection, NumberOfEdgesDirected, TotalReciprocity, OutDegree, InDegree
from pyERGM.datasets import sampson_matrix
from pyERGM.ergm import ERGM
from scipy.linalg import eigh

from matplotlib import pyplot as plt
import sys
import math


class GeneralUtilsTester(unittest.TestCase):
    def test_get_sorted_type_pairs(self):
        types = ["A", "B", "B", "B"]
        sorted_type_pairs = get_sorted_type_pairs(types)
        expected_sorted_type_pairs = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
        self.assertTrue(sorted_type_pairs == expected_sorted_type_pairs)

    def test_convert_non_flat_non_diag_to_i_j(self):
        n = 4
        i_j = convert_flat_no_diag_idx_to_i_j([7], n)
        expected_i_j = np.array([[2],
                                 [1]])
        self.assertTrue(np.all(expected_i_j == i_j))

        flat_no_diags = np.array([0, 5, 11])
        expected_i_js = np.array([[0, 1, 3],
                                  [1, 3, 2]])
        i_js = convert_flat_no_diag_idx_to_i_j(flat_no_diags, n)
        self.assertTrue(np.all(expected_i_js == i_js))

        with self.assertRaises(IndexError):
            convert_flat_no_diag_idx_to_i_j([13], n)

    def test_num_dyads_to_num_nodes(self):
        W = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [1, 1, 0]])

        num_dyads = math.comb(W.shape[0], 2)

        real_num_nodes = W.shape[0]
        num_nodes = num_dyads_to_num_nodes(num_dyads)

        self.assertEqual(real_num_nodes, num_nodes)

    def test_exact_marginals_from_dyads_distributions(self):
        set_seed(9873645)
        metrics = [NumberOfEdgesDirected(), OutDegree(), InDegree(), TotalReciprocity()]
        n_nodes = sampson_matrix.shape[0]

        p1_sampson_model = ERGM(n_nodes, metrics, is_directed=True)

        convergence_result = p1_sampson_model.fit(sampson_matrix)

        sample_size = 50000
        sampled_networks = p1_sampson_model.generate_networks_for_sample(sampling_method="exact",
                                                                         sample_size=sample_size)
        sample_mean = sampled_networks.mean(axis=-1)
        exact_marginals = get_exact_marginals_from_dyads_distrubution(p1_sampson_model._exact_dyadic_distributions)

        self.assertEqual(sampled_networks.shape, (n_nodes, n_nodes, sample_size))
        self.assertEqual(convergence_result["success"], True)
        self.assertTrue(np.abs(exact_marginals - sample_mean).max() < 1e-2)
        self.assertTrue(pearsonr(exact_marginals[~np.eye(n_nodes, dtype=bool)].flatten(),
                                 sampled_networks.mean(axis=-1)[
                                     ~np.eye(n_nodes, dtype=bool)].flatten()).statistic > 0.99)


class TestGreatestConvexMinorant(unittest.TestCase):
    DO_PLOT = False

    def test_not_change_values_of_convex_functions(self):
        xs = np.arange(10) + 1
        convex_func_vals = 1 / xs ** 2

        minorant_vals = get_greatest_convex_minorant(xs, convex_func_vals)

        if TestGreatestConvexMinorant.DO_PLOT:
            plt.plot(xs, convex_func_vals, '.k', label='input values')
            plt.plot(xs, minorant_vals, label='greatest convex minorant')
            plt.legend()
            plt.show()

        self.assertTrue(np.all(minorant_vals == convex_func_vals))

    def test_engineered_scenario(self):
        def get_expected_value(convex_func_vals, wrapping_indices, perturbed_idx):
            slope = (convex_func_vals[wrapping_indices[1]] - convex_func_vals[wrapping_indices[0]]) / (
                    xs[wrapping_indices[1]] - xs[wrapping_indices[0]])
            return slope * xs[perturbed_idx] + convex_func_vals[wrapping_indices[1]] - xs[wrapping_indices[1]] * slope

        num_points = 10
        xs = np.arange(num_points) + 1
        convex_func_vals = np.exp(-xs)
        num_scenarios = 4
        perturbation = 0.5
        indices_to_increase = np.array([[1, 3, 7], [1, 2, 3], [0, 2, 4], [5, 6, 9]]).astype(int)
        expected_values = np.array([convex_func_vals, ] * num_scenarios)
        expected_values[0, 1] = get_expected_value(convex_func_vals, (0, 2), 1)
        expected_values[0, 3] = get_expected_value(convex_func_vals, (2, 4), 3)
        expected_values[0, 7] = get_expected_value(convex_func_vals, (6, 8), 7)

        expected_values[1, 1] = get_expected_value(convex_func_vals, (0, 4), 1)
        expected_values[1, 2] = get_expected_value(convex_func_vals, (0, 4), 2)
        expected_values[1, 3] = get_expected_value(convex_func_vals, (0, 4), 3)

        expected_values[2, 0] = convex_func_vals[0] + perturbation
        expected_values[2, 2] = get_expected_value(convex_func_vals, (1, 3), 2)
        expected_values[2, 4] = get_expected_value(convex_func_vals, (3, 5), 4)

        expected_values[3, 5] = get_expected_value(convex_func_vals, (4, 7), 5)
        expected_values[3, 6] = get_expected_value(convex_func_vals, (4, 7), 6)
        expected_values[3, 9] = convex_func_vals[9] + perturbation

        for i in range(num_scenarios):
            cur_vals = convex_func_vals.copy()
            cur_vals[indices_to_increase[i]] += perturbation
            minorant_vals = get_greatest_convex_minorant(xs, cur_vals)
            if TestGreatestConvexMinorant.DO_PLOT:
                plt.plot(xs, cur_vals, '.k', label='input values')
                plt.plot(xs, minorant_vals, label='greatest convex minorant')
                plt.legend()
                plt.title(f'scenario {i}')
                plt.show()

            for j in range(num_points):
                self.assertAlmostEqual(minorant_vals[j], expected_values[i, j], places=8)

    def test_random_points(self):
        set_seed(8972634)
        num_points = 100
        xs = np.arange(num_points) + 1
        values = 10 * np.random.rand(100)
        minorant_vals = get_greatest_convex_minorant(xs, values)
        if TestGreatestConvexMinorant.DO_PLOT:
            plt.plot(xs, values, '.k', label='input values')
            plt.plot(xs, minorant_vals, label='greatest convex minorant')
            plt.legend()
            plt.show()
        self.assertTrue(np.all(minorant_vals <= values))
        self.assertTrue(np.all(np.diff(minorant_vals, n=2) >= -10 ** -10))


class TestSparseTensorUtilities(unittest.TestCase):
    def test_transpose_sample_matrices(self):
        np_tensor = np.random.choice([0, 1], size=(5, 5, 1), p=[0.8, 0.2])
        expected_np_tensor_T = np.transpose(np_tensor, axes=(1, 0, 2))

        sparse_tensor = np_tensor_to_sparse_tensor(np_tensor)

        self.assertTrue(sparse_tensor.is_sparse)

        transposed_sparse_tensor = transpose_sparse_sample_matrices(sparse_tensor)
        transposed_sparse_tensor_as_np = transposed_sparse_tensor.to_dense().numpy()

        self.assertTrue(np.all(expected_np_tensor_T == transposed_sparse_tensor_as_np))


class TestCovarianceMatrixEstimation(unittest.TestCase):

    @staticmethod
    def validate_test_sample_validity(metrics_collection, sample):
        features_of_sample = metrics_collection.calculate_sample_statistics(sample)
        gamma_hats = approximate_auto_correlation_function(features_of_sample)
        capital_gammas = calc_capital_gammas(gamma_hats)
        Sigma_0 = -gamma_hats[0] + 2 * capital_gammas[0]
        Sigma_1 = Sigma_0 + 2 * capital_gammas[1]
        Sigma_2 = Sigma_1 + 2 * capital_gammas[2]

        is_Sigma_0_positive_definite = eigh(Sigma_0)[0].min() > 0
        is_Sigma_1_positive_definite = eigh(Sigma_1)[0].min() > 0
        is_Sigma_2_positive_definite = eigh(Sigma_2)[0].min() > 0

        det_Sigma_0 = np.linalg.det(Sigma_0)
        det_Sigma_1 = np.linalg.det(Sigma_1)
        det_Sigma_2 = np.linalg.det(Sigma_2)

        if ((is_Sigma_0_positive_definite and det_Sigma_0 < det_Sigma_1) or
                (is_Sigma_1_positive_definite and det_Sigma_1 < det_Sigma_2)):
            return True
        return False

    @staticmethod
    def find_sample_for_multivariate_initial_seq_covariance_matrix_estimation_test(num_trials=1000, sample_size=6, n=3,
                                                                                   p=0.5):
        metrics_collection = MetricsCollection([NumberOfEdgesDirected(), TotalReciprocity()], True, n)
        for i in range(num_trials):
            sample = np.zeros((n, n, sample_size))
            for j in range(sample_size):
                cur_mat = nx.to_numpy_array(nx.erdos_renyi_graph(n, p, directed=True))
                sample[:, :, j] = cur_mat
                if TestCovarianceMatrixEstimation.validate_test_sample_validity(metrics_collection, sample):
                    return sample
        print("No sample was valid")
        return None

    @staticmethod
    def _construct_sample():
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

        return sample

    def test_batches_covariance_estimation(self):
        sample = self._construct_sample()
        n = sample.shape[0]

        expected_covariance_batch_estimation = np.ones((2, 2)) * 4 / 9

        network_statistics = MetricsCollection([NumberOfEdgesDirected(), TotalReciprocity()], True, n)
        features_of_sample = network_statistics.calculate_sample_statistics(sample)
        mean_features = features_of_sample.mean(axis=1)

        sys.setrecursionlimit(2000)
        batch_estimation = covariance_matrix_estimation(features_of_sample, mean_features, method='batch',
                                                        num_batches=3)
        self.assertTrue(np.abs(expected_covariance_batch_estimation - batch_estimation).max() < 10 ** -15)

    def test_naive_covariance_matrix_estimation(self):
        sample = self._construct_sample()
        n = sample.shape[0]
        expected_covariance_naive_estimation = np.array([[17 / 36, 8 / 36], [8 / 36, 8 / 36]])

        network_statistics = MetricsCollection([NumberOfEdgesDirected(), TotalReciprocity()], True, n)
        features_of_sample = network_statistics.calculate_sample_statistics(sample)
        mean_features = features_of_sample.mean(axis=1)

        sys.setrecursionlimit(2000)
        naive_estimation = covariance_matrix_estimation(features_of_sample, mean_features, method='naive')
        self.assertTrue(np.abs(expected_covariance_naive_estimation - naive_estimation).max() < 10 ** -14)

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

        network_statistics = MetricsCollection([NumberOfEdgesDirected(), TotalReciprocity()], True, n)

        features_of_sample = network_statistics.calculate_sample_statistics(sample)
        gammas = approximate_auto_correlation_function(features_of_sample)

        self.assertTrue(np.abs(expected_gamma_hat_0 - gammas[0]).max() < 10 ** -15)
        self.assertTrue(np.abs(expected_gamma_hat_1 - gammas[1]).max() < 10 ** -15)
        self.assertTrue(np.abs(expected_gamma_hat_2 - gammas[2]).max() < 10 ** -15)

    def test_capital_gammas(self):
        n = 3
        sample_size = 4

        W1 = np.array([[0, 0, 1],
                       [0, 0, 1],
                       [1, 1, 0]])

        W2 = np.array([[0, 0, 1],
                       [0, 0, 1],
                       [1, 1, 0]])

        W3 = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 1, 0]])

        W4 = np.array([[0, 0, 1],
                       [1, 0, 0],
                       [1, 0, 0]])

        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2
        sample[:, :, 2] = W3
        sample[:, :, 3] = W4

        expected_capital_gamma_0 = 5 / 16 * np.ones((2, 2))
        expected_capital_gamma_1 = -3 / 16 * np.ones((2, 2))

        network_statistics = MetricsCollection([NumberOfEdgesDirected(), TotalReciprocity()], True, n)

        features_of_sample = network_statistics.calculate_sample_statistics(sample)
        gammas = approximate_auto_correlation_function(features_of_sample)
        capital_gammas = calc_capital_gammas(gammas)

        self.assertTrue(np.abs(capital_gammas[0] - expected_capital_gamma_0).max() < 10 ** -15)
        self.assertTrue(np.abs(capital_gammas[1] - expected_capital_gamma_1).max() < 10 ** -15)

    def test_single_auto_correlation_function(self):
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
        expected_gammas = [expected_gamma_hat_0, expected_gamma_hat_1, expected_gamma_hat_2]

        network_statistics = MetricsCollection([NumberOfEdgesDirected(), TotalReciprocity()], True, n)

        features_of_sample = network_statistics.calculate_sample_statistics(sample)
        features_of_sample_mean = features_of_sample.mean(axis=1)
        diff_from_mean = features_of_sample - features_of_sample_mean[:, None]
        for k in range(3):
            cur_gamma = approximate_kth_auto_correlation_function(diff_from_mean, k)
            self.assertTrue(np.abs(expected_gammas[k] - cur_gamma).max() < 10 ** -15)

    def test_kth_capital_gamma(self):
        n = 3
        sample_size = 4

        W1 = np.array([[0, 0, 1],
                       [0, 0, 1],
                       [1, 1, 0]])

        W2 = np.array([[0, 0, 1],
                       [0, 0, 1],
                       [1, 1, 0]])

        W3 = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 1, 0]])

        W4 = np.array([[0, 0, 1],
                       [1, 0, 0],
                       [1, 0, 0]])

        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2
        sample[:, :, 2] = W3
        sample[:, :, 3] = W4

        expected_capital_gamma_0 = 5 / 16 * np.ones((2, 2))
        expected_capital_gamma_1 = -3 / 16 * np.ones((2, 2))

        network_statistics = MetricsCollection([NumberOfEdgesDirected(), TotalReciprocity()], True, n)

        features_of_sample = network_statistics.calculate_sample_statistics(sample)
        features_of_sample_mean = features_of_sample.mean(axis=1)
        diff_from_mean = features_of_sample - features_of_sample_mean[:, None]
        capital_gamma_0 = calc_kth_capital_gamma(diff_from_mean, 0)
        capital_gamma_1 = calc_kth_capital_gamma(diff_from_mean, 1)
        self.assertTrue(np.abs(capital_gamma_0 - expected_capital_gamma_0).max() < 10 ** -15)
        self.assertTrue(np.abs(capital_gamma_1 - expected_capital_gamma_1).max() < 10 ** -15)

    def test_multivariate_initial_seq_covariance_matrix_estimation(self):
        n = 3
        sample_size = 6
        W1 = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [1, 0, 0]])
        W2 = np.array([[0, 1, 0],
                       [0, 0, 0],
                       [1, 1, 0]])
        W3 = np.array([[0, 1, 1],
                       [0, 0, 1],
                       [1, 0, 0]])
        W4 = np.array([[0, 1, 0],
                       [0, 0, 0],
                       [0, 1, 0]])
        W5 = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 0]])
        W6 = np.array([[0, 1, 1],
                       [1, 0, 0],
                       [1, 1, 0]])
        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2
        sample[:, :, 2] = W3
        sample[:, :, 3] = W4
        sample[:, :, 4] = W5
        sample[:, :, 5] = W6

        # NOTE! this matrix was calculated after automatically calculating the capital gammas (using the dedicated
        # function that has a dedicated test). I gave up on calculating everything from scratch after a long battle.
        # Maybe we should come to that in the future.
        expected_covariance_multivariate_initial_seq_estimation = 1 / (6 * 36) * np.array([[264, 192], [192, 144]])

        network_statistics = MetricsCollection([NumberOfEdgesDirected(), TotalReciprocity()], True, n)
        features_of_sample = network_statistics.calculate_sample_statistics(sample)
        mean_features = features_of_sample.mean(axis=1)

        sys.setrecursionlimit(2000)
        multivariate_initial_seq_estimation = covariance_matrix_estimation(features_of_sample, mean_features,
                                                                           method='multivariate_initial_sequence')
        self.assertTrue(np.abs(
            expected_covariance_multivariate_initial_seq_estimation - multivariate_initial_seq_estimation).max() <
                        10 ** -14)
