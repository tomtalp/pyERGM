import unittest

from scipy.stats import pearsonr

from pyERGM.utils import *
from pyERGM.metrics import (
    MetricsCollection, NumberOfEdgesDirected,
    TotalReciprocity, OutDegree, InDegree
)
from pyERGM.datasets import sampson_matrix
from pyERGM.ergm import ERGM
from pyERGM.constants import SamplingMethod, CovMatrixEstimationMethod, Reduction
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
        sampled_networks = p1_sampson_model.generate_networks_for_sample(sampling_method=SamplingMethod.EXACT,
                                                                         sample_size=sample_size)
        sample_mean = sampled_networks.mean(axis=-1)
        exact_marginals = get_exact_marginals_from_dyads_distribution(p1_sampson_model._exact_dyadic_distributions)

        self.assertEqual(sampled_networks.shape, (n_nodes, n_nodes, sample_size))
        self.assertEqual(convergence_result.success, True)
        self.assertTrue(np.abs(exact_marginals - sample_mean).max() < 1e-2)
        self.assertTrue(pearsonr(exact_marginals[~np.eye(n_nodes, dtype=bool)].flatten(),
                                 sampled_networks.mean(axis=-1)[
                                     ~np.eye(n_nodes, dtype=bool)].flatten()).statistic > 0.99)

    def test_flatten_square_matrix_to_edge_list(self):
        set_seed(49876)
        n = 10
        matrix = np.random.rand(n ** 2).reshape(n, n)
        flattened_directed = flatten_square_matrix_to_edge_list(matrix, True)
        flattened_undirected = flatten_square_matrix_to_edge_list(matrix + matrix.T, False)
        self.assertEqual(flattened_directed.shape, (n ** 2 - n, ))
        self.assertEqual(flattened_undirected.shape, ((n ** 2 - n) // 2,))
        idx_directed = 0
        idx_undirected = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                self.assertEqual(flattened_directed[idx_directed], matrix[i, j])
                idx_directed += 1
                if j > i:
                    self.assertEqual(flattened_undirected[idx_undirected], matrix[i, j] + matrix[j, i])
                    idx_undirected += 1



    def test_reshape_flattened_off_diagonal_elements_to_square(self):
        set_seed(39876)
        n = 10
        matrix = np.random.rand(n ** 2).reshape(n, n)
        matrix[np.diag_indices(n)] = 0
        flattened = flatten_square_matrix_to_edge_list(matrix, True)
        reshaped = reshape_flattened_off_diagonal_elements_to_square(flattened, is_directed=True)
        self.assertTrue(np.allclose(reshaped, matrix))

        matrix = matrix + matrix.T
        flattened = flatten_square_matrix_to_edge_list(matrix, False)
        reshaped = reshape_flattened_off_diagonal_elements_to_square(flattened, is_directed=False)
        self.assertTrue(np.allclose(reshaped, matrix))


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
                cur_mat = generate_erdos_renyi_matrix(n, p, is_directed=True)
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
        batch_estimation = covariance_matrix_estimation(features_of_sample, mean_features, method=CovMatrixEstimationMethod.BATCH,
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
        naive_estimation = covariance_matrix_estimation(features_of_sample, mean_features, method=CovMatrixEstimationMethod.NAIVE)
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
                                                                           method=CovMatrixEstimationMethod.MULTIVARIATE_INITIAL_SEQUENCE)
        self.assertTrue(np.abs(
            expected_covariance_multivariate_initial_seq_estimation - multivariate_initial_seq_estimation).max() <
                        10 ** -14)


class SamplingWithoutReplacementTester(unittest.TestCase):
    """Tests for sampling without replacement functionality."""

    def test_sample_without_replacement_produces_unique_networks(self):
        """Verify that replace=False produces only unique networks."""
        set_seed(12345)
        n_nodes = 5
        sample_size = 20
        prob_matrix = np.full((n_nodes, n_nodes), 0.5)
        np.fill_diagonal(prob_matrix, 0)

        sample = sample_from_independent_probabilities_matrix(
            prob_matrix, sample_size, is_directed=True, replace=False
        )

        # Convert each network to hashable representation
        hashes = set()
        for i in range(sample_size):
            net_hash = network_to_hashable(sample[:, :, i], is_directed=True)
            hashes.add(net_hash)

        self.assertEqual(len(hashes), sample_size, "Some networks are duplicates!")
        self.assertEqual(sample.shape, (n_nodes, n_nodes, sample_size))

    def test_sample_dyads_without_replacement_produces_unique_networks(self):
        """Test uniqueness for dyadic (reciprocity) models."""
        set_seed(12346)
        num_dyads = 6  # 4 nodes
        sample_size = 15

        # Uniform distribution over dyadic states
        dyads_distributions = np.full((num_dyads, 4), 0.25)

        sample = sample_from_dyads_distribution(
            dyads_distributions, sample_size=sample_size, replace=False
        )

        # Verify uniqueness
        hashes = set()
        for i in range(sample_size):
            net_hash = network_to_hashable(sample[:, :, i], is_directed=True)
            hashes.add(net_hash)

        self.assertEqual(len(hashes), sample_size, "Some networks are duplicates!")
        n_nodes = num_dyads_to_num_nodes(num_dyads)
        self.assertEqual(sample.shape, (n_nodes, n_nodes, sample_size))

    def test_sample_without_replacement_fails_gracefully_for_low_entropy(self):
        """Verify that low-entropy models raise informative errors."""
        set_seed(12347)
        n_nodes = 4
        sample_size = 50

        # Very low entropy: almost all edges present
        prob_matrix = np.full((n_nodes, n_nodes), 0.99)
        np.fill_diagonal(prob_matrix, 0)

        with self.assertRaises(RuntimeError) as context:
            sample_from_independent_probabilities_matrix(
                prob_matrix, sample_size, is_directed=True, replace=False
            )
        self.assertIn("duplication rate", str(context.exception))

    def test_sample_without_replacement_with_masks(self):
        """Test that masked edges are handled correctly."""
        set_seed(12349)
        n_nodes = 4
        prob_matrix = np.full((n_nodes, n_nodes), 0.5)
        np.fill_diagonal(prob_matrix, 0)
        prob_matrix[0, 1] = np.nan  # Masked edge
        prob_matrix[1, 0] = np.nan  # Masked edge

        sample = sample_from_independent_probabilities_matrix(
            prob_matrix, sample_size=10, is_directed=True, replace=False
        )

        # Verify masked edges are NaN in output
        self.assertTrue(np.all(np.isnan(sample[0, 1, :])))
        self.assertTrue(np.all(np.isnan(sample[1, 0, :])))

        # Verify networks are unique (hash function handles NaN automatically)
        hashes = set()
        for i in range(10):
            net_hash = network_to_hashable(sample[:, :, i], is_directed=True)
            self.assertTrue(all(not np.isnan(x) for x in net_hash))
            hashes.add(net_hash)

        self.assertEqual(len(hashes), 10, "Some networks are duplicates!")

    def test_sample_with_replacement_produces_expected_shape(self):
        """Verify that replace=True still works correctly (backward compatibility)."""
        set_seed(12350)
        n_nodes = 5
        sample_size = 20
        prob_matrix = np.full((n_nodes, n_nodes), 0.5)
        np.fill_diagonal(prob_matrix, 0)

        sample = sample_from_independent_probabilities_matrix(
            prob_matrix, sample_size, is_directed=True, replace=True
        )

        self.assertEqual(sample.shape, (n_nodes, n_nodes, sample_size))

    def test_network_to_hashable_consistency(self):
        """Verify that identical networks produce identical hashes."""
        set_seed(12351)
        n_nodes = 4
        network = np.random.binomial(1, 0.5, (n_nodes, n_nodes))
        np.fill_diagonal(network, 0)

        hash1 = network_to_hashable(network.copy(), is_directed=True)
        hash2 = network_to_hashable(network.copy(), is_directed=True)

        self.assertEqual(hash1, hash2)

    def test_network_to_hashable_different_networks(self):
        """Verify that different networks produce different hashes."""
        set_seed(12352)
        n_nodes = 4
        network1 = np.random.binomial(1, 0.3, (n_nodes, n_nodes))
        network2 = np.random.binomial(1, 0.3, (n_nodes, n_nodes))
        np.fill_diagonal(network1, 0)
        np.fill_diagonal(network2, 0)

        # Make sure they're actually different
        if np.allclose(network1, network2):
            network2[0, 1] = 1 - network2[0, 1]

        hash1 = network_to_hashable(network1, is_directed=True)
        hash2 = network_to_hashable(network2, is_directed=True)

        self.assertNotEqual(hash1, hash2)


class TestEntropyCalculations(unittest.TestCase):
    """
    Tests for entropy calculation utility functions in utils.py.

    Tests validate:
    1. Direct utility functions: calc_entropy_independent_probability_matrix
       - Uniform probability matrices with known ground truth
       - Deterministic edges (p→0 or p→1)
       - All Reduction modes (SUM, MEAN, NONE)
       - Masked edges (NaN handling)

    2. Dyadic distribution entropy: calc_entropy_dyads_dists
       - Known dyadic distributions with ground truth values
       - All Reduction modes (SUM, MEAN, NONE)

    Note: ERGM model entropy integration tests are located in test_ergm.py (TestERGMEntropy).
    Those tests validate ERGM.calc_model_entropy() using BruteForceERGM for exact ground truth.
    """

    def setUp(self):
        set_seed(30976)

    # ==================== Direct Probability Matrix Tests ====================

    def test_entropy_independent_probability_matrix_with_known_values(self):
        """Test entropy calculation for uniform probability matrices with known ground truth."""
        test_cases = [
            # (scenario_name, n_nodes, is_directed, p_value, expected_entropy)
            ("3node_directed_p50", 3, True, 0.5, 6.0),  # 6 edges * 1.0 bit/edge
            ("3node_undirected_p50", 3, False, 0.5, 3.0),  # 3 edges * 1.0 bit/edge
            ("4node_directed_p10", 4, True, 0.1, 5.628),  # 12 edges * ~0.469 bits/edge
            ("4node_directed_p90", 4, True, 0.9, 5.628),  # Symmetric around 0.5
        ]

        for scenario_name, n_nodes, is_directed, p_value, expected_entropy in test_cases:
            with self.subTest(scenario=scenario_name):
                # Create probability matrix with uniform probability
                prob_mat = np.full((n_nodes, n_nodes), p_value)
                np.fill_diagonal(prob_mat, 0)  # No self-loops

                # Calculate entropy
                entropy = calc_entropy_independent_probability_matrix(prob_mat, is_directed)

                # Assert matches expected value
                self.assertAlmostEqual(entropy, expected_entropy, places=3,
                                      msg=f"Entropy mismatch for scenario: {scenario_name}")

    def test_entropy_deterministic_edges(self):
        """Test entropy for deterministic edges (p→0 or p→1)."""
        test_cases = [
            ("p_near_one", 1 - 1e-9),
            ("p_near_zero", 1e-9),
        ]

        for scenario_name, p_value in test_cases:
            with self.subTest(scenario=scenario_name):
                n_nodes = 4
                prob_mat = np.full((n_nodes, n_nodes), p_value)
                np.fill_diagonal(prob_mat, 0)

                entropy = calc_entropy_independent_probability_matrix(prob_mat, True)

                # Entropy should be very small (limited by epsilon clipping to 1e-10)
                self.assertLess(entropy, 1e-6,
                               msg=f"Entropy not near zero for deterministic case: {scenario_name}")

    def test_entropy_probability_matrix_with_reduction_modes(self):
        """Test all Reduction modes for probability matrix entropy."""
        test_cases = [
            ("sum_reduction", Reduction.SUM, 6.0),
            ("mean_reduction", Reduction.MEAN, 1.0),
            ("none_reduction", Reduction.NONE, None),  # Array check
        ]

        n_nodes = 3
        is_directed = True
        prob_mat = np.full((n_nodes, n_nodes), 0.5)
        np.fill_diagonal(prob_mat, 0)

        for scenario_name, reduction, expected_value in test_cases:
            with self.subTest(reduction_mode=scenario_name):
                entropy = calc_entropy_independent_probability_matrix(prob_mat, is_directed, reduction=reduction)

                if reduction == Reduction.NONE:
                    # Should be array of per-edge entropies
                    self.assertIsInstance(entropy, np.ndarray)
                    self.assertEqual(entropy.shape, (6,))  # 3-node directed has 6 edges
                    self.assertTrue(np.allclose(entropy, np.ones(6), atol=1e-5))
                else:
                    self.assertAlmostEqual(entropy, expected_value, places=5,
                                          msg=f"Entropy reduction mismatch: {scenario_name}")

    def test_entropy_with_masked_edges(self):
        """Test entropy calculation correctly ignores NaN (masked) edges."""
        n_nodes = 3
        prob_mat = np.full((n_nodes, n_nodes), 0.5)
        np.fill_diagonal(prob_mat, 0)
        # Mask half the edges
        prob_mat[0, 1] = np.nan
        prob_mat[0, 2] = np.nan
        prob_mat[1, 2] = np.nan

        entropy = calc_entropy_independent_probability_matrix(prob_mat, True)

        # Only 3 non-NaN edges, each with entropy 1.0 bit
        self.assertAlmostEqual(entropy, 3.0, places=5,
                              msg="Entropy should only count non-NaN edges")

    # ==================== Dyadic Distribution Tests ====================

    def test_entropy_dyadic_distribution_with_known_values(self):
        """Test entropy calculation for dyadic state distributions with known ground truth."""
        test_cases = [
            # (scenario_name, dyad_distribution, num_dyads, expected_entropy)
            ("uniform_4state", [0.25, 0.25, 0.25, 0.25], 3, 6.0),  # 3 dyads * 2.0 bits each
            ("deterministic_empty", [1, 0, 0, 0], 3, 0.0),  # All empty dyads
            ("mixed_distributions", None, 3, 3.0),  # Custom - set in test
        ]

        for scenario_name, dyad_dist, num_dyads, expected_entropy in test_cases:
            with self.subTest(scenario=scenario_name):
                if scenario_name == "mixed_distributions":
                    # Create mixed dyadic distributions: [1,0,0,0], [0.5,0.5,0,0], [0.25,0.25,0.25,0.25]
                    dyads = np.array([
                        [1, 0, 0, 0],          # entropy ≈ 0
                        [0.5, 0.5, 0, 0],      # entropy ≈ 1.0
                        [0.25, 0.25, 0.25, 0.25]  # entropy ≈ 2.0
                    ])
                    expected = 3.0
                else:
                    # Create uniform dyadic distribution
                    dyads = np.tile(dyad_dist, (num_dyads, 1))
                    expected = expected_entropy

                entropy = calc_entropy_dyads_dists(dyads)

                self.assertAlmostEqual(entropy, expected, places=2,
                                      msg=f"Dyadic entropy mismatch: {scenario_name}")

    def test_entropy_dyadic_distribution_with_reduction_modes(self):
        """Test all Reduction modes for dyadic distribution entropy."""
        test_cases = [
            ("sum_reduction", Reduction.SUM, 6.0),
            ("mean_reduction", Reduction.MEAN, 2.0),
            ("none_reduction", Reduction.NONE, None),
        ]

        num_dyads = 3
        dyads = np.tile([0.25, 0.25, 0.25, 0.25], (num_dyads, 1))

        for scenario_name, reduction, expected_value in test_cases:
            with self.subTest(reduction_mode=scenario_name):
                entropy = calc_entropy_dyads_dists(dyads, reduction=reduction)

                if reduction == Reduction.NONE:
                    self.assertIsInstance(entropy, np.ndarray)
                    self.assertEqual(entropy.shape, (3,))
                    self.assertTrue(np.allclose(entropy, np.ones(3) * 2.0, atol=1e-5))
                else:
                    self.assertAlmostEqual(entropy, expected_value, places=5,
                                          msg=f"Dyadic entropy reduction mismatch: {scenario_name}")

