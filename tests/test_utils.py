import unittest
from utils import *

from matplotlib import pyplot as plt


# class TestNetworkStatistics(unittest.TestCase):
#     # def setUp(self):
#     #     pass      

#     def test_initialization_params_validation(self):
#         with self.assertRaises(ValueError):
#             net_stats = NetworkStatistics()

#         with self.assertRaises(ValueError):
#             net_stats = NetworkStatistics(metric_names=["fake"])

#         with self.assertRaises(ValueError):
#             net_stats = NetworkStatistics(metric_names=["num_edges"], custom_metrics={"fake": 1})

#     def test_calculate_statistics(self):
#         net_stats = NetworkStatistics(metric_names=["num_edges", "num_triangles"], directed=False)
#         W = np.array([[0, 1, 1], 
#                       [1, 0, 1], 
#                       [1, 1, 0]])
#         stats = net_stats.calculate_statistics(W)
#         self.assertEqual(stats[0], 3)
#         self.assertEqual(stats[1], 1)

#         # Assert that num_triangles can't be run on a directed graph
#         with self.assertRaises(ValueError): 
#             net_stats = NetworkStatistics(metric_names=["num_edges", "num_triangles"], directed=True)

#         net_stats = NetworkStatistics(metric_names=["num_edges"], directed=True)
#         W = np.array([[0, 1, 1], 
#                       [1, 0, 0], 
#                       [1, 0, 1]])

#         stats = net_stats.calculate_statistics(W)
#         self.assertEqual(stats[0], 5)

#         net_stats = NetworkStatistics(metric_names=["num_edges", "num_triangles"], directed=False)
#         W = np.array([[0, 0, 0], 
#                       [0, 0, 0], 
#                       [0, 0, 0]])
#         stats = net_stats.calculate_statistics(W)
#         self.assertEqual(stats[0], 0)
#         self.assertEqual(stats[1], 0)


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
        np.random.seed(8972634)
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
