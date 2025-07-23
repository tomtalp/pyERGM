import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from pyERGM.utils import *
from pyERGM.metrics import *
from pyERGM.datasets import sampson_matrix

import networkx as nx
import math
import pandas as pd


class TestNumberOfEdgesUndirected(unittest.TestCase):
    def test_num_of_edges(self):
        metric = NumberOfEdgesUndirected()
        W = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])

        result = metric.calculate(W)
        expected_result = 3

        self.assertEqual(result, expected_result)


class TestNumberOfEdgesDirected(unittest.TestCase):
    def test_num_of_edges(self):
        metric = NumberOfEdgesDirected()
        W = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 0]
        ])

        result = metric.calculate(W)
        expected_result = 4

        self.assertEqual(result, expected_result)

    def test_num_of_edges_on_sample(self):
        metric = NumberOfEdgesDirected()
        n = 3
        sample_size = 2
        W1 = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 0]
        ])
        W2 = np.array([
            [0, 0, 0],
            [1, 0, 1],
            [1, 0, 0]
        ])

        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2

        result = metric.calculate_for_sample(sample)
        expected_result = np.array([4, 3])

        self.assertTrue(np.all(result == expected_result))


class TestNumberOfTriangles(unittest.TestCase):
    def test_validation(self):
        metric = NumberOfTriangles()

        self.assertTrue(not metric.requires_graph)

        W = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 0]
        ])

        with self.assertRaises(ValueError):
            metric.calculate(W)

    def test_num_of_triangles(self):
        metric = NumberOfTriangles()
        W = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])

        result = metric.calculate(W)
        expected_result = 1

        self.assertEqual(result, expected_result)

        random_graph = nx.fast_gnp_random_graph(6, 0.5, directed=False, seed=42)
        triangles = nx.triangles(random_graph)
        total_num_triangles = sum(triangles.values()) / 3

        G_as_W = nx.to_numpy_array(random_graph)
        calculated_number_of_triangles = metric.calculate(G_as_W)

        self.assertEqual(total_num_triangles, calculated_number_of_triangles)

    def test_calc_change_score(self):
        metric = NumberOfTriangles()

        W = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])

        idx_to_toggle = (0, 1)
        result = metric.calc_change_score(W, idx_to_toggle)
        self.assertEqual(result, -1)

        W = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 0]
        ])

        result = metric.calc_change_score(W, idx_to_toggle)
        self.assertEqual(result, 1)

        W = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ])

        idx_to_toggle = (1, 3)
        result = metric.calc_change_score(W, idx_to_toggle)
        self.assertEqual(result, 2)

    def test_calculate_for_sample(self):
        set_seed(678)
        sample_size = 50
        n = 10
        networks_sample = np.zeros((n, n, sample_size))
        expected_triangles = np.zeros(sample_size)
        for i in range(sample_size):
            cur_graph = nx.fast_gnp_random_graph(n, 0.5, seed=np.random)
            expected_triangles[i] = sum(nx.triangles(cur_graph).values()) / 3
            networks_sample[:, :, i] = nx.to_numpy_array(cur_graph)

        res = NumberOfTriangles().calculate_for_sample(networks_sample)
        self.assertTrue(np.all(res == expected_triangles))


class TestReciprocity(unittest.TestCase):
    def test_calculate(self):
        metrics = [Reciprocity()]
        n = 3
        collection = MetricsCollection(metrics, is_directed=True, n_nodes=n)

        matrices_to_test = [
            {
                "W": np.array([
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0]
                ]),
                "expected_recip": np.array([0, 0, 0])
            },
            {
                "W": np.array([
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 1, 0]
                ]),
                "expected_recip": np.array([0, 1, 0])
            },
            {
                "W": np.array([
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0]
                ]),
                "expected_recip": np.array([1, 1, 1])
            }
        ]

        for test_data in matrices_to_test:
            W = test_data["W"]
            expected_reciprocity_vector = test_data["expected_recip"]
            g_y = collection.calculate_statistics(W)
            self.assertTrue(np.array_equal(g_y, expected_reciprocity_vector))

        n = 30
        collection = MetricsCollection(metrics, is_directed=True, n_nodes=n)
        W = np.random.randint(0, 2, (n, n))

        G = nx.from_numpy_array(W, create_using=nx.DiGraph)

        total_edges = np.sum(W)

        reciprocity_vector = collection.calculate_statistics(W)
        total_reciprocity = np.sum(reciprocity_vector)

        reciprocty_fraction = total_reciprocity / total_edges

        nx_reciprocity = nx.algorithms.reciprocity(G) / 2  # nx counts each reciprocity twice
        self.assertEqual(reciprocty_fraction, nx_reciprocity)


class TestDegreeMetrics(unittest.TestCase):
    def test_calculate(self):
        W = np.array([
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 0]
        ])

        G = nx.from_numpy_array(W, create_using=nx.DiGraph)
        expected_in_degrees = list(dict(G.in_degree()).values())

        receiver = InDegree()
        indegree = receiver.calculate(W)

        self.assertTrue(all(indegree == expected_in_degrees))

        expected_out_degrees = list(dict(G.out_degree()).values())

        sender = OutDegree()
        outdegree = sender.calculate(W)

        self.assertTrue(all(outdegree == expected_out_degrees))

        undirected_degree = UndirectedDegree()
        W = np.array([
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 0]
        ])

        G = nx.from_numpy_array(W)

        degrees = undirected_degree.calculate(W)
        expected_degrees = list(dict(G.degree()).values())
        self.assertTrue(all(degrees == expected_degrees))


class TestNumberOfEdgesTypesDirected(unittest.TestCase):
    def test_calc_edge_weights(self):
        neuronal_types = ['A', 'B', 'A', 'B']
        metric = NumberOfEdgesTypesDirected(neuronal_types)
        expected_edge_weights = np.array([[0, 1, 0, 1],
                                          [2, 3, 2, 3],
                                          [0, 1, 0, 1],
                                          [2, 3, 2, 3]])

        expected_edge_weights += 1  # Increment because we also want a bin for nonexisting edges (which have the entry 0)

        self.assertTrue(np.all(metric._edge_type_idx_assignment == expected_edge_weights))

        neuronal_types = ['A', 'B', 'B', 'A']
        metric = NumberOfEdgesTypesDirected(neuronal_types)
        expected_edge_weights = np.array([[0, 1, 1, 0],
                                          [2, 3, 3, 2],
                                          [2, 3, 3, 2],
                                          [0, 1, 1, 0]])

        expected_edge_weights += 1  # Increment because we also want a bin for nonexisting edges (which have the entry 0)

        print(metric._edge_type_idx_assignment)
        print(expected_edge_weights)
        self.assertTrue(np.all(metric._edge_type_idx_assignment == expected_edge_weights))

    def test_calculate(self):
        W = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        neuronal_types = ['A', 'B', 'A', 'B']
        expected_num_edges = np.array([1, 2, 2, 1])
        metric = NumberOfEdgesTypesDirected(neuronal_types)
        calculated_num_edges = metric.calculate(W)
        self.assertTrue(np.all(expected_num_edges == calculated_num_edges))

    def test_calculate_for_sample(self):
        n = 4
        sample_size = 2
        W1 = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        W2 = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 1, 0]
        ])

        neuronal_types = ['A', 'B', 'A', 'B']
        expected_num_edges = np.array([
            [1, 2, 2, 1],
            [2, 1, 1, 2]
        ]).T

        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2
        metric = NumberOfEdgesTypesDirected(neuronal_types)
        calculated_num_edges = metric.calculate_for_sample(sample)
        self.assertTrue(np.all(expected_num_edges == calculated_num_edges))

    def test_calc_change_score(self):
        W1 = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        W2 = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 1, 0]
        ])

        neuronal_types = ['A', 'B', 'A', 'B']
        metric = NumberOfEdgesTypesDirected(neuronal_types)

        calculated_change_score = metric.calc_change_score(W1, (0, 1))
        expected_change_score = np.array([0, -1, 0, 0])
        self.assertTrue(np.all(expected_change_score == calculated_change_score))

        calculated_change_score = metric.calc_change_score(W1, (2, 3))
        expected_change_score = np.array([0, 1, 0, 0])
        self.assertTrue(np.all(expected_change_score == calculated_change_score))

        calculated_change_score = metric.calc_change_score(W2, (0, 2))
        expected_change_score = np.array([-1, 0, 0, 0])
        self.assertTrue(np.all(expected_change_score == calculated_change_score))

        calculated_change_score = metric.calc_change_score(W2, (1, 2))
        expected_change_score = np.array([0, 0, 1, 0])
        self.assertTrue(np.all(expected_change_score == calculated_change_score))

    def test_calculate_change_score_full_network(self):
        types = ['A', 'B', 'A', 'B']
        metric = NumberOfEdgesTypesDirected(types)

        W1 = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        expected_full_change_score = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, -1, 0]
        ])

        full_change_score = metric.calculate_change_score_full_network(W1)
        self.assertTrue(np.all(expected_full_change_score == full_change_score))


class TestNodeAttrSums(unittest.TestCase):
    def test_calc_edge_weights(self):
        node_attr = np.array([1, 2, 3, 4])
        metric_both = NodeAttrSum(node_attr, is_directed=True)
        expected_edge_weights = np.array([
            [0, 3, 4, 5],
            [3, 0, 5, 6],
            [4, 5, 0, 7],
            [5, 6, 7, 0]
        ])
        self.assertTrue(np.all(expected_edge_weights == metric_both.edge_weights))

        metric_out = NodeAttrSumOut(node_attr)
        expected_edge_weights = np.array([
            [0, 1, 1, 1],
            [2, 0, 2, 2],
            [3, 3, 0, 3],
            [4, 4, 4, 0]
        ])
        self.assertTrue(np.all(expected_edge_weights == metric_out.edge_weights))

        metric_in = NodeAttrSumIn(node_attr)
        self.assertTrue(np.all(expected_edge_weights.T == metric_in.edge_weights))

    def test_calculate(self):
        W = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 1, 0]
        ])

        node_attr = np.array([1, 2, 3, 4])
        metric_both = NodeAttrSum(node_attr, is_directed=True)
        expected_statistic = 3 + 4 + 6 + 4 + 6 + 7
        self.assertEqual(expected_statistic, metric_both.calculate(W)[0])

        metric_out = NodeAttrSumOut(node_attr)
        expected_statistic = 1 + 1 + 2 + 3 + 4 + 4
        self.assertEqual(expected_statistic, metric_out.calculate(W)[0])

        metric_in = NodeAttrSumIn(node_attr)
        expected_statistic = 1 + 2 + 2 + 3 + 3 + 4
        self.assertEqual(expected_statistic, metric_in.calculate(W)[0])

    def test_calculate_for_sample(self):
        n = 3
        sample_size = 3

        W1 = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 0]
        ])
        W2 = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [1, 1, 0]
        ])
        W3 = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])

        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2
        sample[:, :, 2] = W3

        node_attributes = np.array([2, 1, 1])

        metric_both = NodeAttrSum(node_attributes, is_directed=True)
        expected_stats_sample = np.array([11, 11, 5])
        calculated_stats_sample = metric_both.calculate_for_sample(sample)
        self.assertTrue(np.all(expected_stats_sample == calculated_stats_sample))

        metric_out = NodeAttrSumOut(node_attributes)
        expected_stats_sample = np.array([5, 5, 2])
        calculated_stats_sample = metric_out.calculate_for_sample(sample)
        self.assertTrue(np.all(expected_stats_sample == calculated_stats_sample))

        metric_in = NodeAttrSumIn(node_attributes)
        expected_stats_sample = np.array([6, 6, 3])
        calculated_stats_sample = metric_in.calculate_for_sample(sample)
        self.assertTrue(np.all(expected_stats_sample == calculated_stats_sample))

    def test_calc_change_score(self):
        W_off = np.zeros((3, 3))
        W_on = np.ones((3, 3))

        node_attributes = np.array([1, 0.5, 2])

        metric_both = NodeAttrSum(node_attributes, is_directed=True)
        metric_out = NodeAttrSumOut(node_attributes)
        metric_in = NodeAttrSumIn(node_attributes)

        calculated_change_score = metric_both.calc_change_score(W_off, (2, 1))
        expected_change_score = 2.5
        self.assertEqual(expected_change_score, calculated_change_score)

        calculated_change_score = metric_both.calc_change_score(W_on, (2, 0))
        expected_change_score = -3
        self.assertEqual(expected_change_score, calculated_change_score)

        calculated_change_score = metric_out.calc_change_score(W_off, (1, 0))
        expected_change_score = 0.5
        self.assertEqual(expected_change_score, calculated_change_score)

        calculated_change_score = metric_out.calc_change_score(W_on, (0, 2))
        expected_change_score = -1
        self.assertEqual(expected_change_score, calculated_change_score)

        calculated_change_score = metric_in.calc_change_score(W_off, (0, 1))
        expected_change_score = 0.5
        self.assertEqual(expected_change_score, calculated_change_score)

        calculated_change_score = metric_in.calc_change_score(W_on, (1, 2))
        expected_change_score = -2
        self.assertEqual(expected_change_score, calculated_change_score)

    def test_calculate_mple_regressors(self):
        # directed both (in+out)
        node_attr = np.array([3, 5, 7])
        W = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [1, 0, 0]])

        attr_sum_mat = np.array([[0, 8, 10],
                                 [8, 0, 12],
                                 [10, 12, 0]])

        metric_both = NodeAttrSum(node_attr, is_directed=True)
        indices_lims = (0, 6)
        expected_regressors_1_1 = np.array([[8.],
                                            [10.],
                                            [8.],
                                            [12.],
                                            [10.],
                                            [12.]])
        self.assertTrue(np.all(metric_both.calculate_mple_regressors(W, indices_lims) == expected_regressors_1_1))

        indices_lims = (1, 4)
        expected_regressors_1_2 = np.array([[10.],
                                            [8.],
                                            [12.]])
        self.assertTrue(np.all(metric_both.calculate_mple_regressors(W, indices_lims) == expected_regressors_1_2))

        # directed in
        attr_sum_mat_in = np.array([[0, 5, 7],
                                    [3, 0, 7],
                                    [3, 5, 0]])

        metric_in = NodeAttrSumIn(node_attr)
        indices_lims = (0, 6)
        expected_regressors_2_1 = np.array([[5.],
                                            [7.],
                                            [3.],
                                            [7.],
                                            [3.],
                                            [5.]])
        self.assertTrue(np.all(metric_in.calculate_mple_regressors(W, indices_lims) == expected_regressors_2_1))

        indices_lims = (1, 4)
        expected_regressors_2_2 = np.array([[7.],
                                            [3.],
                                            [7.]])
        self.assertTrue(np.all(metric_in.calculate_mple_regressors(W, indices_lims) == expected_regressors_2_2))

        # directed out
        attr_sum_mat_out = np.array([[0, 3, 3],
                                     [5, 0, 5],
                                     [7, 7, 0]])

        metric_out = NodeAttrSumOut(node_attr)
        indices_lims = (0, 6)
        expected_regressors_3_1 = np.array([[3.],
                                            [3.],
                                            [5.],
                                            [5.],
                                            [7.],
                                            [7.]])

        self.assertTrue(np.all(metric_out.calculate_mple_regressors(W, indices_lims) == expected_regressors_3_1))

        indices_lims = (1, 4)
        expected_regressors_3_2 = np.array([[3.],
                                            [5.],
                                            [5.]])
        self.assertTrue(np.all(metric_out.calculate_mple_regressors(W, indices_lims) == expected_regressors_3_2))

        # undirected (both)
        node_attr = np.array([3, 5, 7])
        W = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]])
        attr_sum_mat = np.array([[0, 8, 10],
                                 [8, 0, 12],
                                 [10, 12, 0]])

        metric_both = NodeAttrSum(node_attr, is_directed=False)
        indices_lims = (0, 3)
        # in undirected graphs the expected outcome is the upper triangle (without the diagonal)
        expected_regressors_1 = np.array([[8.],
                                          [10.],
                                          [12.]])
        self.assertTrue(np.all(metric_both.calculate_mple_regressors(W, indices_lims) == expected_regressors_1))

        indices_lims = (1, 2)
        expected_regressors_2 = np.array([[10.]])
        self.assertTrue(np.all(metric_both.calculate_mple_regressors(W, indices_lims) == expected_regressors_2))


class TestSumDistancesConnectedNeurons(unittest.TestCase):
    def test_sum_distances(self):
        positions = pd.DataFrame({"x_pos": [0, 0, 4], "y_pos": [0, 3, 0], "z_pos": [2, 2, 2]})
        W = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [1, 0, 0]])

        # Dataframe with multiple columns
        metric_1 = SumDistancesConnectedNeurons(positions, is_directed=True)
        expected_res_1 = 3 + 3 + 5 + 4
        self.assertTrue(metric_1.calculate(W) == expected_res_1)

        # 2D numpy array
        metric_2 = SumDistancesConnectedNeurons(positions.to_numpy(), is_directed=True)
        expected_res_2 = 3 + 3 + 5 + 4
        self.assertTrue(metric_2.calculate(W) == expected_res_2)

        # Series
        metric_3 = SumDistancesConnectedNeurons(positions.x_pos, is_directed=True)
        expected_res_3 = 4 + 4
        self.assertTrue(metric_3.calculate(W) == expected_res_3)

        # 1D numpy array
        metric_4 = SumDistancesConnectedNeurons(positions.z_pos.to_numpy(), is_directed=True)
        expected_res_4 = 0
        self.assertTrue(metric_4.calculate(W) == expected_res_4)

        # list
        positions_y_list = [0, 3, 0]
        metric_5 = SumDistancesConnectedNeurons(positions_y_list, is_directed=True)
        expected_res_5 = 3 + 3 + 3
        self.assertTrue(metric_5.calculate(W) == expected_res_5)

        # tuple
        positions_y_tuple = (0, 3, 0)
        metric_6 = SumDistancesConnectedNeurons(positions_y_tuple, is_directed=True)
        expected_res_6 = 3 + 3 + 3
        self.assertTrue(metric_6.calculate(W) == expected_res_6)

        # undirected_graph
        W_undirected = np.array([[0, 1, 1],
                                 [1, 0, 0],
                                 [1, 0, 0]])

        metric_7 = SumDistancesConnectedNeurons(positions, is_directed=False)
        expected_res_7 = 3 + 4
        self.assertTrue(metric_7.calculate(W_undirected) == expected_res_7)

    def test_calculate_mple_regressors(self):
        positions = pd.DataFrame({"x_pos": [0, 0, 4], "y_pos": [0, 3, 0], "z_pos": [2, 2, 2]})
        W = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [1, 0, 0]])

        # multiple columns
        metric_1 = SumDistancesConnectedNeurons(positions, is_directed=True)
        distances = np.array([[0, 3, 4],
                              [3, 0, 5],
                              [4, 5, 0]])
        indices_lims = (0, 6)
        expected_regressors_1_1 = np.array([[3.],
                                            [4.],
                                            [3.],
                                            [5.],
                                            [4.],
                                            [5.]])
        self.assertTrue(np.all(metric_1.calculate_mple_regressors(W, indices_lims) == expected_regressors_1_1))

        indices_lims = (1, 4)
        expected_regressors_1_2 = np.array([[4.],
                                            [3.],
                                            [5.]])
        self.assertTrue(np.all(metric_1.calculate_mple_regressors(W, indices_lims) == expected_regressors_1_2))

        # Series
        metric_2 = SumDistancesConnectedNeurons(positions.x_pos, is_directed=True)
        distances = np.array([[0, 0, 4],
                              [0, 0, 4],
                              [4, 4, 0]])
        indices_lims = (0, 6)
        expected_regressors_2_1 = np.array([[0.],
                                            [4.],
                                            [0.],
                                            [4.],
                                            [4.],
                                            [4.]])
        self.assertTrue(np.all(metric_2.calculate_mple_regressors(W, indices_lims) == expected_regressors_2_1))

        indices_lims = (1, 4)
        expected_regressors_2_2 = np.array([[4.],
                                            [0.],
                                            [4.]])
        self.assertTrue(np.all(metric_2.calculate_mple_regressors(W, indices_lims) == expected_regressors_2_2))

        metric_3 = SumDistancesConnectedNeurons(positions.z_pos, is_directed=True)
        distances = np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]])
        indices_lims = (0, 6)
        expected_regressors_3 = np.array([[0.],
                                          [0.],
                                          [0.],
                                          [0.],
                                          [0.],
                                          [0.]])
        self.assertTrue(np.all(metric_3.calculate_mple_regressors(W, indices_lims) == expected_regressors_3))

        # undirected
        metric_4 = SumDistancesConnectedNeurons(positions, is_directed=False)
        distances = np.array([[0, 3, 4],
                              [3, 0, 5],
                              [4, 5, 0]])
        indices_lims = (0, 3)
        expected_regressors_4_1 = np.array([[3.],
                                            [4.],
                                            [5.]])
        self.assertTrue(np.all(metric_4.calculate_mple_regressors(W, indices_lims) == expected_regressors_4_1))

        indices_lims = (1, 2)
        expected_regressors_4_2 = np.array([[4.]])
        self.assertTrue(np.all(metric_4.calculate_mple_regressors(W, indices_lims) == expected_regressors_4_2))

        metric_5 = SumDistancesConnectedNeurons(positions.z_pos, is_directed=False)
        distances = np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]])
        indices_lims = (0, 3)
        expected_regressors_5 = np.array([[0.],
                                          [0.],
                                          [0.]])
        self.assertTrue(np.all(metric_5.calculate_mple_regressors(W, indices_lims) == expected_regressors_5))


class TestNumberOfNodesPerType(unittest.TestCase):
    def test_calculate(self):
        V = np.array([[1, 0],
                      [2, 1],
                      [1, 1],
                      [0, 2]]) # n=4, k=2

        expected_num_neurons_per_type = np.array([1, 2, 1])
        metric = NumberOfNodesPerType(metric_node_feature={'morphology'}, n_node_categories=3)
        calculated_num_neurons_per_type = metric.calculate(V[:, [0]])
        self.assertTrue(np.all(expected_num_neurons_per_type[:-1] == calculated_num_neurons_per_type))

    def test_calculate_for_sample(self):
        n = 4
        sample_size = 2
        V1 = np.array([[1, 0],
                       [2, 1],
                       [1, 1],
                       [0, 2]])
        V2 = np.array([[2, 0],
                       [0, 1],
                       [0, 1],
                       [0, 1]])

        expected_num_neurons_per_type = np.array([
            [1, 2, 1],
            [3, 0, 1]
        ]).T

        sample = np.zeros((n, 1, sample_size), dtype=int)
        sample[:, :, 0] = V1[:, [0]]
        sample[:, :, 1] = V2[:, [0]]
        metric = NumberOfNodesPerType(metric_node_feature={'morphology'}, n_node_categories=3)
        calculated_num_neurons_per_type = metric.calculate_for_sample(sample)
        self.assertTrue(np.all(expected_num_neurons_per_type[:-1] == calculated_num_neurons_per_type))

    def test_calc_change_score(self):
        V1 = np.array([[1, 0],
                       [2, 1],
                       [1, 1],
                       [0, 2]])
        V2 = np.array([[2, 0],
                       [0, 1],
                       [0, 1],
                       [0, 1]])

        metric = NumberOfNodesPerType(metric_node_feature={'morphology'}, n_node_categories=3)

        calculated_change_score = metric.calc_change_score(V1[:, [0]], idx=2, new_category=0)
        expected_change_score = np.array([1, -1, 0])
        self.assertTrue(np.all(expected_change_score[:-1] == calculated_change_score))

        calculated_change_score = metric.calc_change_score(V1[:, [1]], idx=0, new_category=2)
        expected_change_score = np.array([-1, 0, 1])
        self.assertTrue(np.all(expected_change_score[:-1] == calculated_change_score))

        calculated_change_score = metric.calc_change_score(V2[:, [0]], idx=1, new_category=1)
        expected_change_score = np.array([-1, 1, 0])
        self.assertTrue(np.all(expected_change_score[:-1] == calculated_change_score))

        calculated_change_score = metric.calc_change_score(V2[:, [1]], idx=0, new_category=1)
        expected_change_score = np.array([-1, 1, 0])
        self.assertTrue(np.all(expected_change_score[:-1] == calculated_change_score))


class TestMetricsCollection(unittest.TestCase):

    def test_get_effective_feature_count(self):
        n = 18
        receiver = InDegree()

        collection = MetricsCollection([receiver], is_directed=True, n_nodes=n, do_copy_metrics=False)
        self.assertEqual(receiver._get_effective_feature_count(), n)

        receiver = InDegree(indices_from_user=[0])
        collection = MetricsCollection([receiver], is_directed=True, n_nodes=n, do_copy_metrics=False)
        self.assertEqual(receiver._get_effective_feature_count(), n - 1)

        sender = OutDegree()

        collection = MetricsCollection([sender], is_directed=True, n_nodes=n, do_copy_metrics=False)
        self.assertEqual(sender._get_effective_feature_count(), n)

        sender = OutDegree(indices_from_user=[0])
        collection = MetricsCollection([sender], is_directed=True, n_nodes=n, do_copy_metrics=False)
        self.assertEqual(sender._get_effective_feature_count(), n - 1)

    def test_metrics_setup(self):
        metrics = [NumberOfEdgesDirected(), TotalReciprocity(), InDegree()]
        collection = MetricsCollection(metrics, is_directed=True, n_nodes=3)

        expected_metric_names = tuple([str(NumberOfEdgesDirected()), str(TotalReciprocity()), str(InDegree())])

        self.assertTrue(collection.metric_names == expected_metric_names)

        metrics = [InDegree(), UndirectedDegree()]
        with self.assertRaises(ValueError):
            collection = MetricsCollection(metrics, is_directed=True,
                                           n_nodes=3)  # Should fail because we have an undirected metric in a directed metrics collection

        with self.assertRaises(ValueError):
            collection = MetricsCollection(metrics, is_directed=False,
                                           n_nodes=3)  # Should fail because we have a directed metric in an undirected metrics collection

    def test_collinearity_fixer(self):
        n = 18

        possible_types = ['A', 'B', 'C', 'D']
        # sampled once using np.random.choice(len(possible_types), size=n)
        type_idx_per_node = [3, 3, 0, 0, 0, 0, 1, 3, 1, 1, 3, 0, 3, 2, 0, 3, 3, 1]
        neuronal_types = [possible_types[type_idx_per_node[i]] for i in range(n)]

        n_for_multitypes_test = 6
        multitypes_1 = ['A', 'B']
        multitypes_2 = ['C', 'D']
        type_1_idx_per_node = [0, 0, 0, 1, 1, 1]
        type_2_idx_per_node = [0, 0, 0, 0, 1, 1]
        types_1 = [multitypes_1[type_1_idx_per_node[i]] for i in range(n_for_multitypes_test)]
        types_2 = [multitypes_2[type_2_idx_per_node[i]] for i in range(n_for_multitypes_test)]

        multitypes_3 = ['A', 'B']
        multitypes_4 = ['C', 'D', 'E']
        type_3_idx_per_node = [0, 0, 1, 1, 1, 1]
        type_4_idx_per_node = [0, 0, 1, 1, 2, 2]
        types_3 = [multitypes_3[type_3_idx_per_node[i]] for i in range(n_for_multitypes_test)]
        types_4 = [multitypes_4[type_4_idx_per_node[i]] for i in range(n_for_multitypes_test)]

        test_scenarios = {
            "num_edges_twice": {
                "metrics": [NumberOfEdgesDirected(), NumberOfEdgesDirected()],
                "expected_num_of_features": 1,
                "expected_trimmed_metrics": {},
                "expected_eliminated_metrics": []
            },
            "num_edges__total_reciprocity__indegree": {
                "metrics": [NumberOfEdgesDirected(), TotalReciprocity(), InDegree()],
                "expected_num_of_features": 1 + 1 + n - 1,
                "expected_trimmed_metrics": {},
                "expected_eliminated_metrics": []
            },
            "indegree": {
                "metrics": [InDegree()],
                "expected_num_of_features": n,  # There is no collinearity since InDegree() is the only feature
                "expected_trimmed_metrics": {},
                "expected_eliminated_metrics": []
            },
            "outdegree": {
                "metrics": [OutDegree()],
                "expected_num_of_features": n,
                "expected_trimmed_metrics": {},
                "expected_eliminated_metrics": []
            },
            "in_outdegree": {
                "metrics": [InDegree(), OutDegree()],
                "expected_num_of_features": n + n - 1,
                "expected_trimmed_metrics": {},
                "expected_eliminated_metrics": []
            },
            "num_edges__total_reciprocity__indegree_outdegree": {
                "metrics": [NumberOfEdgesDirected(), TotalReciprocity(), InDegree(), OutDegree()],
                "expected_num_of_features": 1 + 1 + n - 1 + n - 1,
                "expected_trimmed_metrics": {},
                "expected_eliminated_metrics": []
            },
            "num_edges__exogenous_types": {
                "metrics": [NumberOfEdgesDirected(), NumberOfEdgesTypesDirected(neuronal_types)],
                "expected_num_of_features": 1 + len(set(neuronal_types)) ** 2 - 2,
                "expected_trimmed_metrics": {},
                "expected_eliminated_metrics": []
            },
            "sum_attr_both__sum_attr_in__sum_attr_out": {
                "metrics": [NodeAttrSum(np.arange(n), is_directed=True), NodeAttrSumIn(np.arange(n)),
                            NodeAttrSumOut(np.arange(n))],
                "expected_num_of_features": 2,
                "expected_trimmed_metrics": {},
                "expected_eliminated_metrics": []
            },
            "multiple_types": {
                "n": n_for_multitypes_test,
                "metrics": [NumberOfEdgesDirected(), NumberOfEdgesTypesDirected(types_1),
                            NumberOfEdgesTypesDirected(types_2)],
                "expected_num_of_features": 1 + len(set(types_1)) ** 2 + len(set(types_2)) ** 2 - 2,
                "expected_trimmed_metrics": {},
                "expected_eliminated_metrics": []
            },
            "multiple_types_2": {
                "n": n_for_multitypes_test,
                "metrics": [NumberOfEdgesDirected(), NumberOfEdgesTypesDirected(types_3),
                            NumberOfEdgesTypesDirected(types_4)],
                "expected_num_of_features": 1 + len(set(types_3)) ** 2 + len(set(types_4)) ** 2 - 5,
                "expected_trimmed_metrics": {},
                "expected_eliminated_metrics": []
            }
        }

        # We want to make sure that the changes (trimming and complete removal) are made for the right metrics, so we
        # update the scenarios after initializing the lists of Metrics, to pass the class instances by reference when
        # asserting stuff later.
        test_scenarios["num_edges_twice"]["expected_eliminated_metrics"] = [
            test_scenarios["num_edges_twice"]["metrics"][0]]

        test_scenarios["num_edges__total_reciprocity__indegree"]["expected_trimmed_metrics"] = {
            # InDegree is trimmed by 1 because of collinearity with num_of_edges
            test_scenarios["num_edges__total_reciprocity__indegree"]["metrics"][2]: [0]
        }

        test_scenarios["in_outdegree"]["expected_trimmed_metrics"] = {
            test_scenarios["in_outdegree"]["metrics"][0]: [0]
        }

        test_scenarios["num_edges__total_reciprocity__indegree_outdegree"]["expected_trimmed_metrics"] = {
            test_scenarios["num_edges__total_reciprocity__indegree_outdegree"]["metrics"][2]: [0],
            test_scenarios["num_edges__total_reciprocity__indegree_outdegree"]["metrics"][3]: [0]
        }

        test_scenarios["num_edges__exogenous_types"]["expected_trimmed_metrics"] = {
            test_scenarios["num_edges__exogenous_types"]["metrics"][1]: [0, 10]
        }

        test_scenarios["sum_attr_both__sum_attr_in__sum_attr_out"]["expected_eliminated_metrics"] = [
            test_scenarios["sum_attr_both__sum_attr_in__sum_attr_out"]["metrics"][0]]

        test_scenarios["multiple_types"]["expected_trimmed_metrics"] = {
            test_scenarios["multiple_types"]["metrics"][1]: [0],
            test_scenarios["multiple_types"]["metrics"][2]: [0]}

        test_scenarios["multiple_types_2"]["expected_trimmed_metrics"] = {
            # A->A, A->B, B->A trimmed
            test_scenarios["multiple_types_2"]["metrics"][1]: [0, 1, 2],
            # C->C, D->D trimmed
            test_scenarios["multiple_types_2"]["metrics"][2]: [0, 4]}

        for name, scenario_data in test_scenarios.items():
            net_size = scenario_data.get("n", n)

            collection = MetricsCollection(scenario_data["metrics"], is_directed=True, fix_collinearity=True,
                                           n_nodes=net_size, do_copy_metrics=False)

            # Check the general number of features matches the expectation
            self.assertEqual(collection.num_of_features, scenario_data["expected_num_of_features"])

            # Check if the correct metrics were trimmed
            for metric in scenario_data["metrics"]:
                if metric in scenario_data["expected_trimmed_metrics"].keys():
                    self.assertEqual(set(np.where(metric._indices_to_ignore)[0]),
                                     set(scenario_data["expected_trimmed_metrics"][metric]))
                elif hasattr(metric,
                             "_indices_to_ignore"):  # Only metrics with a base_idx can be trimmed. The others aren't tested
                    self.assertFalse(np.any(metric._indices_to_ignore))
                elif metric in scenario_data["expected_eliminated_metrics"]:
                    self.assertTrue(metric not in [m for m in collection.metrics])
                    self.assertEqual(collection.num_of_metrics, len(collection.metrics))

    def test_calculate_statistics(self):
        # Test undirected graphs
        metrics = [NumberOfEdgesUndirected(), NumberOfTriangles()]
        collection = MetricsCollection(metrics, is_directed=False, n_nodes=3)

        W = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])

        stats = collection.calculate_statistics(W)
        expected_stats = np.array([3, 1])

        np.testing.assert_array_equal(stats, expected_stats)

        # Test directed graphs
        metrics = [NumberOfEdgesDirected()]
        collection = MetricsCollection(metrics, is_directed=True, n_nodes=3)
        W = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 0]
        ])

        stats = collection.calculate_statistics(W)
        expected_stats = np.array([4])

        np.testing.assert_array_equal(stats, expected_stats)

    def test_calculate_statistics_with_node_features(self):
        # Test undirected graphs
        metrics = [NumberOfEdgesUndirected(), NumberOfTriangles()]
        collection = MetricsCollection(metrics, is_directed=False, n_nodes=3)

        # Now, W is an (n, n+k) with n=3, k=2. Result shouldn't change.
        W = np.array([
            [0, 1, 1, 1, 2],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0]
        ])

        stats = collection.calculate_statistics(W)
        expected_stats = np.array([3, 1])

        np.testing.assert_array_equal(stats, expected_stats)

        # Test directed graphs
        metrics = [NumberOfEdgesDirected()]
        collection = MetricsCollection(metrics, is_directed=True, n_nodes=3)
        W = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 0]
        ])

        stats = collection.calculate_statistics(W)
        expected_stats = np.array([4])

        np.testing.assert_array_equal(stats, expected_stats)

    def test_get_num_of_features(self):
        n = 4
        metrics = [NumberOfEdgesUndirected(), NumberOfTriangles()]
        collection = MetricsCollection(metrics, is_directed=False, n_nodes=n)
        num_of_features = collection.num_of_features

        # NumberOfEdges and NumberOfTriangles each produce 1 features
        expected_num_of_features = 2

        self.assertEqual(num_of_features, expected_num_of_features)

        metrics = [Reciprocity(), NumberOfEdgesDirected()]
        collection = MetricsCollection(metrics, is_directed=True, n_nodes=n)

        num_of_features = collection.num_of_features

        expected_num_of_features = math.comb(n, 2) + 1

        self.assertEqual(num_of_features, expected_num_of_features)

    def test_calc_change_scores(self):
        metrics = [NumberOfEdgesDirected(), Reciprocity()]
        collection = MetricsCollection(metrics, is_directed=True, n_nodes=3)

        W1 = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [1, 1, 0]
        ])

        W2 = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])

        flipped_indices = (2, 0)
        edge_flip_info = {
            'edge': flipped_indices
        }
        result = collection.calc_change_scores(W1, edge_flip_info=edge_flip_info)

        # 1st is -1 because we lost an edge, and 3rd entry is -1 because node #2 lost it's reciprocity
        expected_result = [-1, 0, -1, 0]

        self.assertTrue(all(result == expected_result))

    def test_calculate_change_scores_all_edges(self):
        types = ['A', 'B', 'A', 'B']
        metrics = [NumberOfEdgesDirected(), NumberOfEdgesTypesDirected(types)]

        W1 = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        n_nodes = W1.shape[0]

        expected_full_change_score = np.array([
            [-1, 0, -1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [-1, 0, 0, -1, 0],
            [1, 0, 0, 1, 0],
            [-1, 0, 0, 0, -1],
            [-1, -1, 0, 0, 0],
            [-1, 0, -1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [-1, 0, 0, -1, 0]
        ])

        collection = MetricsCollection(metrics, is_directed=True, n_nodes=n_nodes)

        # The collinearity fixer is supposed to remove one attribute from the NumberOfEdgesTypesDirected metric
        # self.assertEqual(len(collection.metrics[1]._indices_to_ignore), 1)
        self.assertEqual(np.sum(collection.metrics[1]._indices_to_ignore), 1)

        idx_to_ignore = np.where(collection.metrics[1]._indices_to_ignore)[0][0]

        res = collection.calculate_change_scores_all_edges(W1)

        # Deleting the 1+idx_to_ignore because the first entry is the NumberOfEdgesDirected metric
        expected_full_change_score = np.delete(expected_full_change_score, 1 + idx_to_ignore, axis=1)
        self.assertTrue(np.all(expected_full_change_score == res))

    def test_prepare_mple_data(self):
        types = ['A', 'B', 'A', 'B']
        metrics = [NumberOfEdgesDirected(), NumberOfEdgesTypesDirected(types)]

        W1 = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        n_nodes = W1.shape[0]
        expected_flattened_mat = np.array([1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1]).reshape(n_nodes ** 2 - n_nodes, 1)

        expected_mple_regressors = np.array([
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0]
        ])

        collection = MetricsCollection(metrics, is_directed=True, n_nodes=n_nodes)

        # The collinearity fixer is supposed to remove one attribute from the NumberOfEdgesTypesDirected metric
        self.assertEqual(np.sum(collection.metrics[1]._indices_to_ignore), 1)

        idx_to_ignore = np.where(collection.metrics[1]._indices_to_ignore)[0][0]

        Xs_full = collection.prepare_mple_regressors(W1)
        ys_full = collection.prepare_mple_labels(W1[..., np.newaxis])

        Xs_half = collection.prepare_mple_regressors(W1, edges_indices_lims=(0, expected_mple_regressors.shape[0] // 2))
        ys_half = collection.prepare_mple_labels(
            W1[..., np.newaxis],
            edges_indices_lims=(0, expected_mple_regressors.shape[0] // 2)
        )

        # Deleting the 1+idx_to_ignore because the first entry is the NumberOfEdgesDirected metric
        expected_mple_regressors = np.delete(expected_mple_regressors, 1 + idx_to_ignore, axis=1)
        self.assertTrue(np.all(expected_mple_regressors == Xs_full))
        self.assertTrue(np.all(expected_mple_regressors[:expected_mple_regressors.shape[0] // 2] == Xs_half))
        self.assertTrue(np.all(expected_flattened_mat == ys_full))
        self.assertTrue(np.all(expected_flattened_mat[:expected_mple_regressors.shape[0] // 2] == ys_half))

    def test_prepare_mple_reciprocity_data(self):
        W = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        metrics = [NumberOfEdgesDirected(), OutDegree(), InDegree(), TotalReciprocity()]
        n_nodes = W.shape[0]

        expected_statistics = [7, 2, 2, 2, 2, 1, 2, 3] # just a sanity check for the actual statistics

        collection = MetricsCollection(metrics, is_directed=True, n_nodes=n_nodes)
        statistics = collection.calculate_statistics(W)
        self.assertTrue(np.all(statistics == expected_statistics))

        X = collection.prepare_mple_reciprocity_regressors()
        y = collection.prepare_mple_reciprocity_labels(expand_net_dims(W))

        # expected_X is an array of shape (6, 4, 8) - 6 dyads, 4 options per dyad, 8 p1 features after collinearity_fixer (10 before)
        # For each dyad we calculate its changescore for all 4 options, on all p1 features (i.e. a (4,8) matrix)
        dyad_1_2_X = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [2, 1, 0, 0, 1, 0, 0, 1],
        ])

        dyad_1_3_X = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [2, 0, 1, 0, 0, 1, 0, 1],
        ])

        dyad_1_4_X = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 0],
            [2, 0, 0, 1, 0, 0, 1, 1],
        ])

        dyad_2_3_X = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0],
            [2, 1, 1, 0, 1, 1, 0, 1],
        ])

        dyad_2_4_X = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 1, 0, 0, 0],
            [2, 1, 0, 1, 1, 0, 1, 1],
        ])

        dyad_3_4_X = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 0],
            [2, 0, 1, 1, 0, 1, 1, 1],
        ])

        expected_X = np.array([dyad_1_2_X, dyad_1_3_X, dyad_1_4_X, dyad_2_3_X, dyad_2_4_X, dyad_3_4_X])
        self.assertTrue(np.all(X == expected_X))

        # expected_dyads is an array of 4choose2 dyads, with a one-hot encoding of length 4.
        # 0 = empty, 1 = i->j , 2 = j->i , 3=reciprocal
        expected_dyads = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ])

        self.assertTrue(np.all(y == expected_dyads))


    def test_get_parameter_names(self):
        n = 18
        metrics = [NumberOfEdgesDirected(), InDegree(), OutDegree()]
        collection = MetricsCollection(metrics, is_directed=True, n_nodes=n)

        param_names = collection.get_parameter_names()

        expected_names = ()
        for metric in metrics:
            metric_name = str(metric)
            if metric_name == "num_edges_directed":
                expected_names += (metric_name,)
            elif metric_name == "indegree" or metric_name == "outdegree":
                for i in range(1, n):
                    expected_names += (f"{metric_name}_{i + 1}",)

        self.assertTrue(param_names == expected_names)

        n = 4
        metrics = [NumberOfEdgesTypesDirected(['A', 'B', 'A', 'B'])]
        collection = MetricsCollection(metrics, is_directed=True, n_nodes=n)

        param_names = collection.get_parameter_names()

        expected_names = ("num_edges_between_types_directed_A__A", "num_edges_between_types_directed_A__B",
                          "num_edges_between_types_directed_B__A", "num_edges_between_types_directed_B__B")
        self.assertTrue(param_names == expected_names)

    def test_get_ignored_features(self):
        n = 18
        metrics = [NumberOfEdgesDirected(), InDegree(), OutDegree()]
        collection = MetricsCollection(metrics, is_directed=True, n_nodes=n)

        ignored_features = collection.get_ignored_features()
        expected_ignored_features = ("indegree_1", "outdegree_1")
        self.assertTrue(ignored_features == expected_ignored_features)

    @patch('pyERGM.metrics.split_network_for_bootstrapping')
    def test_bootstrapped_features(self, mock_split_network_for_bootstrapping):
        W = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])

        W_symmetric = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 1, 0]
        ])

        mock_split_network_for_bootstrapping.return_value = (
            np.array([0, 2]).reshape((2, 1)), np.array([1, 3]).reshape((2, 1)))

        metrics = [NumberOfEdgesDirected()]
        metrics_collection = MetricsCollection(metrics, is_directed=True, n_nodes=W.shape[0])
        bootstrapped_features = metrics_collection.bootstrap_observed_features(W, 1)
        expected_bootstrapped_features = np.array([6]).reshape(1, 1)  # np.sum(W[[0,2],[0,2].T]) / 2 * 12
        self.assertTrue(np.all(bootstrapped_features == expected_bootstrapped_features))

        metrics = [NumberOfEdgesUndirected()]
        metrics_collection = MetricsCollection(metrics, is_directed=False, n_nodes=W_symmetric.shape[0])
        bootstrapped_features = metrics_collection.bootstrap_observed_features(W_symmetric, 1)
        expected_bootstrapped_features = np.array([0]).reshape(1, 1)  # nodes 0 and 2 are not connected
        self.assertTrue(np.all(bootstrapped_features == expected_bootstrapped_features))

        metrics = [TotalReciprocity()]
        metrics_collection = MetricsCollection(metrics, is_directed=True, n_nodes=W.shape[0])
        bootstrapped_features = metrics_collection.bootstrap_observed_features(W, 1)
        expected_bootstrapped_features = np.array([0]).reshape(1, 1)
        self.assertTrue(np.all(bootstrapped_features == expected_bootstrapped_features))

        metrics = [UndirectedDegree()]
        metrics_collection = MetricsCollection(metrics, is_directed=False, n_nodes=W_symmetric.shape[0])
        bootstrapped_features = metrics_collection.bootstrap_observed_features(W_symmetric, 1)
        expected_bootstrapped_features = np.array([0, 3, 0, 3]).reshape(4,
                                                                        1)  # nodes 0,2 are not connected, and nodes 1,3 are.
        self.assertTrue(np.all(bootstrapped_features == expected_bootstrapped_features))

        metrics = [InDegree()]
        metrics_collection = MetricsCollection(metrics, is_directed=True, n_nodes=W.shape[0])
        bootstrapped_features = metrics_collection.bootstrap_observed_features(W, 1)
        expected_bootstrapped_features = np.array([3, 3, 0, 3]).reshape(4,
                                                                        1)  # e.g., the first is given by np.sum(W[[0,2],[0,2].T], axis=0) / (2-1) * (4-1)
        self.assertTrue(np.all(bootstrapped_features == expected_bootstrapped_features))

        metrics = [NumberOfEdgesDirected(), OutDegree()]
        metrics_collection = MetricsCollection(metrics, is_directed=True, n_nodes=W.shape[0])
        bootstrapped_features = metrics_collection.bootstrap_observed_features(W, 1)
        expected_bootstrapped_features = np.array([6, 3, 3, 3]).reshape(4,
                                                                        1)  # Ignoring the out-degree of the first node
        self.assertTrue(np.all(bootstrapped_features == expected_bootstrapped_features))

        mock_split_network_for_bootstrapping.return_value = (
            np.array([2, 3]).reshape((2, 1)), np.array([0, 1]).reshape((2, 1)))
        metrics = [TotalReciprocity()]
        metrics_collection = MetricsCollection(metrics, is_directed=True, n_nodes=W.shape[0])
        bootstrapped_features = metrics_collection.bootstrap_observed_features(W, 1)
        expected_bootstrapped_features = np.array([6]).reshape(1, 1)
        self.assertTrue(np.all(bootstrapped_features == expected_bootstrapped_features))
