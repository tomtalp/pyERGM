import unittest

import numpy as np

from utils import *
from metrics import *

import networkx as nx
import math


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
    def test_get_effective_feature_count(self):
        receiver = InDegree()
        n = 18
        receiver._n_nodes = n
        self.assertEqual(receiver._get_effective_feature_count(), n)

        receiver = InDegree(indices_to_ignore=[0])
        n = 18
        receiver._n_nodes = n
        self.assertEqual(receiver._get_effective_feature_count(), n - 1)

        sender = OutDegree()
        n = 18
        sender._n_nodes = n
        self.assertEqual(sender._get_effective_feature_count(), n)

        sender = OutDegree(indices_to_ignore=[0])
        n = 18
        sender._n_nodes = n
        self.assertEqual(sender._get_effective_feature_count(), n - 1)

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
    def test_get_num_weight_mats(self):
        neuronal_types = ['A', 'B', 'C']
        metric = NumberOfEdgesTypesDirected(neuronal_types)
        self.assertTrue(metric._get_num_weight_mats() == 9)

        neuronal_types = ['A', 'B', 'C', 'A', 'B', 'C']
        metric = NumberOfEdgesTypesDirected(neuronal_types)
        self.assertTrue(metric._get_num_weight_mats() == 9)

    def test_calc_edge_weights(self):
        neuronal_types = ['A', 'B', 'B', 'A']
        expected_edge_weights = np.array([
            # A->A connections
            [[0, 0, 0, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 0]],

            # A->B connections
            [[0, 1, 1, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 1, 0]],

            # B->A connections
            [[0, 0, 0, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 0, 0, 0]],

            # B->B connections
            [[0, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 0]]
        ])

        metric = NumberOfEdgesTypesDirected(neuronal_types)

        self.assertTrue(np.all(metric.edge_weights == expected_edge_weights))

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


class TestMetricsCollection(unittest.TestCase):
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
                # InDegree is trimmed by 1 because of collinearity with num_of_edges
                "expected_trimmed_metrics": {str(InDegree()): [0]},
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
                "expected_trimmed_metrics": {str(InDegree()): [0]},
                "expected_eliminated_metrics": []
            },
            "num_edges__total_reciprocity__indegree_outdegree": {
                "metrics": [NumberOfEdgesDirected(), TotalReciprocity(), InDegree(), OutDegree()],
                "expected_num_of_features": 1 + 1 + n - 1 + n - 1,
                "expected_trimmed_metrics": {str(InDegree()): [0], str(OutDegree()): [0]},
                "expected_eliminated_metrics": []
            },
            "num_edges__exogenous_types": {
                "metrics": [NumberOfEdgesDirected(), NumberOfEdgesTypesDirected(neuronal_types)],
                "expected_num_of_features": 1 + len(set(neuronal_types)) ** 2 - 2,
                "expected_trimmed_metrics": {str(NumberOfEdgesTypesDirected(neuronal_types)): [0, 10]},
                "expected_eliminated_metrics": []
            },
            "sum_attr_both__sum_attr_in__sum_attr_out": {
                "metrics": [NodeAttrSum(np.arange(n), is_directed=True), NodeAttrSumIn(np.arange(n)),
                            NodeAttrSumOut(np.arange(n))],
                "expected_num_of_features": 2,
                "expected_trimmed_metrics": {},
                "expected_eliminated_metrics": []
            }
        }

        # In a case where the collinearity fixer is going to eliminate a specific metric, we want to verify that the
        # correct metric was removed by checking the pointer of the removed metric (since some metrics might have the same name).
        # For this reason, we're going to put the metric that's going to be removed in a list and check if the pointer
        test_scenarios["num_edges_twice"]["expected_eliminated_metrics"] = [test_scenarios["num_edges_twice"]["metrics"][0]]
        test_scenarios["sum_attr_both__sum_attr_in__sum_attr_out"]["expected_eliminated_metrics"] = [test_scenarios["sum_attr_both__sum_attr_in__sum_attr_out"]["metrics"][0]]

        for scenario_data in test_scenarios.values():
            collection = MetricsCollection(scenario_data["metrics"], is_directed=True, fix_collinearity=True, n_nodes=n)

            # Check the general number of features matches the expectation
            self.assertEqual(collection.num_of_features, scenario_data["expected_num_of_features"])

            # Check if the correct metrics were trimmed
            for metric in scenario_data["metrics"]:
                if str(metric) in scenario_data["expected_trimmed_metrics"].keys():
                    self.assertEqual(set(collection.get_metric(str(metric))._indices_to_ignore),
                                     set(scenario_data["expected_trimmed_metrics"][str(metric)]))
                elif hasattr(metric,
                             "_indices_to_ignore"):  # Only metrics with a base_idx can be trimmed. The others aren't tested
                    self.assertEqual(collection.get_metric(str(metric))._indices_to_ignore, [])
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
        result = collection.calc_change_scores(W1, indices=flipped_indices)

        # 1st is -1 because we lost an edge, and 3rd entry is -1 because node #2 lost it's reciprocity
        expected_result = [-1, 0, -1, 0]

        self.assertTrue(all(result == expected_result))
