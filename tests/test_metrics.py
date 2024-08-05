import unittest
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
        collection = MetricsCollection(metrics, is_directed=True)

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
        self.assertEqual(receiver.get_effective_feature_count(n), n)

        receiver = InDegree(base_idx=1)
        n = 18
        self.assertEqual(receiver.get_effective_feature_count(n), n-1)

        sender = OutDegree()
        n = 18
        self.assertEqual(sender.get_effective_feature_count(n), n)

        sender = OutDegree(base_idx=1)
        n = 18
        self.assertEqual(sender.get_effective_feature_count(n), n-1)
    
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

        self.assertTrue(np.all(indegree == expected_in_degrees))

        expected_out_degrees = list(dict(G.out_degree()).values())

        sender = OutDegree()
        outdegree = sender.calculate(W)

        self.assertTrue(np.all(outdegree == expected_out_degrees))


class TestMetricsCollection(unittest.TestCase):
    def test_metrics_setup(self):
        metrics = [NumberOfEdgesDirected(), TotalReciprocity(), InDegree()]
        collection = MetricsCollection(metrics, is_directed=True)
        
        expected_metric_names = tuple([str(NumberOfEdgesDirected()), str(TotalReciprocity()), str(InDegree())])

        self.assertTrue(np.all(collection.metric_names == expected_metric_names))

    def test_collinearity_fixer(self):
        n = 18

        test_scenarios = {
            "num_edges__total_reciprocity__indegree": {
                "metrics": [NumberOfEdgesDirected(), TotalReciprocity(), InDegree()],
                "expected_num_of_features": 1 + 1 + n-1, # InDegree is trimmed by 1 because of collinearity with num_of_edges
                "expected_trimmed_metrics": [str(InDegree())]
            },
            "indegree": {
                "metrics": [InDegree()],
                "expected_num_of_features": n, # There is no collinearity since InDegree() is the only feature
                "expected_trimmed_metrics": []
            },
            "outdegree": {
                "metrics": [OutDegree()],
                "expected_num_of_features": n, # There is no collinearity since InDegree() is the only feature
                "expected_trimmed_metrics": []
            },
            "in_outdegree": {
                "metrics": [InDegree(), OutDegree()],
                "expected_num_of_features": n+n-1, # There is no collinearity since InDegree() is the only feature
                "expected_trimmed_metrics": [str(InDegree())]
            },
            "num_edges__total_reciprocity__indegree_outdegree": {
                "metrics": [NumberOfEdgesDirected(), TotalReciprocity(), InDegree(), OutDegree()],
                "expected_num_of_features": 1 + 1 + n-1 + n-1,
                "expected_trimmed_metrics": [str(InDegree()), str(OutDegree())]
            }
            
        }

        for scenario_data in test_scenarios.values():
            collection = MetricsCollection(scenario_data["metrics"], is_directed=True, fix_collinearity=True)
            
            # Check the general number of features matches the expectation
            self.assertEqual(collection.get_num_of_features(n), scenario_data["expected_num_of_features"])

            # Check if the correct metrics were trimmed
            for metric in scenario_data["metrics"]:
                if str(metric) in scenario_data["expected_trimmed_metrics"]:
                    self.assertEqual(collection.get_metric(str(metric)).base_idx, 1)
                elif hasattr(metric, "base_idx"): # Only metrics with a base_idx can be trimmed. The others aren't tested
                    self.assertEqual(collection.get_metric(str(metric)).base_idx, 0)

    def test_calculate_statistics(self):
        # Test undirected graphs
        metrics = [NumberOfEdgesUndirected(), NumberOfTriangles()]
        collection = MetricsCollection(metrics, is_directed=False)

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
        collection = MetricsCollection(metrics, is_directed=True)
        W = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 0]
        ])

        stats = collection.calculate_statistics(W)
        expected_stats = np.array([4])

        np.testing.assert_array_equal(stats, expected_stats)

    def test_get_num_of_features(self):
        metrics = [NumberOfEdgesUndirected(), NumberOfTriangles()]
        collection = MetricsCollection(metrics, is_directed=False)
        num_of_features = collection.get_num_of_features(n=4)

        # NumberOfEdges and NumberOfTriangles each produce 1 features
        expected_num_of_features = 2

        self.assertEqual(num_of_features, expected_num_of_features)

        metrics = [Reciprocity(), NumberOfEdgesDirected()]
        collection = MetricsCollection(metrics, is_directed=True)

        n = 4
        num_of_features = collection.get_num_of_features(n)

        expected_num_of_features = math.comb(n, 2) + 1

        self.assertEqual(num_of_features, expected_num_of_features)
    
    def test_calc_change_scores(self):
        metrics = [NumberOfEdgesDirected(), Reciprocity()]
        collection = MetricsCollection(metrics, is_directed=True)

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
        result = collection.calc_change_scores(W1, W2, indices=flipped_indices)
        
        # 1st is -1 because we lost an edge, and 3rd entry is -1 because node #2 lost it's reciprocity
        expected_result = [-1, 0, -1, 0] 

        self.assertTrue(np.all(result == expected_result))

