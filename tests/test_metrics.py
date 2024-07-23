import unittest
from utils import *
from metrics import *

import networkx as nx

class TestNumberOfEdges(unittest.TestCase):  
    def test_num_of_edges(self):
        metric = NumberOfEdges()
        W = np.array([
                [0, 1, 1], 
                [1, 0, 1], 
                [1, 1, 0]
            ])
        
        result = metric.calculate(W, is_directed=False)
        expected_result = 3

        self.assertEqual(result, expected_result)
    
        W = np.array([
                [0, 1, 0], 
                [1, 0, 1], 
                [1, 0, 0]
            ])
        
        result = metric.calculate(W, is_directed=True)
        expected_result = 4

        self.assertEqual(result, expected_result)

class TestNumberOfTriangles(unittest.TestCase):
    def test_validation(self):
        metric = NumberOfTriangles()

        self.assertTrue(metric.requires_graph)

        W = np.array([
                [0, 1, 0], 
                [1, 0, 1], 
                [1, 0, 0]
            ])
        
        G = nx.from_numpy_array(W, create_using=nx.DiGraph)

        with self.assertRaises(ValueError):
            metric.calculate(G, is_directed=False)
        
        G = nx.from_numpy_array(W)
        with self.assertRaises(ValueError):
            metric.calculate(G, is_directed=True)

    def test_num_of_triangles(self):
        metric = NumberOfTriangles()
        W = np.array([
                [0, 1, 1], 
                [1, 0, 1], 
                [1, 1, 0]
            ])
        
        G = nx.from_numpy_array(W)

        result = metric.calculate(G, is_directed=False)
        expected_result = 1

        self.assertEqual(result, expected_result)
    
class TestMetricsCollection(unittest.TestCase):
    def test_calculate_statistics(self):
        ## Test undirected graphs
        metrics = [NumberOfEdges(), NumberOfTriangles()]
        collection = MetricsCollection(metrics, is_directed=False)

        W = np.array([
                [0, 1, 1], 
                [1, 0, 1], 
                [1, 1, 0]
            ])

        stats = collection.calculate_statistics(W)
        expected_stats = np.array([3, 1])

        np.testing.assert_array_equal(stats, expected_stats)

        ## Test directed graphs
        metrics = [NumberOfEdges()]
        collection = MetricsCollection(metrics, is_directed=True)
        W = np.array([
                [0, 1, 0], 
                [1, 0, 1], 
                [1, 0, 0]
            ])

        stats = collection.calculate_statistics(W)
        expected_stats = np.array([4])

        np.testing.assert_array_equal(stats, expected_stats)