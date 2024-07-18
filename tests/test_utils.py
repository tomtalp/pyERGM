import unittest
from utils import *

class TestNetworkStatistics(unittest.TestCase):
    # def setUp(self):
    #     pass      
        
    def test_initialization_params_validation(self):
        with self.assertRaises(ValueError):
            net_stats = NetworkStatistics()

        with self.assertRaises(ValueError):
            net_stats = NetworkStatistics(metric_names=["fake"])
        
        with self.assertRaises(ValueError):
            net_stats = NetworkStatistics(metric_names=["num_edges"], custom_metrics={"fake": 1})

    def test_calculate_statistics(self):
        net_stats = NetworkStatistics(metric_names=["num_edges", "num_triangles"], directed=False)
        W = np.array([[0, 1, 1], 
                      [1, 0, 1], 
                      [1, 1, 0]])
        stats = net_stats.calculate_statistics(W)
        self.assertEqual(stats[0], 3)
        self.assertEqual(stats[1], 1)

        # Assert that num_triangles can't be run on a directed graph
        with self.assertRaises(ValueError): 
            net_stats = NetworkStatistics(metric_names=["num_edges", "num_triangles"], directed=True)

        net_stats = NetworkStatistics(metric_names=["num_edges"], directed=True)
        W = np.array([[0, 1, 1], 
                      [1, 0, 0], 
                      [1, 0, 1]])
        
        stats = net_stats.calculate_statistics(W)
        self.assertEqual(stats[0], 5)

        net_stats = NetworkStatistics(metric_names=["num_edges", "num_triangles"], directed=False)
        W = np.array([[0, 0, 0], 
                      [0, 0, 0], 
                      [0, 0, 0]])
        stats = net_stats.calculate_statistics(W)
        self.assertEqual(stats[0], 0)
        self.assertEqual(stats[1], 0)
    
    def test_get_num_of_statistics(self):
        net_stats = NetworkStatistics(metric_names=["num_edges"])
        self.assertEqual(net_stats.get_num_of_statistics(), 1)

        net_stats = NetworkStatistics(metric_names=["num_edges", "num_triangles"])
        self.assertEqual(net_stats.get_num_of_statistics(), 2)

        

        

