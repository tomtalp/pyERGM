import unittest
from utils import *
from ergm import ERGM

class TestERGM(unittest.TestCase):  
    def setUp(self):
        self.net_stats = NetworkStatistics(metric_names=["num_edges", "num_triangles"])      
        self.n_nodes = 3

        self.K = 100
        self.thetas = np.ones(self.net_stats.get_num_of_statistics())
        
    def test_calculate_weight(self):
        ergm = ERGM(self.n_nodes, self.net_stats)
        ergm.fit(precalculated_thetas=self.thetas, precalculated_normalization_factor=self.K)

        W = np.array([[0, 1, 1], 
                      [1, 0, 1], 
                      [1, 1, 0]])
        weight = ergm.calculate_weight(W)

        expected_num_edges = 3
        expected_num_triangles = 1
        expected_weight = np.exp(expected_num_edges*1 + expected_num_triangles*1)
        
        self.assertEqual(weight, expected_weight)

        W = np.array([[0, 0, 0], 
                      [0, 0, 0], 
                      [0, 0, 0]])
        weight = ergm.calculate_weight(W)

        expected_num_edges = 0
        expected_num_triangles = 0
        expected_weight = np.exp(expected_num_edges*1 + expected_num_triangles*1)
        
        self.assertEqual(weight, expected_weight)
    
    def test_calculate_probability(self):
        ergm = ERGM(self.n_nodes, self.net_stats)
        ergm.fit(precalculated_thetas=self.thetas, precalculated_normalization_factor=self.K)

        W = np.array([[0, 1, 1], 
                      [1, 0, 1], 
                      [1, 1, 0]])
        probability = ergm.calculate_probability(W)

        expected_num_edges = 3
        expected_num_triangles = 1
        expected_weight = np.exp(expected_num_edges*1 + expected_num_triangles*1)
        expected_probability = expected_weight / self.K
        
        self.assertEqual(probability, expected_probability)

    def test_calculate_probability_wiki_example(self):
        ergm = ERGM(self.n_nodes, self.net_stats)
        
        thetas = [-np.log(2), np.log(3)]
        K = 29 / 8

        ergm.fit(precalculated_thetas=thetas, precalculated_normalization_factor=K)

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
    
    # def test_sample_network(self):
    #     n_nodes = 6
    #     net_stats = NetworkStatistics(metric_names=["num_edges", "num_triangles"], directed=False)      

    #     ergm = ERGM(n_nodes, net_stats, is_directed=False)

    #     thetas = [-np.log(2), np.log(3)]
    #     K = 129 / 8

    #     ergm.fit(precalculated_thetas=thetas)

    #     sampled_net = ergm.sample_network(steps=30)
    #     print(sampled_net)

    # def test_sample_network_directed(self):
    #     # n_nodes = 6
    #     # net_stats = NetworkStatistics(metric_names=["num_edges"], directed=True)      

    #     # ergm = ERGM(n_nodes, net_stats, is_directed=True)

    #     # thetas = [-np.log(2)]
    #     # K = 129 / 8

    #     # ergm.fit(precalculated_thetas=thetas)

    #     # sampled_net = ergm.sample_network(steps=30)
    #     # print(sampled_net)