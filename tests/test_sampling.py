import unittest
import numpy as np
import networkx as nx

from utils import *
from metrics import *
from ergm import ERGM, BruteForceERGM
import sampling

class Test_MetropolisHastings(unittest.TestCase):  
    def setUp(self):
        pass

    def test_flip_network_edge(self):
        stats_calculator = MetricsCollection([NumberOfEdges()], is_directed=False)
        thetas = np.array([np.log(2)])

        ## UNDIRECTED VERSION
        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, network_stats_calculator=stats_calculator, is_directed=False)

        test_W = np.array([
            [0., 0., 0., 1.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.]
        ])

        node_i = 0
        node_j = 2
        W_plus = sampler.flip_network_edge(test_W, node_i, node_j)

        expected_W = np.array([
            [0., 0., 1., 1.],
            [0., 0., 1., 0.],
            [1., 1., 0., 0.],
            [1., 0., 0., 0.]
        ])

        self.assertTrue((W_plus == expected_W).all())

        node_i = 0
        node_j = 3
        W_minus = sampler.flip_network_edge(test_W, node_i, node_j)

        expected_W = np.array([
            [0., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.]
        ])

        self.assertTrue((W_minus == expected_W).all())

        ## DIRECTED VERSION
        stats_calculator = MetricsCollection([NumberOfEdges()], is_directed=True)
        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, network_stats_calculator=stats_calculator, is_directed=True)

        test_W = np.array([[0, 1], [0, 0]])
        node_i = 1
        node_j = 0
        W_plus = sampler.flip_network_edge(test_W, node_i, node_j)
        expected_W = np.array([[0, 1], [1, 0]])
        
        self.assertTrue((W_plus == expected_W).all())

        node_i = 0
        node_j = 1
        W_minus = sampler.flip_network_edge(test_W, node_i, node_j)
        expected_W = np.array([[0, 0], [0, 0]])

        self.assertTrue((W_minus == expected_W).all())
    
    def test__calculate_weighted_change_score_undirected_single_variable(self):
        """
        Test the change score calculation for an undirected graph, based on a single variable - num_edges
        """
        stats_calculator = MetricsCollection([NumberOfEdges()], is_directed=False)

        theta_edges = 0.5
        thetas = np.array([theta_edges])

        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, network_stats_calculator=stats_calculator, is_directed=False)

        test_W_plus = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])

        test_W_minus = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0]
        ])

        change_score = sampler._calculate_weighted_change_score(test_W_plus, test_W_minus)
        expected_change_score = 1*theta_edges
        self.assertEqual(change_score, expected_change_score)
        
    def test__calculate_weighted_change_score_undirected_multiple_variables(self):
        """
        Test the change score calculation for a undirected graph, based on two variables - num_edges & num_triangles
        """
        stats_calculator = MetricsCollection([NumberOfEdges(), NumberOfTriangles()], is_directed=False)

        theta_edges = 2
        theta_triangles = 0.5
        thetas = np.array([theta_edges, theta_triangles])

        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, network_stats_calculator=stats_calculator, is_directed=False)

        test_W_plus = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])

        test_W_minus = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0]
        ])

        change_score = sampler._calculate_weighted_change_score(test_W_plus, test_W_minus)
        
        changed_edges = 1
        changed_triangles = 1
        expected_change_score = changed_edges*theta_edges + changed_triangles*theta_triangles

        self.assertEqual(change_score, expected_change_score)

    def test__calculate_weighted_change_score_directed(self):
        """
        Test the change score calculation for a directed graph, based on num_edges
        """
        stats_calculator = MetricsCollection([NumberOfEdges()], is_directed=True)

        theta_edges = -1
        thetas = np.array([theta_edges])

        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, network_stats_calculator=stats_calculator, is_directed=True)

        test_W_plus = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])

        test_W_minus = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 0]
        ])

        change_score = sampler._calculate_weighted_change_score(test_W_plus, test_W_minus)
        
        changed_edges = 2
        expected_change_score = changed_edges*theta_edges

        self.assertEqual(change_score, expected_change_score)

    def normalization_approximation_benchmarks(self):
        np.random.seed(9873645)
        n = 4
        p = 0.25
        true_theta = -np.log(2)
        is_directed = False

        stats_calculator = MetricsCollection([NumberOfEdges()], is_directed=is_directed)
        
        true_model = BruteForceERGM(n,  stats_calculator, is_directed=is_directed, initial_thetas=[true_theta])
        true_norm_factor = true_model._normalization_factor
        print(f"Normalization factor for a network with {n} nodes, theta = {true_theta}: {true_norm_factor}")

        nets_for_approx = n*n*1000
        mcmc_steps_for_sample = n*10
        model = ERGM(n, stats_calculator, is_directed=is_directed, initial_thetas=[true_theta], n_networks_for_norm=nets_for_approx, n_mcmc_steps=mcmc_steps_for_sample)
        model._approximate_normalization_factor()
        approximated_normalization_factor = model._normalization_factor
        print(f"Approximated normalization factor: {approximated_normalization_factor}")

        print(f"Simulated network with {n} nodes, p={p}")
        W = nx.to_numpy_array(nx.erdos_renyi_graph(n, p, seed=9873645))
        print(W)
        
        true_proba = true_model.calculate_probability(W)
        estimated_proba = model.calculate_probability(W)

        print(f"True model params - thetas = {true_model._thetas}, Z = {true_model._normalization_factor}. Probability - {true_proba:.8f}")
        print(f"Approx. model params - thetas = {model._thetas}, Z = {model._normalization_factor}. Probability - {estimated_proba:.8f}")
        print("Done")