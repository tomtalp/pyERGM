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
        stats_calculator = MetricsCollection([NumberOfEdgesUndirected()], is_directed=False, n_nodes=4)
        thetas = np.array([np.log(2)])

        # UNDIRECTED VERSION
        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, network_stats_calculator=stats_calculator)

        test_W = np.array([
            [0., 0., 0., 1.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.]
        ])

        node_i = 0
        node_j = 2
        W_plus = test_W.copy()
        sampler._flip_network_edge(W_plus, node_i, node_j)

        expected_W = np.array([
            [0., 0., 1., 1.],
            [0., 0., 1., 0.],
            [1., 1., 0., 0.],
            [1., 0., 0., 0.]
        ])

        self.assertTrue(np.all(W_plus == expected_W))

        node_i = 0
        node_j = 3
        W_minus = test_W.copy()
        sampler._flip_network_edge(W_minus, node_i, node_j)

        expected_W = np.array([
            [0., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.]
        ])

        self.assertTrue(np.all(W_minus == expected_W))

        # DIRECTED VERSION
        stats_calculator = MetricsCollection([NumberOfEdgesDirected()], is_directed=True, n_nodes=2)
        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, network_stats_calculator=stats_calculator)

        test_W = np.array([[0, 1], [0, 0]])
        node_i = 1
        node_j = 0
        W_plus = test_W.copy()
        sampler._flip_network_edge(W_plus, node_i, node_j)
        expected_W = np.array([[0, 1], [1, 0]])

        self.assertTrue(np.all(W_plus == expected_W))

        node_i = 0
        node_j = 1
        W_minus = test_W.copy()
        sampler._flip_network_edge(W_minus, node_i, node_j)
        expected_W = np.array([[0, 0], [0, 0]])

        self.assertTrue(np.all(W_minus == expected_W))

    def test__calculate_weighted_change_score_undirected_single_variable(self):
        """
        Test the change score calculation for an undirected graph, based on a single variable - num_edges
        """
        stats_calculator = MetricsCollection([NumberOfEdgesUndirected()], is_directed=False, n_nodes=3)

        theta_edges = 0.5
        thetas = np.array([theta_edges])

        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, network_stats_calculator=stats_calculator)

        current_W = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0]
        ])

        change_score = sampler._calculate_weighted_change_score(current_W, (0, 1))
        expected_change_score = 1 * theta_edges
        self.assertEqual(change_score, expected_change_score)

    def test__calculate_weighted_change_score_undirected_multiple_variables(self):
        """
        Test the change score calculation for a undirected graph, based on two variables - num_edges & num_triangles
        """
        stats_calculator = MetricsCollection([NumberOfEdgesUndirected(), NumberOfTriangles()], is_directed=False, n_nodes=3)

        theta_edges = 2
        theta_triangles = 0.5
        thetas = np.array([theta_edges, theta_triangles])

        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, network_stats_calculator=stats_calculator)

        current_W = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0]
        ])

        change_score = sampler._calculate_weighted_change_score(current_W, (0, 1))

        changed_edges = 1
        changed_triangles = 1
        expected_change_score = changed_edges * theta_edges + changed_triangles * theta_triangles

        self.assertEqual(change_score, expected_change_score)

    def test__calculate_weighted_change_score_directed(self):
        """
        Test the change score calculation for a directed graph, based on num_edges
        """
        stats_calculator = MetricsCollection([NumberOfEdgesDirected()], is_directed=True, n_nodes=3)

        theta_edges = -1
        thetas = np.array([theta_edges])

        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, network_stats_calculator=stats_calculator)
        
        current_W = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 0]
        ])

        current_W_2 = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 0]
        ])

        total_change_score = (sampler._calculate_weighted_change_score(current_W, (0, 1)) +
                              sampler._calculate_weighted_change_score(current_W_2, (2, 0)))

        changed_edges = 2
        expected_change_score = changed_edges * theta_edges

        self.assertEqual(total_change_score, expected_change_score)
