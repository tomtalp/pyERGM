import unittest
import numpy as np

from pyERGM.utils import *
from pyERGM.metrics import *
from pyERGM import sampling


class Test_MetropolisHastings(unittest.TestCase):
    def setUp(self):
        pass

    def test_flip_network_edge(self):
        stats_calculator = MetricsCollection([NumberOfEdgesUndirected()], is_directed=False, n_nodes=4)
        thetas = np.array([np.log(2)])

        # UNDIRECTED VERSION
        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, metrics_collection=stats_calculator)

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
        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, metrics_collection=stats_calculator)

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

        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, metrics_collection=stats_calculator)

        current_W = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0]
        ])

        change_score = sampler._calculate_weighted_change_score(current_W, {'edge': (0, 1)})
        expected_change_score = 1 * theta_edges
        self.assertEqual(change_score, expected_change_score)

    def test__calculate_weighted_change_score_undirected_multiple_variables(self):
        """
        Test the change score calculation for a undirected graph, based on two variables - num_edges & num_triangles
        """
        stats_calculator = MetricsCollection([NumberOfEdgesUndirected(), NumberOfTriangles()], is_directed=False,
                                             n_nodes=3)

        theta_edges = 2
        theta_triangles = 0.5
        thetas = np.array([theta_edges, theta_triangles])

        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, metrics_collection=stats_calculator)

        current_W = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0]
        ])

        change_score = sampler._calculate_weighted_change_score(current_W, {'edge': (0, 1)})

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

        sampler = sampling.NaiveMetropolisHastings(thetas=thetas, metrics_collection=stats_calculator)

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

        total_change_score = (sampler._calculate_weighted_change_score(current_W, {'edge': (0, 1)}) +
                              sampler._calculate_weighted_change_score(current_W_2, {'edge': (2, 0)}))

        changed_edges = 2
        expected_change_score = changed_edges * theta_edges

        self.assertEqual(total_change_score, expected_change_score)

    def test__calc_edge_influence_on_features(self):
        set_seed(5678234)
        n = 10
        adj_mat = generate_erdos_renyi_matrix(n, 0.5, is_directed=False)
        sampler = sampling.NaiveMetropolisHastings(np.zeros(n),
                                                   MetricsCollection([NumberOfEdgesDirected(), OutDegree()],
                                                                     is_directed=True, n_nodes=n))
        # The contribution of degree is n, and of number of edges is 1, for each edge
        expected_total_influence = (1 + n) * np.ones(n * (n - 1))
        # The collinearity fixed removed the out degree of the first node, so no contribution for its edges.
        expected_total_influence[:n - 1] = 1
        total_influence = sampler._calc_edge_influence_on_features(adj_mat)
        self.assertTrue(np.all(expected_total_influence == total_influence))

    def test_sample_non_uniform_proposals_smoke_test(self):
        set_seed(5678234)
        n = 10
        adj_mat = generate_erdos_renyi_matrix(n, 0.5, is_directed=False)
        sampler = sampling.NaiveMetropolisHastings(np.zeros(n),
                                                   MetricsCollection([NumberOfEdgesDirected(), OutDegree()],
                                                                     is_directed=True, n_nodes=n))
        sampler.sample(initial_state=adj_mat, num_of_nets=10, edge_proposal_method='features_influence__sum')
        sampler.sample(initial_state=adj_mat, num_of_nets=10, edge_proposal_method='features_influence__softmax')

    def test_custom_edge_proposal_distribution_sampling(self):
        """
        Verify that get_custom_distribution_random_edges_to_flip samples
        edges according to the specified probability distribution.

        Approach:
        1. Create a model with non-uniform edge influences (using OutDegree)
        2. Calculate the proposal distribution
        3. Sample many edges
        4. Compare empirical frequencies to theoretical probabilities using chi-squared test
        """
        from scipy.stats import chi2

        set_seed(12345)
        n = 5
        is_directed = True
        # OutDegree creates non-uniform influence: edges from node i affect
        # node i's out-degree differently than other nodes
        metrics_collection = MetricsCollection(
            [NumberOfEdgesDirected(), OutDegree()],
            is_directed=is_directed,
            n_nodes=n
        )

        sampler = sampling.NaiveMetropolisHastings(
            thetas=np.zeros(metrics_collection.num_of_features),
            metrics_collection=metrics_collection
        )

        adj_mat = generate_erdos_renyi_matrix(n, 0.5, is_directed=is_directed)
        # Test both normalization methods
        for method in ['features_influence__sum', 'features_influence__softmax']:
            if method == 'features_influence__sum':
                sampler._calc_proposal_dist_features_influence__sum(adj_mat)
            else:
                sampler._calc_proposal_dist_features_influence__softmax(adj_mat)

            edge_probs = sampler._edge_proposal_dists[method]

            # Verify valid probability distribution
            self.assertAlmostEqual(edge_probs.sum(), 1.0, places=10)
            self.assertTrue(np.all(edge_probs >= 0))

            # Sample many edges
            num_samples = 100000
            sampled_edges = get_custom_distribution_random_edges_to_flip(num_samples, edge_probs, is_directed=True)

            # Convert (i,j) pairs back to flat-no-diagonal indices
            flat_indices = sampled_edges[0] * (n - 1) + sampled_edges[1]
            flat_indices[sampled_edges[1] > sampled_edges[0]] -= 1

            # Count empirical frequencies
            empirical_counts = np.bincount(flat_indices, minlength=n*(n-1))

            # Chi-squared test for goodness of fit
            expected_counts = edge_probs * num_samples
            # Only test bins with expected count > 5 (chi-squared assumption)
            valid_bins = expected_counts > 5
            self.assertGreater(valid_bins.sum(), 1,
                f"Not enough bins with expected count > 5 for chi-squared test ({valid_bins.sum()} bins)")

            chi_squared = np.sum((empirical_counts[valid_bins] - expected_counts[valid_bins])**2
                                 / expected_counts[valid_bins])
            dof = valid_bins.sum() - 1

            p_value = 1 - chi2.cdf(chi_squared, dof)

            self.assertGreater(p_value, 0.01,
                f"Chi-squared test failed for {method}: chi2={chi_squared:.2f}, dof={dof}, p={p_value:.4f}")

    def test_proposal_distribution_creates_non_uniform_probs(self):
        """
        Verify that the proposal distribution is actually non-uniform when using
        metrics that create different influences for different edges.
        """
        set_seed(67890)
        n = 6

        # OutDegree should create non-uniform influence since the first node's
        # OutDegree is removed due to collinearity, making edges from node 0
        # have different total influence than edges from other nodes
        metrics_collection = MetricsCollection(
            [NumberOfEdgesDirected(), OutDegree()],
            is_directed=True,
            n_nodes=n
        )

        sampler = sampling.NaiveMetropolisHastings(
            thetas=np.zeros(metrics_collection.num_of_features),
            metrics_collection=metrics_collection
        )

        adj_mat = generate_erdos_renyi_matrix(n, 0.5, is_directed=True)

        sampler._calc_proposal_dist_features_influence__softmax(adj_mat)
        edge_probs = sampler._edge_proposal_dists['features_influence__softmax']

        # The distribution should NOT be uniform
        uniform_prob = 1.0 / (n * (n - 1))
        max_deviation_from_uniform = np.abs(edge_probs - uniform_prob).max()

        # With OutDegree, we expect meaningful deviation from uniform
        self.assertGreater(max_deviation_from_uniform, 0.001,
            "Expected non-uniform distribution but got approximately uniform")
