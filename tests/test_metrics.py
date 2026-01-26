import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from pyparsing import empty

from pyERGM.utils import *
from pyERGM.metrics import *
from pyERGM.datasets import sampson_matrix

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

    def test_masked_num_of_edges(self):
        metric = NumberOfEdgesUndirected()
        W1 = np.array([
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0]
        ])
        W2 = 1 - W1
        mask = np.array([
            [False, True, False],
            [True, False, True],
            [False, True, False]
        ])[np.triu_indices(3, 1)]

        result = metric.calculate_for_sample(np.stack([W1, W2], axis=-1), mask)
        expected_result = np.array([1, 1])
        self.assertTrue(np.array_equal(result, expected_result))


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

        mask = flatten_square_matrix_to_edge_list(
            np.array([
                [False, True, False],
                [True, False, True],
                [False, True, False]
            ]),
            True
        )
        result = metric.calculate_for_sample(sample, mask)
        expected_result = np.array([3, 2])
        self.assertTrue(np.all(result == expected_result))

    def test_calc_mple_regressors_masked(self):
        n = 10
        mask = flatten_square_matrix_to_edge_list(
            generate_binomial_tensor(n, 0, 1).astype(bool)[..., -1],
            True,
        ).reshape(-1, 1)
        m = NumberOfEdgesDirected()
        Xs = np.zeros((mask.sum(), 1))
        m.calculate_mple_regressors(Xs, np.array([0]), mask)
        self.assertTrue(np.allclose(Xs, np.ones(mask.sum())))


class TestNumberOfTriangles(unittest.TestCase):
    def test_validation(self):
        metric = NumberOfTriangles()

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

        # Generate random adjacency matrix
        np.random.seed(42)
        G_as_W = generate_erdos_renyi_matrix(6, 0.5, is_directed=False)

        # Calculate expected triangles directly from adjacency matrix
        # Number of triangles = trace(A^3) / 6 for undirected graphs
        total_num_triangles = np.trace(np.linalg.matrix_power(G_as_W, 3)) / 6

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
            cur_mat = generate_erdos_renyi_matrix(n, 0.5, is_directed=False)
            expected_triangles[i] = np.trace(np.linalg.matrix_power(cur_mat, 3)) / 6
            networks_sample[:, :, i] = cur_mat

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

        total_edges = np.sum(W)

        reciprocity_vector = collection.calculate_statistics(W)
        total_reciprocity = np.sum(reciprocity_vector)

        reciprocty_fraction = total_reciprocity / total_edges

        # Calculate reciprocity directly: fraction of edges that are reciprocated
        # Reciprocated edges are where both W[i,j] and W[j,i] are 1
        reciprocated_edges = np.sum(W * W.T) / 2
        expected_reciprocity = reciprocated_edges / total_edges
        self.assertEqual(reciprocty_fraction, expected_reciprocity)


class TestDegreeMetrics(unittest.TestCase):
    def test_calculate(self):
        W = np.array([
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 0]
        ])

        # Calculate in-degrees and out-degrees directly from adjacency matrix
        expected_in_degrees = np.sum(W, axis=0)  # Sum over columns
        expected_out_degrees = np.sum(W, axis=1)  # Sum over rows

        receiver = InDegree()
        indegree = receiver.calculate(W)

        self.assertTrue(np.all(indegree == expected_in_degrees))

        sender = OutDegree()
        outdegree = sender.calculate(W)

        self.assertTrue(np.all(outdegree == expected_out_degrees))

        undirected_degree = UndirectedDegree()
        W = np.array([
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 0]
        ])

        # For undirected graphs, degree is sum of each row (or column, they're the same)
        expected_degrees = np.sum(W, axis=1)

        degrees = undirected_degree.calculate(W)
        self.assertTrue(np.all(degrees == expected_degrees))

    def test_out_degree_on_sample(self):
        metric = OutDegree()
        n = 4
        sample_size = 2

        # W1 row sums: [2, 1, 3, 0]
        W1 = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 0, 0]
        ])
        # W2 row sums: [1, 2, 0, 2]
        W2 = np.array([
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0]
        ])

        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2

        # Case 1: No Mask
        # Expected: Shape (4, 2) -> Rows are nodes, Columns are samples
        result = metric.calculate_for_sample(sample)
        expected_result = np.array([
            [2, 1],  # Node 0
            [1, 2],  # Node 1
            [3, 0],  # Node 2
            [0, 2]  # Node 3
        ])
        self.assertTrue(np.all(result == expected_result))

        # Case 2: With Mask
        # Mask out the last column (index 3).
        mask_mat = np.ones((n, n), dtype=bool)
        mask_mat[:, 3] = False

        mask = flatten_square_matrix_to_edge_list(mask_mat, True)

        result = metric.calculate_for_sample(sample, mask)
        # W1 row sums without col 3: [2, 0, 2, 0]
        # W2 row sums without col 3: [0, 2, 0, 2]
        expected_masked = np.array([
            [2, 0],
            [0, 2],
            [2, 0],
            [0, 2]
        ])
        self.assertTrue(np.all(result == expected_masked))

    def test_in_degree_on_sample(self):
        metric = InDegree()
        n = 4
        sample_size = 2

        # W1 col sums: [1, 2, 1, 2]
        W1 = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 0, 0]
        ])
        # W2 col sums: [2, 1, 1, 1]
        W2 = np.array([
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0]
        ])

        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2

        # Case 1: No Mask
        result = metric.calculate_for_sample(sample)
        expected_result = np.array([
            [1, 2],  # Node 0 in-degree
            [2, 1],  # Node 1 in-degree
            [1, 1],  # Node 2 in-degree
            [2, 1]  # Node 3 in-degree
        ])
        self.assertTrue(np.all(result == expected_result))

        # Case 2: With Mask
        # Mask out the first row (index 0).
        mask_mat = np.ones((n, n), dtype=bool)
        mask_mat[0, :] = False

        mask = flatten_square_matrix_to_edge_list(mask_mat, True)

        result = metric.calculate_for_sample(sample, mask)
        # W1 col sums without row 0: [1, 1, 0, 2]
        # W2 col sums without row 0: [2, 1, 1, 0]
        expected_masked = np.array([
            [1, 2],
            [1, 1],
            [0, 1],
            [2, 0]
        ])
        self.assertTrue(np.all(result == expected_masked))

    def test_undirected_degree_on_sample(self):
        metric = UndirectedDegree()
        n = 4
        sample_size = 2

        # W1 is symmetric. Row/Col sums: [2, 2, 2, 2]
        W1 = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        # W2 is symmetric. Row/Col sums: [1, 2, 2, 1]
        W2 = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])

        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2

        # Case 1: No Mask
        # Expected result shape: (nodes, sample_size) -> (4, 2)
        result = metric.calculate_for_sample(sample)
        expected_result = np.array([
            [2, 1],  # Node 0 degree
            [2, 2],  # Node 1 degree
            [2, 2],  # Node 2 degree
            [2, 1]  # Node 3 degree
        ])
        self.assertTrue(np.all(result == expected_result))

        # Case 2: With Symmetric Mask
        # Mask edge between Node 0 and Node 1.
        mask_mat = np.ones((n, n), dtype=bool)
        mask_mat[0, 1] = False
        mask_mat[1, 0] = False

        mask = flatten_square_matrix_to_edge_list(mask_mat, is_directed=False)

        result = metric.calculate_for_sample(sample, mask)

        # W1: Edge (0,1) removed. Node 0 sum: 2->1. Node 1 sum: 2->1.
        # W2: Edge (0,1) removed. Node 0 sum: 1->0. Node 1 sum: 2->1.
        expected_masked = np.array([
            [1, 0],  # Node 0
            [1, 1],  # Node 1
            [2, 2],  # Node 2 (unaffected)
            [2, 1]  # Node 3 (unaffected)
        ])

        self.assertTrue(np.all(result == expected_masked))

    @staticmethod
    def get_expected_Xs_in_out_degs(n, metric):
        zero_network = np.zeros((n, n))
        expected_Xs = np.zeros((n ** 2 - n, n))
        idx = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                expected_Xs[idx] = metric.calc_change_score(zero_network, (i, j))
                idx += 1
        return expected_Xs

    @staticmethod
    def get_expected_Xs_undirected_degs(n):
        metric = UndirectedDegree()
        zero_network = np.zeros((n, n))
        expected_Xs = np.zeros(((n ** 2 - n) // 2, n))
        idx = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                expected_Xs[idx] = metric.calc_change_score(zero_network, (i, j))
                idx += 1
        return expected_Xs

    def test_calculate_mple_regressors(self):
        n = 50
        mask = flatten_square_matrix_to_edge_list(np.ones((n, n), dtype=bool), True).reshape(-1, 1)
        Xs = np.zeros((n ** 2 - n, n))

        in_degree = InDegree()
        in_degree.calculate_mple_regressors(Xs, np.arange(n, dtype=int), mask, None)
        self.assertTrue(np.allclose(Xs, self.get_expected_Xs_in_out_degs(n, in_degree)))

        out_degree = OutDegree()
        out_degree.calculate_mple_regressors(Xs, np.arange(n, dtype=int), mask, None)
        self.assertTrue(np.allclose(Xs, self.get_expected_Xs_in_out_degs(n, out_degree)))

        undirected_degree = UndirectedDegree()
        mask = flatten_square_matrix_to_edge_list(np.ones((n, n), dtype=bool), False).reshape(-1, 1)
        Xs = np.zeros(((n ** 2 - n) // 2, n))
        undirected_degree.calculate_mple_regressors(Xs, np.arange(n, dtype=int), mask, None)
        self.assertTrue(np.allclose(Xs, self.get_expected_Xs_undirected_degs(n)))

    def test_in_degree_mple_regressors_masked_ignored_indices(self):
        n = 50
        ignored_indices = [0, 5, 11, 23]
        set_seed(348976)
        in_degree_ignored_indices = InDegree(indices_from_user=ignored_indices)
        global_mask = flatten_square_matrix_to_edge_list(
            generate_binomial_tensor(
                net_size=n, num_samples=1
            )[..., -1],
            True,
        ).astype(bool)
        dummy_metrics_collection = MetricsCollection(
            metrics=[in_degree_ignored_indices], is_directed=True, n_nodes=n, mask=global_mask, fix_collinearity=False,
        )
        edge_indices_lims = (5, 22)
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(edge_indices_lims)
        Xs = dummy_metrics_collection.prepare_mple_regressors(edge_indices_lims=edge_indices_lims)

        expected_xs = self.get_expected_Xs_in_out_degs(n, InDegree())
        expected_xs = expected_xs[mask][:, [i for i in range(n) if i not in ignored_indices]]
        self.assertTrue(np.allclose(Xs, expected_xs))

    def test_out_degree_mple_regressors_masked_ignored_indices(self):
        n = 50
        ignored_indices = [17, 29, 34, 46, 47]
        set_seed(94538)
        out_degree_ignored_indices = OutDegree(indices_from_user=ignored_indices)
        global_mask = flatten_square_matrix_to_edge_list(
            generate_binomial_tensor(
                net_size=n, num_samples=1
            )[..., -1],
            True,
        ).astype(bool)
        dummy_metrics_collection = MetricsCollection(
            metrics=[out_degree_ignored_indices], is_directed=True, n_nodes=n, mask=global_mask,
            fix_collinearity=False,
        )
        edge_indices_lims = (112, 1024)
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(edge_indices_lims)
        Xs = dummy_metrics_collection.prepare_mple_regressors(edge_indices_lims=edge_indices_lims)

        expected_xs = self.get_expected_Xs_in_out_degs(n, OutDegree())
        expected_xs = expected_xs[mask][:, [i for i in range(n) if i not in ignored_indices]]
        self.assertTrue(np.allclose(Xs, expected_xs))

    def test_undirected_degree_mple_regressors_masked_ignored_indices(self):
        n = 50
        set_seed(384976)
        ignored_indices = np.random.choice(n, 12, replace=False)
        undirected_degree_ignored_indices = UndirectedDegree(indices_from_user=ignored_indices)
        global_mask = generate_binomial_tensor(net_size=n, num_samples=1)[..., -1]
        global_mask = (global_mask + global_mask.T) // 2
        global_mask = flatten_square_matrix_to_edge_list(global_mask, False).astype(bool)
        dummy_metrics_collection = MetricsCollection(
            metrics=[undirected_degree_ignored_indices], is_directed=False, n_nodes=n, mask=global_mask,
            fix_collinearity=False,
        )
        edge_indices_lims = (58, 132)
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(edge_indices_lims)
        Xs = dummy_metrics_collection.prepare_mple_regressors(edge_indices_lims=edge_indices_lims)

        expected_xs = self.get_expected_Xs_undirected_degs(n)
        expected_xs = expected_xs[mask][:, [i for i in range(n) if i not in ignored_indices]]
        self.assertTrue(np.allclose(Xs, expected_xs))


class TestNumberOfEdgesTypesDirected(unittest.TestCase):
    def test_calc_edge_type_idx_assignment(self):
        neuronal_types = ['A', 'B', 'A', 'B']
        metric = NumberOfEdgesTypesDirected(neuronal_types)
        expected_edge_type_indices = np.array([[0, 1, 0, 1],
                                               [2, 3, 2, 3],
                                               [0, 1, 0, 1],
                                               [2, 3, 2, 3]])

        expected_edge_type_indices += 1  # Increment because we also want a bin for nonexisting edges (which have the entry 0)

        self.assertTrue(np.all(metric._edge_type_idx_assignment == expected_edge_type_indices))

        neuronal_types = ['A', 'B', 'B', 'A']
        metric = NumberOfEdgesTypesDirected(neuronal_types)
        expected_edge_type_indices = np.array([[0, 1, 1, 0],
                                               [2, 3, 3, 2],
                                               [2, 3, 3, 2],
                                               [0, 1, 1, 0]])

        expected_edge_type_indices += 1  # Increment because we also want a bin for nonexisting edges (which have the entry 0)

        self.assertTrue(np.all(metric._edge_type_idx_assignment == expected_edge_type_indices))

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

    def test_calculate_for_sample_masked(self):
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

        mask = np.array([
            [False, True, False, False],
            [True, False, True, False],
            [False, False, False, False],
            [True, False, False, False]
        ])

        neuronal_types = ['A', 'B', 'A', 'B']
        expected_num_edges = np.array([
            [0, 1, 1, 0],
            [0, 1, 0, 0]
        ]).T

        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2
        metric = NumberOfEdgesTypesDirected(neuronal_types)
        calculated_num_edges = metric.calculate_for_sample(sample,
                                                           mask=flatten_square_matrix_to_edge_list(mask, True))
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

    def test_calc_mple_regressors(self):
        neuronal_types = ['A', 'B', 'A', 'B', 'C', 'C']
        metric = NumberOfEdgesTypesDirected(neuronal_types)
        n_nodes = len(neuronal_types)
        empty_matrix = np.zeros((n_nodes, n_nodes))
        num_types = len(set(neuronal_types))
        expected_mple_regressors = np.zeros((n_nodes ** 2 - n_nodes, num_types ** 2))
        idx = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                expected_mple_regressors[idx] = metric.calc_change_score(empty_matrix, (i, j))
                idx += 1

        mple_regressors = np.zeros((n_nodes ** 2 - n_nodes, num_types ** 2))
        metric.calculate_mple_regressors(
            Xs_out=mple_regressors,
            feature_col_indices=np.arange(mple_regressors.shape[1], dtype=int),
            edge_indices_mask=np.ones(mple_regressors.shape[0], dtype=bool),
        )
        self.assertTrue(np.all(expected_mple_regressors == mple_regressors))

        edge_indices = (2, 6)
        dummy_metrics_collection = MetricsCollection(metrics=[metric], is_directed=True, n_nodes=n_nodes)
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(edge_indices)
        mple_regressors = np.zeros((edge_indices[1] - edge_indices[0], num_types ** 2))
        metric.calculate_mple_regressors(
            Xs_out=mple_regressors,
            feature_col_indices=np.arange(mple_regressors.shape[1], dtype=int),
            edge_indices_mask=mask,
        )
        self.assertTrue(np.all(expected_mple_regressors[edge_indices[0]:edge_indices[1]] == mple_regressors))

        set_seed(92349)
        global_mask = flatten_square_matrix_to_edge_list(
            generate_binomial_tensor(
                net_size=n_nodes, num_samples=1
            )[..., -1],
            True,
        ).astype(bool)
        dummy_metrics_collection = MetricsCollection(
            metrics=[metric], is_directed=True, n_nodes=n_nodes, mask=global_mask
        )
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(None)
        mple_regressors = np.zeros((global_mask.sum(), num_types ** 2))
        metric.calculate_mple_regressors(
            Xs_out=mple_regressors,
            feature_col_indices=np.arange(mple_regressors.shape[1], dtype=int),
            edge_indices_mask=mask,
        )
        self.assertTrue(np.all(expected_mple_regressors[global_mask] == mple_regressors))

        edge_indices = (1, 3)
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(edge_indices)
        mple_regressors = np.zeros((mask.sum(), num_types ** 2))
        metric.calculate_mple_regressors(
            Xs_out=mple_regressors,
            feature_col_indices=np.arange(mple_regressors.shape[1], dtype=int),
            edge_indices_mask=mask,
        )
        self.assertTrue(
            np.all(
                expected_mple_regressors[global_mask][edge_indices[0]:edge_indices[1]] == mple_regressors
            )
        )

    def test_directed_with_ignored_indices(self):
        """Test NumberOfEdgesTypesDirected with ignored feature indices."""
        neuronal_types = ['A', 'B', 'A', 'B']
        W = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        metric = NumberOfEdgesTypesDirected(neuronal_types, indices_from_user=[0])
        n = len(neuronal_types)
        metric._n_nodes = n
        metric.initialize_indices_to_ignore()

        # Calculate features
        result = metric.calculate(W)

        # Should have fewer features due to ignored index
        n_features = metric._get_effective_feature_count()
        self.assertEqual(len(result), n_features)
        self.assertEqual(n_features, 3)

    def test_directed_calc_change_score_with_ignored_indices(self):
        """Test change score calculation with ignored indices."""
        neuronal_types = ['A', 'B', 'A', 'B']
        W = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        metric = NumberOfEdgesTypesDirected(neuronal_types, indices_from_user=[0])
        n = len(neuronal_types)
        metric._n_nodes = n
        metric.initialize_indices_to_ignore()

        n_features = metric._get_effective_feature_count()
        self.assertEqual(n_features, 3)

        # Edge (0,1) is A->B which is type pair index 1 in full features.
        # After removing ignored index 0, it maps to effective index 0.
        # W[0,1]=1, so flipping removes the edge, change score = -1
        change_score = metric.calc_change_score(W, (0, 1))
        expected_change_score = np.array([-1, 0, 0])  # A->B at effective index 0
        np.testing.assert_array_equal(change_score, expected_change_score)

        # Edge (0,2) is A->A which is the ignored type pair (index 0).
        # Change score should be all zeros since this type is ignored.
        change_score_ignored = metric.calc_change_score(W, (0, 2))
        np.testing.assert_array_equal(change_score_ignored, np.zeros(3))


class TestNumberOfEdgesTypesUndirected(unittest.TestCase):
    def test_calc_edge_type_idx_assignment(self):
        neuronal_types = ['A', 'B', 'A', 'B']
        metric = NumberOfEdgesTypesUndirected(neuronal_types)
        expected_edge_type_indices = np.array([[0, 1, 0, 1],
                                               [1, 2, 1, 2],
                                               [0, 1, 0, 1],
                                               [1, 2, 1, 2]])

        expected_edge_type_indices += 1  # Increment because we also want a bin for nonexisting edges (which have the entry 0)

        self.assertTrue(np.all(metric._edge_type_idx_assignment == expected_edge_type_indices))

        neuronal_types = ['A', 'B', 'B', 'A']
        metric = NumberOfEdgesTypesUndirected(neuronal_types)
        expected_edge_type_indices = np.array([[0, 1, 1, 0],
                                               [1, 2, 2, 1],
                                               [1, 2, 2, 1],
                                               [0, 1, 1, 0]])

        expected_edge_type_indices += 1  # Increment because we also want a bin for nonexisting edges (which have the entry 0)

        self.assertTrue(np.all(metric._edge_type_idx_assignment == expected_edge_type_indices))

    def test_calculate(self):
        W = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        neuronal_types = ['A', 'B', 'A', 'B']
        expected_num_edges = np.array([1, 1, 1])
        metric = NumberOfEdgesTypesUndirected(neuronal_types)
        calculated_num_edges = metric.calculate(W)
        self.assertTrue(np.all(expected_num_edges == calculated_num_edges))

    def test_calculate_for_sample(self):
        n = 4
        sample_size = 2
        W1 = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])
        W2 = np.array([
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 0]
        ])

        neuronal_types = ['A', 'B', 'A', 'B']
        expected_num_edges = np.array([
            [0, 3, 0],
            [1, 2, 1]
        ]).T

        sample = np.zeros((n, n, sample_size))
        sample[:, :, 0] = W1
        sample[:, :, 1] = W2
        metric = NumberOfEdgesTypesUndirected(neuronal_types)
        calculated_num_edges = metric.calculate_for_sample(sample)
        self.assertTrue(np.all(expected_num_edges == calculated_num_edges))

        mask = np.array([
            [False, False, False, False],
            [False, False, True, True],
            [False, True, False, True],
            [False, True, True, False],
        ])

        expected_num_edges_masked = np.array([
            [0, 2, 0],
            [0, 1, 1]
        ]).T
        calculated_num_edges = metric.calculate_for_sample(sample, mask=flatten_square_matrix_to_edge_list(mask, False))
        self.assertTrue(np.all(expected_num_edges_masked == calculated_num_edges))

    def test_calc_change_score(self):
        W1 = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])
        W2 = np.array([
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 0]
        ])

        neuronal_types = ['A', 'B', 'A', 'B']
        metric = NumberOfEdgesTypesUndirected(neuronal_types)

        calculated_change_score = metric.calc_change_score(W1, (0, 1))
        expected_change_score = np.array([0, -1, 0])
        self.assertTrue(np.all(expected_change_score == calculated_change_score))

        calculated_change_score = metric.calc_change_score(W1, (1, 3))
        expected_change_score = np.array([0, 0, 1])
        self.assertTrue(np.all(expected_change_score == calculated_change_score))

        calculated_change_score = metric.calc_change_score(W2, (0, 2))
        expected_change_score = np.array([-1, 0, 0])
        self.assertTrue(np.all(expected_change_score == calculated_change_score))

        calculated_change_score = metric.calc_change_score(W2, (1, 2))
        expected_change_score = np.array([0, 1, 0])
        self.assertTrue(np.all(expected_change_score == calculated_change_score))

    def test_calc_mple_regressors(self):
        neuronal_types = ['B', 'B', 'A', 'C', 'C', 'A']
        metric = NumberOfEdgesTypesUndirected(neuronal_types)
        n_nodes = len(neuronal_types)
        empty_matrix = np.zeros((n_nodes, n_nodes))
        num_types = len(set(neuronal_types))
        expected_mple_regressors = np.zeros(((n_nodes ** 2 - n_nodes) // 2, (num_types ** 2 + num_types) // 2))
        idx = 0
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                expected_mple_regressors[idx] = metric.calc_change_score(empty_matrix, (i, j))
                idx += 1

        mple_regressors = np.zeros_like(expected_mple_regressors)
        metric.calculate_mple_regressors(
            Xs_out=mple_regressors,
            feature_col_indices=np.arange(mple_regressors.shape[1], dtype=int),
            edge_indices_mask=np.ones(mple_regressors.shape[0], dtype=bool),
        )
        self.assertTrue(np.all(expected_mple_regressors == mple_regressors))

        set_seed(349876)
        global_mask = generate_binomial_tensor(n_nodes, 0, 1)[..., -1][
            np.triu_indices(n_nodes, k=1)
        ].astype(bool)
        dummy_metrics_collection = MetricsCollection(
            metrics=[metric], is_directed=False, n_nodes=n_nodes, mask=global_mask, fix_collinearity=False
        )
        with self.assertRaises(ValueError):
            edge_indices_limits = (4, 11)
            dummy_metrics_collection._get_mple_data_chunk_mask(edge_indices_limits)

        edge_indices_limits = (2, 5)
        mask_for_metric = dummy_metrics_collection._get_mple_data_chunk_mask(edge_indices_limits)
        mple_regressors = np.zeros((mask_for_metric.sum(), expected_mple_regressors.shape[1]))
        metric.calculate_mple_regressors(
            Xs_out=mple_regressors,
            feature_col_indices=np.arange(mple_regressors.shape[1], dtype=int),
            edge_indices_mask=mask_for_metric,
        )
        self.assertTrue(
            np.all(
                expected_mple_regressors[global_mask][edge_indices_limits[0]:edge_indices_limits[1]] == mple_regressors
            )
        )

    def test_undirected_with_ignored_indices(self):
        """Test NumberOfEdgesTypesUndirected with ignored indices."""
        neuronal_types = ['A', 'B', 'A', 'B']
        W = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # For undirected with types ['A', 'B'], canonical pairs are:
        # [('A','A'), ('A','B'), ('B','B')] = indices [0, 1, 2]
        # indices_from_user=[1] ignores ('A','B'), so effective features are 2
        metric = NumberOfEdgesTypesUndirected(neuronal_types, indices_from_user=[1])
        n = len(neuronal_types)
        metric._n_nodes = n
        metric.initialize_indices_to_ignore()

        n_features = metric._get_effective_feature_count()
        self.assertEqual(n_features, 2)

        # Calculate features
        result = metric.calculate(W)
        self.assertEqual(len(result), 2)

        # Count edges by type in upper triangle (undirected):
        # (0,1): A-B (ignored), (0,2): A-A = 1, (0,3): A-B (ignored)
        # (1,2): B-A (ignored), (1,3): B-B = 1
        # (2,3): A-B (ignored)
        # So A-A count = 1, B-B count = 1
        expected_result = np.array([1, 1])  # [A-A, B-B] after removing A-B
        np.testing.assert_array_equal(result, expected_result)

    def test_undirected_calc_change_score_with_ignored_indices(self):
        """Test change score calculation with ignored indices."""
        neuronal_types = ['A', 'B', 'A', 'B']
        W = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # For undirected with types ['A', 'B'], canonical pairs are:
        # [('A','A'), ('A','B'), ('B','B')] = indices [0, 1, 2]
        # indices_from_user=[1] ignores ('A','B'), so effective features are 2
        metric = NumberOfEdgesTypesUndirected(neuronal_types, indices_from_user=[1])
        n = len(neuronal_types)
        metric._n_nodes = n
        metric.initialize_indices_to_ignore()

        n_features = metric._get_effective_feature_count()
        self.assertEqual(n_features, 2)

        # Edge (0,1) is A-B which is the ignored type pair (index 1).
        # W[0,1]=1, so flipping removes the edge, but since it's ignored,
        # change score should be all zeros.
        change_score = metric.calc_change_score(W, (0, 1))
        np.testing.assert_array_equal(change_score, np.zeros(2))

        # Edge (0,2) is A-A which is type pair index 0 (effective index 0).
        # W[0,2]=1, so flipping removes the edge, change score = -1
        change_score_aa = metric.calc_change_score(W, (0, 2))
        expected_change_score = np.array([-1, 0])  # A-A at effective index 0
        np.testing.assert_array_equal(change_score_aa, expected_change_score)


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

        # 2D numpy array with mask
        mask = flatten_square_matrix_to_edge_list(
            np.array([
                [False, False, False],
                [True, False, False],
                [True, True, False]
            ]),
            True,
        )
        expected_res_2_mask = 0 + 3 + 0 + 4
        self.assertTrue(metric_2.calculate(W, mask=mask) == expected_res_2_mask)

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

        # multiple columns
        metric_1 = SumDistancesConnectedNeurons(positions, is_directed=True)
        # distance matrix for reference
        # np.array([[0, 3, 4],
        #           [3, 0, 5],
        #           [4, 5, 0]])
        indices_lims = (0, 6)
        expected_regressors_1_1 = np.array([[3.],
                                            [4.],
                                            [3.],
                                            [5.],
                                            [4.],
                                            [5.]])
        Xs_out = np.zeros_like(expected_regressors_1_1)
        dummy_metrics_collection = MetricsCollection(
            metrics=[metric_1], n_nodes=3, is_directed=True, fix_collinearity=False
        )
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(indices_lims)
        metric_1.calculate_mple_regressors(Xs_out, np.array([0]), edge_indices_mask=mask)
        self.assertTrue(np.all(Xs_out == expected_regressors_1_1))

        indices_lims = (1, 4)
        expected_regressors_1_2 = np.array([[4.],
                                            [3.],
                                            [5.]])
        Xs_out = np.zeros_like(expected_regressors_1_2)
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(indices_lims)
        metric_1.calculate_mple_regressors(Xs_out, np.array([0]), edge_indices_mask=mask)
        self.assertTrue(np.all(Xs_out == expected_regressors_1_2))

        set_seed(349876)
        global_mask = flatten_square_matrix_to_edge_list(
            generate_binomial_tensor(3, 0, 1, 0.6)[..., -1].astype(bool),
            True,
        )
        assert not np.all(global_mask), "the randomly sampled global mask for testing shouldn't be all-True"
        dummy_metrics_collection = MetricsCollection(
            metrics=[metric_1], n_nodes=3, is_directed=True, fix_collinearity=False, mask=global_mask
        )
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(indices_lims)
        Xs_out = np.zeros((mask.sum(), 1))
        metric_1.calculate_mple_regressors(Xs_out, np.array([0]), edge_indices_mask=mask)
        self.assertTrue(
            np.all(Xs_out == expected_regressors_1_1[global_mask][indices_lims[0]:indices_lims[1]])
        )

        # Series
        metric_2 = SumDistancesConnectedNeurons(positions.x_pos, is_directed=True)
        # distance matrix for reference
        # np.array([[0, 0, 4],
        #           [0, 0, 4],
        #           [4, 4, 0]])
        indices_lims = (0, 6)
        expected_regressors_2_1 = np.array([[0.],
                                            [4.],
                                            [0.],
                                            [4.],
                                            [4.],
                                            [4.]])
        Xs_out = np.zeros_like(expected_regressors_2_1)
        dummy_metrics_collection = MetricsCollection(
            metrics=[metric_2], n_nodes=3, is_directed=True, fix_collinearity=False
        )
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(indices_lims)
        metric_2.calculate_mple_regressors(Xs_out, np.array([0]), edge_indices_mask=mask)
        self.assertTrue(np.all(Xs_out == expected_regressors_2_1))

        indices_lims = (1, 4)
        expected_regressors_2_2 = np.array([[4.],
                                            [0.],
                                            [4.]])
        Xs_out = np.zeros_like(expected_regressors_2_2)
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(indices_lims)
        metric_2.calculate_mple_regressors(Xs_out, np.array([0]), edge_indices_mask=mask)
        self.assertTrue(np.all(Xs_out == expected_regressors_2_2))

        metric_3 = SumDistancesConnectedNeurons(positions.z_pos, is_directed=True)
        # distance matrix for reference
        # np.array([[0, 0, 0],
        #           [0, 0, 0],
        #           [0, 0, 0]])
        indices_lims = (0, 6)
        expected_regressors_3 = np.array([[0.],
                                          [0.],
                                          [0.],
                                          [0.],
                                          [0.],
                                          [0.]])
        Xs_out = np.zeros_like(expected_regressors_3)
        dummy_metrics_collection = MetricsCollection(
            metrics=[metric_3], n_nodes=3, is_directed=True, fix_collinearity=False
        )
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(indices_lims)
        metric_3.calculate_mple_regressors(Xs_out, np.array([0]), edge_indices_mask=mask)
        self.assertTrue(np.all(Xs_out == expected_regressors_3))

        # undirected
        metric_4 = SumDistancesConnectedNeurons(positions, is_directed=False)
        # distance matrix for reference
        # np.array([[--, 3, 4],
        #           [--, --, 5],
        #           [--, --, --]])
        indices_lims = (0, 3)
        expected_regressors_4_1 = np.array([[3.],
                                            [4.],
                                            [5.]])
        Xs_out = np.zeros_like(expected_regressors_4_1)
        dummy_metrics_collection = MetricsCollection(
            metrics=[metric_4], n_nodes=3, is_directed=False, fix_collinearity=False
        )
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(indices_lims)
        metric_4.calculate_mple_regressors(Xs_out, np.array([0]), edge_indices_mask=mask)
        self.assertTrue(np.all(Xs_out == expected_regressors_4_1))

        indices_lims = (1, 2)
        expected_regressors_4_2 = np.array([[4.]])
        Xs_out = np.zeros_like(expected_regressors_4_2)
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(indices_lims)
        metric_4.calculate_mple_regressors(Xs_out, np.array([0]), edge_indices_mask=mask)
        self.assertTrue(np.all(Xs_out == expected_regressors_4_2))

        global_mask = (
            generate_binomial_tensor(3, 0, 1, 0.6)[..., -1].astype(bool)
        )[np.triu_indices(3, k=1)]
        assert not np.all(global_mask), "the randomly sampled global mask for testing shouldn't be all-True"
        dummy_metrics_collection = MetricsCollection(
            metrics=[metric_4], n_nodes=3, is_directed=False, fix_collinearity=False, mask=global_mask
        )
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(indices_lims)
        Xs_out = np.zeros((mask.sum(), 1))
        metric_4.calculate_mple_regressors(Xs_out, np.array([0]), edge_indices_mask=mask)
        self.assertTrue(
            np.all(Xs_out == expected_regressors_4_1[global_mask][indices_lims[0]:indices_lims[1]])
        )

        metric_5 = SumDistancesConnectedNeurons(positions.z_pos, is_directed=False)
        # distance matrix for reference
        # np.array([[--, 0, 0],
        #           [--, --, 0],
        #           [--, --, --]])
        indices_lims = (0, 3)
        expected_regressors_5 = np.array([[0.],
                                          [0.],
                                          [0.]])
        Xs_out = np.zeros_like(expected_regressors_5)
        dummy_metrics_collection = MetricsCollection(
            metrics=[metric_5], n_nodes=3, is_directed=False, fix_collinearity=False
        )
        mask = dummy_metrics_collection._get_mple_data_chunk_mask(indices_lims)
        metric_5.calculate_mple_regressors(Xs_out, np.array([0]), edge_indices_mask=mask)
        self.assertTrue(np.all(Xs_out == expected_regressors_5))


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
        expected_result = np.array([-1, 0, -1, 0])

        self.assertTrue(np.all(result == expected_result))

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

        Xs_half = collection.prepare_mple_regressors(W1, edge_indices_lims=(0, expected_mple_regressors.shape[0] // 2))
        ys_half = collection.prepare_mple_labels(
            W1[..., np.newaxis],
            edge_indices_lims=(0, expected_mple_regressors.shape[0] // 2)
        )

        # Deleting the 1+idx_to_ignore because the first entry is the NumberOfEdgesDirected metric
        expected_mple_regressors_ignored_features = np.delete(expected_mple_regressors, 1 + idx_to_ignore, axis=1)
        self.assertTrue(np.all(expected_mple_regressors_ignored_features == Xs_full))
        self.assertTrue(
            np.all(expected_mple_regressors_ignored_features[:expected_mple_regressors.shape[0] // 2] == Xs_half))
        self.assertTrue(np.all(expected_flattened_mat == ys_full))
        self.assertTrue(np.all(expected_flattened_mat[:expected_mple_regressors.shape[0] // 2] == ys_half))

        mask = flatten_square_matrix_to_edge_list(
            np.array([
                [False, True, False, True],
                [False, False, True, False],
                [True, True, False, False],
                [False, True, True, False]
            ]),
            True,
        )
        collection = MetricsCollection(metrics, is_directed=True, n_nodes=n_nodes, mask=mask)
        # The collinearity fixer is supposed to remove one attribute from the NumberOfEdgesTypesDirected metric
        # Note - the mask is designed to not eliminate all connections of a specific pair of types, as in such a case
        # another attribute will be ignored (the one corresponding to the degenerate pair that never appears).
        # For reference - the edge type idx assignment matrix for the list of types in this test is
        # array([[1, 2, 1, 2],
        #        [3, 4, 3, 4],
        #        [1, 2, 1, 2],
        #        [3, 4, 3, 4]])
        self.assertEqual(np.sum(collection.metrics[1]._indices_to_ignore), 1)

        idx_to_ignore = np.where(collection.metrics[1]._indices_to_ignore)[0][0]

        Xs_full = collection.prepare_mple_regressors(W1)
        ys_full = collection.prepare_mple_labels(W1[..., np.newaxis])

        Xs_half = collection.prepare_mple_regressors(W1, edge_indices_lims=(0, mask.sum() // 2))
        ys_half = collection.prepare_mple_labels(W1, edge_indices_lims=(0, mask.sum() // 2))

        # Deleting the 1+idx_to_ignore because the first entry is the NumberOfEdgesDirected metric
        expected_mple_regressors_ignored_features = np.delete(expected_mple_regressors[mask], 1 + idx_to_ignore, axis=1)
        self.assertTrue(np.all(expected_mple_regressors_ignored_features == Xs_full))
        self.assertTrue(
            np.all(expected_mple_regressors_ignored_features[:mask.sum() // 2] == Xs_half))
        self.assertTrue(np.all(expected_flattened_mat[mask] == ys_full))
        self.assertTrue(np.all(expected_flattened_mat[mask][:mask.sum() // 2] == ys_half))

    def test_prepare_mple_reciprocity_data(self):
        W = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        metrics = [NumberOfEdgesDirected(), OutDegree(), InDegree(), TotalReciprocity()]
        n_nodes = W.shape[0]

        expected_statistics = [7, 2, 2, 2, 2, 1, 2, 3]  # just a sanity check for the actual statistics

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


class TestTotalReciprocity(unittest.TestCase):
    """Tests for TotalReciprocity metric methods."""

    def test_total_reciprocity_calculate(self):
        """Test direct calculation of total reciprocity."""
        metric = TotalReciprocity()

        # Test case 1: No reciprocal edges
        W1 = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        result = metric.calculate(W1)
        self.assertEqual(result, 0)

        # Test case 2: One reciprocal pair
        W2 = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0]
        ])
        result = metric.calculate(W2)
        self.assertEqual(result, 1)

        # Test case 3: All edges reciprocal
        W3 = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        result = metric.calculate(W3)
        self.assertEqual(result, 3)

    def test_total_reciprocity_calc_change_score(self):
        """Test change score when toggling edges."""
        metric = TotalReciprocity()

        # Test case 1: Adding an edge that completes a reciprocal pair
        W1 = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        change_score = metric.calc_change_score(W1, (1, 0))
        self.assertEqual(change_score, 1)  # Creates reciprocal pair

        # Test case 2: Removing an edge from a reciprocal pair
        W2 = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])
        change_score = metric.calc_change_score(W2, (0, 1))
        self.assertEqual(change_score, -1)  # Destroys reciprocal pair

        # Test case 3: Adding an edge that doesn't complete a pair
        W3 = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        change_score = metric.calc_change_score(W3, (0, 1))
        self.assertEqual(change_score, 0)  # No reciprocal exists

    def test_total_reciprocity_calculate_for_sample(self):
        """Test batch calculation on sample."""
        metric = TotalReciprocity()

        W1 = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        W2 = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        W3 = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])

        sample = np.stack([W1, W2, W3], axis=-1)
        result = metric.calculate_for_sample(sample)
        expected_result = np.array([2, 3, 0])
        self.assertTrue(np.array_equal(result, expected_result))

    def test_total_reciprocity_calculate_bootstrapped_features(self):
        """Test bootstrap feature generation."""
        metric = TotalReciprocity()

        # Create a network with known reciprocity
        W = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 1, 0]
        ])

        # Bootstrap with specific node splits
        first_half_indices = np.array([0, 1])
        second_half_indices = np.array([2, 3])

        # Create subnetwork samples
        # first_half_W = W[[0,1], :][:, [2,3]] = [[1,0], [0,1]]
        first_half_W = W[np.ix_(first_half_indices, second_half_indices)]
        second_half_W = W[np.ix_(second_half_indices, first_half_indices)]

        # Stack samples
        first_halves = first_half_W.reshape(2, 2, 1)
        second_halves = second_half_W.reshape(2, 2, 1)

        result = metric.calculate_bootstrapped_features(first_halves, second_halves, first_half_indices, second_half_indices)

        # TotalReciprocity.calculate_for_sample on first_halves [[1,0],[0,1]]:
        # einsum("ijk,jik->k", sample, sample) / 2 counts reciprocal pairs
        # Reciprocal: (0,0)->1, (1,1)->1, total=2, /2 = 1
        # Normalization: result * (n_obs*(n_obs-1)/2) / (n_half*(n_half-1)/2)
        #              = 1 * (4*3/2) / (2*1/2) = 1 * 6 / 1 = 6
        expected_result = np.array([6.0])
        np.testing.assert_array_equal(result, expected_result)

    def test_total_reciprocity_calculate_mple_regressors(self):
        """Test MPLE regressor generation."""
        n = 4
        metric = TotalReciprocity()
        metric._n_nodes = n

        W = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 0]
        ])

        # MPLE regressors indicate whether each edge would be reciprocated if it existed
        n_edges = n * (n - 1)
        Xs = np.zeros((n_edges, 1))
        feature_indices = np.array([0])
        mask = np.ones(n_edges, dtype=bool).reshape(-1, 1)

        metric.calculate_mple_regressors(Xs, feature_indices, mask, W)

        self.assertEqual(Xs.shape, (n_edges, 1))


        # MPLE regressors = 1 if the reverse edge exists (would be reciprocal), 0 otherwise
        expected_Xs = np.array([
            1, 0, 0, 1, 0, 1,  # edges from node 0, 1
            0, 1, 1, 0, 0, 1   # edges from node 2, 3
        ]).reshape(-1, 1)
        np.testing.assert_array_equal(Xs, expected_Xs)


class TestReciprocity(unittest.TestCase):
    """Tests for Reciprocity metric methods."""

    def test_reciprocity_calc_change_score(self):
        """Test change score calculation for Reciprocity."""
        n = 4
        metric = Reciprocity()
        metric._n_nodes = n
        metric.initialize_indices_to_ignore()

        W = np.array([
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0]
        ])

        # Calculate change score for adding edge (1, 0) which would create reciprocity with (0, 1)
        change_score = metric.calc_change_score(W, (1, 0))

        # Reciprocity uses n choose 2 features (unordered pairs via upper triangular indices):
        # pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        # Adding edge (1,0) when W[0,1]=1 creates reciprocity for pair (0,1) only
        expected_change_score = np.array([1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(change_score, expected_change_score)

    def test_reciprocity_calculate_for_sample(self):
        """Test batch calculation for Reciprocity."""
        n = 3
        metric = Reciprocity()
        metric._n_nodes = n
        metric.initialize_indices_to_ignore()

        W1 = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0]
        ])
        W2 = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])

        sample = np.stack([W1, W2], axis=-1)
        result = metric.calculate_for_sample(sample)

        # Should return reciprocity vector for each network (n choose 2 features)
        self.assertEqual(result.shape, (n * (n - 1) // 2, 2))

        # W1 has one reciprocal pair: (0,1)
        expected_W1 = np.array([1, 0, 0])  # pairs: (0,1), (0,2), (1,2)
        # W2 has all pairs reciprocal
        expected_W2 = np.array([1, 1, 1])

        self.assertTrue(np.array_equal(result[:, 0], expected_W1))
        self.assertTrue(np.array_equal(result[:, 1], expected_W2))

    def test_reciprocity_calculate_mple_regressors(self):
        """Test MPLE regressor calculation for Reciprocity."""
        n = 3
        metric = Reciprocity()
        metric._n_nodes = n
        metric.initialize_indices_to_ignore()

        W = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])

        n_edges = n * (n - 1)
        n_features = n * (n - 1) // 2  # Reciprocity has n choose 2 features
        mask = np.ones(n_edges, dtype=bool).reshape(-1, 1)
        Xs = np.zeros((n_edges, n_features))
        feature_indices = np.arange(n_features)

        metric.calculate_mple_regressors(Xs, feature_indices, mask, W)

        # MPLE regressors: Xs[i, j] = 1 if directed edge i would make unordered pair j reciprocal
        self.assertEqual(Xs.shape, (n_edges, n_features))

        # Edge list order (directed): (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)
        # Feature list (unordered pairs): pair(0,1), pair(0,2), pair(1,2)
        # Current edges in W: (0,1), (1,2), (2,0)

        # Xs[0, :] for edge (0,1): reverse (1,0) doesn't exist, so pair(0,1) would NOT be reciprocal
        # But pair(0,2) and pair(1,2) are unaffected
        self.assertEqual(Xs[0, 0], 0, "Edge (0,1): pair(0,1) not reciprocal (W[1,0]=0)")
        self.assertEqual(Xs[0, 1], 0, "Edge (0,1): doesn't affect pair(0,2)")
        self.assertEqual(Xs[0, 2], 0, "Edge (0,1): doesn't affect pair(1,2)")

        # Xs[1, :] for edge (0,2): reverse (2,0) exists, so pair(0,2) WOULD be reciprocal
        self.assertEqual(Xs[1, 0], 0, "Edge (0,2): doesn't affect pair(0,1)")
        self.assertEqual(Xs[1, 1], 1, "Edge (0,2): pair(0,2) would be reciprocal (W[2,0]=1)")
        self.assertEqual(Xs[1, 2], 0, "Edge (0,2): doesn't affect pair(1,2)")

        # Xs[2, :] for edge (1,0): reverse (0,1) exists, so pair(0,1) WOULD be reciprocal
        self.assertEqual(Xs[2, 0], 1, "Edge (1,0): pair(0,1) would be reciprocal (W[0,1]=1)")
        self.assertEqual(Xs[2, 1], 0, "Edge (1,0): doesn't affect pair(0,2)")
        self.assertEqual(Xs[2, 2], 0, "Edge (1,0): doesn't affect pair(1,2)")

        # Xs[5, :] for edge (2,1): reverse (1,2) exists, so pair(1,2) WOULD be reciprocal
        self.assertEqual(Xs[5, 0], 0, "Edge (2,1): doesn't affect pair(0,1)")
        self.assertEqual(Xs[5, 1], 0, "Edge (2,1): doesn't affect pair(0,2)")
        self.assertEqual(Xs[5, 2], 1, "Edge (2,1): pair(1,2) would be reciprocal (W[1,2]=1)")


class TestNumberOfTrianglesMPLE(unittest.TestCase):
    """Test MPLE regressors for NumberOfTriangles."""

    def test_triangles_calculate_mple_regressors(self):
        """Test MPLE regressor computation for triangles."""
        n = 4
        metric = NumberOfTriangles()
        metric._n_nodes = n

        W = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0]
        ])

        # Number of potential edges in undirected graph
        n_edges = n * (n - 1) // 2
        mask = np.ones(n_edges, dtype=bool).reshape(-1, 1)
        Xs = np.zeros((n_edges, 1))
        feature_indices = np.array([0])

        metric.calculate_mple_regressors(Xs, feature_indices, mask, W)

        # Each regressor value indicates how many triangles would be added/removed
        self.assertEqual(Xs.shape, (n_edges, 1))
        # Values should be non-negative integers (number of triangles affected)
        self.assertTrue(np.all(Xs >= -3))  # Can't remove more than 3 triangles with one edge


class TestEdgeCases(unittest.TestCase):
    """Test metrics on edge case networks (empty, full, single node)."""

    def test_empty_network_directed(self):
        """Test all metrics on empty directed network."""
        n = 5
        W = np.zeros((n, n))

        # Test NumberOfEdgesDirected
        metric = NumberOfEdgesDirected()
        result = metric.calculate(W)
        self.assertEqual(result, 0)

        # Test TotalReciprocity
        metric = TotalReciprocity()
        result = metric.calculate(W)
        self.assertEqual(result, 0)

        # Test InDegree
        metric = InDegree()
        result = metric.calculate(W)
        self.assertTrue(np.all(result == 0))

        # Test OutDegree
        metric = OutDegree()
        result = metric.calculate(W)
        self.assertTrue(np.all(result == 0))

    def test_empty_network_undirected(self):
        """Test metrics on empty undirected network."""
        n = 5
        W = np.zeros((n, n))

        # Test NumberOfEdgesUndirected
        metric = NumberOfEdgesUndirected()
        result = metric.calculate(W)
        self.assertEqual(result, 0)

        # Test NumberOfTriangles
        metric = NumberOfTriangles()
        result = metric.calculate(W)
        self.assertEqual(result, 0)

        # Test UndirectedDegree
        metric = UndirectedDegree()
        result = metric.calculate(W)
        self.assertTrue(np.all(result == 0))

    def test_fully_connected_directed(self):
        """Test metrics on fully connected directed network."""
        n = 4
        W = np.ones((n, n))
        np.fill_diagonal(W, 0)  # No self-loops

        # Test NumberOfEdgesDirected
        metric = NumberOfEdgesDirected()
        result = metric.calculate(W)
        self.assertEqual(result, n * (n - 1))

        # Test TotalReciprocity
        metric = TotalReciprocity()
        result = metric.calculate(W)
        self.assertEqual(result, n * (n - 1) / 2)

        # Test InDegree
        metric = InDegree()
        result = metric.calculate(W)
        self.assertTrue(np.all(result == n - 1))

        # Test OutDegree
        metric = OutDegree()
        result = metric.calculate(W)
        self.assertTrue(np.all(result == n - 1))

    def test_fully_connected_undirected(self):
        """Test metrics on fully connected undirected network."""
        n = 4
        W = np.ones((n, n))
        np.fill_diagonal(W, 0)

        # Test NumberOfEdgesUndirected
        metric = NumberOfEdgesUndirected()
        result = metric.calculate(W)
        self.assertEqual(result, n * (n - 1) // 2)

        # Test NumberOfTriangles - complete graph of n nodes has C(n,3) triangles
        metric = NumberOfTriangles()
        result = metric.calculate(W)
        expected_triangles = n * (n - 1) * (n - 2) // 6
        self.assertEqual(result, expected_triangles)

        # Test UndirectedDegree
        metric = UndirectedDegree()
        result = metric.calculate(W)
        self.assertTrue(np.all(result == n - 1))

    def test_single_node_network(self):
        """Test metrics on single node network."""
        n = 1
        W = np.zeros((n, n))

        # Test NumberOfEdgesDirected
        metric = NumberOfEdgesDirected()
        result = metric.calculate(W)
        self.assertEqual(result, 0)

        # Test TotalReciprocity
        metric = TotalReciprocity()
        result = metric.calculate(W)
        self.assertEqual(result, 0)

        # Test NumberOfEdgesUndirected
        metric = NumberOfEdgesUndirected()
        result = metric.calculate(W)
        self.assertEqual(result, 0)

        # Test degree metrics
        metric = InDegree()
        result = metric.calculate(W)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 0)
