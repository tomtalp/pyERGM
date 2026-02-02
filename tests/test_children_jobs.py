import os
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pyERGM.metrics import MetricsCollection, NumberOfEdgesDirected, InDegree
from logistic_regression_distributed_calcs import main as logistic_regression_main
from sample_statistics_distributed_calcs import main as sample_statistics_main
from mple_data_distributed_paging import main as mple_data_paging_main


class TestLogisticRegressionDistributedCalcs(unittest.TestCase):
    """Tests for logistic_regression_distributed_calcs.py main function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.num_samples = 20
        self.num_features = 3

        # Create input data directory
        chunks_dir = os.path.join(self.temp_dir, 'mple_data_paged_chunks')
        os.makedirs(chunks_dir, exist_ok=True)

        # Create test data
        np.random.seed(42)
        self.Xs_chunk = np.random.randn(self.num_samples, self.num_features).astype(np.float32)
        self.ys_chunk = np.random.randint(0, 2, (self.num_samples, 1)).astype(np.float32)
        self.weights_chunk = np.random.rand(self.num_samples, 1).astype(np.float32)

        # Save input file (func_id=0)
        np.savez_compressed(
            os.path.join(chunks_dir, '0.npz'),
            Xs_chunk=self.Xs_chunk,
            ys_chunk=self.ys_chunk
        )

        self.thetas = np.random.randn(self.num_features).astype(np.float32)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _assert_logistic_regression_outputs(self, func_id=0):
        """Helper to verify all logistic regression outputs for a given func_id."""
        thetas_col = self.thetas[:, None]
        predictions = 1 / (1 + np.exp(-self.Xs_chunk @ thetas_col))

        # Check predictions
        output_path = os.path.join(self.temp_dir, 'predictions', f'{func_id}.pkl')
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                actual = pickle.load(f)
            np.testing.assert_array_almost_equal(actual, predictions, decimal=5)

        # Check log likelihood
        output_path = os.path.join(self.temp_dir, 'log_likelihood', f'{func_id}.pkl')
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                actual = pickle.load(f)
            eps = 1e-10
            predictions_clipped = np.clip(predictions, eps, 1 - eps)
            expected_ll = (self.ys_chunk * np.log(predictions_clipped) +
                           (1 - self.ys_chunk) * np.log(1 - predictions_clipped)).sum()
            np.testing.assert_almost_equal(actual, expected_ll, decimal=4)

        # Check gradient
        output_path = os.path.join(self.temp_dir, 'log_likelihood_gradient', f'{func_id}.pkl')
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                actual = pickle.load(f)
            residuals = self.ys_chunk - predictions
            expected_gradient = self.Xs_chunk.T @ residuals
            np.testing.assert_array_almost_equal(actual, expected_gradient, decimal=5)
            np.testing.assert_array_equal(actual.shape, (self.num_features, 1))

        # Check hessian
        output_path = os.path.join(self.temp_dir, 'log_likelihood_hessian', f'{func_id}.pkl')
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                actual = pickle.load(f)
            diag_values = predictions * (1 - predictions)
            expected_hessian = self.Xs_chunk.T @ (diag_values * self.Xs_chunk)
            np.testing.assert_array_almost_equal(actual, expected_hessian, decimal=5)
            np.testing.assert_array_equal(actual.shape, (self.num_features, self.num_features))

    @patch.dict(os.environ, {'LSB_JOBINDEX': '1'})
    def test_main_predictions(self):
        """Test that predictions are correctly calculated and saved."""
        test_args = [
            'script_name',
            '--out_dir_path', self.temp_dir,
            '--functions', 'predictions',
            '--thetas', ','.join(map(str, self.thetas))
        ]

        with patch('sys.argv', test_args):
            logistic_regression_main()

        # Check output file exists
        output_path = os.path.join(self.temp_dir, 'predictions', '0.pkl')
        self.assertTrue(os.path.exists(output_path))

        # Load and verify predictions
        with open(output_path, 'rb') as f:
            predictions = pickle.load(f)

        # Calculate expected predictions manually
        thetas_col = self.thetas[:, None]
        expected_predictions = 1 / (1 + np.exp(-self.Xs_chunk @ thetas_col))

        np.testing.assert_array_almost_equal(predictions, expected_predictions, decimal=5)

    @patch.dict(os.environ, {'LSB_JOBINDEX': '1'})
    def test_main_log_likelihood(self):
        """Test that log likelihood is correctly calculated and saved."""
        test_args = [
            'script_name',
            '--out_dir_path', self.temp_dir,
            '--functions', 'log_likelihood',
            '--thetas', ','.join(map(str, self.thetas))
        ]

        with patch('sys.argv', test_args):
            logistic_regression_main()

        # Check output file exists
        output_path = os.path.join(self.temp_dir, 'log_likelihood', '0.pkl')
        self.assertTrue(os.path.exists(output_path))

        # Load and verify log likelihood
        with open(output_path, 'rb') as f:
            log_likelihood = pickle.load(f)

        # Calculate expected log likelihood manually
        thetas_col = self.thetas[:, None]
        predictions = 1 / (1 + np.exp(-self.Xs_chunk @ thetas_col))
        eps = 1e-10
        predictions_clipped = np.clip(predictions, eps, 1 - eps)
        expected_ll = (self.ys_chunk * np.log(predictions_clipped) +
                       (1 - self.ys_chunk) * np.log(1 - predictions_clipped)).sum()

        np.testing.assert_almost_equal(log_likelihood, expected_ll, decimal=4)

    @patch.dict(os.environ, {'LSB_JOBINDEX': '1'})
    def test_main_log_likelihood_gradient(self):
        """Test that log likelihood gradient is correctly calculated and saved."""
        test_args = [
            'script_name',
            '--out_dir_path', self.temp_dir,
            '--functions', 'log_likelihood_gradient',
            '--thetas', ','.join(map(str, self.thetas))
        ]

        with patch('sys.argv', test_args):
            logistic_regression_main()

        # Check output file exists
        output_path = os.path.join(self.temp_dir, 'log_likelihood_gradient', '0.pkl')
        self.assertTrue(os.path.exists(output_path))

        # Load and verify gradient
        with open(output_path, 'rb') as f:
            gradient = pickle.load(f)

        # Calculate expected gradient manually
        thetas_col = self.thetas[:, None]
        predictions = 1 / (1 + np.exp(-self.Xs_chunk @ thetas_col))
        residuals = self.ys_chunk - predictions
        expected_gradient = self.Xs_chunk.T @ residuals

        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=5)
        np.testing.assert_array_equal(gradient.shape, (self.num_features, 1))

    @patch.dict(os.environ, {'LSB_JOBINDEX': '1'})
    def test_main_log_likelihood_hessian(self):
        """Test that log likelihood hessian is correctly calculated and saved."""
        test_args = [
            'script_name',
            '--out_dir_path', self.temp_dir,
            '--functions', 'log_likelihood_hessian',
            '--thetas', ','.join(map(str, self.thetas))
        ]

        with patch('sys.argv', test_args):
            logistic_regression_main()

        # Check output file exists
        output_path = os.path.join(self.temp_dir, 'log_likelihood_hessian', '0.pkl')
        self.assertTrue(os.path.exists(output_path))

        # Load and verify hessian
        with open(output_path, 'rb') as f:
            hessian = pickle.load(f)

        # Calculate expected hessian manually
        thetas_col = self.thetas[:, None]
        predictions = 1 / (1 + np.exp(-self.Xs_chunk @ thetas_col))
        diag_values = predictions * (1 - predictions)
        expected_hessian = self.Xs_chunk.T @ (diag_values * self.Xs_chunk)

        np.testing.assert_array_almost_equal(hessian, expected_hessian, decimal=5)
        np.testing.assert_array_equal(hessian.shape, (self.num_features, self.num_features))

    @patch.dict(os.environ, {'LSB_JOBINDEX': '1'})
    def test_main_all_functions(self):
        """Test that all functions are calculated when specified together."""
        test_args = [
            'script_name',
            '--out_dir_path', self.temp_dir,
            '--functions', 'predictions,log_likelihood,log_likelihood_gradient,log_likelihood_hessian',
            '--thetas', ','.join(map(str, self.thetas))
        ]

        with patch('sys.argv', test_args):
            logistic_regression_main()

        # Check all output files exist and have correct content
        for func_name in ['predictions', 'log_likelihood', 'log_likelihood_gradient', 'log_likelihood_hessian']:
            output_path = os.path.join(self.temp_dir, func_name, '0.pkl')
            self.assertTrue(os.path.exists(output_path), msg=f"Output file for {func_name} not found")

        self._assert_logistic_regression_outputs(func_id=0)

    @patch.dict(os.environ, {'LSB_JOBINDEX': '1'})
    def test_main_with_weights(self):
        """Test that weights are correctly used when provided."""
        # Re-create input file with weights
        chunks_dir = os.path.join(self.temp_dir, 'mple_data_paged_chunks')
        np.savez_compressed(
            os.path.join(chunks_dir, '0.npz'),
            Xs_chunk=self.Xs_chunk,
            ys_chunk=self.ys_chunk,
            weights_chunk=self.weights_chunk
        )

        test_args = [
            'script_name',
            '--out_dir_path', self.temp_dir,
            '--functions', 'log_likelihood,log_likelihood_gradient',
            '--thetas', ','.join(map(str, self.thetas))
        ]

        with patch('sys.argv', test_args):
            logistic_regression_main()

        # Verify weighted gradient
        output_path = os.path.join(self.temp_dir, 'log_likelihood_gradient', '0.pkl')
        with open(output_path, 'rb') as f:
            gradient = pickle.load(f)

        # Calculate expected weighted gradient
        thetas_col = self.thetas[:, None]
        predictions = 1 / (1 + np.exp(-self.Xs_chunk @ thetas_col))
        residuals = self.weights_chunk * (self.ys_chunk - predictions)
        expected_gradient = self.Xs_chunk.T @ residuals

        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=5)


class TestSampleStatisticsDistributedCalcs(unittest.TestCase):
    """Tests for sample_statistics_distributed_calcs.py main function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.n_nodes = 5
        self.num_samples_per_job = 10
        self.p = 0.3

        # Create data directory
        data_dir = os.path.join(self.temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Create a metric collection with multiple metrics
        metrics = [NumberOfEdgesDirected(), InDegree()]
        self.metric_collection = MetricsCollection(
            metrics=metrics,
            n_nodes=self.n_nodes,
            is_directed=True
        )

        # Save metric collection
        with open(os.path.join(data_dir, 'metric_collection.pkl'), 'wb') as f:
            pickle.dump(self.metric_collection, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _assert_sample_statistics_output(self, func_id, seed):
        """Helper to verify sample statistics output for a given func_id."""
        output_path = os.path.join(self.temp_dir, 'sample_statistics', f'{func_id}.pkl')
        self.assertTrue(os.path.exists(output_path))

        with open(output_path, 'rb') as f:
            sample_statistics = pickle.load(f)

        # Verify shape
        num_features = self.metric_collection.num_of_features
        np.testing.assert_array_equal(
            sample_statistics.shape, (num_features, self.num_samples_per_job)
        )

        # Calculate expected statistics
        np.random.seed(seed)
        expected_statistics = self.metric_collection.calc_statistics_for_binomial_tensor_local(
            self.num_samples_per_job, p=self.p
        )
        np.testing.assert_array_almost_equal(sample_statistics, expected_statistics, decimal=10)

    @patch.dict(os.environ, {'LSB_JOBINDEX': '1'})
    def test_main_creates_output(self):
        """Test that sample statistics are correctly calculated and saved."""
        # Use a fixed seed to get reproducible results
        np.random.seed(123)

        test_args = [
            'script_name',
            '--out_dir_path', self.temp_dir,
            '--num_samples_per_job', str(self.num_samples_per_job),
            '--p', str(self.p)
        ]

        with patch('sys.argv', test_args):
            sample_statistics_main()

        # Check output file exists
        output_path = os.path.join(self.temp_dir, 'sample_statistics', '0.pkl')
        self.assertTrue(os.path.exists(output_path))

        # Load output
        with open(output_path, 'rb') as f:
            sample_statistics = pickle.load(f)

        # Verify shape
        num_features = self.metric_collection.num_of_features
        np.testing.assert_array_equal(sample_statistics.shape, (num_features, self.num_samples_per_job))

        # Calculate expected statistics by running the same method directly
        np.random.seed(123)  # Reset seed to get same random networks
        expected_statistics = self.metric_collection.calc_statistics_for_binomial_tensor_local(
            self.num_samples_per_job, p=self.p
        )

        np.testing.assert_array_almost_equal(sample_statistics, expected_statistics, decimal=10)

    @patch.dict(os.environ, {'LSB_JOBINDEX': '2'})
    def test_main_different_job_index(self):
        """Test that different job indices create different output files."""
        np.random.seed(456)

        test_args = [
            'script_name',
            '--out_dir_path', self.temp_dir,
            '--num_samples_per_job', str(self.num_samples_per_job),
            '--p', str(self.p)
        ]

        with patch('sys.argv', test_args):
            sample_statistics_main()

        # Check output file exists with correct func_id (2-1=1) and verify content
        self._assert_sample_statistics_output(func_id=1, seed=456)


class TestMpleDataDistributedPaging(unittest.TestCase):
    """Tests for mple_data_distributed_paging.py main function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.n_nodes = 5
        self.num_edges_per_job = 11  # 20 edges total, so chunks of 11 and 9 to test residual

        # Ensure we're actually testing chunking (at least 2 chunks)
        max_edges = self.n_nodes ** 2 - self.n_nodes
        assert self.num_edges_per_job < max_edges, \
            f"num_edges_per_job ({self.num_edges_per_job}) must be < max_edges ({max_edges}) to test chunking"

        # Create data directory
        data_dir = os.path.join(self.temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Create a metric collection with multiple metrics
        metrics = [NumberOfEdgesDirected(), InDegree()]
        self.metric_collection = MetricsCollection(
            metrics=metrics,
            n_nodes=self.n_nodes,
            is_directed=True
        )

        # Create observed network
        np.random.seed(42)
        self.observed_networks = np.random.randint(0, 2, (self.n_nodes, self.n_nodes, 1)).astype(float)
        np.fill_diagonal(self.observed_networks[:, :, 0], 0)

        # Save metric collection and observed networks
        with open(os.path.join(data_dir, 'metric_collection.pkl'), 'wb') as f:
            pickle.dump(self.metric_collection, f)
        with open(os.path.join(data_dir, 'observed_networks.pkl'), 'wb') as f:
            pickle.dump(self.observed_networks, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch.dict(os.environ, {'LSB_JOBINDEX': '1'})
    def test_main_creates_output(self):
        """Test that MPLE data chunks are correctly created and saved."""
        test_args = [
            'script_name',
            '--out_dir_path', self.temp_dir,
            '--num_edges_per_job', str(self.num_edges_per_job)
        ]

        with patch('sys.argv', test_args):
            mple_data_paging_main()

        # Check output file exists
        output_path = os.path.join(self.temp_dir, 'mple_data_paged_chunks', '0.npz')
        self.assertTrue(os.path.exists(output_path))

        # Load output
        data = np.load(output_path)
        self.assertIn('Xs_chunk', data)
        self.assertIn('ys_chunk', data)

        Xs_chunk = data['Xs_chunk']
        ys_chunk = data['ys_chunk']

        # Calculate expected values using metric_collection methods
        edge_indices = (0, self.num_edges_per_job)
        expected_Xs = self.metric_collection.prepare_mple_regressors(
            observed_network=None, edge_indices_lims=edge_indices
        )
        expected_ys = self.metric_collection.prepare_mple_labels(
            self.observed_networks, edge_indices
        )

        np.testing.assert_array_almost_equal(Xs_chunk, expected_Xs, decimal=10)
        np.testing.assert_array_almost_equal(ys_chunk, expected_ys, decimal=10)

        # Verify shapes
        np.testing.assert_array_equal(Xs_chunk.shape[1], self.metric_collection.num_of_features)

        data.close()

    @patch.dict(os.environ, {'LSB_JOBINDEX': '1'})
    def test_main_with_sample_weights(self):
        """Test that sample weights are correctly chunked when provided."""
        # Create and save sample weights
        np.random.seed(42)
        max_edges = self.n_nodes ** 2 - self.n_nodes
        sample_weights = np.random.rand(max_edges).astype(np.float32)
        with open(os.path.join(self.temp_dir, 'data', 'sample_weights.pkl'), 'wb') as f:
            pickle.dump(sample_weights, f)

        test_args = [
            'script_name',
            '--out_dir_path', self.temp_dir,
            '--num_edges_per_job', str(self.num_edges_per_job)
        ]

        with patch('sys.argv', test_args):
            mple_data_paging_main()

        # Check output contains weights
        output_path = os.path.join(self.temp_dir, 'mple_data_paged_chunks', '0.npz')
        data = np.load(output_path)

        self.assertIn('weights_chunk', data)
        weights_chunk = data['weights_chunk']
        Xs_chunk = data['Xs_chunk']
        ys_chunk = data['ys_chunk']

        # Calculate expected values
        edge_indices = (0, self.num_edges_per_job)
        expected_weights = self.metric_collection.slice_flat_array_by_edge_indices(
            sample_weights, edge_indices
        ).reshape(-1, 1)
        expected_ys = self.metric_collection.prepare_mple_labels(
            self.observed_networks, edge_indices
        )

        np.testing.assert_array_almost_equal(weights_chunk, expected_weights, decimal=10)
        np.testing.assert_array_almost_equal(ys_chunk, expected_ys, decimal=10)

        # Verify shape matches Xs/ys
        np.testing.assert_array_equal(weights_chunk.shape[0], Xs_chunk.shape[0])
        np.testing.assert_array_equal(weights_chunk.shape[1], 1)

        data.close()

    @patch.dict(os.environ, {'LSB_JOBINDEX': '2'})
    def test_main_second_chunk(self):
        """Test that the second job index creates the correct chunk."""
        test_args = [
            'script_name',
            '--out_dir_path', self.temp_dir,
            '--num_edges_per_job', str(self.num_edges_per_job)
        ]

        with patch('sys.argv', test_args):
            mple_data_paging_main()

        # Check output file exists with correct func_id (2-1=1)
        output_path = os.path.join(self.temp_dir, 'mple_data_paged_chunks', '1.npz')
        self.assertTrue(os.path.exists(output_path))

        # Load and verify values for second (residual) chunk
        data = np.load(output_path)
        Xs_chunk = data['Xs_chunk']
        ys_chunk = data['ys_chunk']

        # Calculate expected values for second chunk (func_id=1)
        # This is the residual chunk: edges 11-20 (9 edges, not 11)
        max_edges = self.n_nodes ** 2 - self.n_nodes
        edge_indices = (self.num_edges_per_job, max_edges)
        expected_Xs = self.metric_collection.prepare_mple_regressors(
            observed_network=None, edge_indices_lims=edge_indices
        )
        expected_ys = self.metric_collection.prepare_mple_labels(
            self.observed_networks, edge_indices
        )

        # Verify this is indeed a residual chunk (smaller than num_edges_per_job)
        expected_residual_size = max_edges - self.num_edges_per_job
        self.assertEqual(Xs_chunk.shape[0], expected_residual_size)
        self.assertLess(expected_residual_size, self.num_edges_per_job)

        np.testing.assert_array_almost_equal(Xs_chunk, expected_Xs, decimal=10)
        np.testing.assert_array_almost_equal(ys_chunk, expected_ys, decimal=10)

        data.close()


if __name__ == '__main__':
    unittest.main()
