import unittest
import numpy as np
from pyERGM.ergm import ERGM
from pyERGM.metrics import NumberOfEdgesDirected, InDegree, OutDegree, Reciprocity


class TestMPLENoneHandling(unittest.TestCase):
    """Tests for handling observed_networks=None in get_mple_prediction()"""

    def setUp(self):
        """Set up test fixtures"""
        self.n_nodes = 10
        np.random.seed(42)
        # Create a simple directed network for testing
        self.observed_network = np.random.randint(0, 2, (self.n_nodes, self.n_nodes))
        np.fill_diagonal(self.observed_network, 0)  # No self-loops

    def test_dyadic_independent_multiple_metrics_with_none(self):
        """Test dyadic-independent model with multiple metrics and observed_networks=None"""
        model = ERGM(
            self.n_nodes,
             [NumberOfEdgesDirected(), InDegree(), OutDegree()],
            is_directed=True
        )
        model.fit(self.observed_network, method='MPLE')

        # This should work without error
        probs = model.get_mple_prediction()

        # Verify output
        self.assertEqual(probs.shape, (self.n_nodes, self.n_nodes))
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))

    def test_dyadic_dependent_with_none_raises(self):
        """Test that dyadic-dependent models raise error with observed_networks=None"""
        # Create model with dyadic-dependent metric
        model = ERGM(
            self.n_nodes,
             [NumberOfEdgesDirected(), Reciprocity()],
            is_directed=True
        )

        # Fit the model first
        model.fit(self.observed_network, method='MPLE')

        # This should raise ValueError
        with self.assertRaises(ValueError) as context:
            model.get_mple_prediction(observed_networks=None)

        # Check error message content
        error_msg = str(context.exception)
        self.assertIn("dyadic-dependent", error_msg)
        self.assertIn("reciprocity", error_msg.lower())


    def test_3d_array_handling_with_none_check(self):
        """Test that 3D array dimension extraction works correctly"""
        # Create model with dyadic-independent metric
        model = ERGM(
            self.n_nodes,
             [NumberOfEdgesDirected()],
            is_directed=True
        )

        # Create 3D array (multiple networks)
        networks_3d = np.stack([self.observed_network] * 3, axis=-1)

        # Fit the model
        model.fit(self.observed_network, method='MPLE')

        # This should work and extract first network
        probs = model.get_mple_prediction(observed_networks=networks_3d)

        # Verify output
        self.assertEqual(probs.shape, (self.n_nodes, self.n_nodes))

    def test_distributed_dyadic_dependent_raises(self):
        """Test that distributed mode raises error for dyadic-dependent models"""
        # Create model with dyadic-dependent metric
        model = ERGM(
            self.n_nodes,
             [NumberOfEdgesDirected(), Reciprocity()],
            is_directed=True
        )

        # Fit the model
        model.fit(self.observed_network, method='MPLE')

        # Enable distributed optimization
        model._is_distributed_optimization = True

        # This should raise ValueError about distributed incompatibility
        with self.assertRaises(ValueError) as context:
            model.get_mple_prediction(observed_networks=self.observed_network)

        # Check error message
        error_msg = str(context.exception)
        self.assertIn("Distributed", error_msg)
        self.assertIn("dyadic-independent", error_msg)

if __name__ == '__main__':
    unittest.main()
