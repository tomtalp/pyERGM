import unittest
from unittest.mock import MagicMock

import numpy as np

from pyERGM.convergence import (
    ConvergenceTester,
    HotellingTester,
    ZeroGradNormTester,
    ObservedBootstrapTester,
    ModelBootstrapTester,
)
from pyERGM.constants import ConvergenceCriterion, OptimizationResult, CovMatrixEstimationMethod, \
    DataBootstrapSplittingMethod


class TestFactory(unittest.TestCase):
    """Tests for the ConvergenceTester.create factory method."""

    def _make_kwargs(self):
        return dict(
            observed_features=np.array([1.0, 2.0]),
            l2_grad_thresh=1e-3,
            sliding_grad_window_k=5,
            opt_steps=100,
            num_of_features=2,
            hotelling_confidence=0.5,
            observed_networks=np.zeros((3, 3, 1)),
            metrics_collection=MagicMock(),
            data_splitting_method=DataBootstrapSplittingMethod.UNIFORM,
            num_subsamples_data=10,
            num_model_sub_samples=10,
            model_subsample_size=10,
            bootstrap_convergence_confidence=0.95,
            bootstrap_convergence_num_stds_away_thr=1,
            observed_cov_mat_est_method=CovMatrixEstimationMethod.NAIVE,
            model_boot_cov_mat_est_method=CovMatrixEstimationMethod.NAIVE,
        )

    def test_factory_returns_hotelling(self):
        tester = ConvergenceTester.create(ConvergenceCriterion.HOTELLING, **self._make_kwargs())
        self.assertIsInstance(tester, HotellingTester)

    def test_factory_returns_zero_grad_norm(self):
        tester = ConvergenceTester.create(ConvergenceCriterion.ZERO_GRAD_NORM, **self._make_kwargs())
        self.assertIsInstance(tester, ZeroGradNormTester)

    def test_factory_returns_model_bootstrap(self):
        tester = ConvergenceTester.create(ConvergenceCriterion.MODEL_BOOTSTRAP, **self._make_kwargs())
        self.assertIsInstance(tester, ModelBootstrapTester)

    def test_factory_raises_for_unknown_criterion(self):
        with self.assertRaises(ValueError):
            ConvergenceTester.create("not_a_real_criterion", **self._make_kwargs())


class TestRequiresCovarianceEstimation(unittest.TestCase):
    def test_hotelling_requires_covariance(self):
        tester = HotellingTester(observed_features=np.array([1.0]), confidence=0.5)
        self.assertTrue(tester.requires_covariance_estimation)

    def test_zero_grad_norm_does_not_require_covariance(self):
        tester = ZeroGradNormTester(l2_grad_thresh=1e-3, sliding_grad_window_k=5,
                                    opt_steps=10, num_of_features=2)
        self.assertFalse(tester.requires_covariance_estimation)

    def test_model_bootstrap_does_not_require_covariance(self):
        tester = ModelBootstrapTester(
            observed_features=np.array([1.0]),
            num_model_sub_samples=10,
            model_subsample_size=5,
            confidence=0.9,
            stds_away_thr=1.0,
            cov_mat_est_method=CovMatrixEstimationMethod.NAIVE,
        )
        self.assertFalse(tester.requires_covariance_estimation)


class TestHotellingTester(unittest.TestCase):
    def test_converges_with_identical_features(self):
        """When observed == model mean, the test statistic is 0 and should converge."""
        features = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)
        tester = HotellingTester(observed_features=features, confidence=0.5)
        tester.update(mean_features=features, inv_estimated_cov_matrix=cov, sample_size=100)
        result = tester.test()
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.statistic, 0.0)

    def test_does_not_converge_with_distant_features(self):
        """When features are far apart, should not converge."""
        observed = np.array([0.0, 0.0])
        mean = np.array([100.0, 100.0])
        cov = np.eye(2)
        tester = HotellingTester(observed_features=observed, confidence=0.5)
        tester.update(mean_features=mean, inv_estimated_cov_matrix=cov, sample_size=100)
        result = tester.test()
        self.assertFalse(result.success)
        self.assertGreater(result.statistic, result.threshold)


class TestZeroGradNormTester(unittest.TestCase):
    def test_converges_with_zero_gradients(self):
        tester = ZeroGradNormTester(l2_grad_thresh=0.01, sliding_grad_window_k=3,
                                    opt_steps=10, num_of_features=2)
        for step in range(3):
            tester.update(grad=np.array([0.0, 0.0]), step=step)
        result = tester.test()
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.statistic, 0.0)

    def test_does_not_converge_with_large_gradients(self):
        tester = ZeroGradNormTester(l2_grad_thresh=0.01, sliding_grad_window_k=3,
                                    opt_steps=10, num_of_features=2)
        tester.update(grad=np.array([10.0, 10.0]), step=0)
        result = tester.test()
        self.assertFalse(result.success)
        self.assertGreater(result.statistic, 0.01)

    def test_sliding_window_averages_correctly(self):
        """With window_k=2, should average last 2 gradients."""
        tester = ZeroGradNormTester(l2_grad_thresh=0.01, sliding_grad_window_k=2,
                                    opt_steps=10, num_of_features=1)
        tester.update(grad=np.array([10.0]), step=0)
        tester.update(grad=np.array([0.0]), step=1)
        result = tester.test()
        # Mean of [10.0, 0.0] = 5.0, norm = 5.0
        self.assertAlmostEqual(result.statistic, 5.0)

    def test_always_returns_result(self):
        """Even when not converged, should return an OptimizationResult (bug fix)."""
        tester = ZeroGradNormTester(l2_grad_thresh=0.01, sliding_grad_window_k=3,
                                    opt_steps=10, num_of_features=2)
        tester.update(grad=np.array([1.0, 1.0]), step=0)
        result = tester.test()
        self.assertIsInstance(result, OptimizationResult)
        self.assertFalse(result.success)


class TestObservedBootstrapTester(unittest.TestCase):
    def test_raises_for_multi_network_input(self):
        """Should raise ValueError when observed_networks has more than 1 network."""
        observed_networks = np.zeros((3, 3, 2))  # 2 networks
        metrics_collection = MagicMock()
        metrics_collection.metrics = []
        with self.assertRaises(ValueError, msg="observed_bootstrap doesn't support multiple networks!"):
            ObservedBootstrapTester(
                observed_features=np.array([1.0]),
                observed_networks=observed_networks,
                metrics_collection=metrics_collection,
                data_splitting_method=DataBootstrapSplittingMethod.UNIFORM,
                num_subsamples_data=10,
                num_model_sub_samples=10,
                model_subsample_size=10,
                confidence=0.95,
                stds_away_thr=1,
                observed_cov_mat_est_method=CovMatrixEstimationMethod.NAIVE,
            )

    def test_raises_for_missing_bootstrap_method(self):
        """Should raise when a metric lacks calculate_bootstrapped_features."""
        observed_networks = np.zeros((3, 3, 1))
        mock_metric = MagicMock(spec=['metric_name'])  # no attributes at all
        mock_metric.name = "bad_metric"
        metrics_collection = MagicMock()
        metrics_collection.metrics = [mock_metric]
        with self.assertRaises(ValueError, msg="calculate_bootstrapped_features"):
            ObservedBootstrapTester(
                observed_features=np.array([1.0]),
                observed_networks=observed_networks,
                metrics_collection=metrics_collection,
                data_splitting_method=DataBootstrapSplittingMethod.UNIFORM,
                num_subsamples_data=10,
                num_model_sub_samples=10,
                model_subsample_size=10,
                confidence=0.95,
                stds_away_thr=1,
                observed_cov_mat_est_method=CovMatrixEstimationMethod.NAIVE,
            )


class TestModelBootstrapTester(unittest.TestCase):
    def test_smoke_test_with_synthetic_data(self):
        """Smoke test: ModelBootstrapTester runs without errors on synthetic data."""
        np.random.seed(42)
        num_features = 3
        num_samples = 200
        observed_features = np.array([1.0, 2.0, 3.0])
        # Generate features that are close to observed
        sampled_features = observed_features[:, None] + np.random.randn(num_features, num_samples) * 0.01

        tester = ModelBootstrapTester(
            observed_features=observed_features,
            num_model_sub_samples=5,
            model_subsample_size=20,
            confidence=0.95,
            stds_away_thr=5,
            cov_mat_est_method=CovMatrixEstimationMethod.NAIVE,
        )
        tester.update(features_of_net_samples=sampled_features)
        result = tester.test()
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsNotNone(result.statistic)
        self.assertIsNotNone(result.threshold)


if __name__ == "__main__":
    unittest.main()
