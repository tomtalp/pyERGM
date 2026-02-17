"""
Convergence testers for ERGM MCMLE optimization.

Provides an abstract base class and concrete implementations for different
convergence criteria used during MCMLE fitting.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import f

from pyERGM.constants import ConvergenceCriterion, OptimizationResult, CovMatrixEstimationMethod, \
    DataBootstrapSplittingMethod
from pyERGM.metrics import MetricsCollection
from pyERGM.utils import covariance_matrix_estimation, expand_net_dims

__all__ = [
    'ConvergenceTester',
    'HotellingTester',
    'ZeroGradNormTester',
    'ObservedBootstrapTester',
    'ModelBootstrapTester',
]


def _get_subsample_features(
        sampled_networks_features: np.ndarray,
        num_subsamples: int,
        subsample_size: int,
) -> np.ndarray:
    """
    Receives a sample of networks, and subsample for `num_subsamples` times, each time with `subsample_size` networks.
    For each subsample, calculates the sample statistics and reshapes the result to a tensor of shape
    (num_of_features, num_subsamples, subsample_size).

    Parameters
    ----------
    sampled_networks_features : np.ndarray
        Features of a sample of networks that will be used for subsampling

    num_subsamples : int
        The number of subsamples to draw from the sample.

    subsample_size : int
        The size of each subsample.

    Returns
    -------
    sub_samples_features : np.ndarray
        A tensor of shape (num_of_features, num_subsamples, subsample_size) containing the features of all subsamples.
    """
    sample_size = sampled_networks_features.shape[1]

    sub_sample_indices = np.random.choice(np.arange(sample_size), size=num_subsamples * subsample_size)
    sub_samples_features = sampled_networks_features[:, sub_sample_indices]
    sub_samples_features = sub_samples_features.reshape(-1, num_subsamples, subsample_size)

    return sub_samples_features


class ConvergenceTester(ABC):
    """
    Abstract base class for ERGM convergence testers.

    Each subclass encapsulates a specific convergence criterion, maintaining
    its own state and providing a uniform API for updating per-iteration data
    and testing for convergence.
    """

    @property
    @abstractmethod
    def requires_covariance_estimation(self) -> bool:
        """Whether this tester requires a covariance matrix to be estimated each iteration."""
        ...

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Receive per-iteration data from the optimization loop.

        Each subclass picks out the keyword arguments it needs.
        """
        ...

    @abstractmethod
    def test(self) -> OptimizationResult:
        """
        Test for convergence and return an OptimizationResult.

        Always returns a result (including when not converged, with success=False).
        """
        ...

    @staticmethod
    def create(criterion: ConvergenceCriterion, **kwargs) -> "ConvergenceTester":
        """
        Factory method that creates the appropriate ConvergenceTester subclass.

        Parameters
        ----------
        criterion : ConvergenceCriterion
            The convergence criterion to use.
        **kwargs
            Keyword arguments forwarded to the relevant subclass constructor.

        Returns
        -------
        ConvergenceTester
            An instance of the appropriate subclass.
        """
        if criterion == ConvergenceCriterion.HOTELLING:
            return HotellingTester(
                observed_features=kwargs["observed_features"],
                confidence=kwargs["hotelling_confidence"],
            )
        elif criterion == ConvergenceCriterion.ZERO_GRAD_NORM:
            return ZeroGradNormTester(
                l2_grad_thresh=kwargs["l2_grad_thresh"],
                sliding_grad_window_k=kwargs["sliding_grad_window_k"],
                opt_steps=kwargs["opt_steps"],
                num_of_features=kwargs["num_of_features"],
            )
        elif criterion == ConvergenceCriterion.OBSERVED_BOOTSTRAP:
            return ObservedBootstrapTester(
                observed_features=kwargs["observed_features"],
                observed_networks=kwargs["observed_networks"],
                metrics_collection=kwargs["metrics_collection"],
                data_splitting_method=kwargs["data_splitting_method"],
                num_subsamples_data=kwargs["num_subsamples_data"],
                num_model_sub_samples=kwargs["num_model_sub_samples"],
                model_subsample_size=kwargs["model_subsample_size"],
                confidence=kwargs["bootstrap_convergence_confidence"],
                stds_away_thr=kwargs["bootstrap_convergence_num_stds_away_thr"],
                observed_cov_mat_est_method=kwargs["observed_cov_mat_est_method"],
            )
        elif criterion == ConvergenceCriterion.MODEL_BOOTSTRAP:
            return ModelBootstrapTester(
                observed_features=kwargs["observed_features"],
                num_model_sub_samples=kwargs["num_model_sub_samples"],
                model_subsample_size=kwargs["model_subsample_size"],
                confidence=kwargs["bootstrap_convergence_confidence"],
                stds_away_thr=kwargs["bootstrap_convergence_num_stds_away_thr"],
                cov_mat_est_method=kwargs["model_boot_cov_mat_est_method"],
            )
        else:
            raise ValueError(f"Unknown convergence criterion: {criterion}")


class HotellingTester(ConvergenceTester):
    """
    Convergence tester using Hotelling's T-squared test.

    Tests H0: E[g(Y)] = g(y_obs), i.e., the expected features under the model
    equal the observed features. Convergence is declared when we fail to reject H0.

    Following Krivitsky (2017), a high alpha (like 0.5) is recommended because
    we want to *not reject* the null hypothesis (convergence) rather than reject it.
    """

    def __init__(self, observed_features: np.ndarray, confidence: float):
        self._observed_features = observed_features
        self._confidence = confidence
        self._mean_features = None
        self._inv_estimated_cov_matrix = None
        self._sample_size = None

    @property
    def requires_covariance_estimation(self) -> bool:
        return True

    def update(self, **kwargs) -> None:
        self._mean_features = kwargs["mean_features"]
        self._inv_estimated_cov_matrix = kwargs["inv_estimated_cov_matrix"]
        self._sample_size = kwargs["sample_size"]

    def test(self) -> OptimizationResult:
        dist = mahalanobis(self._observed_features, self._mean_features, self._inv_estimated_cov_matrix)
        hotelling_t_statistic = self._sample_size * dist * dist

        num_of_features = self._observed_features.shape[0]

        hotelling_as_f_statistic = (
                (self._sample_size - num_of_features) /
                (num_of_features * (self._sample_size - 1)) *
                hotelling_t_statistic
        )

        hotelling_critical_value = f.ppf(
            1 - self._confidence, num_of_features, self._sample_size - num_of_features
        )

        return OptimizationResult(
            success=hotelling_as_f_statistic <= hotelling_critical_value,
            statistic=hotelling_as_f_statistic,
            threshold=hotelling_critical_value,
        )


class ZeroGradNormTester(ConvergenceTester):
    """
    Convergence tester based on the L2 norm of the gradient sliding window mean.

    Maintains an internal gradient history and sliding window. Convergence is
    declared when the norm of the sliding window mean gradient falls below a threshold.
    """

    def __init__(self, l2_grad_thresh: float, sliding_grad_window_k: int, opt_steps: int, num_of_features: int):
        self._l2_grad_thresh = l2_grad_thresh
        self._sliding_grad_window_k = sliding_grad_window_k
        self._grads = np.zeros((opt_steps, num_of_features))
        self._sliding_window_mean = None

    @property
    def requires_covariance_estimation(self) -> bool:
        return False

    def update(self, **kwargs) -> None:
        step = kwargs["step"]
        grad = kwargs["grad"]
        self._grads[step] = grad
        idx_start = max(0, step - self._sliding_grad_window_k + 1)
        self._sliding_window_mean = self._grads[idx_start:step + 1].mean(axis=0)

    def test(self) -> OptimizationResult:
        cur_window_norm = np.linalg.norm(self._sliding_window_mean)
        return OptimizationResult(
            success=cur_window_norm <= self._l2_grad_thresh,
            statistic=cur_window_norm,
            threshold=self._l2_grad_thresh,
        )


class ObservedBootstrapTester(ConvergenceTester):
    """
    Convergence tester using bootstrapped Mahalanobis distance
    from observed features, under a data noise model based on
    bootstrapped subnetworks from the data.

    Pre-computes the inverse observed covariance matrix from bootstrapped features
    of the observed data ("noise model"). Each iteration, subsamples from model-generated
    network features and checks Mahalanobis distance against the observed features.
    """

    def __init__(
            self,
            observed_features: np.ndarray,
            observed_networks: np.ndarray,
            metrics_collection: MetricsCollection,
            data_splitting_method: DataBootstrapSplittingMethod,
            num_subsamples_data: int,
            num_model_sub_samples: int,
            model_subsample_size: int,
            confidence: float,
            stds_away_thr: float,
            observed_cov_mat_est_method: CovMatrixEstimationMethod,
    ):
        # Normalize input to 3D using the existing utility for consistent 2D->3D conversion
        observed_networks = expand_net_dims(observed_networks)

        # Now check for multiple networks
        if observed_networks.shape[-1] > 1:
            raise ValueError("ConvergenceCriterion.OBSERVED_BOOTSTRAP doesn't support multiple networks!")
        metrics_collection.validate_supports_observed_bootstrap()

        self._observed_features = observed_features
        self._num_model_sub_samples = num_model_sub_samples
        self._model_subsample_size = model_subsample_size
        self._confidence = confidence
        self._stds_away_thr = stds_away_thr

        bootstrapped_features = metrics_collection.bootstrap_observed_features(
            observed_networks, num_subsamples=num_subsamples_data, splitting_method=data_splitting_method
        )
        observed_covariance = covariance_matrix_estimation(
            bootstrapped_features, bootstrapped_features.mean(axis=1), method=observed_cov_mat_est_method
        )
        if np.all(observed_covariance == 0):
            raise RuntimeError("The observed covariance matrix is all-zeros (bootstrapped features are identical), "
                               "the convergence test can never pass.")
        self._inv_observed_covariance = np.linalg.pinv(observed_covariance)

        self._features_of_net_samples = None

    @property
    def requires_covariance_estimation(self) -> bool:
        return False

    def update(self, **kwargs) -> None:
        self._features_of_net_samples = kwargs["features_of_net_samples"]

    def test(self) -> OptimizationResult:
        mahalanobis_dists = np.zeros(self._num_model_sub_samples)

        sub_samples_features = _get_subsample_features(
            self._features_of_net_samples, self._num_model_sub_samples, self._model_subsample_size
        )
        mean_per_subsample = sub_samples_features.mean(axis=2)

        for cur_subsam_idx in range(self._num_model_sub_samples):
            cur_subsample_feat_mean = mean_per_subsample[:, cur_subsam_idx]
            mahalanobis_dists[cur_subsam_idx] = mahalanobis(
                self._observed_features, cur_subsample_feat_mean, self._inv_observed_covariance
            )

        empirical_threshold = np.quantile(mahalanobis_dists, self._confidence)

        return OptimizationResult(
            success=empirical_threshold < self._stds_away_thr,
            statistic=empirical_threshold,
            threshold=self._stds_away_thr,
        )


class ModelBootstrapTester(ConvergenceTester):
    """
    Convergence tester using bootstrapped Mahalanobis distance from model samples.

    Repeatedly subsamples from model-generated network features, computes
    per-subsample covariance, and checks whether the Mahalanobis distance
    to the observed features is within acceptable bounds.
    """

    def __init__(
            self,
            observed_features: np.ndarray,
            num_model_sub_samples: int,
            model_subsample_size: int,
            confidence: float,
            stds_away_thr: float,
            cov_mat_est_method: CovMatrixEstimationMethod,
    ):
        self._observed_features = observed_features
        self._num_model_sub_samples = num_model_sub_samples
        self._model_subsample_size = model_subsample_size
        self._confidence = confidence
        self._stds_away_thr = stds_away_thr
        self._cov_mat_est_method = cov_mat_est_method
        self._features_of_net_samples = None

    @property
    def requires_covariance_estimation(self) -> bool:
        return False

    def update(self, **kwargs) -> None:
        self._features_of_net_samples = kwargs["features_of_net_samples"]

    def test(self) -> OptimizationResult:
        mahalanobis_dists = np.zeros(self._num_model_sub_samples)

        sub_samples_features = _get_subsample_features(
            self._features_of_net_samples, self._num_model_sub_samples, self._model_subsample_size
        )
        mean_per_subsample = sub_samples_features.mean(axis=2)

        for cur_subsam_idx in range(self._num_model_sub_samples):
            cur_subsample_feat = sub_samples_features[:, cur_subsam_idx, :]
            cur_subsample_feat_mean = mean_per_subsample[:, cur_subsam_idx]
            model_covariance_matrix = covariance_matrix_estimation(
                cur_subsample_feat, cur_subsample_feat_mean, method=self._cov_mat_est_method
            )

            if np.all(model_covariance_matrix == 0):
                mahalanobis_dists[cur_subsam_idx] = np.inf
                continue

            inv_model_cov_matrix = np.linalg.pinv(model_covariance_matrix)
            mahalanobis_dists[cur_subsam_idx] = mahalanobis(
                self._observed_features, cur_subsample_feat_mean, inv_model_cov_matrix
            )

        empirical_threshold = np.quantile(mahalanobis_dists, self._confidence)

        return OptimizationResult(
            success=empirical_threshold < self._stds_away_thr,
            statistic=empirical_threshold,
            threshold=self._stds_away_thr,
        )
