import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit

import pyERGM.mple_optimization as mo
from pyERGM.constants import MPLEOptimizationMethod
from pyERGM.ergm import ERGM
from pyERGM.metrics import (
    NodeAttrSum,
    NumberOfEdgesDirected,
    NumberOfEdgesTypesDirected,
    SumDistancesConnectedNeurons,
    TotalReciprocity,
)
from pyERGM.mple_optimization import (
    analytical_minus_log_like_grad_local,
    analytical_minus_log_likelihood_hessian_distributed,
    analytical_minus_log_likelihood_hessian_local,
    analytical_minus_log_likelihood_local,
    calc_logistic_regression_log_likelihood_grad,
    calc_logistic_regression_log_likelihood_hessian,
    calc_logistic_regression_predictions,
    calc_logistic_regression_predictions_log_likelihood,
    mple_logistic_regression_optimization,
    mple_reciprocity_logistic_regression_optimization,
)


def _hessian_test_data():
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(2000, 4))
    ys = (rng.random((2000, 1)) < expit(Xs @ np.array([0.5, -1, 0.3, 0.8])).reshape(-1, 1)).astype(float)
    thetas = rng.normal(size=4)
    return Xs, ys, thetas


def _central_differences(grad_fn, thetas, h=1e-6):
    cols = []
    for j in range(thetas.size):
        e = np.zeros(thetas.size)
        e[j] = h
        cols.append((grad_fn(thetas + e) - grad_fn(thetas - e)) / (2 * h))
    return np.stack(cols, axis=1)


class TestMPLEHessian(unittest.TestCase):
    def test_log_likelihood_hessian_matches_finite_differences_and_is_negative_definite(self):
        Xs, ys, thetas = _hessian_test_data()

        def grad(t):
            p = calc_logistic_regression_predictions(Xs, t[:, None])
            return calc_logistic_regression_log_likelihood_grad(Xs, p, ys).reshape(-1)

        p = calc_logistic_regression_predictions(Xs, thetas[:, None])
        H = calc_logistic_regression_log_likelihood_hessian(Xs, p)
        H_fd = _central_differences(grad, thetas)
        np.testing.assert_allclose(H, H_fd, rtol=1e-5, atol=1e-6)
        self.assertTrue(np.all(np.linalg.eigvalsh((H + H.T) / 2) < 0))

    def test_minus_log_likelihood_hessian_local_matches_finite_differences_and_is_positive_definite(self):
        Xs, ys, thetas = _hessian_test_data()
        H = analytical_minus_log_likelihood_hessian_local(thetas, Xs, ys, 1e-10, np.empty(0))
        H_fd = _central_differences(
            lambda t: analytical_minus_log_like_grad_local(t, Xs, ys, 1e-10, np.empty(0)), thetas)
        np.testing.assert_allclose(H, H_fd, rtol=1e-5, atol=1e-6)
        self.assertTrue(np.all(np.linalg.eigvalsh((H + H.T) / 2) > 0))

    def test_newton_cg_converges_on_ill_conditioned_problem(self):
        rng = np.random.default_rng(0)
        N, F = 20000, 6
        Xs = rng.normal(size=(N, F)) * np.array([1, 1, 10, 0.1, 30, 1])
        Xs[:, 0] = 1
        ys = (rng.random((N, 1)) < expit(Xs @ np.array([-2, -1, 0.05, 3, 0.02, 0.8])).reshape(-1, 1)).astype(float)
        # Column scales 0.1 to 30 (cond(H) about 9e4): Newton-CG with a wrongly signed Hessian does not converge here.
        res = minimize(analytical_minus_log_likelihood_local, np.zeros(F), args=(Xs, ys, 1e-10, np.empty(0)),
                       jac=analytical_minus_log_like_grad_local, hess=analytical_minus_log_likelihood_hessian_local,
                       method="Newton-CG")
        self.assertTrue(res.success)
        self.assertLess(np.abs(res.jac).max(), 1e-2)

    def test_distributed_hessian_sign_and_call(self):
        M = np.array([[2., 0.], [0., 3.]])
        with patch("pyERGM.mple_optimization.distributed_logistic_regression_optimization_step",
                   side_effect=lambda data_path, thetas, funcs: (-M,)) as mock_step:
            result = analytical_minus_log_likelihood_hessian_distributed(np.zeros(2), Path("/nonexistent"))
        np.testing.assert_array_equal(result, M)
        self.assertEqual(len(mock_step.call_args.args), 3)


# ---------------------------------------------------------------------------------------------------------------------
# Helpers for the mocked-minimize tests
# ---------------------------------------------------------------------------------------------------------------------

def _small_directed_problem(metrics, reciprocity_regressors=False):
    """
    A fixed 5-node directed network, its MetricsCollection, the number of features F and the feature scales that the
    optimization functions compute from the regressors (`prepare_mple_reciprocity_regressors` if
    `reciprocity_regressors`, else `prepare_mple_regressors`).
    """
    net = np.zeros((5, 5, 1))
    net[0, 1, 0] = net[1, 0, 0] = net[2, 3, 0] = net[3, 4, 0] = 1
    metrics_collection = ERGM(5, metrics, is_directed=True, fix_collinearity=False)._metrics_collection
    if reciprocity_regressors:
        Xs = metrics_collection.prepare_mple_reciprocity_regressors()
    else:
        Xs = metrics_collection.prepare_mple_regressors(net[..., 0])
    feature_scales = mo._feature_scales(Xs)
    # With s = 1, `* s` and `/ s` are indistinguishable and the mapping assertions would be vacuous.
    assert not np.all(feature_scales == 1)
    return metrics_collection, net, metrics_collection.calc_num_of_features(), feature_scales


@contextmanager
def _mock_minimize(returned_x, success=True, message="ok"):
    """Replace the `minimize` used by mple_optimization with a mock that returns `returned_x`; yields the mock."""
    result = OptimizeResult(x=returned_x, jac=np.zeros_like(returned_x), success=success, message=message)
    with patch("pyERGM.mple_optimization.minimize", MagicMock(return_value=result)) as mock_minimize:
        yield mock_minimize


def _x0_passed_to(mock_minimize):
    """The starting point (second positional argument) of the last call to the mocked `minimize`."""
    return mock_minimize.call_args.args[1]


def _kwargs_passed_to(mock_minimize):
    return mock_minimize.call_args.kwargs


@contextmanager
def _mock_distributed_io():
    """Stub out everything the distributed path does besides calling `minimize`: data chunks, predictions, cleanup."""
    with patch.object(mo, "distributed_mple_data_chunks_calculations", return_value=Path("/nonexistent")), \
            patch.object(mo, "analytical_logistic_regression_predictions_distributed", return_value=np.zeros(1)), \
            patch("shutil.rmtree"):
        yield


class TestMPLEOptimizerCalls(unittest.TestCase):
    """
    Pins down what scipy's `minimize` receives from the MPLE optimization functions (method, options, and the starting
    point x0 in scaled coordinates theta' = theta * s) and that the results come back in theta units.
    `minimize` is mocked, so no optimization runs: the mock returns a chosen theta' and the tests check the
    function's output and the arguments of the call.
    """

    def _check_lbfgsb_call_and_scaling(self, optimize_fn, metrics_collection, net, F, feature_scales):
        """Shared checks for the two local L-BFGS-B functions (plain MPLE and reciprocity)."""
        optimizer_solution_in_scaled_coords = 2 * feature_scales
        expected_thetas = np.full(F, 2.0)
        with _mock_minimize(optimizer_solution_in_scaled_coords) as mock_minimize:
            thetas, _, _ = optimize_fn(metrics_collection, net)
            self.assertEqual(_kwargs_passed_to(mock_minimize)["method"], "L-BFGS-B")
            self.assertEqual(_kwargs_passed_to(mock_minimize)["options"], mo._MPLE_LBFGSB_OPTIONS)
            # A copy, so that scipy can't modify the module-level constant.
            self.assertIsNot(_kwargs_passed_to(mock_minimize)["options"], mo._MPLE_LBFGSB_OPTIONS)
            np.testing.assert_array_equal(_x0_passed_to(mock_minimize), np.zeros(F))
            np.testing.assert_array_equal(thetas, expected_thetas)

            user_initial_thetas = np.full(F, 0.5)
            optimize_fn(metrics_collection, net, initial_thetas=user_initial_thetas)
            np.testing.assert_array_equal(_x0_passed_to(mock_minimize), user_initial_thetas * feature_scales)

    def test_reciprocity_lbfgsb_call(self):
        """
        The mock returns theta' = 2 * s, so the function must return theta = 2 (theta' / s); a caller's initial
        theta = 0.5 must be sent as theta' = 0.5 * s; the default start is zeros; options are the strict L-BFGS-B ones.
        """
        metrics_collection, net, F, feature_scales = _small_directed_problem(
            [NumberOfEdgesDirected(), TotalReciprocity()], reciprocity_regressors=True)
        self._check_lbfgsb_call_and_scaling(
            mple_reciprocity_logistic_regression_optimization, metrics_collection, net, F, feature_scales)

    def test_local_mple_lbfgsb_call(self):
        """Same checks as the reciprocity test, for the plain local MPLE function."""
        metrics_collection, net, F, feature_scales = _small_directed_problem(
            [NumberOfEdgesDirected(), TotalReciprocity()])
        self._check_lbfgsb_call_and_scaling(
            mple_logistic_regression_optimization, metrics_collection, net, F, feature_scales)

    def _check_call_without_options(self, optimization_method, is_distributed):
        """
        Shared checks for the calls that get no L-BFGS-B options: zero start, no "options" kwarg, and a "hess" kwarg
        for Newton-CG. The local path runs in scaled coordinates (returned theta' = 2 * s comes back as theta = 2);
        the distributed path is not rescaled (the returned theta = 2 is returned as is).
        """
        metrics_collection, net, F, feature_scales = _small_directed_problem(
            [NumberOfEdgesDirected(), TotalReciprocity()])
        optimizer_solution = np.full(F, 2.0) if is_distributed else 2 * feature_scales
        with _mock_minimize(optimizer_solution) as mock_minimize:
            if is_distributed:
                with _mock_distributed_io():
                    thetas, _, _ = mple_logistic_regression_optimization(
                        metrics_collection, net, optimization_method=optimization_method, is_distributed=True)
            else:
                thetas, _, _ = mple_logistic_regression_optimization(
                    metrics_collection, net, optimization_method=optimization_method)
        self.assertNotIn("options", _kwargs_passed_to(mock_minimize))
        if optimization_method == MPLEOptimizationMethod.NEWTON_CG:
            self.assertIn("hess", _kwargs_passed_to(mock_minimize))
        np.testing.assert_array_equal(_x0_passed_to(mock_minimize), np.zeros(F))
        np.testing.assert_array_equal(thetas, np.full(F, 2.0))

    def test_local_newton_cg_call(self):
        """
        Newton-CG is also run in scaled coordinates (returned theta' = 2 * s comes back as theta = 2) from a zero
        start, gets a Hessian, and gets no L-BFGS-B options.
        """
        self._check_call_without_options(MPLEOptimizationMethod.NEWTON_CG, is_distributed=False)

    def test_distributed_lbfgsb_call(self):
        """
        The distributed path gets only the zero start: no options, and no rescaling (the returned theta = 2 is
        returned as is).
        """
        self._check_call_without_options(MPLEOptimizationMethod.L_BFGS_B, is_distributed=True)

    def test_warning_on_failure(self):
        """When scipy reports failure, scipy's message is logged at WARNING."""
        metrics_collection, net, F, _ = _small_directed_problem([NumberOfEdgesDirected(), TotalReciprocity()])
        with _mock_minimize(np.zeros(F), success=False, message="ABNORMAL: test"):
            with self.assertLogs("pyERGM", level="WARNING") as logs:
                mple_logistic_regression_optimization(metrics_collection, net)
        self.assertTrue(any("ABNORMAL: test" in message for message in logs.output))

    def test_feature_scales(self):
        """
        Root-mean-square per feature; an all-zero feature gets scale 1; accepts the plain 2-D layout and the 3-D
        reciprocity layout (dyads, 4, F); any other layout raises ValueError.
        """
        Xs = np.array([[3.0, 0.0, 1.0],
                       [4.0, 0.0, -1.0]])
        expected_scales = [np.sqrt(12.5), 1.0, 1.0]
        np.testing.assert_allclose(mo._feature_scales(Xs), expected_scales)
        # Reciprocity layout (2 dyads, 4 states, 3 features): each row repeated over the 4 states, so same scales.
        Xs_reciprocity = np.stack([Xs] * 4, axis=1)
        np.testing.assert_allclose(mo._feature_scales(Xs_reciprocity), expected_scales)
        for invalid_Xs in (np.ones(3), np.ones((2, 3, 3)), np.ones((2, 4, 3, 1))):
            with self.assertRaises(ValueError):
                mo._feature_scales(invalid_Xs)

    def test_scaled_coordinates_wrappers_and_jac_mapping(self):
        """
        Checks `_minimize_in_scaled_coordinates` directly. The objective, gradient and Hessian wrappers that scipy
        receives must equal the library functions evaluated at theta = theta' / s (with the chain-rule factors 1/s and
        1/(s s^T)), and the result must be mapped back: x = theta' / s, jac = jac' * s.
        """
        Xs, ys, _ = _hessian_test_data()
        Xs = Xs * np.array([1.0, 5.0, 0.2, 30.0])  # features of very different magnitudes, so s is far from 1
        library_args = (Xs, ys, 1e-10, np.empty(0))
        feature_scales = mo._feature_scales(Xs)
        # With s = 1, `* s` and `/ s` are indistinguishable and the mapping assertions would be vacuous.
        self.assertFalse(np.all(feature_scales == 1))
        user_initial_thetas = np.array([0.1, -0.2, 0.3, 0.05])
        # Arbitrary non-zero values returned by the mocked minimize, to check the mapping back to theta units.
        solution_in_scaled_coords = np.array([1.0, 2.0, 3.0, 4.0])
        jac_in_scaled_coords = np.array([0.5, -1.0, 2.0, 0.25])
        mock_result = OptimizeResult(x=solution_in_scaled_coords, jac=jac_in_scaled_coords, success=True, message="ok")
        with patch("pyERGM.mple_optimization.minimize", MagicMock(return_value=mock_result)) as mock_minimize:
            res = mo._minimize_in_scaled_coordinates(
                analytical_minus_log_likelihood_local, analytical_minus_log_like_grad_local, user_initial_thetas,
                args=library_args, method="Newton-CG", callback=None,
                hess=analytical_minus_log_likelihood_hessian_local)

        np.testing.assert_allclose(_x0_passed_to(mock_minimize), user_initial_thetas * feature_scales)
        # The wrappers are checked at an arbitrary point in scaled coordinates, against the library functions at
        # the corresponding theta.
        eval_point_in_scaled_coords = np.array([0.3, -0.7, 1.1, 0.4])
        theta_at_eval_point = eval_point_in_scaled_coords / feature_scales
        scaled_fun = mock_minimize.call_args.args[0]
        scaled_jac = _kwargs_passed_to(mock_minimize)["jac"]
        scaled_hess = _kwargs_passed_to(mock_minimize)["hess"]
        np.testing.assert_allclose(scaled_fun(eval_point_in_scaled_coords, *library_args),
                                   analytical_minus_log_likelihood_local(theta_at_eval_point, *library_args))
        np.testing.assert_allclose(scaled_jac(eval_point_in_scaled_coords, *library_args),
                                   analytical_minus_log_like_grad_local(theta_at_eval_point, *library_args)
                                   / feature_scales)
        np.testing.assert_allclose(scaled_hess(eval_point_in_scaled_coords, *library_args),
                                   analytical_minus_log_likelihood_hessian_local(theta_at_eval_point, *library_args)
                                   / np.outer(feature_scales, feature_scales))
        np.testing.assert_allclose(res.x, solution_in_scaled_coords / feature_scales)
        np.testing.assert_allclose(res.jac, jac_in_scaled_coords * feature_scales)


class TestRescalingInvariance(unittest.TestCase):
    """
    End-to-end: the same model with a feature measured in different units must give the same fit. Positions in
    meters instead of micrometers shrink the distance feature by 1e-6, so its coefficient must grow by 1e6 and all the
    other coefficients must stay the same.
    """
    N_NODES = 60
    DISTANCE_COEF_IDX = 1  # in the plain model; shifted by one when TotalReciprocity is included

    def _metrics(self, positions, node_attr, types, reciprocity):
        return ([NumberOfEdgesDirected()]
                + ([TotalReciprocity()] if reciprocity else [])
                + [SumDistancesConnectedNeurons(positions, is_directed=True),
                   NodeAttrSum(node_attr, is_directed=True),
                   NumberOfEdgesTypesDirected(types, indices_from_user=[0])])

    def _check_units_invariance(self, reciprocity, sample_size):
        rng = np.random.default_rng(12345)
        positions_um = rng.uniform(0, 1000, size=(self.N_NODES, 3))
        node_attr = rng.normal(0, 100, size=self.N_NODES)
        types = rng.integers(0, 3, self.N_NODES)
        metrics_um = self._metrics(positions_um, node_attr, types, reciprocity)
        metrics_m = self._metrics(positions_um * 1e-6, node_attr, types, reciprocity)

        # Observed networks, generated once from the micrometer model with a plausible ground truth.
        ground_truth_head = [-3.0] + ([2.0] if reciprocity else []) + [-1.0 / 1000, 0.2 / 100]
        model_um = ERGM(self.N_NODES, metrics_um, is_directed=True)
        ground_truth = np.concatenate([ground_truth_head,
                                       rng.normal(0, 1, model_um._metrics_collection.calc_num_of_features()
                                                  - len(ground_truth_head))])
        model_um._thetas = ground_truth
        observed_networks = model_um.generate_networks_for_sample(sample_size=sample_size, seed=777)

        fit_um = ERGM(self.N_NODES, metrics_um, is_directed=True)
        fit_m = ERGM(self.N_NODES, metrics_m, is_directed=True)
        result_um = fit_um.fit(observed_networks, seed=1)
        result_m = fit_m.fit(observed_networks, seed=1)
        self.assertTrue(result_um.success)
        self.assertTrue(result_m.success)

        distance_coef_idx = self.DISTANCE_COEF_IDX + int(reciprocity)
        expected_thetas_m = fit_um._thetas.copy()
        expected_thetas_m[distance_coef_idx] *= 1e6
        # Measured max relative disagreement: 6e-12 (plain), 2e-15 (reciprocity); rtol is >100x above that.
        np.testing.assert_allclose(fit_m._thetas, expected_thetas_m, rtol=1e-9)

    def test_plain_mple_units_invariance(self):
        """Plain MPLE: fitting with positions in meters equals the micrometer fit with the distance coef x 1e6."""
        self._check_units_invariance(reciprocity=False, sample_size=8)

    def test_reciprocity_units_invariance(self):
        """Same as the plain test, for the MPLE-reciprocity path."""
        self._check_units_invariance(reciprocity=True, sample_size=16)


class TestLogLikelihoodClip(unittest.TestCase):
    """The log-likelihood clip depends on the dtype, so float32 workers cannot produce NaN or -inf."""

    def test_float32_saturated_predictions_are_finite(self):
        """A float32 sigmoid saturated at exactly 1.0 still gives a finite log-likelihood, for y = 1 and y < 1."""
        Xs = np.ones((3, 1), np.float32)
        thetas = np.array([[40.]], np.float32)
        pred = calc_logistic_regression_predictions(Xs, thetas)
        self.assertEqual(pred[0, 0], 1.0)
        for y in (1.0, 0.999):
            ys = np.full((3, 1), y, np.float32)
            self.assertTrue(np.isfinite(calc_logistic_regression_predictions_log_likelihood(pred, ys)[0][0]))

    def _clip_bounds(self, dtype, **kwargs):
        """The (a_min, a_max) that the log-likelihood passes to np.clip."""
        pred = np.full((3, 1), 0.5, dtype)
        ys = np.ones((3, 1), dtype)
        with patch("pyERGM.mple_optimization.np.clip", wraps=np.clip) as spy:
            calc_logistic_regression_predictions_log_likelihood(pred, ys, **kwargs)
        spy.assert_called_once()
        return spy.call_args.args[1], spy.call_args.args[2]

    def test_float64_default_eps_is_kept(self):
        """float64 with the default eps=1e-10: eps is above the dtype's eps, so it is the clip."""
        self.assertEqual(self._clip_bounds(np.float64), (1e-10, 1 - 1e-10))

    def test_float32_larger_eps_is_kept(self):
        """float32 with eps=1e-3: eps is above the dtype's eps, so it is the clip."""
        self.assertEqual(self._clip_bounds(np.float32, eps=1e-3), (1e-3, 1 - 1e-3))

    def test_float32_default_eps_is_replaced_by_dtype_eps(self):
        """float32 with the default eps=1e-10: eps is below the dtype's eps, so the dtype's eps is the clip."""
        e = float(np.finfo(np.float32).eps)
        self.assertEqual(self._clip_bounds(np.float32), (e, 1 - e))

    def test_float64_zero_eps_is_replaced_by_dtype_eps(self):
        """float64 with eps=0: eps is below the dtype's eps, so the dtype's eps is the clip."""
        e = float(np.finfo(np.float64).eps)
        self.assertEqual(self._clip_bounds(np.float64, eps=0), (e, 1 - e))


if __name__ == "__main__":
    unittest.main()
