import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from pyERGM.mple_optimization import (
    analytical_minus_log_like_grad_local,
    analytical_minus_log_likelihood_hessian_distributed,
    analytical_minus_log_likelihood_hessian_local,
    analytical_minus_log_likelihood_local,
    calc_logistic_regression_log_likelihood_grad,
    calc_logistic_regression_log_likelihood_hessian,
    calc_logistic_regression_predictions,
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


if __name__ == "__main__":
    unittest.main()
