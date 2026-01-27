import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.special import softmax
import glob

from pyERGM.logging_config import logger
from pyERGM.metrics import *


@njit
def sigmoid(x: np.ndarray | float):
    """
    Compute the sigmoid (logistic) function.

    Parameters
    ----------
    x : np.ndarray or float
        Input value(s).

    Returns
    -------
    np.ndarray or float
        Sigmoid of x: 1 / (1 + exp(-x)).
    """
    return 1 / (1 + np.exp(-x))


@njit
def calc_logistic_regression_predictions(Xs: np.ndarray, thetas: np.ndarray):
    """
    Calculate logistic regression predictions.

    Computes the predicted probabilities by applying the sigmoid function to
    the linear combination of features and parameters.

    Parameters
    ----------
    Xs : np.ndarray
        Feature matrix (design matrix) of shape (num_samples, num_features).
    thetas : np.ndarray
        Model parameters of shape (num_features, 1) or (num_features,).

    Returns
    -------
    np.ndarray
        Predicted probabilities of shape (num_samples, 1), computed as sigmoid(Xs @ thetas).
    """
    return sigmoid(Xs @ thetas)


@njit
def calc_logistic_regression_predictions_log_likelihood(predictions: np.ndarray, ys: np.ndarray, eps=1e-10,
                                                        reduction: str = 'sum', log_base: float = np.exp(1),
                                                        sample_weights: np.ndarray = np.empty(0)):
    """
    Calculate the log-likelihood of observations given model predictions.

    Computes the binary cross-entropy loss (negative log-likelihood) between
    predicted probabilities and observed labels.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities of shape (num_samples, 1). Values should be in [0, 1].
    ys : np.ndarray
        Observed labels of shape (num_samples, 1). Can be binary (0/1) or fractional
        probabilities when averaging over multiple observed networks.
    eps : float, optional
        Small constant to avoid log(0). Default is 1e-10.
    reduction : str, optional
        How to aggregate the log-likelihood: 'sum' (default), 'mean', or 'none'.
    log_base : float, optional
        Base for logarithm. Default is e (natural log).
    sample_weights : np.ndarray, optional
        Per-sample weights of shape (num_samples, 1). If empty (default), all samples
        are weighted equally.

    Returns
    -------
    np.ndarray or float
        Log-likelihood value(s). Shape depends on reduction parameter.
    """
    trimmed_predictions = np.clip(predictions, eps, 1 - eps)
    minus_binary_cross_entropy_per_edge = (ys * np.log(trimmed_predictions) + (1 - ys) * np.log(
        1 - trimmed_predictions)) / np.log(log_base)
    if sample_weights.size > 0:
        minus_binary_cross_entropy_per_edge = sample_weights * minus_binary_cross_entropy_per_edge
    if reduction == 'none':
        return minus_binary_cross_entropy_per_edge
    # The wrapping into a numpy array and reshape to 2D is necessary for numba to compile the function properly
    # (returned types must be unified).
    elif reduction == 'sum':
        return np.array([minus_binary_cross_entropy_per_edge.sum()]).reshape(1, 1)
    elif reduction == 'mean':
        return np.array([minus_binary_cross_entropy_per_edge.mean()]).reshape(1, 1)
    else:
        raise ValueError(f"{reduction} is an unsupported reduction method, options are 'none', 'sum', or 'mean'")


@njit
def calc_logistic_regression_log_likelihood_grad(Xs: np.ndarray, predictions: np.ndarray, ys: np.ndarray,
                                                  sample_weights: np.ndarray = np.empty(0)):
    """
    Calculate the gradient of log-likelihood with respect to model parameters.

    Computes the partial derivatives of the log-likelihood function with respect
    to each parameter (theta).

    Parameters
    ----------
    Xs : np.ndarray
        Feature matrix of shape (num_samples, num_features).
    predictions : np.ndarray
        Predicted probabilities of shape (num_samples, 1).
    ys : np.ndarray
        Observed labels of shape (num_samples, 1).
    sample_weights : np.ndarray, optional
        Per-sample weights of shape (num_samples, 1). If empty (default), all samples
        are weighted equally.

    Returns
    -------
    np.ndarray
        Gradient vector of shape (num_features, 1).
    """
    residuals = ys - predictions
    if sample_weights.size > 0:
        residuals = sample_weights * residuals
    return Xs.T @ residuals


@njit
def calc_logistic_regression_log_likelihood_hessian(Xs: np.ndarray, predictions: np.ndarray,
                                                     sample_weights: np.ndarray = np.empty(0)):
    """
    Calculate the Hessian matrix of log-likelihood.

    Computes the matrix of second partial derivatives of the log-likelihood
    with respect to model parameters. Used in Newton-type optimization methods.

    Parameters
    ----------
    Xs : np.ndarray
        Feature matrix of shape (num_samples, num_features).
    predictions : np.ndarray
        Predicted probabilities of shape (num_samples, 1).
    sample_weights : np.ndarray, optional
        Per-sample weights of shape (num_samples, 1). If empty (default), all samples
        are weighted equally.

    Returns
    -------
    np.ndarray
        Hessian matrix of shape (num_features, num_features).
    """
    diag_values = predictions * (1 - predictions)
    if sample_weights.size > 0:
        diag_values = sample_weights * diag_values
    return Xs.T @ (diag_values * Xs)


def calc_logistic_regression_log_likelihood_from_x_thetas(Xs: np.ndarray, thetas: np.ndarray, ys: np.ndarray):
    """
    Calculate log-likelihood directly from features and parameters.

    Alternative formulation that computes log-likelihood without explicitly
    computing predictions first. Can be numerically more stable.

    Parameters
    ----------
    Xs : np.ndarray
        Feature matrix of shape (num_samples, num_features).
    thetas : np.ndarray
        Model parameters of shape (num_features, 1) or (num_features,).
    ys : np.ndarray
        Observed labels of shape (num_samples, 1).

    Returns
    -------
    float
        Total log-likelihood (summed over all samples).
    """
    return (-np.log(1 + np.exp(-Xs @ thetas)) + (ys - 1) * Xs @ thetas).sum()


def analytical_minus_log_likelihood_local(thetas, Xs, ys, eps=1e-10, sample_weights=np.empty(0)):
    """
    Compute negative log-likelihood for local (non-distributed) optimization.

    Wrapper function compatible with scipy.optimize.minimize.

    Parameters
    ----------
    thetas : np.ndarray
        Flattened parameter vector.
    Xs : np.ndarray
        Feature matrix.
    ys : np.ndarray
        Observed labels.
    eps : float, optional
        Clipping value to avoid numerical issues. Default is 1e-10.
    sample_weights : np.ndarray, optional
        Per-sample weights. If empty (default), all samples are weighted equally.

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    pred = np.clip(calc_logistic_regression_predictions(Xs, thetas.reshape(thetas.size, 1)), eps, 1 - eps)
    return -calc_logistic_regression_predictions_log_likelihood(pred, ys, sample_weights=sample_weights)[0][0]


def analytical_minus_log_likelihood_distributed(thetas, data_path):
    """
    Compute negative log-likelihood and gradient for distributed optimization.

    Parameters
    ----------
    thetas : np.ndarray
        Flattened parameter vector.
    data_path : Path
        Path to distributed data chunks.

    Returns
    -------
    tuple
        (negative_log_likelihood, negative_gradient).
    """
    log_like, grad = distributed_logistic_regression_optimization_step(
        data_path,
        thetas.reshape(thetas.size, 1),
        ('log_likelihood', 'log_likelihood_gradient'),
    )
    return -log_like, -grad.reshape(thetas.size, )


def analytical_logistic_regression_predictions_distributed(thetas, data_path):
    """
    Compute predictions using distributed computation.

    Parameters
    ----------
    thetas : np.ndarray
        Model parameters.
    data_path : Path
        Path to distributed data chunks.

    Returns
    -------
    np.ndarray
        Predicted probabilities.
    """
    return distributed_logistic_regression_optimization_step(
        data_path,
        thetas.reshape(thetas.size, 1),
        ('predictions',),
    )[0]


def analytical_minus_log_like_grad_local(thetas, Xs, ys, eps=1e-10, sample_weights=np.empty(0)):
    """
    Compute negative log-likelihood gradient for local optimization.

    Parameters
    ----------
    thetas : np.ndarray
        Flattened parameter vector.
    Xs : np.ndarray
        Feature matrix.
    ys : np.ndarray
        Observed labels.
    eps : float, optional
        Clipping value. Default is 1e-10.
    sample_weights : np.ndarray, optional
        Per-sample weights. If empty (default), all samples are weighted equally.

    Returns
    -------
    np.ndarray
        Negative gradient vector.
    """
    pred = np.clip(calc_logistic_regression_predictions(Xs, thetas.reshape(thetas.size, 1)), eps, 1 - eps)
    return -calc_logistic_regression_log_likelihood_grad(Xs, pred, ys, sample_weights=sample_weights).reshape(thetas.size, )


def analytical_minus_log_likelihood_hessian_local(thetas, Xs, ys, eps=1e-10, sample_weights=np.empty(0)):
    """
    Compute negative log-likelihood Hessian for local optimization.

    Parameters
    ----------
    thetas : np.ndarray
        Flattened parameter vector.
    Xs : np.ndarray
        Feature matrix.
    ys : np.ndarray
        Observed labels.
    eps : float, optional
        Clipping value. Default is 1e-10.
    sample_weights : np.ndarray, optional
        Per-sample weights. If empty (default), all samples are weighted equally.

    Returns
    -------
    np.ndarray
        Negative Hessian matrix.
    """
    pred = np.clip(calc_logistic_regression_predictions(Xs, thetas.reshape(thetas.size, 1)), eps, 1 - eps)
    return -calc_logistic_regression_log_likelihood_hessian(Xs, pred, sample_weights=sample_weights)


def analytical_minus_log_likelihood_hessian_distributed(thetas, data_path, num_edges_per_job):
    """
    Compute negative log-likelihood Hessian using distributed computation.

    Parameters
    ----------
    thetas : np.ndarray
        Model parameters.
    data_path : Path
        Path to distributed data chunks.
    num_edges_per_job : int
        Number of edges per distributed job.

    Returns
    -------
    np.ndarray
        Negative Hessian matrix.
    """
    return -distributed_logistic_regression_optimization_step(data_path, thetas.reshape(thetas.size, 1),
                                                              ('log_likelihood_hessian',),
                                                              num_edges_per_job)[0]


def mple_logistic_regression_optimization(metrics_collection: MetricsCollection, observed_networks: np.ndarray,
                                          initial_thetas: np.ndarray | None = None,
                                          is_distributed: bool = False, optimization_method: str = 'L-BFGS-B',
                                          sample_weights: np.ndarray | None = None,
                                          **kwargs):
    """
    Optimize the parameters of a Logistic Regression model by maximizing the likelihood using scipy.optimize.minimize.
    Parameters
    ----------
    metrics_collection
        The `MetricsCollection` with relation to which the optimization is carried out.
        # TODO: we can't add a type hint for this, due to circular import (utils can't import from metrics, as metrics
            already imports from utils). This might suggest that this isn't the right place for this function.
    observed_networks
        The observed network used as data for the optimization, or an array of observed networks.
    initial_thetas
        The initial vector of parameters. If `None`, the initial state is randomly sampled from (0,1)
    is_distributed
        Whether the calculations are carried locally or distributed over many compute nodes of an IBM LSF cluster.
    optimization_method
        The optimization method to use. Currently only 'L-BFGS-B' and 'Newton-CG' are supported.
    sample_weights
        An (n, n) matrix of non-negative edge weights. If provided, each edge's contribution to the
        log-likelihood is scaled by its weight. If `None`, all edges are weighted equally.
    num_edges_per_job
        The number of graph edges (representing data points in this optimization) to consider for each job. Relevant
        only for distributed optimization.

    Returns
    -------
    thetas: np.ndarray
        The optimized parameters of the model
    pred: np.ndarray
        The predictions of the model on the observed network
    success: bool
        Whether the optimization was successful
    """

    # TODO: this code is duplicated, but the scoping of the nonlocal variables makes it not trivial to export out of
    #  the scope of each function using it.
    def _after_optim_iteration_callback(intermediate_result: OptimizeResult):
        nonlocal iteration
        iteration += 1
        cur_time = time.time()
        logger.info(f'iteration: {iteration}, time from start '
              f'training: {cur_time - start_time:.2f} '
              f'log10 likelihood: {-intermediate_result.fun / np.log(10):.4f}')
        if is_distributed:
            checkpoint_path_getter = lambda idx: (
                    Path.cwd().parent / "OptimizationIntermediateCalculations" / f"checkpoint_iter_{idx}.pkl"
            ).resolve()
            with open(checkpoint_path_getter(iteration), 'wb') as f:
                pickle.dump({'metrics_collection': metrics_collection, 'thetas': intermediate_result.x, }, f)
            if iteration > 1:
                os.unlink(checkpoint_path_getter(iteration - 1))

    iteration = 0
    start_time = time.time()
    logger.info("MPLE optimization started")

    observed_networks = expand_net_dims(observed_networks)

    num_features = metrics_collection.calc_num_of_features()
    if initial_thetas is None:
        thetas = np.random.rand(num_features)
    else:
        thetas = initial_thetas.copy()

    if not is_distributed:
        Xs = metrics_collection.prepare_mple_regressors(observed_networks[..., 0])
        ys = metrics_collection.prepare_mple_labels(observed_networks)

        if sample_weights is not None:
            flat_weights = flatten_square_matrix_to_edge_list(sample_weights, metrics_collection.is_directed)
            if metrics_collection.mask is not None:
                flat_weights = flat_weights[metrics_collection.mask]
            flat_weights = flat_weights.reshape(-1, 1)
        else:
            flat_weights = np.empty(0)

        if optimization_method == "Newton-CG":
            res = minimize(analytical_minus_log_likelihood_local, thetas, args=(Xs, ys, 1e-10, flat_weights),
                           jac=analytical_minus_log_like_grad_local, hess=analytical_minus_log_likelihood_hessian_local,
                           callback=_after_optim_iteration_callback, method=optimization_method)
        elif optimization_method == "L-BFGS-B":
            res = minimize(analytical_minus_log_likelihood_local, thetas, args=(Xs, ys, 1e-10, flat_weights),
                           jac=analytical_minus_log_like_grad_local, method=optimization_method,
                           callback=_after_optim_iteration_callback)
        else:
            raise ValueError(
                f"Unsupported optimization method: {optimization_method}. Options are: Newton-CG, L-BFGS-B")
        pred = calc_logistic_regression_predictions(Xs, res.x.reshape(-1, 1)).flatten()
    else:
        data_path = distributed_mple_data_chunks_calculations(metrics_collection, observed_networks,
                                                            kwargs.get('num_edges_per_job', 100000),
                                                            sample_weights=sample_weights)
        if optimization_method == "L-BFGS-B":
            res = minimize(analytical_minus_log_likelihood_distributed, thetas, args=(data_path,),
                           jac=True, callback=_after_optim_iteration_callback, method=optimization_method)
        elif optimization_method == "Newton-CG":
            res = minimize(analytical_minus_log_likelihood_distributed, thetas, args=(data_path,),
                           jac=True,
                           hess=analytical_minus_log_likelihood_hessian_distributed,
                           callback=_after_optim_iteration_callback, method=optimization_method)
        else:
            raise ValueError(
                f"Unsupported optimization method: {optimization_method} for distributed optimization. "
                f"Options are: Newton-CG, L-BFGS-B")
        pred = analytical_logistic_regression_predictions_distributed(res.x.reshape(-1, 1), data_path).flatten()

        # Clean up MPLE data (metrics_collection, observed_networks and Xs,ys chunks).
        shutil.rmtree(data_path)
        shutil.rmtree((data_path.parent / 'mple_data_paged_chunks').resolve())

    logger.debug(f"Optimization result: {res}")

    return res.x, pred, res.success


def distributed_logistic_regression_optimization_step(data_path, thetas, funcs_to_calc):
    # Arrange files and send the children jobs
    num_jobs, out_path, job_array_ids, children_logs_dir = _run_distributed_logistic_regression_children_jobs(
        data_path,
        thetas,
        funcs_to_calc,
    )

    # Wait for all jobs to finish.
    chunks_paths = [(out_path / func_to_calc).resolve() for func_to_calc in funcs_to_calc]
    for chunks_path in chunks_paths:
        os.makedirs(chunks_path, exist_ok=True)

    logger.debug("Start waiting for children jobs in MPLE optimization")
    wait_for_distributed_children_outputs(num_jobs, chunks_paths, job_array_ids, "__".join(funcs_to_calc),
                                          children_logs_dir)
    logger.debug("Done waiting for children jobs in MPLE optimization")

    aggregated_funcs = []
    for func_to_calc in funcs_to_calc:
        # Aggregate results
        if func_to_calc == "predictions":
            aggregated_funcs.append(cat_children_jobs_outputs(num_jobs, (out_path / func_to_calc).resolve()))
        else:
            aggregated_funcs.append(sum_children_jobs_outputs(num_jobs, (out_path / func_to_calc).resolve()))

        # Clean current outputs
        shutil.rmtree((out_path / func_to_calc).resolve())

    # Clean current scripts
    shutil.rmtree((out_path / "scripts").resolve())
    return tuple(aggregated_funcs)


def distributed_mple_data_chunks_calculations(
        metrics_collection: MetricsCollection,
        observed_networks: np.ndarray,
        num_edges_per_job,
        sample_weights: np.ndarray | None = None,
) -> Path:
    out_dir_path = (Path.cwd().parent / "OptimizationIntermediateCalculations").resolve()
    data_path = (out_dir_path / "data").resolve()
    os.makedirs(data_path, exist_ok=True)

    # Copy the `MetricsCollection` and the observed network to provide its path to children jobs, so they will be
    # able to access it.
    metric_collection_path = os.path.join(data_path, 'metric_collection.pkl')
    logger.debug("Dumping metrics collection")
    with open(metric_collection_path, 'wb') as f:
        pickle.dump(metrics_collection, f)
    logger.debug("Dumped metrics collection")
    observed_networks_path = os.path.join(data_path, 'observed_networks.pkl')
    logger.debug("Dumping observed networks")
    with open(observed_networks_path, 'wb') as f:
        pickle.dump(observed_networks, f)
    logger.debug("Dumped observed networks")
    if sample_weights is not None:
        sample_weights_path = os.path.join(data_path, 'sample_weights.pkl')
        logger.debug("Dumping sample weights")
        with open(sample_weights_path, 'wb') as f:
            pickle.dump(sample_weights, f)
        logger.debug("Dumped sample weights")
    cmd_line_single_batch = (f'python ./mple_data_distributed_paging.py '
                             f'--out_dir_path={out_dir_path} '
                             f'--num_edges_per_job={num_edges_per_job} ')

    num_nodes = observed_networks.shape[0]
    num_data_points = num_nodes * num_nodes - num_nodes
    num_jobs = int(np.ceil(num_data_points / num_edges_per_job))

    logger.debug("Sending children jobs to calculate MPLE data chunks")
    job_array_ids, children_logs_dir = run_distributed_children_jobs(
        out_dir_path,
        cmd_line_single_batch,
        "distributed_mple_data_paging.sh",
        num_jobs, 'data_paging',
    )

    chunks_path = (out_dir_path / 'mple_data_paged_chunks').resolve()
    os.makedirs(chunks_path, exist_ok=True)

    logger.debug("Start waiting for children jobs in MPLE data paging")
    wait_for_distributed_children_outputs(num_jobs, [chunks_path], job_array_ids, 'data_paging',
                                          children_logs_dir)
    logger.debug("Done waiting for children jobs in MPLE data paging")
    # Clean current scripts
    shutil.rmtree((out_dir_path / "scripts").resolve())
    return data_path


def _run_distributed_logistic_regression_children_jobs(data_path, cur_thetas, funcs_to_calculate):
    out_path = data_path.parent

    cmd_line_single_batch = construct_single_batch_bash_cmd_logistic_regression(
        out_path,
        cur_thetas,
        funcs_to_calculate,
    )

    paged_chunks_path = os.path.join(out_path, "mple_data_paged_chunks")
    num_jobs = len(glob.glob(f"{paged_chunks_path}/[0-9]*.npz"))

    logger.debug("Sending children jobs to calculate MPLE likelihood grad")
    job_array_ids, children_logs_dir = run_distributed_children_jobs(out_path, cmd_line_single_batch,
                                                                     "distributed_logistic_regression.sh",
                                                                     num_jobs, "__".join(funcs_to_calculate))
    return num_jobs, out_path, job_array_ids, children_logs_dir


def predict_multi_class_logistic_regression(Xs, thetas):
    """
    Predict class probabilities using multinomial logistic regression.

    Applies the softmax function to compute probability distributions over
    multiple classes. Used for MPLE with reciprocity (4 dyadic states).

    Parameters
    ----------
    Xs : np.ndarray
        Feature tensor of shape (num_samples, num_classes, num_features).
    thetas : np.ndarray
        Model parameters of shape (num_features,).

    Returns
    -------
    np.ndarray
        Probability distributions of shape (num_samples, num_classes).
    """
    return softmax(Xs @ thetas, axis=1)


def log_likelihood_multi_class_logistic_regression(true_labels, predictions, reduction='sum', log_base=np.exp(1),
                                                   eps: float = 1e-10):
    """
    Calculate log-likelihood for multinomial logistic regression.

    Computes the categorical cross-entropy between true labels and predictions.

    Parameters
    ----------
    true_labels : np.ndarray
        One-hot encoded labels or probability distributions of shape (num_samples, num_classes).
    predictions : np.ndarray
        Predicted probability distributions of shape (num_samples, num_classes).
    reduction : str, optional
        How to aggregate: 'sum' (default), 'mean', or 'none'.
    log_base : float, optional
        Base for logarithm. Default is e.
    eps : float, optional
        Small constant to avoid log(0). Default is 1e-10.

    Returns
    -------
    float or np.ndarray
        Log-likelihood value(s).
    """
    predictions = np.maximum(predictions, eps)
    individual_data_samples_minus_cross_ent = ((np.log(predictions) / np.log(log_base)) * true_labels).sum(axis=1)
    if reduction == 'none':
        return individual_data_samples_minus_cross_ent
    elif reduction == 'sum':
        return individual_data_samples_minus_cross_ent.sum()
    elif reduction == 'mean':
        return individual_data_samples_minus_cross_ent.mean()
    else:
        raise ValueError(f"reduction {reduction} not supported, options are 'none', 'sum', or 'mean'")


def minus_log_likelihood_multi_class_logistic_regression(thetas, Xs, ys):
    """
    Compute negative log-likelihood for multinomial logistic regression.

    Wrapper for optimization with scipy.optimize.minimize.

    Parameters
    ----------
    thetas : np.ndarray
        Model parameters.
    Xs : np.ndarray
        Feature tensor.
    ys : np.ndarray
        True labels.

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    return -log_likelihood_multi_class_logistic_regression(ys, predict_multi_class_logistic_regression(Xs, thetas))


def minus_log_likelihood_gradient_multi_class_logistic_regression(thetas, Xs, ys):
    """
    Compute gradient of negative log-likelihood for multinomial logistic regression.

    Parameters
    ----------
    thetas : np.ndarray
        Model parameters.
    Xs : np.ndarray
        Feature tensor.
    ys : np.ndarray
        True labels.

    Returns
    -------
    np.ndarray
        Negative gradient vector.
    """
    prediction = predict_multi_class_logistic_regression(Xs, thetas)
    num_features = Xs.shape[-1]
    return -(ys - prediction).flatten() @ Xs.reshape(-1, num_features)


def mple_reciprocity_logistic_regression_optimization(
        metrics_collection: MetricsCollection,
        observed_networks: np.ndarray,
        initial_thetas: np.ndarray | None = None,
        optimization_method: str = 'L-BFGS-B',
):
    """
    Optimize ERGM parameters using MPLE for reciprocity models.

    This function performs Maximum Pseudo-Likelihood Estimation for directed ERGMs
    that contain reciprocity dependencies but are otherwise dyadic independent.
    Uses multinomial logistic regression over the 4 possible dyadic states.

    Parameters
    ----------
    metrics_collection : MetricsCollection
        Collection of metrics defining the ERGM model.
    observed_networks : np.ndarray
        Observed network(s) of shape (n, n) or (n, n, num_networks).
    initial_thetas : np.ndarray, optional
        Initial parameter values. If None, randomly initialized from [0, 1].
    optimization_method : str, optional
        Optimization method. Currently only 'L-BFGS-B' is supported. Default is 'L-BFGS-B'.

    Returns
    -------
    thetas : np.ndarray
        Optimized model parameters.
    pred : np.ndarray
        Predicted dyadic state probabilities of shape (n_choose_2, 4).
    success : bool
        Whether optimization converged successfully.
    """
    def _after_optim_iteration_callback(intermediate_result: OptimizeResult):
        nonlocal iteration
        iteration += 1
        cur_time = time.time()
        logger.info(f'iteration: {iteration}, time from start '
              f'training: {cur_time - start_time:.2f} '
              f'log10 likelihood: {-intermediate_result.fun / np.log(10):.4f}')

    iteration = 0
    start_time = time.time()
    logger.info("MPLE optimization started")

    observed_networks = expand_net_dims(observed_networks)
    Xs = metrics_collection.prepare_mple_reciprocity_regressors()
    ys = metrics_collection.prepare_mple_reciprocity_labels(observed_networks)

    num_features = metrics_collection.calc_num_of_features()
    if initial_thetas is None:
        thetas = np.random.rand(num_features)
    else:
        thetas = initial_thetas.copy()

    if optimization_method == "L-BFGS-B":
        res = minimize(minus_log_likelihood_multi_class_logistic_regression, thetas, args=(Xs, ys),
                       jac=minus_log_likelihood_gradient_multi_class_logistic_regression, method="L-BFGS-B",
                       callback=_after_optim_iteration_callback)
    else:
        raise ValueError(
            f"Unsupported optimization method: {optimization_method}. Options are: L-BFGS-B")
    pred = predict_multi_class_logistic_regression(Xs, res.x)
    logger.debug(f"Optimization result: {res}")
    return res.x, pred, res.success
