import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.special import softmax

from pyERGM.metrics import *


@njit
def sigmoid(x: np.ndarray | float):
    return 1 / (1 + np.exp(-x))


@njit
def calc_logistic_regression_predictions(Xs: np.ndarray, thetas: np.ndarray):
    """
    Calculate the predictions of a Logistic Regression model with input Xs and parameters thetas
    Parameters
    ----------
    Xs
        The input to the model (the regressors), of shape (num_samples X num_features)
    thetas
        The parameters of the model, of shape (num_features X 1)
    Returns
        sigmoid(Xs @ thetas)
    -------
    """
    return sigmoid(Xs @ thetas)


@njit
def calc_logistic_regression_predictions_log_likelihood(predictions: np.ndarray, ys: np.ndarray, eps=1e-10,
                                                        reduction: str = 'sum', log_base: float = np.exp(1)):
    """
    Calculates the log-likelihood of labeled data with regard to the predictions of a Logistic Regression model.
    Parameters
    ----------
    predictions
        The predictions of a Logistic Regression model (the probabilities it assigns to feature vectors to be
        labeled 1). Of shape (num_samples X 1)
    ys
        The labeled data (probability for each feature vector). The values are floats between 0 and 1 (and are
        calculated as the fraction of networks in the observed ensemble where an edge exists. If training on a single
        network, labels are binary). Of shape (num_samples X 1)
    Returns
    -------
    The probability to observe the vector `ys` under the distribution induced by `predictions`.
    """
    trimmed_predictions = np.clip(predictions, eps, 1 - eps)
    minus_binary_cross_entropy_per_edge = (ys * np.log(trimmed_predictions) + (1 - ys) * np.log(
        1 - trimmed_predictions)) / np.log(log_base)
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
def calc_logistic_regression_log_likelihood_grad(Xs: np.ndarray, predictions: np.ndarray, ys: np.ndarray):
    """
    Calculates the gradient of the log-likelihood of labeled data with regard to the predictions of a Logistic
    Regression model.
    Parameters
    ----------
    Xs
        The input to the model (the regressors), of shape (num_samples X num_features)
    predictions
        The predictions of a Logistic Regression model (the probabilities it assigns to feature vectors to be
        labeled 1). Of shape (num_samples X 1)
    ys
         The labeled data (zeros and ones, for each feature vector). Of shape (num_samples X 1)
    Returns
    -------
    The gradient (partial derivatives with relation to thetas - the model parameters) of the log-likelihood of the data
    given the model. Of shape (num_features X 1)
    """
    return Xs.T @ (ys - predictions)


@njit
def calc_logistic_regression_log_likelihood_hessian(Xs: np.ndarray, predictions: np.ndarray):
    """
    Calculates the hessian of the log-likelihood of labeled data with regard to the predictions of a Logistic
    Regression model.
    Parameters
    ----------
    Xs
        The input to the model (the regressors), of shape (num_samples X num_features)
    predictions
        The predictions of a Logistic Regression model (the probabilities it assigns to feature vectors to be
        labeled 1). Of shape (num_samples X 1)
    Returns
    -------
    The hessian (partial second derivatives with relation to thetas - the model parameters) of the log-likelihood of the
    data given the model. Of shape (num_features X num_features)
    """
    return Xs.T @ (predictions * (1 - predictions) * Xs)


def calc_logistic_regression_log_likelihood_from_x_thetas(Xs: np.ndarray, thetas: np.ndarray, ys: np.ndarray):
    """
    Calculates the log-likelihood of labeled data with regard to the predictions of a Logistic Regression model.
    Parameters
    ----------
    Xs
        The input to the model (the regressors), of shape (num_samples X num_features)
    thetas
        The parameters of the model, of shape (num_features X 1)
    ys
        The labeled data (zeros and ones, for each feature vector). Of shape (num_samples X 1)
    Returns
    -------
    The probability to observe the vector `ys` under the distribution induced by the model.
    """
    return (-np.log(1 + np.exp(-Xs @ thetas)) + (ys - 1) * Xs @ thetas).sum()


def analytical_minus_log_likelihood_local(thetas, Xs, ys, eps=1e-10):
    pred = np.clip(calc_logistic_regression_predictions(Xs, thetas.reshape(thetas.size, 1)), eps, 1 - eps)
    return -calc_logistic_regression_predictions_log_likelihood(pred, ys)[0][0]


def analytical_minus_log_likelihood_distributed(thetas, data_path, num_edges_per_job):
    log_like, grad = distributed_logistic_regression_optimization_step(
        data_path,
        thetas.reshape(thetas.size, 1),
        ('log_likelihood', 'log_likelihood_gradient'),
        num_edges_per_job,
    )
    return -log_like, -grad.reshape(thetas.size, )


def analytical_logistic_regression_predictions_distributed(thetas, data_path, num_edges_per_job):
    return distributed_logistic_regression_optimization_step(data_path, thetas.reshape(thetas.size, 1),
                                                             ('predictions',), num_edges_per_job)[0]


def analytical_minus_log_like_grad_local(thetas, Xs, ys, eps=1e-10):
    pred = np.clip(calc_logistic_regression_predictions(Xs, thetas.reshape(thetas.size, 1)), eps, 1 - eps)
    return -calc_logistic_regression_log_likelihood_grad(Xs, pred, ys).reshape(thetas.size, )


def analytical_minus_log_likelihood_hessian_local(thetas, Xs, ys, eps=1e-10):
    pred = np.clip(calc_logistic_regression_predictions(Xs, thetas.reshape(thetas.size, 1)), eps, 1 - eps)
    return -calc_logistic_regression_log_likelihood_hessian(Xs, pred)


def analytical_minus_log_likelihood_hessian_distributed(thetas, data_path, num_edges_per_job):
    return -distributed_logistic_regression_optimization_step(data_path, thetas.reshape(thetas.size, 1),
                                                              ('log_likelihood_hessian',),
                                                              num_edges_per_job)[0]


def mple_logistic_regression_optimization(metrics_collection: MetricsCollection, observed_networks: np.ndarray,
                                          initial_thetas: np.ndarray | None = None,
                                          is_distributed: bool = False, optimization_method: str = 'L-BFGS-B',
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
        print(f'iteration: {iteration}, time from start '
              f'training: {cur_time - start_time} '
              f'log10 likelihood: {-intermediate_result.fun / np.log(10)}')
        sys.stdout.flush()
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
    print("optimization started")
    sys.stdout.flush()

    observed_networks = expand_net_dims(observed_networks)
    if not is_distributed:
        Xs = metrics_collection.prepare_mple_regressors(observed_networks[..., 0])
        ys = metrics_collection.prepare_mple_labels(observed_networks)
    else:
        out_dir_path = (Path.cwd().parent / "OptimizationIntermediateCalculations").resolve()
        data_path = (out_dir_path / "data").resolve()
        os.makedirs(data_path, exist_ok=True)

        # Copy the `MetricsCollection` and the observed network to provide its path to children jobs, so they will be
        # able to access it.
        metric_collection_path = os.path.join(data_path, 'metric_collection.pkl')
        with open(metric_collection_path, 'wb') as f:
            pickle.dump(metrics_collection, f)

    num_features = metrics_collection.calc_num_of_features()
    if initial_thetas is None:
        thetas = np.random.rand(num_features)
    else:
        thetas = initial_thetas.copy()

    if not is_distributed:
        if optimization_method == "Newton-CG":
            res = minimize(analytical_minus_log_likelihood_local, thetas, args=(Xs, ys),
                           jac=analytical_minus_log_like_grad_local, hess=analytical_minus_log_likelihood_hessian_local,
                           callback=_after_optim_iteration_callback, method=optimization_method)
        elif optimization_method == "L-BFGS-B":
            res = minimize(analytical_minus_log_likelihood_local, thetas, args=(Xs, ys),
                           jac=analytical_minus_log_like_grad_local, method=optimization_method,
                           callback=_after_optim_iteration_callback)
        else:
            raise ValueError(
                f"Unsupported optimization method: {optimization_method}. Options are: Newton-CG, L-BFGS-B")
        pred = calc_logistic_regression_predictions(Xs, res.x.reshape(-1, 1)).flatten()
    else:
        num_edges_per_job = kwargs.get('num_edges_per_job', 100000)
        if optimization_method == "L-BFGS-B":
            res = minimize(analytical_minus_log_likelihood_distributed, thetas, args=(data_path, num_edges_per_job),
                           jac=True, callback=_after_optim_iteration_callback, method=optimization_method)
        elif optimization_method == "Newton-CG":
            res = minimize(analytical_minus_log_likelihood_distributed, thetas, args=(data_path, num_edges_per_job),
                           jac=True,
                           hess=analytical_minus_log_likelihood_hessian_distributed,
                           callback=_after_optim_iteration_callback, method=optimization_method)
        else:
            raise ValueError(
                f"Unsupported optimization method: {optimization_method} for distributed optimization. "
                f"Options are: Newton-CG, L-BFGS-B")
        pred = analytical_logistic_regression_predictions_distributed(res.x.reshape(-1, 1), data_path,
                                                                      num_edges_per_job).flatten()

    print(res)
    sys.stdout.flush()
    return res.x, pred, res.success


def distributed_logistic_regression_optimization_step(data_path, thetas, funcs_to_calc, num_edges_per_job=5000):
    # Arrange files and send the children jobs
    num_jobs, out_path, job_array_ids, children_logs_dir = _run_distributed_logistic_regression_children_jobs(
        data_path,
        thetas,
        funcs_to_calc,
        num_edges_per_job,
    )

    # Wait for all jobs to finish.
    chunks_paths = [(out_path / func_to_calc).resolve() for func_to_calc in funcs_to_calc]
    for chunks_path in chunks_paths:
        os.makedirs(chunks_path, exist_ok=True)

    print("start waiting for children jobs in MPLE optimization")
    sys.stdout.flush()
    wait_for_distributed_children_outputs(num_jobs, chunks_paths, job_array_ids, "__".join(funcs_to_calc),
                                          children_logs_dir)
    print("done waiting for children jobs in MPLE optimization")
    sys.stdout.flush()

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


def _run_distributed_logistic_regression_children_jobs(data_path, cur_thetas, funcs_to_calculate, num_edges_per_job):
    out_path = data_path.parent

    cmd_line_single_batch = construct_single_batch_bash_cmd_logistic_regression(
        out_path, cur_thetas,
        funcs_to_calculate, num_edges_per_job)

    with open(os.path.join(data_path, "observed_networks.pkl"), 'rb') as f:
        observed_networks = pickle.load(f)

    num_nodes = observed_networks.shape[0]
    num_data_points = num_nodes * num_nodes - num_nodes
    num_jobs = int(np.ceil(num_data_points / num_edges_per_job))

    print("sending children jobs to calculate MPLE likelihood grad")
    sys.stdout.flush()
    job_array_ids, children_logs_dir = run_distributed_children_jobs(out_path, cmd_line_single_batch,
                                                                     "distributed_logistic_regression.sh",
                                                                     num_jobs, "__".join(funcs_to_calculate))
    return num_jobs, out_path, job_array_ids, children_logs_dir


def predict_multi_class_logistic_regression(Xs, thetas):
    return softmax(Xs @ thetas, axis=1)


def log_likelihood_multi_class_logistic_regression(true_labels, predictions, reduction='sum', log_base=np.exp(1),
                                                   eps: float = 1e-10):
    predictions = np.maximum(predictions, eps)
    individual_data_samples_minus_cross_ent = ((np.log(predictions) / np.log(log_base)) * true_labels).sum(axis=0)
    if reduction == 'none':
        return individual_data_samples_minus_cross_ent
    elif reduction == 'sum':
        return individual_data_samples_minus_cross_ent.sum()
    elif reduction == 'mean':
        return individual_data_samples_minus_cross_ent.mean()
    else:
        raise ValueError(f"reduction {reduction} not supported, options are 'none', 'sum', or 'mean'")


def minus_log_likelihood_multi_class_logistic_regression(thetas, Xs, ys):
    return -log_likelihood_multi_class_logistic_regression(ys, predict_multi_class_logistic_regression(Xs, thetas))


def minus_log_likelihood_gradient_multi_class_logistic_regression(thetas, Xs, ys):
    prediction = predict_multi_class_logistic_regression(Xs, thetas)
    num_features = Xs.shape[-1]
    return -(ys - prediction).flatten() @ Xs.reshape(-1, num_features)


def mple_reciprocity_logistic_regression_optimization(
        metrics_collection: MetricsCollection,
        observed_networks: np.ndarray,
        initial_thetas: np.ndarray | None = None,
        optimization_method: str = 'L-BFGS-B',
):
    def _after_optim_iteration_callback(intermediate_result: OptimizeResult):
        nonlocal iteration
        iteration += 1
        cur_time = time.time()
        print(f'iteration: {iteration}, time from start '
              f'training: {cur_time - start_time} '
              f'log10 likelihood: {-intermediate_result.fun / np.log(10)}')
        sys.stdout.flush()

    iteration = 0
    start_time = time.time()
    print("optimization started")
    sys.stdout.flush()

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
    return res.x, pred, res.success
