import itertools
from collections import Counter
from typing import Collection
import numpy as np
import networkx as nx
from numba import njit
from scipy.spatial.distance import mahalanobis
import torch
import random
import time


def perturb_network_by_overriding_edge(network, value, i, j, is_directed):
    perturbed_net = network.copy()
    perturbed_net[i, j] = value
    if not is_directed:
        perturbed_net[j, i] = value

    return perturbed_net


def connectivity_matrix_to_G(W: np.ndarray, directed=False):
    """
    Convert a connectivity matrix to a graph object.
    
    Parameters
    ----------
    W : np.ndarray
        A connectivity matrix.
        
    Returns
    -------
    G : nx.Graph
        A graph object.

    """
    if directed:
        G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(W)
    return G


def get_random_nondiagonal_matrix_entry(n: int):
    """
    Get a random entry for a non-diagonal entry of a matrix.
    
    Parameters
    ----------
    n : int
        The size of the matrix.
        
    Returns
    -------
    entry : tuple
        The row and column index of the entry.
    """
    xs = list(range(n))
    return tuple(random.sample(xs, 2))


def construct_adj_mat_from_int(int_code: int, num_nodes: int, is_directed: bool) -> np.ndarray:
    num_pos_connects = num_nodes * (num_nodes - 1)
    if not is_directed:
        num_pos_connects //= 2
    adj_mat_str = f'{int_code:0{num_pos_connects}b}'
    cur_adj_mat = np.zeros((num_nodes, num_nodes))
    mat_entries_arr = np.array(list(adj_mat_str), 'uint8')
    if is_directed:
        cur_adj_mat[~np.eye(num_nodes, dtype=bool)] = mat_entries_arr
    else:
        upper_triangle_indices = np.triu_indices(num_nodes, k=1)
        cur_adj_mat[upper_triangle_indices] = mat_entries_arr
        lower_triangle_indices_aligned = (upper_triangle_indices[1], upper_triangle_indices[0])
        cur_adj_mat[lower_triangle_indices_aligned] = mat_entries_arr
    return cur_adj_mat


def construct_int_from_adj_mat(adj_mat: np.ndarray, is_directed: bool) -> int:
    if len(adj_mat.shape) != 2 or adj_mat.shape[0] != adj_mat.shape[1]:
        raise ValueError(f"The dimensions of the given adjacency matrix {adj_mat.shape} are not valid for an "
                         f"adjacency matrix (should be a 2D squared matrix)")
    num_nodes = adj_mat.shape[0]
    adj_mat_no_diag = adj_mat[~np.eye(num_nodes, dtype=bool)]
    if not is_directed:
        adj_mat_no_diag = adj_mat_no_diag[:adj_mat_no_diag.size // 2]
    return round((adj_mat_no_diag * 2 ** np.arange(adj_mat_no_diag.size - 1, -1, -1).astype(np.ulonglong)).sum())


def _calc_line_slope_2_points(xs: np.ndarray, ys: np.ndarray):
    if xs.size != 2 or ys.size != 2:
        raise ValueError(
            f"The sizes of xs and ys must be 2 to form a well defined line! Got {xs.size}, {ys.size} instead")
    return (xs)


def get_greatest_convex_minorant(xs: np.ndarray, ys: np.ndarray):
    if xs.size != ys.size:
        raise ValueError("Arrays must have the same size!")
    # For numerical considerations we don't force the sequence of slopes to really not decrease
    slope_diff_thr = -10 ** -10
    x_diffs = np.diff(xs)
    # TODO: this case be handled (reduce the larger y value to the lower one, perform the same logic for the effective
    #  n-1 points, and duplicate the relevant point at the end). But is seems a very specific case and I'm not sure it's
    #  worth the complication.
    if np.any(x_diffs == 0):
        raise ValueError("xs array must contain unique values!")
    cur_proposed_minorant = ys.copy()
    cur_spline_slopes = np.diff(cur_proposed_minorant) / x_diffs
    cur_decreasing_slope_indices = np.where(np.diff(cur_spline_slopes) < slope_diff_thr)[0]
    # Convexity condition - the slopes are monotonically increasing
    while cur_decreasing_slope_indices.size > 0:
        # The indices of the points that violate the condition, and whose y values need to be decreases are in 1 shift
        # from the problematic slopes (if the second slope is smaller than the first slope, cur_decreasing_slope_indices
        # would include the index 0, of the first slopes difference. This corresponds to the point that takes part in
        # the calculations of both slopes, the first and the second, which is the second point, with index 1).
        cur_problematic_indices = cur_decreasing_slope_indices + 1

        # These are the indices that wrap the problematic ones (i.e., all the indices that are between wrapping_indices
        # are problematic).
        # Note: problematic indices may be subsequent (indices 2 and 3 are problematic, so we want 1, 4 to wrap them).
        cur_wrapping_indices = np.array([i for i in range(xs.size) if (i not in cur_problematic_indices and (
                i + 1 in cur_problematic_indices or i - 1 in cur_problematic_indices))]).astype(int)

        # To get an equation of a line from the slope and a point on in we can rearrange the formula:
        # y-y0 = m(x-x0) -> y = mx + (y0-m*x0)
        cur_wrapping_lines_slopes = np.diff(cur_proposed_minorant[cur_wrapping_indices]) / np.diff(
            xs[cur_wrapping_indices])
        cur_wrapping_lines_intersepts = (cur_proposed_minorant[cur_wrapping_indices[:-1]] -
                                         cur_wrapping_lines_slopes * xs[cur_wrapping_indices[:-1]])

        # Update the values in problematic indices to lie on the corresponding wrapping line
        cur_assignment_problematic_idx_to_line = np.searchsorted(cur_wrapping_indices, cur_problematic_indices) - 1
        cur_proposed_minorant[cur_problematic_indices] = (
                cur_wrapping_lines_slopes[cur_assignment_problematic_idx_to_line] * xs[cur_problematic_indices] +
                cur_wrapping_lines_intersepts[cur_assignment_problematic_idx_to_line])

        # Update the slopes and the indices where they decrease
        cur_spline_slopes = np.diff(cur_proposed_minorant) / x_diffs
        cur_decreasing_slope_indices = np.where(np.diff(cur_spline_slopes) < slope_diff_thr)[0]

    return cur_proposed_minorant


@njit
def get_random_edges_to_flip(num_nodes, num_pairs):
    """
    Create a matrix of size (2 x num_pairs), where each column represents a pair of nodes.
    These nodes represent the edge we wish to flip.
    """

    edges_to_flip = np.zeros((2, num_pairs), dtype=np.int32)

    edges_to_flip[0, :] = np.random.choice(num_nodes, size=num_pairs)

    diff = np.random.choice(num_nodes - 1, size=num_pairs) + 1

    edges_to_flip[1, :] = (edges_to_flip[0, :] - diff) % num_nodes

    return edges_to_flip


def np_tensor_to_sparse_tensor(np_tensor: np.ndarray) -> torch.Tensor:
    """
    Receives a numpy tensor and converts it to a sparse Tensor, using Torch.
    TODO - Support different types of Sparse Matrix data structures? More efficient conversion?
    """
    return torch.from_numpy(np_tensor).to_sparse()


def transpose_sparse_sample_matrices(sparse_tensor: torch.Tensor) -> torch.Tensor:
    """
    Transpose a sparse tensor that represents k matrices of dimension n x n.
    The transpose operation occurs along the dimension of sample (i.e. each matrix is transposed separately)

    Parameters
    ----------
    sparse_tensor: torch.Tensor
        A sparse tensor of dimension (n, n, k) representing k matrices of dim (n,n)

    Returns
    -------
    transposed_tensor: torch.Tensor
        The same tensor but matrices are transposed

    """
    n = sparse_tensor.shape[0]
    k = sparse_tensor.shape[2]

    indices = sparse_tensor.indices().type(torch.int64)
    transposed_indices = torch.stack([indices[1], indices[0], indices[2]])
    values = sparse_tensor.values()

    return torch.sparse_coo_tensor(transposed_indices, values, (n, n, k))


def calc_for_sample_njit():
    def wrapper(func):
        def inner(sample):
            if isinstance(sample, np.ndarray):
                return njit(func)(sample)
            return func(sample)

        return inner

    return wrapper


def approximate_auto_correlation_function(features_of_net_samples: np.ndarray) -> np.ndarray:
    """
    This is gamma hat from Geyer's handbook of mcmc (1D) and Dai and Jones 2017 (multi-D).
    """
    # TODO: it must be possible to vectorize this calculation and spare the for loop. Maybe somehow use the
    #  convolution theorem and go back and forth to the frequency domain using FFT for calculating correlations.
    features_mean_diff = features_of_net_samples - features_of_net_samples.mean(axis=1)[:, None]
    num_features = features_of_net_samples.shape[0]
    sample_size = features_of_net_samples.shape[1]
    auto_correlation_func = np.zeros((sample_size, num_features, num_features))
    for k in range(sample_size):
        auto_correlation_func[k] = 1 / sample_size * (
                features_mean_diff[:, :sample_size - k].T.reshape(sample_size - k, num_features, 1) @
                features_mean_diff[:, k:].T.reshape(sample_size - k, 1, num_features)
        ).sum(axis=0)
    return auto_correlation_func


@njit
def approximate_kth_auto_correlation_function(features_mean_diff: np.ndarray, lag: int) -> np.ndarray:
    num_features = features_mean_diff.shape[0]
    sample_size = features_mean_diff.shape[1]
    head = features_mean_diff[:, :sample_size - lag]
    tail = features_mean_diff[:, lag:]
    mean_feat_diff_outer_prods = np.zeros((num_features, num_features, sample_size - lag))
    for i in range(sample_size - lag):
        mean_feat_diff_outer_prods[:, :, i] = np.outer(head[:, i], tail[:, i])
    return 1 / sample_size * mean_feat_diff_outer_prods.sum(axis=2)


def calc_capital_gammas(auto_corr_funcs: np.ndarray) -> np.ndarray:
    """
    This is the capital gammas hat from Geyer's handbook of mcmc (1D) and Dai and Jones 2017 (multi-D).
    They are simply summations over pairs of consecutive even and odd indices of the auto correlation function (gammas).
    """
    # From Dai and Jones 2017 - a mean of gamma with its transpose (which corresponds to the negative index with the
    # same abs value).
    gamma_tilde = (auto_corr_funcs + np.transpose(auto_corr_funcs, [0, 2, 1])) / 2

    # Note - we assume here an even sample_size, it is forced elsewhere (everytime the sample size is updated).
    sample_size = gamma_tilde.shape[0]
    return (gamma_tilde[np.arange(0, sample_size - 1, 2, dtype=int)] +
            gamma_tilde[np.arange(1, sample_size, 2, dtype=int)])


@njit
def calc_kth_capital_gamma(features_mean_diff: np.ndarray, k: int) -> np.ndarray:
    two_k_gamma_hat = approximate_kth_auto_correlation_function(features_mean_diff, 2 * k)
    two_k_plus_one_gamma_hat = approximate_kth_auto_correlation_function(features_mean_diff, 2 * k + 1)
    return (two_k_gamma_hat + two_k_gamma_hat.T + two_k_plus_one_gamma_hat + two_k_plus_one_gamma_hat.T) / 2


@njit
def covariance_matrix_estimation(features_of_net_samples: np.ndarray, mean_features_of_net_samples: np.ndarray,
                                 method='batch', num_batches=25) -> np.ndarray:
    """
    Approximate the covariance matrix of the model's features
    Parameters
    ----------
    features_of_net_samples
        The calculated features of the networks that are used for the approximation. Of dimensions
        (num_features X sample_size)
    mean_features_of_net_samples
        The mean of the features across the samples (a vector of size num_features)
    method
        the method to use for approximating the covariance matrix
        currently supported options are:
            naive
                A naive estimation from the sample: E[gi*gj] - E[gi]E[gj]
            batch
                based on difference of means of sample batches from the total mean, as in Geyer's handbook of
                MCMC (there it is stated for the univariate case, but the generalization is straight forward).
            multivariate_initial_sequence
                Following Dai and Jones 2017 - the first estimator in section 3.1 (denoted mIS).
    TODO: implement a mechanism that allows to pass arguments that are customized for each method

    Returns
    -------
    The covariance matrix estimation (num_features X num_features).
    """
    num_features = features_of_net_samples.shape[0]
    sample_size = features_of_net_samples.shape[1]
    if method == 'naive':
        # An outer product of the means (E[gi]E[gj])
        cross_prod_mean_features = (mean_features_of_net_samples.reshape(num_features, 1) @
                                    mean_features_of_net_samples.T.reshape(1, num_features))
        # A mean of the outer products of the sample (E[gi*gj])
        features_cross_prods = np.zeros((sample_size, num_features, num_features))
        for i in range(sample_size):
            features_cross_prods[i] = np.outer(features_of_net_samples[:, i], features_of_net_samples[:, i])
        mean_features_cross_prod = features_cross_prods.sum(axis=0) / sample_size

        return mean_features_cross_prod - cross_prod_mean_features

    elif method == 'batch':
        # Verify that the sample is nicely divided into non-overlapping batches.
        while sample_size % num_batches != 0:
            num_batches += 1
        batch_size = sample_size // num_batches

        # Divide the sample into batches, and calculate the mean of each one of them
        batches_means = np.zeros((num_features, num_batches))
        for i in range(num_batches):
            batches_means[:, i] = features_of_net_samples[:, i * batch_size:(i + 1) * batch_size].sum(
                axis=1) / batch_size

        diff_of_global_mean = batches_means - mean_features_of_net_samples.reshape(num_features, 1)

        # Average the outer products of the differences between batch means and the global mean and multiply by the
        # batch size to compensate for the aggregation into batches
        diff_of_global_mean_outer_prods = np.zeros((num_batches, num_features, num_features))
        for i in range(num_batches):
            diff_of_global_mean_outer_prods[i] = np.outer(diff_of_global_mean[:, i], diff_of_global_mean[:, i])
        batches_cov_mat_est = batch_size * diff_of_global_mean_outer_prods.sum(axis=0) / num_batches

        return batches_cov_mat_est

    elif method == "multivariate_initial_sequence":
        features_mean_diff = features_of_net_samples - mean_features_of_net_samples.reshape(num_features, 1)

        # In this method, we sum up capital gammas, and choose where to cut the tail (which corresponds to estimates
        # of auto-correlations with large lags within the chain. Naturally, as the lag increases the estimation
        # becomes worse, so the magic here is to determine where to cut). So we calculate the estimates one by one and
        # evaluate the conditions for cutting.
        # The first condition is to have an index where the estimation is positive-definite, namely all eigen-values
        # are positive. As both gamma_0 (which is auto_corr_funcs[0]) and the capital gammas are symmetric, all
        # the sum of them is allways symmetric, which ensures real eigen values, and we can simply calculate the
        # eigen value with the smallest algebraic value to determine whether all of them are positive.
        is_positive = False
        first_pos_def_idx = 0
        cur_pos_cov_mat_est = np.zeros((num_features, num_features))
        gamma_hat_0 = approximate_kth_auto_correlation_function(features_mean_diff, 0)
        while not is_positive:
            if first_pos_def_idx == sample_size // 2:
                # TODO: ValueError? probably should throw something else. And maybe it is better to try alone some
                #  of the remediations suggested here and just notify the user...
                raise ValueError("Got a sample with no valid multivariate_initial_sequence covariance matrix "
                                 "estimation (no possibility is positive-definite). Consider increasing sample size"
                                 " or using a different covariance matrix estimation method.")
            cur_capital_gamma = calc_kth_capital_gamma(features_mean_diff, first_pos_def_idx)
            if first_pos_def_idx == 0:
                cur_pos_cov_mat_est = -gamma_hat_0 + 2 * cur_capital_gamma
            else:
                cur_pos_cov_mat_est += 2 * cur_capital_gamma
            cur_smallest_eigen_val = np.linalg.eigvalsh(cur_pos_cov_mat_est)[0]
            if cur_smallest_eigen_val > 0:
                is_positive = True
            else:
                first_pos_def_idx += 1

        # Now we find the farthest idx after first_pos_def_idx for which the sequence of determinants is strictly
        # monotonically increasing.
        do_dets_increase = True
        cutting_idx = first_pos_def_idx
        cur_det = np.linalg.det(cur_pos_cov_mat_est)
        while do_dets_increase and cutting_idx < sample_size // 2:
            cur_capital_gamma = calc_kth_capital_gamma(features_mean_diff, cutting_idx + 1)
            next_pos_cov_mat_est = cur_pos_cov_mat_est + 2 * cur_capital_gamma
            next_det = np.linalg.det(next_pos_cov_mat_est)
            if next_det <= cur_det:
                do_dets_increase = False
            else:
                cutting_idx += 1
                cur_pos_cov_mat_est = next_pos_cov_mat_est
                cur_det = next_det

        return cur_pos_cov_mat_est

    else:
        raise ValueError(f"{method} is an unsupported method for covariance matrix estimation")


@njit
def calc_nll_gradient(observed_features, mean_features_of_net_samples):
    return mean_features_of_net_samples - observed_features


def get_edge_density_per_type_pairs(W: np.ndarray, types: Collection):
    """
    Calculate the density of edges between each pair of types in the network.

    Parameters
    ----------
    W : np.ndarray
        The adjacency matrix of the network
    
    types : Collection
        A list of types for every node in a network

    Returns
    ----------
    densities : np.ndarray
        An array of edge density per type pairs, which is calculated as follows -    
        p_(type1, type2) = (number of edges between type1 and type2) / (number of potential edges between type1 and type2).

        Array size is k^2 where k is number of types
    """

    sorted_types = sorted(list(set(types)))
    n = W.shape[0]
    real_frequencies = {k: 0 for k in itertools.product(sorted_types, sorted_types)}

    types_frequencies = dict(Counter(types))
    potential_frequencies = {}

    # Count how many potential edges can exist between each pair of types
    for pair in itertools.product(sorted_types, sorted_types):
        type_1 = pair[0]
        type_2 = pair[1]

        if type_1 == type_2:
            potential_frequencies[pair] = types_frequencies[type_1] * (types_frequencies[type_1] - 1)
        else:
            potential_frequencies[pair] = types_frequencies[type_1] * types_frequencies[type_2]

    # Count how many actual edges exist between each pair of types
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            type_a = types[i]
            type_b = types[j]

            real_frequencies[(type_a, type_b)] += W[i, j]

    normalized_real_frequencies = {k: 0 if potential_frequencies[k] == 0 else v / potential_frequencies[k] for k, v in
                                   real_frequencies.items()}
    return normalized_real_frequencies


def calc_hotelling_statistic_for_sample(observed_features: np.ndarray, sample_features: np.ndarray,
                                        cov_mat_est_method: str):
    mean_features = np.mean(sample_features, axis=1)
    cov_mat_est = covariance_matrix_estimation(sample_features, mean_features,
                                               method=cov_mat_est_method)
    inv_cov_mat = np.linalg.pinv(cov_mat_est)
    dist = mahalanobis(observed_features, mean_features, inv_cov_mat)
    sample_size = sample_features.shape[1]
    hotelling_t_stat = sample_size * dist * dist
    num_features = sample_features.shape[0]
    hotelling_t_as_f = ((sample_size - num_features) / (
            num_features * (sample_size - 1))) * hotelling_t_stat
    return hotelling_t_as_f


def sigmoid(x: np.ndarray | float):
    return 1 / (1 + np.exp(-x))


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


def calc_logistic_regression_predictions_log_likelihood(predictions: np.ndarray, ys: np.ndarray):
    """
    Calculates the log-likelihood of labeled data with regard to the predictions of a Logistic Regression model.
    Parameters
    ----------
    predictions
        The predictions of a Logistic Regression model (the probabilities it assigns to feature vectors to be
        labeled 1). Of shape (num_samples X 1)
    ys
        The labeled data (zeros and ones, for each feature vector). Of shape (num_samples X 1)
    Returns
    -------
    The probability to observe the vector `ys` under the distribution induced by `predictions`.
    """
    # TODO: trim to (eps, 1-eps) before taking the log? (the model can't give probabilities of strictly 0 or 1 in
    #  theory, but numerics...)?
    return np.dot(np.log(predictions).T, ys) + np.dot(np.log(1 - predictions).T, 1 - ys)


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

# TODO: njit this and all related functions
def logistic_regression_optimization(Xs: np.ndarray, ys: np.ndarray, initial_thetas: np.ndarray | None = None,
                                     lr: float = 1, max_iter: int = 5000, stopping_thr: float = 1e-6):
    """
    Optimize the parameters of a Logistic Regression model by maximizing the likelihood using Newton-Raphson.
    Parameters
    ----------
    Xs
        The input to the model (the regressors), of shape (num_samples X num_features)
    ys
         The labeled data (zeros and ones, for each feature vector). Of shape (num_samples X 1)
    initial_thetas
        The initial vector of parameters. If `None`, the initial state is randomly sampled from (0,1)
    lr
        The learning rate of the gradient-ascent, scales the step size of the optimization.
    max_iter
        The maximum number of optimization iterations to run.
    stopping_thr
        The fraction of change in the objective function (which is the log-likelihood) that is used as stopping
        criterion (i.e., if the percent change of the objective is smaller than this threshold we stop)
    Returns
        Parameters of the trained model.
    -------

    """
    num_samples = Xs.shape[0]
    ys = ys.reshape((num_samples, 1))
    num_features = Xs.shape[1]
    if initial_thetas is None:
        thetas = np.random.rand(num_features, 1)
    else:
        thetas = initial_thetas.copy()
    log_like_history = np.zeros(max_iter)
    prediction = calc_logistic_regression_predictions(Xs, thetas)
    idx = 0
    start = time.time()
    print("Logistic regression optimization started")
    for i in range(max_iter):
        idx = i
        log_like_history[i] = calc_logistic_regression_predictions_log_likelihood(prediction, ys)
        if (i - 1) % 100 == 0:
            print(f"Iteration {i}, log-likelihood: {log_like_history[i]}, time from start: "
                  f"{time.time() - start:.2f} seconds")
        if i > 0:
            log_like_frac_change = (log_like_history[i] - log_like_history[i - 1]) / log_like_history[i - 1]
            if 0 <= log_like_frac_change < stopping_thr:
                print("Optimization terminated successfully! (the log-likelihood doesn't increase)\n"
                      f"Last iteration: {i}, final log-likelihood: {log_like_history[i]}, time from start: "
                      f"{time.time() - start:.2f} seconds")
                break
        grad = calc_logistic_regression_log_likelihood_grad(Xs, prediction, ys)
        hessian = calc_logistic_regression_log_likelihood_hessian(Xs, prediction)
        hessian_inv = np.linalg.pinv(hessian)
        thetas += lr * hessian_inv @ grad
        prediction = calc_logistic_regression_predictions(Xs, thetas)

    if idx == max_iter:
        print(f"Optimization reached max iterations of {max_iter}!\n"
              f"last log-likelihood: {log_like_history[idx]}, time from start: "
              f"{time.time() - start:.2f} seconds")

    # TODO: remove
    from matplotlib import pyplot as plt
    plt.plot(np.arange(idx), log_like_history[:idx])
    plt.show()

    # TODO: don't return the history
    return thetas.flatten(), prediction.flatten(), log_like_history[:idx]
