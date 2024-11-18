import itertools
from collections import Counter
from typing import Collection
import numpy as np
import networkx as nx
from numba import njit, objmode
from scipy.spatial.distance import mahalanobis
import torch
import random
import time
from pathlib import Path
import os
import pickle
import shutil
import subprocess
import sys

# Indices of columns in the output of bjobs -A
JOB_ARRAY_ID_IDX = 0
ARRAY_SPEC_IDX = 1
DONE_IDX = 5
EXIT_IDX = 7

LSF_ID_LIST_LEN_LIMIT = 100


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
def get_uniform_random_edges_to_flip(num_nodes, num_pairs):
    """
    Create a matrix of size (2 x num_pairs), where each column represents a pair of nodes.
    These nodes represent the edge we wish to flip.
    The pairs are sampled randomly.
    """

    edges_to_flip = np.zeros((2, num_pairs), dtype=np.int32)

    edges_to_flip[0, :] = np.random.choice(num_nodes, size=num_pairs)

    diff = np.random.choice(num_nodes - 1, size=num_pairs) + 1

    edges_to_flip[1, :] = (edges_to_flip[0, :] - diff) % num_nodes

    return edges_to_flip


def convert_flat_no_diag_idx_to_i_j(flat_no_diag_idx: Collection[int], full_mat_size: int) -> np.ndarray[int]:
    """
    Converts the index in the flattened square matrix without the main diagonal to the pair of indices in the original
    matrix.
    For a square matrix A, the flattened-no-diagonal form is given by A[~np.eye(A.shape[0], dtype=bool)].
    E.g., given a 3X3 matrix, the third element in its flattened-no-diagonal form (idx 2) is the (1,0) entry.
    Parameters
    ----------
    flat_no_diag_idx
        The index in the flattened-no-diagonal form
        #TODO: force this to be a np.ndarray and use numba?
    full_mat_size
        The number of rows/columns in the original squared matrix.
    Returns
    -------
    The tuple of indices of the entry in the original square matrix.
    """
    flat_no_diag_idx = np.array(flat_no_diag_idx)
    if np.any(flat_no_diag_idx >= full_mat_size * (full_mat_size - 1)):
        raise IndexError(
            f"Got a too large `flat_no_diag_idx` {flat_no_diag_idx} for original matrix size of {full_mat_size}")
    rows = np.array(flat_no_diag_idx) // (full_mat_size - 1)
    cols = np.array(flat_no_diag_idx) % (full_mat_size - 1)
    cols[cols >= rows] += 1
    return np.stack((rows, cols), axis=0).astype(np.int32)


def get_custom_distribution_random_edges_to_flip(num_pairs, edge_probs):
    """
    Sample pairs of indices for edge flips according to a given probability distribution over all possible edges (all
    entries of the adjacency matrix but the main diagonal).
    Parameters
    ----------
    num_pairs
        The sample size
    edge_probs
        The probability distribution over all possible edges. The indexing here is in the flattened-no-diagonal format.
        This is equivalent to having a matrix A of nXn probabilities, where the i,j-th entry is the probability of the
        i,j-th edge to be sampled, and passing A[~np.eye(A.shape[0], dtype=bool)].
    Returns
    -------
    A sample of num_pairs pairs of indices, sampled according to edge_probs.
    """
    num_possible_edges = edge_probs.size
    # The solution for the equation n(n-1)=num_possible_edges
    num_nodes = np.round((1 + np.sqrt(1 + 4 * num_possible_edges)) / 2)  # TODO: validate that this is indeed an int?
    flat_no_diag_indices = np.random.choice(edge_probs.size, p=edge_probs, size=num_pairs)
    # TODO: force the following to be pre-complied with numba, and the current as well?
    return convert_flat_no_diag_idx_to_i_j(flat_no_diag_indices, num_nodes)


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


def get_sorted_type_pairs(types):
    sorted_types = sorted(list(set(types)))
    return list(itertools.product(sorted_types, sorted_types))


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
    type_pairs = get_sorted_type_pairs(types)
    n = W.shape[0]
    real_frequencies = {k: 0 for k in type_pairs}

    types_frequencies = dict(Counter(types))
    potential_frequencies = {}

    # Count how many potential edges can exist between each pair of types
    for pair in type_pairs:
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
def calc_logistic_regression_predictions_log_likelihood(predictions: np.ndarray, ys: np.ndarray, eps=1e-10):
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
    trimmed_likelihoods = np.clip(predictions, eps, 1 - eps)
    data_zero_indices = np.where(ys == 0)[0]
    trimmed_likelihoods[data_zero_indices] = np.ones((data_zero_indices.size, 1)) - trimmed_likelihoods[
        data_zero_indices]

    return np.log(trimmed_likelihoods).sum()


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


def local_mple_logistic_regression_optimization_step(Xs, ys, thetas):
    prediction = calc_logistic_regression_predictions(Xs, thetas)
    log_like = calc_logistic_regression_predictions_log_likelihood(prediction, ys)
    grad = calc_logistic_regression_log_likelihood_grad(Xs, prediction, ys)
    hessian = calc_logistic_regression_log_likelihood_hessian(Xs, prediction)
    return prediction, log_like, grad, hessian


def mple_logistic_regression_optimization(metrics_collection, observed_network: np.ndarray,
                                          initial_thetas: np.ndarray | None = None,
                                          lr: float = 1, max_iter: int = 5000, stopping_thr: float = 1e-6,
                                          is_distributed: bool = False,
                                          minimal_lr=1e-10):
    """
    Optimize the parameters of a Logistic Regression model by maximizing the likelihood using Newton-Raphson.
    Parameters
    ----------
    metrics_collection
        The `MetricsCollection` with relation to which the optimization is carried out.
        # TODO: we can't add a type hint for this, due to circular import (utils can't import from metrics, as metrics
            already imports from utils). This might suggest that this isn't the right place for this function.
    observed_network
        The observed network used as data for the optimization.
    initial_thetas
        The initial vector of parameters. If `None`, the initial state is randomly sampled from (0,1)
    lr
        The learning rate of the gradient-ascent, scales the step size of the optimization.
    max_iter
        The maximum number of optimization iterations to run.
    stopping_thr
        The fraction of change in the objective function (which is the log-likelihood) that is used as stopping
        criterion (i.e., if the percent change of the objective is smaller than this threshold we stop)
    is_distributed
        Whether the calculations are carried locally or distributed over many compute nodes of an IBM LSF cluster.
    Returns
        Parameters of the trained model.
    -------

    """
    num_features = metrics_collection.calc_num_of_features()
    if initial_thetas is None:
        thetas = np.random.rand(num_features, 1)
    else:
        thetas = initial_thetas.copy()

    cur_log_like = -np.inf
    prev_log_like = -np.inf
    prev_thetas = thetas.copy()

    if not is_distributed:
        Xs, ys = metrics_collection.prepare_mple_data(observed_network)
    else:
        out_dir_path = (Path.cwd().parent / "OptimizationIntermediateCalculations").resolve()
        data_path = (out_dir_path / "data").resolve()
        os.makedirs(data_path, exist_ok=True)

        # Copy the `MetricsCollection` and the observed network to provide its path to children jobs, so they will be
        # able to access it.
        metric_collection_path = os.path.join(data_path, 'metric_collection.pkl')
        with open(metric_collection_path, 'wb') as f:
            pickle.dump(metrics_collection, f)
        observed_net_path = os.path.join(data_path, 'observed_network.pkl')
        with open(observed_net_path, 'wb') as f:
            pickle.dump(observed_network, f)

    idx = 0
    with objmode(start='f8'):
        start = time.perf_counter()
    print("Logistic regression optimization started")
    sys.stdout.flush()
    for i in range(max_iter):
        idx = i
        if not is_distributed:
            prediction, cur_log_like, grad, hessian = local_mple_logistic_regression_optimization_step(Xs, ys, thetas)
        else:
            prediction, cur_log_like, grad, hessian = distributed_logistic_regression_optimization_step(data_path,
                                                                                                        thetas)
        if (i - 1) % 1 == 0:
            with objmode():
                print(
                    "Iteration {0}, log-likelihood: {1}, time from start: {2} seconds, lr: {3}".format(i, cur_log_like,
                                                                                                       time.perf_counter() - start,
                                                                                                       lr))
                sys.stdout.flush()
        if i > 0:
            log_like_frac_change = (cur_log_like - prev_log_like)
            if cur_log_like != 0:
                log_like_frac_change /= np.abs(prev_log_like)
            if 0 < log_like_frac_change < stopping_thr:  # TODO - If the lr is extremely small, then "by definition" we will have very small changes in the log-likelihood. So we need to account for small lr's
                with objmode():
                    print(
                        "Optimization terminated successfully! (the log-likelihood doesn't increase)\nLast iteration: {0}, final log-likelihood: {1}, time from start: {2} seconds".format(
                            i, cur_log_like, time.perf_counter() - start))
                break
            elif log_like_frac_change < 0:
                print(
                    f"\tLog-likelihood decreased from {prev_log_like} to {cur_log_like} in iteration {i}! Decreasing learning rate and reverting to previous thetas.")

                thetas = prev_thetas
                if lr > minimal_lr:
                    lr /= 2
                else:
                    with objmode():
                        # raise Exception("Learning rate decreased to minimal value while log-likelihood is still decreasing. Stopping optimization.")
                        print(
                            "Learning rate decreased to minimal value while log-likelihood is still decreasing. Stopping optimization.")
                    break
                continue

        prev_thetas = thetas.copy()
        hessian_inv = np.linalg.pinv(hessian)
        thetas += lr * hessian_inv @ grad
        prev_log_like = cur_log_like

    if idx == max_iter:
        with objmode():
            print(
                "Optimization reached max iterations of {0}!\nlast log-likelihood: {1}, time from start: {2} seconds".format(
                    max_iter, cur_log_like, time.perf_counter() - start))

    if is_distributed:
        shutil.rmtree(data_path)

    return thetas.flatten(), prediction.flatten()


def distributed_logistic_regression_optimization_step(data_path, thetas, num_edges_per_job=5000):
    # Arrange files and send the children jobs
    num_jobs, out_path, job_array_ids = _run_distributed_logistic_regression_children_jobs(data_path, thetas,
                                                                                           num_edges_per_job)

    # Wait for all jobs to finish. Check the hessian path because it is the last to be computed for each data chunk.
    hessian_path = (out_path / "hessian").resolve()
    os.makedirs(hessian_path, exist_ok=True)
    wait_for_distributed_children_outputs(num_jobs, hessian_path, job_array_ids, "log_reg_step")
    # Clean current scripts
    shutil.rmtree((out_path / "scripts").resolve())

    # Aggregate results
    predictions = cat_children_jobs_outputs(num_jobs, (out_path / "prediction").resolve())
    log_likelihood = _sum_children_jobs_outputs(num_jobs, (out_path / "log_like").resolve())
    grad = _sum_children_jobs_outputs(num_jobs, (out_path / "grad").resolve())
    hessian = _sum_children_jobs_outputs(num_jobs, (out_path / "hessian").resolve())

    # Clean current outputs
    shutil.rmtree((out_path / "prediction").resolve())
    shutil.rmtree((out_path / "log_like").resolve())
    shutil.rmtree((out_path / "grad").resolve())
    shutil.rmtree((out_path / "hessian").resolve())
    return predictions, log_likelihood, grad, hessian


def _construct_single_batch_bash_file_logistic_regression(out_path, cur_thetas, num_edges_per_job):
    # Construct a string with the current thetas, to pass using the command line to children jobs.
    thetas_str = ''
    for t in cur_thetas:
        thetas_str += f'{t[0]} '
    thetas_str = thetas_str[:-1]

    cmd_line_for_bsub = (f'python ./logistic_regression_distributed_calcs.py '
                         f'--out_dir_path {out_path} '
                         f'--num_edges_per_job {num_edges_per_job} '
                         f'--thetas {thetas_str}')
    return cmd_line_for_bsub


def run_distributed_children_jobs(out_path, cmd_line_single_batch, single_batch_template_file_name, num_jobs,
                                  array_name):
    # Create current bash scripts to send distributed calculations
    scripts_path = (out_path / "scripts").resolve()
    os.makedirs(scripts_path, exist_ok=True)
    single_batch_bash_path = os.path.join(scripts_path, "single_batch.sh")
    shutil.copyfile(os.path.join(os.getcwd(), "ClusterScripts", single_batch_template_file_name),
                    single_batch_bash_path)
    with open(single_batch_bash_path, 'a') as f:
        f.write(cmd_line_single_batch)

    multiple_batches_bash_path = os.path.join(scripts_path, "multiple_batches.sh")
    with open(multiple_batches_bash_path, 'w') as f:
        num_rows = 1
        while (num_rows - 1) * 2000 < num_jobs:
            f.write(f'bsub < $1 -J {array_name}'
                    f'[{(num_rows - 1) * 2000 + 1}-{min(num_rows * 2000, num_jobs)}]\n')
            num_rows += 1

    # Make sure the logs directory for the children jobs exists and delete previous logs.
    with open(single_batch_bash_path, 'r') as f:
        single_batch_bash_txt = f.read()
    logs_rel_dir_start = single_batch_bash_txt.find('-o') + 3
    log_rel_dir_end = single_batch_bash_txt.find('outs.%J.%I.log')
    logs_rel_dir = single_batch_bash_txt[logs_rel_dir_start:log_rel_dir_end]
    logs_dir = os.path.join(os.getcwd(), logs_rel_dir)
    os.makedirs(logs_dir, exist_ok=True)
    for file_name in os.listdir(logs_dir):
        os.unlink(os.path.join(logs_dir, file_name))

    # Send the jobs
    send_jobs_command = f'bash {multiple_batches_bash_path} {single_batch_bash_path}'
    jobs_sending_res = subprocess.run(send_jobs_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    job_array_ids = _parse_sent_job_array_ids(jobs_sending_res.stdout)

    return job_array_ids


def _run_distributed_logistic_regression_children_jobs(data_path, cur_thetas, num_edges_per_job):
    out_path = data_path.parent

    cmd_line_single_batch = _construct_single_batch_bash_file_logistic_regression(out_path, cur_thetas,
                                                                                  num_edges_per_job)

    with open(os.path.join(data_path, "observed_network.pkl"), 'rb') as f:
        observed_network = pickle.load(f)

    num_nodes = observed_network.shape[0]
    num_data_points = num_nodes * num_nodes - num_nodes
    num_jobs = int(np.ceil(num_data_points / num_edges_per_job))

    job_array_ids = run_distributed_children_jobs(out_path, cmd_line_single_batch, "distributed_logistic_regression.sh",
                                                  num_jobs, "log_reg_step")
    return num_jobs, out_path, job_array_ids


def wait_for_distributed_children_outputs(num_jobs: int, out_path: Path, job_array_ids: list, array_name: str):
    is_all_done = False
    time_of_last_job_status_check = time.time()
    num_jobs_to_listen_to = num_jobs
    are_there_missing_jobs = True
    # Wait that all output files will be there (all jobs finished successfully).
    while are_there_missing_jobs:
        # Wait for the current running jobs to finish
        while not is_all_done:
            if time.time() - time_of_last_job_status_check > 60:
                is_all_done = _check_if_all_done(job_array_ids, num_jobs_to_listen_to)
                time_of_last_job_status_check = time.time()
        print("is_all_done is True")
        sys.stdout.flush()

        # Reset the ids of arrays and number of jobs to listen to, as previous ones are done.
        job_array_ids = []
        num_jobs_to_listen_to = 0

        # Identify missing output files
        out_dir_files_list = os.listdir(out_path)
        missing_jobs = [i for i in range(num_jobs) if f'{i}.pkl' not in out_dir_files_list]
        are_there_missing_jobs = (len(missing_jobs) > 0)
        if are_there_missing_jobs:
            resent_job_array_ids = _resend_failed_jobs(out_path.parent, missing_jobs, array_name)
            print(f"sent missing jobs: {missing_jobs}")
            sys.stdout.flush()
            job_array_ids += resent_job_array_ids
            num_jobs_to_listen_to += len(missing_jobs)
            is_all_done = False
    print("All output files exist")
    sys.stdout.flush()


def _sum_children_jobs_outputs(num_jobs: int, out_path: Path):
    measure = None
    for j in range(num_jobs):
        with open(os.path.join(out_path, f'{j}.pkl'), 'rb') as f:
            content = pickle.load(f)
            if measure is None:
                measure = content
            else:
                measure += content
    return measure


def cat_children_jobs_outputs(num_jobs: int, out_path: Path, axis: int = 0):
    measure = None
    for j in range(num_jobs):
        with open(os.path.join(out_path, f'{j}.pkl'), 'rb') as f:
            content = pickle.load(f)
            if measure is None:
                measure = content
            else:
                measure = np.concatenate((measure, content), axis=axis)
    return measure


def _check_if_all_done(job_array_ids: list, num_sent_jobs: int) -> bool:
    job_stats_res = subprocess.run(['bjobs', '-A'], stdout=subprocess.PIPE)
    job_stats = job_stats_res.stdout

    # split into different lines, without the header line (and the last empty line which is an artifact of the
    # splitting).
    lines = job_stats.split(b'\n')[1:-1]

    # remove retry lines
    indices_to_remove = []
    for i in range(len(lines)):
        if b'Batch system concurrent query limit exceeded' in lines[i]:
            indices_to_remove.append(i)
    for i in indices_to_remove:
        lines.remove(lines[i])

    # clean spaces
    for i, line in enumerate(lines):
        lines[i] = [c for c in line.split(b' ') if c != b'']

    # Count finished jobs in relevant lines
    num_finished_jobs = 0
    counted_finish_per_line = {}
    array_ids_from_output = [int(lines[i][JOB_ARRAY_ID_IDX]) for i in range(len(lines))]
    for job_arr_id in job_array_ids:
        try:
            relevant_line_idx_in_output = array_ids_from_output.index(job_arr_id)
        except ValueError:
            # There is no line matching this array id in the output of bjobs -A, so it probably hasn't been sent
            # yet.
            print(f"job array {job_arr_id} is not found in bjobs -A output, found ids: {array_ids_from_output}."
                  f"returning False from _check_if_all_done")
            sys.stdout.flush()
            return False
        num_done = int(lines[relevant_line_idx_in_output][DONE_IDX])
        num_exit = int(lines[relevant_line_idx_in_output][EXIT_IDX])
        num_finished_jobs += num_done
        num_finished_jobs += num_exit
        counted_finish_per_line[job_arr_id] = {'done': num_done, 'exit': num_exit}

    if num_finished_jobs < num_sent_jobs:
        return False
    elif num_finished_jobs == num_sent_jobs:
        return True
    else:
        raise ValueError(f"The number of finished jobs {num_finished_jobs} is larger than the total number of "
                         f"sent jobs {num_sent_jobs}, wrong counting!\ncounts per line: {counted_finish_per_line}")


def _resend_failed_jobs(out_path: Path, job_indices: list, array_name: str) -> list:
    num_failed_jobs = len(job_indices)
    job_array_ids = []

    for i in range(num_failed_jobs // LSF_ID_LIST_LEN_LIMIT + 1):
        cur_job_indices_str = ''
        for j_idx_in_list in range(i * LSF_ID_LIST_LEN_LIMIT, min((i + 1) * LSF_ID_LIST_LEN_LIMIT, num_failed_jobs)):
            cur_job_indices_str += f'{job_indices[j_idx_in_list] + 1},'
        cur_job_indices_str = cur_job_indices_str[:-1]
        single_batch_bash_path = os.path.join(out_path, "scripts", "single_batch.sh")
        resend_job_command = f'bsub -J {array_name}[{cur_job_indices_str}]'
        jobs_sending_res = subprocess.run(resend_job_command.split(), stdin=open(single_batch_bash_path, 'r'),
                                          stdout=subprocess.PIPE)
        job_array_ids += _parse_sent_job_array_ids(jobs_sending_res.stdout)

    return job_array_ids


def _parse_sent_job_array_ids(process_stdout) -> list:
    split_array_lines = process_stdout.split(b'\n')[:-1]
    job_array_ids = []
    for line in split_array_lines:
        array_id = int(line.split(b'<')[1].split(b'>')[0])
        job_array_ids.append(array_id)
    return job_array_ids


def generate_binomial_tensor(net_size, num_samples, p=0.5):
    """
    Generate a tensor of size (net_size, net_size, num_samples) where each element is a binomial random variable
    """
    return np.random.binomial(1, p, (net_size, net_size, num_samples)).astype(np.int8)


def sample_from_independent_probabilities_matrix(probability_matrix, sample_size):
    """
    Sample connectivity matrices from a matrix representing the independent probability of an edge between nodes (i, j)
    """
    n_nodes = probability_matrix.shape[0]
    sample = np.zeros((n_nodes, n_nodes, sample_size))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            sample[i, j, :] = np.random.binomial(1, probability_matrix[i, j], size=sample_size)

    return sample


# TODO: njit?
def split_network_for_bootstrapping(net_size: int, first_part_size: int, splitting_method: str = 'uniform') -> tuple[
    np.ndarray[int], np.ndarray[int]]:
    if splitting_method == 'uniform':
        indices = np.arange(net_size)
        np.random.shuffle(indices)
        first_part_indices = indices[:first_part_size].reshape((first_part_size, 1))
    else:
        raise ValueError(f"splitting method {splitting_method} not supported")

    second_part_indices = np.setdiff1d(np.arange(net_size).astype(int), first_part_indices).reshape(
        (net_size - first_part_size, 1))

    return first_part_indices, second_part_indices
