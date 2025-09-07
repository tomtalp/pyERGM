import itertools
from collections import Counter
from typing import Collection
import numpy as np
import networkx as nx
from numba import njit, objmode
from scipy.spatial.distance import mahalanobis
from scipy.optimize import minimize, OptimizeResult
from scipy.special import softmax
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

# Dyad states indexing convention
EMPTY_IDX = 0
UPPER_IDX = 1
LOWER_IDX = 2
RECIPROCAL_IDX = 3


@njit
def _numba_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    _numba_seed(seed)


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
    pre_edges_to_flip = np.random.choice(num_nodes, size=num_pairs)

    diff = np.random.choice(num_nodes - 1, size=num_pairs) + 1

    post_edges_to_flip = (pre_edges_to_flip - diff) % num_nodes

    return np.stack((pre_edges_to_flip, post_edges_to_flip))


def get_uniform_random_nodes_to_flip(num_nodes, num_flips):
    """
    Create a vector of size num_flips, where each entry represents a node we wish to flip.
    The nodes are sampled randomly.
    """

    nodes_to_flip = np.random.choice(num_nodes, size=num_flips)

    return nodes_to_flip


def get_uniform_random_new_node_feature_categories(node_features_to_flip, node_features_inds_to_n_categories):
    """
    Create a dictionary of all the possible category flips for the predetermined node features.
    keys are node features indices, and values are dictionaries with the key being a category, and the value being
    random new categories (excluding the current category) - a vector of size of the number of appearances of the
    feature in the random feature flips array given as input. Each entry represents the new category to flip to.
    Totally, we save num_flips x (mean_num_categories - 1) numbers.
    The categories are sampled randomly.
    """

    new_node_features_categories = {}
    for feature_ind, n_categories in node_features_inds_to_n_categories.items():
        categories = np.arange(n_categories)
        new_node_feature_categories = {}
        for c in categories:
            categories_without_c = np.delete(categories, c)
            random_new_categories = np.random.choice(categories_without_c,
                                                     size=(node_features_to_flip == feature_ind).sum())
            new_node_feature_categories[c] = random_new_categories
        new_node_features_categories[feature_ind] = new_node_feature_categories
    return new_node_features_categories


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


@njit
def numerically_stable_minus_log_like_and_grad_local(thetas, Xs, ys):
    linear_pred = Xs @ thetas

    # Magic numbers are taken from sklearn:
    # https://github.com/scikit-learn/scikit-learn/blob/72b35a46684c0ecf4182500d3320836607d1f17c/sklearn/_loss/_loss.pyx.tp#L728
    rng_1_idx = np.where(linear_pred <= -37)[0]
    rng_2_idx = np.where((-37 < linear_pred) & (linear_pred <= -2))[0]
    rng_3_idx = np.where((-2 < linear_pred) & (linear_pred <= 18))[0]
    rng_4_idx = np.where(linear_pred > 18)[0]

    minus_log_like = 0
    minus_log_like_der_per_sample = np.zeros(linear_pred.size)
    if rng_1_idx.size > 0:
        exp_rng_1 = np.exp(linear_pred[rng_1_idx])
        minus_log_like += (exp_rng_1 - ys[rng_1_idx, 0] * linear_pred[rng_1_idx]).sum()
        minus_log_like_der_per_sample[:rng_1_idx.size] = exp_rng_1 - ys[rng_1_idx, 0]
    if rng_2_idx.size > 0:
        exp_rng_2 = np.exp(linear_pred[rng_2_idx])
        minus_log_like += (
                np.log(1 + exp_rng_2) - ys[rng_2_idx, 0] * linear_pred[rng_2_idx]).sum()
        minus_log_like_der_per_sample[rng_1_idx.size:rng_1_idx.size + rng_2_idx.size] = ((1 - ys[
            rng_2_idx, 0]) * exp_rng_2 - ys[rng_2_idx, 0]) / (1 + exp_rng_2)
    if rng_3_idx.size > 0:
        exp_rng_3 = np.exp(-linear_pred[rng_3_idx])
        minus_log_like += (
                np.log(1 + exp_rng_3) + (1 - ys[rng_3_idx, 0]) * linear_pred[rng_3_idx]).sum()
        minus_log_like_der_per_sample[
        rng_1_idx.size + rng_2_idx.size: minus_log_like_der_per_sample.size - rng_4_idx.size
        ] = ((1 - ys[rng_3_idx, 0]) - ys[rng_3_idx, 0] * exp_rng_3) / (1 + exp_rng_3)
    if rng_4_idx.size > 0:
        exp_rng_4 = np.exp(-linear_pred[rng_4_idx])
        minus_log_like += (exp_rng_4 + (1 - ys[rng_4_idx, 0]) * linear_pred[rng_4_idx]).sum()
        minus_log_like_der_per_sample[minus_log_like_der_per_sample.size - rng_4_idx.size:
        ] = ((1 - ys[rng_4_idx, 0]) - ys[rng_4_idx, 0] * exp_rng_4) / (1 + exp_rng_4)

    return minus_log_like, Xs.T @ minus_log_like_der_per_sample


def analytical_minus_log_likelihood_distributed(thetas, data_path, num_edges_per_job):
    return -distributed_logistic_regression_optimization_step(data_path, thetas.reshape(thetas.size, 1),
                                                              'log_likelihood', num_edges_per_job)


def analytical_logistic_regression_predictions_distributed(thetas, data_path, num_edges_per_job):
    return distributed_logistic_regression_optimization_step(data_path, thetas.reshape(thetas.size, 1),
                                                             'predictions', num_edges_per_job)


def analytical_minus_log_like_grad_local(thetas, Xs, ys, eps=1e-10):
    pred = np.clip(calc_logistic_regression_predictions(Xs, thetas.reshape(thetas.size, 1)), eps, 1 - eps)
    return -calc_logistic_regression_log_likelihood_grad(Xs, pred, ys).reshape(thetas.size, )


def analytical_minus_log_like_grad_distributed(thetas, data_path, num_edges_per_job):
    return -distributed_logistic_regression_optimization_step(data_path, thetas.reshape(thetas.size, 1),
                                                              'log_likelihood_gradient',
                                                              num_edges_per_job).reshape(thetas.size, )


def analytical_minus_log_likelihood_hessian_local(thetas, Xs, ys, eps=1e-10):
    pred = np.clip(calc_logistic_regression_predictions(Xs, thetas.reshape(thetas.size, 1)), eps, 1 - eps)
    return -calc_logistic_regression_log_likelihood_hessian(Xs, pred)


def analytical_minus_log_likelihood_hessian_distributed(thetas, data_path, num_edges_per_job):
    return -distributed_logistic_regression_optimization_step(data_path, thetas.reshape(thetas.size, 1),
                                                              'log_likelihood_hessian',
                                                              num_edges_per_job)


def mple_logistic_regression_optimization(metrics_collection, observed_networks: np.ndarray,
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
        observed_nets_path = os.path.join(data_path, 'observed_networks.pkl')
        with open(observed_nets_path, 'wb') as f:
            pickle.dump(observed_networks, f)

    num_features = metrics_collection.calc_num_of_features()
    if initial_thetas is None:
        thetas = np.random.rand(num_features)
    else:
        thetas = initial_thetas.copy()

    if not is_distributed:
        if optimization_method == "Newton-CG":
            res = minimize(analytical_minus_log_likelihood_local, thetas, args=(Xs, ys),
                           jac=analytical_minus_log_like_grad_local, hess=analytical_minus_log_likelihood_hessian_local,
                           callback=_after_optim_iteration_callback, method="Newton-CG")
        elif optimization_method == "L-BFGS-B":
            res = minimize(analytical_minus_log_likelihood_local, thetas, args=(Xs, ys),
                           jac=analytical_minus_log_like_grad_local, method="L-BFGS-B",
                           callback=_after_optim_iteration_callback)
        else:
            raise ValueError(
                f"Unsupported optimization method: {optimization_method}. Options are: Newton-CG, L-BFGS-B")
        pred = calc_logistic_regression_predictions(Xs, res.x.reshape(-1, 1)).flatten()
    else:
        # TODO: support L-BFGS-B as an optimization method, and implement a numerically stable version for distributed
        #  calculations as well.
        num_edges_per_job = kwargs.get('num_edges_per_job', 100000)
        if optimization_method == "Newton-CG":
            res = minimize(analytical_minus_log_likelihood_distributed, thetas, args=(data_path, num_edges_per_job),
                           jac=analytical_minus_log_like_grad_distributed,
                           hess=analytical_minus_log_likelihood_hessian_distributed,
                           callback=_after_optim_iteration_callback, method="Newton-CG")
        else:
            raise ValueError(
                f"Unsupported optimization method: {optimization_method} for distributed optimization. "
                f"Options are: Newton-CG")
        pred = analytical_logistic_regression_predictions_distributed(res.x.reshape(-1, 1), data_path,
                                                                      num_edges_per_job).flatten()

    print(res)
    sys.stdout.flush()
    return res.x, pred, res.success


def distributed_logistic_regression_optimization_step(data_path, thetas, func_to_calc, num_edges_per_job=5000):
    # TODO: support calculating multiple functions in a single job array, maybe by getting an array of function
    #  names rather than a single function as func_to_calc. This will enable passing jac=True to scipy.optimize.minimize
    #  when the optimization method is L-BFGS-B, and calculating both log-likelihood and gradient in a single job array,
    #  reducing the number of sent jobs significantly.

    # Arrange files and send the children jobs
    num_jobs, out_path, job_array_ids = _run_distributed_logistic_regression_children_jobs(data_path, thetas,
                                                                                           func_to_calc,
                                                                                           num_edges_per_job)

    # Wait for all jobs to finish.
    chunks_path = (out_path / func_to_calc).resolve()
    os.makedirs(chunks_path, exist_ok=True)
    wait_for_distributed_children_outputs(num_jobs, chunks_path, job_array_ids, func_to_calc)
    # Clean current scripts
    shutil.rmtree((out_path / "scripts").resolve())

    # Aggregate results
    if func_to_calc == "predictions":
        aggregated_func = cat_children_jobs_outputs(num_jobs, (out_path / func_to_calc).resolve())
    else:
        aggregated_func = _sum_children_jobs_outputs(num_jobs, (out_path / func_to_calc).resolve())

    # Clean current outputs
    shutil.rmtree((out_path / func_to_calc).resolve())
    return aggregated_func


def _construct_single_batch_bash_cmd_logistic_regression(out_path, cur_thetas, func_to_calculate, num_edges_per_job):
    # Construct a string with the current thetas, to pass using the command line to children jobs.
    thetas_str = ''
    for t in cur_thetas:
        thetas_str += f'{t[0]} '
    thetas_str = thetas_str[:-1]

    cmd_line_for_bsub = (f'python ./logistic_regression_distributed_calcs.py '
                         f'--out_dir_path {out_path} '
                         f'--num_edges_per_job {num_edges_per_job} '
                         f'--function {func_to_calculate} '
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


def _run_distributed_logistic_regression_children_jobs(data_path, cur_thetas, func_to_calculate, num_edges_per_job):
    out_path = data_path.parent

    cmd_line_single_batch = _construct_single_batch_bash_cmd_logistic_regression(out_path, cur_thetas,
                                                                                 func_to_calculate, num_edges_per_job)

    with open(os.path.join(data_path, "observed_networks.pkl"), 'rb') as f:
        observed_networks = pickle.load(f)

    num_nodes = observed_networks.shape[0]
    num_data_points = num_nodes * num_nodes - num_nodes
    num_jobs = int(np.ceil(num_data_points / num_edges_per_job))

    job_array_ids = run_distributed_children_jobs(out_path, cmd_line_single_batch, "distributed_logistic_regression.sh",
                                                  num_jobs, func_to_calculate)
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


def generate_binomial_tensor(net_size, node_features_size, num_samples, p=0.5):
    """
    Generate a tensor of size (net_size, net_size, num_samples) where each element is a binomial random variable
    """
    return np.random.binomial(1, p, (net_size, net_size + node_features_size, num_samples)).astype(np.int8)


def sample_from_independent_probabilities_matrix(probability_matrix, sample_size, is_directed):
    """
    Sample connectivity matrices from a matrix representing the independent probability of an edge between nodes (i, j)
    """
    n_nodes = probability_matrix.shape[0]
    sample = np.zeros((n_nodes, n_nodes, sample_size))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            elif not is_directed and i > j:
                sample[i, j, :] = sample[j, i, :]
                continue
            else:
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


def predict_multi_class_logistic_regression(Xs, thetas):
    return softmax(Xs @ thetas, axis=1)


def log_likelihood_multi_class_logistic_regression(true_labels, predictions, reduction='sum', log_base=np.exp(1)):
    # TODO: trim predictions to avoid log(0)? If yes make sure the predictions over all dyad states sum up to 1 after
    #  trimming? i.e., only the first row or also the last one?
    #   predictions = np.clip(predictions, a_min=eps, a_max=1-eps)
    #   predictions /= predictions.sum(axis=0)
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


def mple_reciprocity_logistic_regression_optimization(metrics_collection, observed_networks: np.ndarray,
                                                      initial_thetas: np.ndarray | None = None,
                                                      optimization_method: str = 'L-BFGS-B'):
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


def num_dyads_to_num_nodes(num_dyads):
    """
    x = num_dyads
    n(n-1) = 2*x
    n^2-n-2x=0 --> n = \frac{1+\sqrt{1-4\cdot(-2x)}}{2}
    """
    return np.round((1 + np.sqrt(1 + 8 * num_dyads)) / 2).astype(int)


def convert_connectivity_to_dyad_states(connectivity: np.ndarray):
    n_nodes = connectivity.shape[0]
    dyads_states = np.zeros(((n_nodes ** 2 - n_nodes) // 2, 4))
    idx = 0
    for i in range(n_nodes - 1):
        for j in range(i + 1, n_nodes):
            if not connectivity[i, j] and not connectivity[j, i]:
                dyads_states[idx, EMPTY_IDX] = 1
            elif connectivity[i, j] and not connectivity[j, i]:
                dyads_states[idx, UPPER_IDX] = 1
            elif not connectivity[i, j] and connectivity[j, i]:
                dyads_states[idx, LOWER_IDX] = 1
            else:
                dyads_states[idx, RECIPROCAL_IDX] = 1

            idx += 1
    return dyads_states


def convert_dyads_states_to_connectivity(dyads_states):
    num_dyads = dyads_states.shape[0]
    num_nodes = num_dyads_to_num_nodes(num_dyads)
    indices = np.triu_indices(num_nodes, k=1)
    network = np.zeros((num_nodes, num_nodes))
    for i in range(num_dyads):
        is_upper = dyads_states[i, UPPER_IDX] or dyads_states[i, RECIPROCAL_IDX]
        is_lower = dyads_states[i, LOWER_IDX] or dyads_states[i, RECIPROCAL_IDX]
        if is_upper:
            network[indices[0][i], indices[1][i]] = 1
        if is_lower:
            network[indices[1][i], indices[0][i]] = 1
    return network


def convert_dyads_state_indices_to_connectivity(dyads_states_indices):
    num_dyads = dyads_states_indices.shape[0]
    num_nodes = num_dyads_to_num_nodes(num_dyads)
    indices = np.triu_indices(num_nodes, k=1)
    network = np.zeros((num_nodes, num_nodes))
    for i in range(num_dyads):
        is_upper = dyads_states_indices[i] in [UPPER_IDX, RECIPROCAL_IDX]
        is_lower = dyads_states_indices[i] in [LOWER_IDX, RECIPROCAL_IDX]
        if is_upper:
            network[indices[0][i], indices[1][i]] = 1
        if is_lower:
            network[indices[1][i], indices[0][i]] = 1
    return network


def sample_from_dyads_distribution(dyads_distributions, sample_size):
    num_dyads = dyads_distributions.shape[0]
    n_nodes = num_dyads_to_num_nodes(num_dyads)
    dyads_states_indices_sample = np.zeros((num_dyads, sample_size))
    for i in range(num_dyads):
        dyads_states_indices_sample[i] = np.random.choice(np.arange(4), p=dyads_distributions[i], size=sample_size)

    net_sample = np.zeros((n_nodes, n_nodes, sample_size))
    for k in range(sample_size):
        net_sample[:, :, k] = convert_dyads_state_indices_to_connectivity(dyads_states_indices_sample[:, k])
    return net_sample


def get_exact_marginals_from_dyads_distrubution(dyads_distributions):
    num_dyads = dyads_distributions.shape[0]
    num_nodes = num_dyads_to_num_nodes(num_dyads)
    indices = np.triu_indices(num_nodes, k=1)
    exact_marginals = np.zeros((num_nodes, num_nodes))
    for i in range(num_dyads):
        exact_marginals[indices[0][i], indices[1][i]] = dyads_distributions[i, RECIPROCAL_IDX] + dyads_distributions[
            i, UPPER_IDX]
        exact_marginals[indices[1][i], indices[0][i]] = dyads_distributions[i, RECIPROCAL_IDX] + dyads_distributions[
            i, LOWER_IDX]
    return exact_marginals


def remove_main_diagonal_flatten(square_mat):
    # TODO: there are multiple places we can use that that currently duplicate this logic.
    if square_mat.ndim != 2 or square_mat.shape[0] != square_mat.shape[1]:
        raise ValueError("The input must be a square matrix")
    return square_mat[~np.eye(square_mat.shape[0], dtype=bool)].flatten()


def set_off_diagonal_elements_from_array(square_mat, values_to_set):
    values_to_set = values_to_set.flatten()
    if values_to_set.size != square_mat.size - square_mat.shape[0]:
        raise ValueError("The size of the array must be compatible the size of the square matrix")
    square_mat[~np.eye(square_mat.shape[0], dtype=bool)] = values_to_set


def get_edges_indices_lims(edges_indices_lims: tuple[int, int] | None, n_nodes: int, is_directed: bool):
    if edges_indices_lims is None:
        num_edges_to_take = n_nodes * n_nodes - n_nodes
        if not is_directed:
            num_edges_to_take = num_edges_to_take // 2
        edges_indices_lims = (0, num_edges_to_take)
    return edges_indices_lims


def expand_net_dims(net: np.ndarray) -> np.ndarray:
    if net.ndim == 2:
        # a single network
        return net[..., np.newaxis]
    elif net.ndim != 3:
        raise ValueError("Cannot expand dims to an array that is not 2 or 3 dimensional")
    return net
