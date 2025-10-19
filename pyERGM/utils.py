import itertools
from collections import Counter
from typing import Collection
import numpy as np
import networkx as nx
from numba import njit
from scipy.spatial.distance import mahalanobis
import torch
import random

import pickle
from memory_profiler import profile

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


@profile
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


@profile
def profiled_pickle_dump(out_path: str, obj):
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)

def calc_entropy_independent_probability_matrix(
        prob_mat: np.ndarray,
        reduction: str = 'sum',
        eps: float = 1e-10
) -> float | np.ndarray:
    flattened_clipped_no_diag_mat = np.clip(remove_main_diagonal_flatten(prob_mat), a_min=eps, a_max=1 - eps)
    entropy_per_entry = -(
            flattened_clipped_no_diag_mat * np.log2(flattened_clipped_no_diag_mat) +
            (1 - flattened_clipped_no_diag_mat) * np.log2(1 - flattened_clipped_no_diag_mat)
    )
    if reduction == 'none':
        return entropy_per_entry
    elif reduction == 'sum':
        return np.sum(entropy_per_entry)
    elif reduction == 'mean':
        return np.mean(entropy_per_entry)
    else:
        raise ValueError(f"reduction must be 'sum', 'mean', or 'none', got: {reduction}")


def calc_entropy_dyads_dists(
        dyads_distributions: np.ndarray,
        reduction: str = 'sum',
        eps: float = 1e-10
) -> float | np.ndarray:
    clipped_dyads_dists = np.clip(dyads_distributions, a_min=eps, a_max=1 - eps)
    entropy_per_dyad = -(clipped_dyads_dists * np.log2(clipped_dyads_dists)).sum(axis=1)
    if reduction == 'none':
        return entropy_per_dyad
    elif reduction == 'sum':
        return np.sum(entropy_per_dyad)
    elif reduction == 'mean':
        return np.mean(entropy_per_dyad)
    else:
        raise ValueError(f"reduction must be 'sum', 'mean', or 'none', got: {reduction}")
