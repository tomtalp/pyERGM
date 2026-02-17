import itertools
import math
import secrets
from collections import Counter
from typing import Sequence, Any
import numpy as np
from numpy import typing as npt
from numba import njit
import random

from pyERGM.constants import DataBootstrapSplittingMethod, CovMatrixEstimationMethod, DyadStateIdx, Reduction


def generate_short_id(length: int = 6) -> str:
    """Generate a short random hex ID for metric disambiguation."""
    return secrets.token_hex(math.ceil(length / 2))[:length]


@njit
def _numba_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def set_seed(seed):
    """
    Set random seed for reproducibility across all libraries.

    Sets seeds for numpy, Python's random module, and numba-jitted functions.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    _numba_seed(seed)


def generate_erdos_renyi_matrix(n_nodes: int, p: float, is_directed: bool) -> np.ndarray:
    """
    Generate an Erdős-Rényi random graph as an adjacency matrix.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    p : float
        Probability for edge creation (0 <= p <= 1).
    is_directed : bool
        If True, generate a directed graph. If False, generate an undirected graph.

    Returns
    -------
    np.ndarray
        Adjacency matrix of shape (n_nodes, n_nodes) with no self-loops.
    """
    matrix = generate_binomial_tensor(n_nodes, 1, p)[:, :, 0].astype(np.float64)
    np.fill_diagonal(matrix, 0)

    if not is_directed:
        # Symmetrize: use upper triangle and mirror to lower
        matrix = np.triu(matrix, k=1)
        matrix = matrix + matrix.T

    return matrix


def construct_adj_mat_from_int(int_code: int, num_nodes: int, is_directed: bool) -> np.ndarray:
    """
    Convert an integer to its corresponding adjacency matrix.

    Used for exhaustive enumeration in BruteForceERGM. Each integer represents
    a unique network configuration via binary encoding.

    Parameters
    ----------
    int_code : int
        Integer encoding of the network (0 to 2^(n*(n-1)) for directed networks).
    num_nodes : int
        Number of nodes in the network.
    is_directed : bool
        Whether the network is directed.

    Returns
    -------
    np.ndarray
        Adjacency matrix of shape (num_nodes, num_nodes).
    """
    num_pos_connects = num_nodes * (num_nodes - 1)
    if not is_directed:
        num_pos_connects //= 2
    adj_mat_str = f'{int_code:0{num_pos_connects}b}'
    mat_entries_arr = np.array(list(adj_mat_str), 'uint8')
    return reshape_flattened_off_diagonal_elements_to_square(mat_entries_arr, is_directed)


def construct_int_from_adj_mat(adj_mat: np.ndarray, is_directed: bool) -> int:
    """
    Convert an adjacency matrix to its integer encoding.

    Inverse operation of construct_adj_mat_from_int.

    Parameters
    ----------
    adj_mat : np.ndarray
        Adjacency matrix of shape (n, n).
    is_directed : bool
        Whether the network is directed.

    Returns
    -------
    int
        Integer encoding of the network.

    Raises
    ------
    ValueError
        If adjacency matrix dimensions are invalid.
    """
    if len(adj_mat.shape) != 2 or adj_mat.shape[0] != adj_mat.shape[1]:
        raise ValueError(f"The dimensions of the given adjacency matrix {adj_mat.shape} are not valid for an "
                         f"adjacency matrix (should be a 2D squared matrix)")
    adj_mat_no_diag = flatten_square_matrix_to_edge_list(adj_mat, is_directed)
    return round((adj_mat_no_diag * 2 ** np.arange(adj_mat_no_diag.size - 1, -1, -1).astype(np.ulonglong)).sum())


def network_to_hashable(network: np.ndarray, is_directed: bool) -> tuple:
    """
    Convert a network to a hashable representation that handles NaN values.

    Uses only non-NaN off-diagonal elements to create a tuple representation.
    This avoids integer overflow issues and properly handles masked networks.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix of shape (n, n)
    is_directed : bool
        Whether the network is directed

    Returns
    -------
    tuple
        Hashable tuple of boolean values representing non-NaN off-diagonal edges.
        For undirected networks, only uses upper triangle. Uses booleans for memory efficiency.
    """
    # Extract off-diagonal elements
    flattened = flatten_square_matrix_to_edge_list(network, is_directed)

    # Filter out NaN values and convert to tuple of booleans
    # NaN values are excluded from the hash, so two networks with different
    # NaN patterns but same non-NaN edges will hash identically
    non_nan_mask = ~np.isnan(flattened)
    non_nan_edges = flattened[non_nan_mask]

    # Convert to tuple of booleans for hashability and memory efficiency
    return tuple(non_nan_edges.astype(bool))


def find_unique_networks(
        networks: np.ndarray,
        is_directed: bool,
        seen_hashes: set | None = None
) -> tuple[np.ndarray, set]:
    """
    Filter networks to keep only unique ones.

    Parameters
    ----------
    networks : np.ndarray
        Array of shape (n, n, num_samples)
    is_directed : bool
        Whether networks are directed
    seen_hashes : set, optional
        Set of network hashes already seen. If provided, only
        networks not in this set are returned.

    Returns
    -------
    unique_networks : np.ndarray
        Array of shape (n, n, num_unique) containing only unique networks
    network_hashes : set
        Set of hashes for all unique networks (including seen_hashes)
    """
    if seen_hashes is None:
        seen_hashes = set()
    else:
        seen_hashes = seen_hashes.copy()  # Don't modify input

    num_samples = networks.shape[2]
    unique_indices = []

    for i in range(num_samples):
        net_hash = network_to_hashable(networks[:, :, i], is_directed)
        if net_hash not in seen_hashes:
            seen_hashes.add(net_hash)
            unique_indices.append(i)

    if len(unique_indices) == 0:
        # Return empty array with correct shape
        return np.zeros((networks.shape[0], networks.shape[1], 0)), seen_hashes

    unique_networks = networks[:, :, unique_indices]
    return unique_networks, seen_hashes


def get_greatest_convex_minorant(xs: np.ndarray, ys: np.ndarray):
    if xs.size != ys.size:
        raise ValueError("Arrays must have the same size!")
    # For numerical considerations we don't force the sequence of slopes to really not decrease
    slope_diff_thr = -10 ** -10
    x_diffs = np.diff(xs)
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
        cur_wrapping_lines_intercepts = (cur_proposed_minorant[cur_wrapping_indices[:-1]] -
                                         cur_wrapping_lines_slopes * xs[cur_wrapping_indices[:-1]])

        # Update the values in problematic indices to lie on the corresponding wrapping line
        cur_assignment_problematic_idx_to_line = np.searchsorted(cur_wrapping_indices, cur_problematic_indices) - 1
        cur_proposed_minorant[cur_problematic_indices] = (
                cur_wrapping_lines_slopes[cur_assignment_problematic_idx_to_line] * xs[cur_problematic_indices] +
                cur_wrapping_lines_intercepts[cur_assignment_problematic_idx_to_line])

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


def convert_flat_no_diag_idx_to_i_j(flat_no_diag_idx: Sequence[int], full_mat_size: int) -> npt.NDArray[int]:
    """
    Converts the index in the flattened square matrix without the main diagonal to the pair of indices in the original
    matrix.
    For a square matrix A, the flattened-no-diagonal form is given by A[~np.eye(A.shape[0], dtype=bool)].
    E.g., given a 3X3 matrix, the third element in its flattened-no-diagonal form (idx 2) is the (1,0) entry.
    Parameters
    ----------
    flat_no_diag_idx
        The index in the flattened-no-diagonal form
    full_mat_size
        The number of rows/columns in the original squared matrix.
    Returns
    -------
    Array (2 X num_indices) of the entries in the original square matrix.
    """
    flat_no_diag_idx = np.array(flat_no_diag_idx)
    if np.any(flat_no_diag_idx >= full_mat_size * (full_mat_size - 1)):
        raise IndexError(
            f"Got a too large `flat_no_diag_idx` {flat_no_diag_idx} for original matrix size of {full_mat_size}")
    rows = np.array(flat_no_diag_idx) // (full_mat_size - 1)
    cols = np.array(flat_no_diag_idx) % (full_mat_size - 1)
    cols[cols >= rows] += 1
    return np.stack((rows, cols), axis=0).astype(np.int32)


def get_custom_distribution_random_edges_to_flip(num_pairs, edge_probs, is_directed: bool):
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
    is_directed
        Whether the network is directed.
    Returns
    -------
    A sample of num_pairs pairs of indices, sampled according to edge_probs.
    """
    num_possible_edges = edge_probs.size
    num_nodes = num_edges_to_num_nodes(num_possible_edges, is_directed=is_directed)
    flat_no_diag_indices = np.random.choice(edge_probs.size, p=edge_probs, size=num_pairs)
    return convert_flat_no_diag_idx_to_i_j(flat_no_diag_indices, num_nodes)


def approximate_auto_correlation_function(features_of_net_samples: np.ndarray) -> np.ndarray:
    """
    This is gamma hat from Geyer's handbook of mcmc (1D) and Dai and Jones 2017 (multi-D).
    """
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
def covariance_matrix_estimation(
        features_of_net_samples: np.ndarray,
        mean_features_of_net_samples: np.ndarray,
        method: CovMatrixEstimationMethod = CovMatrixEstimationMethod.NAIVE,
        num_batches: int = 25,
) -> np.ndarray:
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
            CovMatrixEstimationMethod.NAIVE
                A naive estimation from the sample: E[gi*gj] - E[gi]E[gj]
            CovMatrixEstimationMethod.BATCH
                based on difference of means of sample batches from the total mean, as in Geyer's handbook of
                MCMC (there it is stated for the univariate case, but the generalization is straight forward).
            CovMatrixEstimationMethod.MULTIVARIATE_INITIAL_SEQUENCE
                Following Dai and Jones 2017 - the first estimator in section 3.1 (denoted mIS).
    num_batches:
        The batch size for CovMatrixEstimationMethod.BATCH (default is 25).

    Returns
    -------
    The covariance matrix estimation (num_features X num_features).
    """
    num_features = features_of_net_samples.shape[0]
    sample_size = features_of_net_samples.shape[1]
    match method:
        case CovMatrixEstimationMethod.NAIVE.value:
            # An outer product of the means (E[gi]E[gj])
            cross_prod_mean_features = (mean_features_of_net_samples.reshape(num_features, 1) @
                                        mean_features_of_net_samples.T.reshape(1, num_features))
            # A mean of the outer products of the sample (E[gi*gj])
            features_cross_prods = np.zeros((sample_size, num_features, num_features))
            for i in range(sample_size):
                features_cross_prods[i] = np.outer(features_of_net_samples[:, i], features_of_net_samples[:, i])
            mean_features_cross_prod = features_cross_prods.sum(axis=0) / sample_size

            return mean_features_cross_prod - cross_prod_mean_features

        case CovMatrixEstimationMethod.BATCH.value:
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

        case CovMatrixEstimationMethod.MULTIVARIATE_INITIAL_SEQUENCE.value:
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
                    raise RuntimeError("Got a sample with no valid multivariate_initial_sequence covariance matrix "
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

        case _:
            raise ValueError(f"{method} is an unsupported method for covariance matrix estimation")


@njit
def calc_nll_gradient(observed_features, mean_features_of_net_samples):
    return mean_features_of_net_samples - observed_features


def get_sorted_type_pairs(types):
    sorted_types = sorted(list(set(types)))
    return list(itertools.product(sorted_types, sorted_types))


def get_edge_density_per_type_pairs(W: np.ndarray, types: Sequence[Any]):
    """
    Calculate the density of edges between each pair of types in the network.

    Parameters
    ----------
    W : np.ndarray
        The adjacency matrix of the network
    
    types : Sequence[Any]
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

def generate_binomial_tensor(net_size, num_samples, p=0.5):
    """
    Generate a tensor of size (net_size, net_size, num_samples) where each element is a binomial random variable
    """
    return np.random.binomial(1, p, (net_size, net_size, num_samples)).astype(np.int8)


def _sample_edges_with_replacement(
        probability_matrix: np.ndarray,
        sample_size: int,
        is_directed: bool
) -> np.ndarray:
    """
    Sample networks using independent edge probabilities.

    This is the core logic for sampling with replacement from edge probabilities.
    Extracted as a separate function for reuse in adaptive batch sampling.

    Parameters
    ----------
    probability_matrix : np.ndarray
        Matrix where entry (i,j) is the probability of edge i→j.
        NaN values indicate masked (ignored) edges.
    sample_size : int
        Number of networks to sample.
    is_directed : bool
        Whether the network is directed.

    Returns
    -------
    np.ndarray
        Sampled networks of shape (n, n, sample_size).
    """
    n_nodes = probability_matrix.shape[0]
    sample = np.full((n_nodes, n_nodes, sample_size), np.nan)
    sample[np.diag_indices(n_nodes)] = 0

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j or np.isnan(probability_matrix[i, j]):
                continue
            elif not is_directed and i > j:
                sample[i, j, :] = sample[j, i, :]
                continue
            else:
                sample[i, j, :] = np.random.binomial(1, probability_matrix[i, j], size=sample_size)

    return sample


def _sample_dyads_with_replacement(
        dyads_distributions: np.ndarray,
        sample_size: int
) -> np.ndarray:
    """
    Sample networks using dyadic state distributions.

    This is the core logic for sampling with replacement from dyadic state distributions.
    Extracted as a separate function for reuse in adaptive batch sampling.

    Parameters
    ----------
    dyads_distributions : np.ndarray
        Probability distributions over dyadic states, shape (n_choose_2, 4).
    sample_size : int
        Number of networks to sample.

    Returns
    -------
    np.ndarray
        Sampled networks of shape (n, n, sample_size).
    """
    num_dyads = dyads_distributions.shape[0]
    n_nodes = num_dyads_to_num_nodes(num_dyads)
    dyads_states_indices_sample = np.zeros((num_dyads, sample_size))

    for i in range(num_dyads):
        dyads_states_indices_sample[i] = np.random.choice(
            np.arange(4), p=dyads_distributions[i], size=sample_size
        )

    net_sample = np.zeros((n_nodes, n_nodes, sample_size))
    for k in range(sample_size):
        net_sample[:, :, k] = convert_dyads_state_indices_to_connectivity(dyads_states_indices_sample[:, k])
    return net_sample


def _sample_unique_networks_adaptive(
        probability_matrix_or_dyad_dist: np.ndarray,
        sample_size: int,
        is_directed: bool,
        is_dyadic: bool,
        initial_batch_multiplier: float = 2.0,
        max_batch_multiplier: float = 10.0,
        min_batch_size: int | None = None,
        max_attempts: int = 50,
        high_dup_threshold: float = 0.95,
        batch_increase_factor: float = 1.5,
        batch_decrease_factor: float = 0.8,
        medium_dup_threshold: float = 0.5,
        low_dup_threshold: float = 0.1
) -> np.ndarray:
    """
    Sample unique networks using adaptive batch sizing.

    Parameters
    ----------
    probability_matrix_or_dyad_dist : np.ndarray
        Either edge probability matrix or dyadic state distributions
    sample_size : int
        Number of unique networks desired
    is_directed : bool
        Whether network is directed
    is_dyadic : bool
        If True, use sample_from_dyads_distribution logic
        If False, use independent edge sampling logic
    initial_batch_multiplier : float, optional
        Initial multiplier for batch size (default: 2.0, sample 2x needed)
    max_batch_multiplier : float, optional
        Maximum batch multiplier to avoid memory issues (default: 10.0)
    min_batch_size : int, optional
        Minimum batch size. Default: max(10, sample_size)
    max_attempts : int, optional
        Maximum number of batches to try (default: 50)
    high_dup_threshold : float, optional
        Stop if duplication rate exceeds this (default: 0.95)
    batch_increase_factor : float, optional
        Factor to increase batch_multiplier when duplicates are high (default: 1.5)
    batch_decrease_factor : float, optional
        Factor to decrease batch_multiplier when duplicates are low (default: 0.8)
    medium_dup_threshold : float, optional
        Threshold above which to increase batch size (default: 0.5)
    low_dup_threshold : float, optional
        Threshold below which to decrease batch size (default: 0.1)

    Returns
    -------
    np.ndarray
        Array of shape (n, n, sample_size) with unique networks

    Raises
    ------
    RuntimeError
        If unable to generate enough unique samples
    """
    n_nodes = (
        probability_matrix_or_dyad_dist.shape[0]
        if not is_dyadic
        else num_dyads_to_num_nodes(probability_matrix_or_dyad_dist.shape[0])
    )

    if min_batch_size is None:
        min_batch_size = max(10, sample_size)

    # Initialize collection
    unique_networks = np.zeros((n_nodes, n_nodes, 0))
    seen_hashes = set()

    batch_multiplier = initial_batch_multiplier
    attempts = 0

    while unique_networks.shape[2] < sample_size and attempts < max_attempts:
        attempts += 1
        remaining = sample_size - unique_networks.shape[2]

        # Calculate batch size
        batch_size = max(min_batch_size, int(remaining * batch_multiplier))

        # Sample a batch (using existing logic)
        if is_dyadic:
            batch = _sample_dyads_with_replacement(probability_matrix_or_dyad_dist, batch_size)
        else:
            batch = _sample_edges_with_replacement(
                probability_matrix_or_dyad_dist, batch_size, is_directed
            )

        # Find unique networks in this batch
        unique_in_batch, seen_hashes = find_unique_networks(batch, is_directed, seen_hashes)

        # Calculate duplication rate for this batch
        num_unique_in_batch = unique_in_batch.shape[2]
        duplication_rate = 1.0 - (num_unique_in_batch / batch_size)

        # Append unique networks
        if num_unique_in_batch > 0:
            # Only keep what we need
            to_keep = min(num_unique_in_batch, remaining)
            unique_networks = np.concatenate(
                [unique_networks, unique_in_batch[:, :, :to_keep]], axis=2
            )

        # Adaptive adjustment of batch size
        if duplication_rate > high_dup_threshold:
            # Very high duplication - model has low entropy
            raise RuntimeError(
                f"Unable to generate {sample_size} unique networks: "
                f"duplication rate is {duplication_rate:.1%} after {attempts} attempts. "
                f"The model entropy may be too low (e.g., probabilities near 0 or 1). "
                f"Consider using replace=True or adjusting model parameters."
            )
        elif duplication_rate > medium_dup_threshold:
            # Moderate-high duplication - increase batch size
            batch_multiplier = min(
                max_batch_multiplier, batch_multiplier * batch_increase_factor
            )
        elif duplication_rate < low_dup_threshold:
            # Low duplication - can decrease batch size to save memory
            batch_multiplier = max(1.0, batch_multiplier * batch_decrease_factor)

    if unique_networks.shape[2] < sample_size:
        raise RuntimeError(
            f"Unable to generate {sample_size} unique networks after {max_attempts} attempts. "
            f"Only obtained {unique_networks.shape[2]} unique networks. "
            f"This may indicate the model has very low entropy."
        )

    return unique_networks


def sample_from_independent_probabilities_matrix(
        probability_matrix: npt.NDArray[np.floating],
        sample_size: int,
        is_directed: bool,
        replace: bool = True
) -> npt.NDArray[np.floating]:
    """
    Sample connectivity matrices from a matrix representing the independent probability of an edge between nodes (i, j).

    Parameters
    ----------
    probability_matrix : np.ndarray
        Matrix where entry (i,j) is the probability of edge i→j.
        NaN values indicate masked (ignored) edges.
    sample_size : int
        Number of networks to sample.
    is_directed : bool
        Whether the network is directed.
    replace : bool, optional
        If True (default), sample with replacement (networks may repeat).
        If False, sample without replacement (all networks unique).
        Uses adaptive batch sampling with uniqueness checking.

    Returns
    -------
    np.ndarray
        Sampled networks of shape (n, n, sample_size).

    Raises
    ------
    RuntimeError
        If replace=False and unable to generate enough unique networks
        (typically due to low model entropy).

    Notes
    -----
    When replace=False, uses adaptive batch sampling:
    - Samples in batches, keeps only unique networks
    - Adjusts batch size based on observed duplication rate
    - Stops if duplication rate becomes too high (>95%)

    If the probability matrix contains np.nan values (that designate masked entries),
    the output sample will have np.nans in corresponding coordinates.
    (i.e., if np.isnan(probability_matrix[i, j]) then np.all(np.isnan(returned_sample[i, j, ...))).
    """
    if replace:
        return _sample_edges_with_replacement(probability_matrix, sample_size, is_directed)

    # Sampling without replacement
    return _sample_unique_networks_adaptive(
        probability_matrix, sample_size, is_directed, is_dyadic=False
    )


def split_network_for_bootstrapping(
        net_size: int,
        first_part_size: int,
        splitting_method: DataBootstrapSplittingMethod = DataBootstrapSplittingMethod.UNIFORM
) -> tuple[npt.NDArray[int], npt.NDArray[int]]:
    if splitting_method == DataBootstrapSplittingMethod.UNIFORM:
        indices = np.arange(net_size)
        np.random.shuffle(indices)
        first_part_indices = indices[:first_part_size].reshape((first_part_size, 1))
    else:
        raise ValueError(f"Received an invalid splitting method: {splitting_method}. "
                         f"Supporting elements of DataBootstrapSplittingMethod")

    second_part_indices = np.setdiff1d(np.arange(net_size).astype(int), first_part_indices).reshape(
        (net_size - first_part_size, 1))

    return first_part_indices, second_part_indices


def num_dyads_to_num_nodes(num_dyads: int, int_tolerance: float = 1e-10) -> int:
    """
    x = num_dyads
    n(n-1) = 2*x
    n^2-n-2x=0 --> n = \\frac{1+\\sqrt{1-4\\cdot(-2x)}}{2}
    """
    num_nodes_candidate = (1 + np.sqrt(1 + 8 * num_dyads)) / 2
    rounded_num_nodes_candidate = np.round(num_nodes_candidate).astype(int)
    if np.abs(rounded_num_nodes_candidate - num_nodes_candidate) > int_tolerance:
        raise ValueError(f"Received invalid number of dyads: {num_dyads} (does not equal n choose 2 for an integer n)")
    return rounded_num_nodes_candidate


def num_edges_to_num_nodes(num_edges: int, is_directed: bool) -> int:
    """
    x = num_edges
    x = n(n-1) OR n(n-1) / 2, depending on directionality
    this function extracts n
    """
    return num_dyads_to_num_nodes(num_edges // 2 if is_directed else num_edges)


def convert_connectivity_to_dyad_states(connectivity: np.ndarray):
    """
    Convert directed network to one-hot encoded dyadic states.

    Each dyad (pair of nodes) can be in one of 4 states:
    - EMPTY (0): No edges
    - UPPER (1): Edge i -> j only
    - LOWER (2): Edge j -> i only
    - RECIPROCAL (3): Both edges exist

    Parameters
    ----------
    connectivity : np.ndarray
        Directed adjacency matrix of shape (n, n).

    Returns
    -------
    np.ndarray
        One-hot encoded dyadic states of shape (n_choose_2, 4).
    """
    n_nodes = connectivity.shape[0]
    dyads_states = np.zeros(((n_nodes ** 2 - n_nodes) // 2, 4))
    idx = 0
    for i in range(n_nodes - 1):
        for j in range(i + 1, n_nodes):
            if not connectivity[i, j] and not connectivity[j, i]:
                dyads_states[idx, DyadStateIdx.EMPTY_IDX.value] = 1
            elif connectivity[i, j] and not connectivity[j, i]:
                dyads_states[idx, DyadStateIdx.UPPER_IDX.value] = 1
            elif not connectivity[i, j] and connectivity[j, i]:
                dyads_states[idx, DyadStateIdx.LOWER_IDX.value] = 1
            else:
                dyads_states[idx, DyadStateIdx.RECIPROCAL_IDX.value] = 1

            idx += 1
    return dyads_states


def convert_dyads_states_to_connectivity(dyads_states):
    """
    Convert one-hot encoded dyadic states to directed adjacency matrix.

    Inverse operation of convert_connectivity_to_dyad_states.

    Parameters
    ----------
    dyads_states : np.ndarray
        One-hot encoded dyadic states of shape (n_choose_2, 4).

    Returns
    -------
    np.ndarray
        Directed adjacency matrix of shape (n, n).
    """
    num_dyads = dyads_states.shape[0]
    num_nodes = num_dyads_to_num_nodes(num_dyads)
    indices = np.triu_indices(num_nodes, k=1)
    network = np.zeros((num_nodes, num_nodes))
    for i in range(num_dyads):
        is_upper = dyads_states[i, DyadStateIdx.UPPER_IDX.value] or dyads_states[i, DyadStateIdx.RECIPROCAL_IDX.value]
        is_lower = dyads_states[i, DyadStateIdx.LOWER_IDX.value] or dyads_states[i, DyadStateIdx.RECIPROCAL_IDX.value]
        if is_upper:
            network[indices[0][i], indices[1][i]] = 1
        if is_lower:
            network[indices[1][i], indices[0][i]] = 1
    return network


def convert_dyads_state_indices_to_connectivity(dyads_states_indices):
    """
    Convert dyadic state indices to directed adjacency matrix.

    Similar to convert_dyads_states_to_connectivity but takes indices (0-3)
    instead of one-hot encoded states.

    Parameters
    ----------
    dyads_states_indices : np.ndarray
        Dyadic state indices of shape (n_choose_2,), where each value is 0-3
        representing EMPTY, UPPER, LOWER, or RECIPROCAL.

    Returns
    -------
    np.ndarray
        Directed adjacency matrix of shape (n, n).
    """
    num_dyads = dyads_states_indices.shape[0]
    num_nodes = num_dyads_to_num_nodes(num_dyads)
    indices = np.triu_indices(num_nodes, k=1)
    network = np.zeros((num_nodes, num_nodes))
    for i in range(num_dyads):
        is_upper = dyads_states_indices[i] in [DyadStateIdx.UPPER_IDX.value, DyadStateIdx.RECIPROCAL_IDX.value]
        is_lower = dyads_states_indices[i] in [DyadStateIdx.LOWER_IDX.value, DyadStateIdx.RECIPROCAL_IDX.value]
        if is_upper:
            network[indices[0][i], indices[1][i]] = 1
        if is_lower:
            network[indices[1][i], indices[0][i]] = 1
    return network


def sample_from_dyads_distribution(dyads_distributions, sample_size, replace: bool = True):
    """
    Sample networks from dyadic state probability distributions.

    Used for exact sampling from MPLE_RECIPROCITY models.

    Parameters
    ----------
    dyads_distributions : np.ndarray
        Probability distributions over dyadic states, shape (n_choose_2, 4).
    sample_size : int
        Number of networks to sample.
    replace : bool, optional
        If True (default), sample with replacement (networks may repeat).
        If False, sample without replacement (all networks unique).
        Uses adaptive batch sampling with uniqueness checking.

    Returns
    -------
    np.ndarray
        Sampled networks of shape (n, n, sample_size).

    Raises
    ------
    RuntimeError
        If replace=False and unable to generate enough unique networks
        (typically due to low model entropy).
    """
    if replace:
        return _sample_dyads_with_replacement(dyads_distributions, sample_size)

    return _sample_unique_networks_adaptive(
        dyads_distributions, sample_size, is_directed=True, is_dyadic=True
    )


def get_exact_marginals_from_dyads_distribution(dyads_distributions):
    num_dyads = dyads_distributions.shape[0]
    num_nodes = num_dyads_to_num_nodes(num_dyads)
    indices = np.triu_indices(num_nodes, k=1)
    exact_marginals = np.zeros((num_nodes, num_nodes))
    for i in range(num_dyads):
        exact_marginals[indices[0][i], indices[1][i]] = (
                dyads_distributions[i, DyadStateIdx.RECIPROCAL_IDX.value] +
                dyads_distributions[i, DyadStateIdx.UPPER_IDX.value]
        )
        exact_marginals[indices[1][i], indices[0][i]] = (
                dyads_distributions[i, DyadStateIdx.RECIPROCAL_IDX.value] +
                dyads_distributions[i, DyadStateIdx.LOWER_IDX.value]
        )
    return exact_marginals


def flatten_square_matrix_to_edge_list(square_mat: np.ndarray, is_directed: bool) -> np.ndarray:
    if square_mat.ndim != 2 or square_mat.shape[0] != square_mat.shape[1]:
        raise ValueError("The input must be a square matrix")
    if is_directed:
        return square_mat[~np.eye(square_mat.shape[0], dtype=bool)].flatten()
    else:
        if np.any(square_mat != square_mat.T):
            raise ValueError("Got an asymmetric matrix as an undirected network to flatten")
        return square_mat[np.triu_indices_from(square_mat, k=1)]


def _set_off_diagonal_elements_from_array(square_mat, values_to_set):
    values_to_set = values_to_set.flatten()
    if values_to_set.size == square_mat.size - square_mat.shape[0]:
        square_mat[~np.eye(square_mat.shape[0], dtype=bool)] = values_to_set
    elif values_to_set.size == (square_mat.size - square_mat.shape[0]) // 2:
        square_mat[np.triu_indices(square_mat.shape[0], k=1)] = values_to_set
        square_mat[np.tril_indices(square_mat.shape[0], k=-1)] = square_mat.T[
            np.tril_indices(square_mat.shape[0], k=-1)
        ]
    else:
        raise ValueError(
            "The size of the array must be compatible the size of the square matrix, "
            "i.e. either (n ** 2 - n) for directed networks or ((n ** 2 - n) / 2) for undirected networks. "
            f"got: {values_to_set.size}")


def reshape_flattened_off_diagonal_elements_to_square(
        flattened_array: np.ndarray,
        is_directed: bool,
        flat_mask: npt.NDArray[bool] | None = None
) -> np.ndarray:
    if flat_mask is not None:
        if flat_mask.sum() != flattened_array.size:
            raise ValueError(
                f"Received incompatible flattened_array and mask. flat_mask.sum(): "
                f"{flat_mask.sum()}, flattened_array.size: {flattened_array.size}, but should be equal."
            )
        full_array = np.full(flat_mask.size, fill_value=np.nan)
        full_array[flat_mask] = flattened_array
        num_nodes = num_edges_to_num_nodes(flat_mask.size, is_directed)
    else:
        full_array = flattened_array
        num_nodes = num_edges_to_num_nodes(flattened_array.size, is_directed)
    reshaped = np.zeros((num_nodes, num_nodes), dtype=flattened_array.dtype)
    _set_off_diagonal_elements_from_array(reshaped, full_array)
    return reshaped


def expand_net_dims(net: np.ndarray) -> np.ndarray:
    """
    Ensure network array has 3 dimensions (n, n, num_networks).

    Adds a singleton dimension if the input is 2D.

    Parameters
    ----------
    net : np.ndarray
        Network array of shape (n, n) or (n, n, num_networks).

    Returns
    -------
    np.ndarray
        Network array of shape (n, n, num_networks).

    Raises
    ------
    ValueError
        If array is not 2D or 3D.
    """
    if net.ndim == 2:
        # a single network
        return net[..., np.newaxis]
    elif net.ndim != 3:
        raise ValueError("Cannot expand dims to an array that is not 2 or 3 dimensional")
    return net


def reduce_individual_elements(flat_array: np.ndarray, reduction: Reduction) -> np.ndarray | float:
    match reduction:
        case Reduction.NONE:
            return flat_array
        case Reduction.SUM:
            return np.sum(flat_array)
        case Reduction.MEAN:
            return np.mean(flat_array)
        case _:
            raise ValueError(f"reduction must be Reduction.SUM, Reduction.MEAN, or Reduction.NONE, got: {reduction}")


def calc_entropy_independent_probability_matrix(
        prob_mat: np.ndarray,
        is_directed: bool,
        reduction: Reduction = Reduction.SUM,
        eps: float = 1e-10
) -> float | np.ndarray:
    flat_no_diag_probs = flatten_square_matrix_to_edge_list(prob_mat, is_directed)
    flat_no_diag_probs = flat_no_diag_probs[~np.isnan(flat_no_diag_probs)]
    flat_clipped_no_diag_probs = np.clip(flat_no_diag_probs, eps, 1 - eps)

    entropy_per_entry = -(
            flat_clipped_no_diag_probs * np.log2(flat_clipped_no_diag_probs) +
            (1 - flat_clipped_no_diag_probs) * np.log2(1 - flat_clipped_no_diag_probs)
    )
    return reduce_individual_elements(entropy_per_entry, reduction)


def calc_entropy_dyads_dists(
        dyads_distributions: np.ndarray,
        reduction: Reduction = Reduction.SUM,
        eps: float = 1e-10
) -> float | np.ndarray:
    clipped_dyads_dists = np.clip(dyads_distributions, a_min=eps, a_max=1 - eps)
    entropy_per_dyad = -(clipped_dyads_dists * np.log2(clipped_dyads_dists)).sum(axis=1)
    return reduce_individual_elements(entropy_per_dyad, reduction)
