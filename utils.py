import numpy as np
import networkx as nx


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

    return tuple(np.random.choice(n, size=2, replace=False))


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
