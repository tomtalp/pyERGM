import numpy as np
import networkx as nx
from numba import njit
from scipy.sparse.linalg import eigsh
from typing import Collection
import random


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

def benchmark_generator(W, metrics, is_directed, **kwargs):
    """
    Takes an example of a network and a set of metrics, and generates a benchmark dataset by performing
    a grid-search across many different optimization parameters, which can be passed as kwargs.

    Parameters
    ----
    W : np.ndarray
        The adjacency matrix of the network.
    metrics : Collection[Metric]
        A collection of metrics to be used in the ERGM.
    
    expected_thetas: dict
        A collection of parameter names and their expected values.
        e.g. {edges": 0.5, "sender_1": 0.1, "sender_2": -0.2}
    
    results_path: str
        The path to save the benchmark results to.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    import time, datetime
    import os
    import itertools

    from ergm import ERGM
    
    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path_name = kwargs.get("results_path", f"benchmarks/{dt}")
    os.makedirs(os.path.dirname(path_name), exist_ok=True)
    print(f"Saving benchmark results to {path_name}")

    n = W.shape[0]
    parameters = {
        "opt_steps": [30],
        "sample_size": [1000, 5000, 10000, 20000],
        "n_mcmc_steps": [n],
        "convergence_criterion": ["hotelling", "zero_grad_norm"],
        "hotelling_conf": [0.99],
        "lr": [0.001, 0.01, 0.1, 0.5, 1],
        "lr_decay_pct": [0.1, 0.01],
        "cov_matrix_estimation_method": ["naive", "batch", "multivariate_initial_sequence"],
        "estimated_p_seed": [np.sum(W) / (n * (n - 1))],
        "optimization_method": "newton_raphson",
        "steps_for_decay": 1,
        "sliding_grad_window_k": 5,
        "sample_pct_growth": 0.05,
    }

    expected_thetas = kwargs.get("expected_thetas", None)

    all_params = list(itertools.product(*parameters.values()))

    data = []
    for i, param_values in enumerate(all_params):
        params = {p: v for p, v in zip(parameters.keys(), param_values)}
        print(f"{i+1}/{len(all_params)} Working on parameters - ")
        print(params)

        start_time = time.time()
        model = ERGM(n, 
                    metrics,
                    is_directed=is_directed, 
                    sample_size=params["sample_size"], 
                    n_mcmc_steps=params["n_mcmc_steps"],
                    seed_MCMC_proba=params["estimated_p_seed"]
                )
    
        grads, hotelling_statistics = model.fit(W, 
                                lr=params["lr"], 
                                lr_decay_pct=params["lr_decay_pct"],
                                opt_steps=params["opt_steps"],
                                convergence_criterion=params["convergence_criterion"],
                                hotelling_confidence=params["hotelling_conf"],
                                cov_matrix_estimation_method=params["cov_matrix_estimation_method"],
                                optimization_method=params["optimization_method"],
                                steps_for_decay=params["steps_for_decay"],
                                sliding_grad_window_k=params["sliding_grad_window_k"],
                                sample_pct_growth=params["sample_pct_growth"],
                                )
    
        end_time = time.time()
        elapsed_time = end_time - start_time
        converged = model._converged
        print(f"Finished in {elapsed_time} seconds, converged = {converged}")

        output_data = params.copy()
        output_data["elapsed_time"] = elapsed_time
        output_data["converged"] = converged
        output_data["thetas"] = model._thetas

        params_path_name = f"{path_name}/params_{i+1}.txt"
        with open(params_path_name, "w") as f:
            f.write(str(params))

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(grads, ax=ax, legend=False)
        ax.set(xlabel='Steps', ylabel='Gradients')
        ax.set_title("Gradients over time (Colors represent metrics)")
        plt.savefig(f"gradients_{i}.png")

        
        hotelling_df = pd.DataFrame(hotelling_statistics)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(hotelling_df["hotelling_F"])
        sns.scatterplot(hotelling_df["dist"])
        sns.scatterplot(hotelling_df["inv_cov_norm"])
        sns.scatterplot(hotelling_df["critical_val"])
        ax.set_yscale("log")

        ax.legend(["F value (Hotelling)", "Mahalanobis dist.", "inv_cov_norm", "critical_val", ], loc="upper right")
        ax.set(xlabel='Iteration', ylabel='convergence metric (logscale)')
        ax.set_title("Convergence metrics over time")
        plt.savefig(f"hotelling_{i}.png")

        if expected_thetas:
            true_predictions = list(expected_thetas.values())
            fitted_thetas = model._thetas
            mse = np.mean((np.array(true_predictions) - np.array(fitted_thetas)) ** 2)            
            output_data["mse"] = mse

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=true_predictions, y=fitted_thetas)

            min_x = min(min(true_predictions), min(fitted_thetas))
            max_x = max(max(true_predictions), max(fitted_thetas))

            sns.lineplot(x=[min_x, max_x], y=[min_x, max_x], color="red", linestyle="--", alpha=0.5)
            ax.set(xlabel='R thetas', ylabel='pyERGM thetas')
            ax.set_title(f"Comparison of pyERGM & R thetas. MSE = {mse}")

            plt.savefig(f"thetas_comparison_{i}.png")



        break
    # Expected usage - 
    # W = ...
    # metrics = ...

    # from utils import benchmark_generator
    # benchmark = benchmark_generator(W, metrics)
    # benchmark.to_csv()

