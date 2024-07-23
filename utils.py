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
    i = np.random.randint(n)
    j = np.random.randint(n)

    while i == j:
        i = np.random.randint(n)
        j = np.random.randint(n)

    return (i, j)


class NetworkStatistics():
    def __init__(self, metric_names=[], custom_metrics={}, directed=False):
        """
        Initialize a NetworkStatistics object with a set of metrics to be calculated.
        
        Parameters
        ----------
        metric_names : list
            A list of names of metrics to calculate. 
            Currently supported metrics are - 
                [num_edges, num_triangles]
        
        custom_metrics : dict
            A custom metric is registered via a key, value - 
                key : str
                    The name of the metric.
                value : function
                    A function that takes a graph object and returns the value of the metric (i.e. the statistic value).
                
            For example - 
                {"sqrt_of_edges": lambda G: G.number_of_edges() ** 0.5}
        
        directed : bool
            Indicate whether the graph is directed or not.

        """
        self._SUPPORTED_METRICS = ["num_edges", "num_triangles"]

        self._is_directed = directed

        self.metric_names = metric_names
        self._validate_predefined_metrics()

        self.custom_metrics = custom_metrics
        self._validate_custom_metrics()

        self._register_metrics()

    def _validate_predefined_metrics(self):
        if len(self.metric_names) == 0:
            raise ValueError("At least one metric must be specified.")
        for metric_name in self.metric_names:
            if metric_name not in self._SUPPORTED_METRICS:
                raise ValueError(f"Metric {metric_name} is not supported.")

            if metric_name == "num_triangles" and self._is_directed:
                raise ValueError("The 'num_triangles' metric is not supported for directed graphs.")

    def _validate_custom_metrics(self):
        for metric_name, stat_func in self.custom_metrics:
            if not callable(stat_func):
                raise ValueError(f"Custom Metric {metric_name} is not a function.")

    def _register_metrics(self):
        self.statistics_functions = {}

        for metric_name in self.metric_names:
            if metric_name == "num_edges":
                func = lambda G: len(G.edges())
            elif metric_name == "num_triangles":
                func = lambda G: sum(
                    nx.triangles(G).values()) // 3  # nx.triangles counts each triangle 3 times, for every node in it

            self.statistics_functions[metric_name] = func

    def get_num_of_statistics(self):
        """
        Get the number of statistics that are registered.
        
        Returns
        -------
        n_stats : int
            The number of statistics.
        """
        return len(self.statistics_functions)

    def calculate_statistics(self, W: np.ndarray, verbose=False):
        """
        Calculate the statistics of a graph. 
        This is equivalent to calculating what is known in the ERGM literature as g(y) for a given graph y.
        
        Parameters
        ----------
        W : np.ndarray
            A connectivity matrix.
        
        Returns
        -------
        stats : np.ndarray or dict 
            An array of statistics, represented as a numpy array of values, or a dict if verbose=True
        """
        G = connectivity_matrix_to_G(W, directed=self._is_directed)

        stats = {name: func(G) for name, func in self.statistics_functions.items()}
        if verbose:
            return stats
        else:
            return np.array(list(stats.values()))


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
