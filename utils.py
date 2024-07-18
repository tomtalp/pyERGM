import numpy as np
import networkx as nx

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
    def __init__(self, metric_names=[], custom_metrics={}):
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

        """
        self._SUPPORTED_METRICS = ["num_edges", "num_triangles"]

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
                func = lambda G: sum(nx.triangles(G))

            self.statistics_functions[metric_name] = func
    
    def calculate_statistics(self, W: np.ndarray, directed=False, verbose=False):
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
        G = connectivity_matrix_to_G(W, directed=directed)
        
        if G.is_directed():
            if "num_triangles" in self.metric_names:
                raise ValueError("The 'num_triangles' metric is not supported for directed graphs.")

        stats = {name: func(G) for name, func in self.statistics_functions.items()}
        if verbose:
            return stats
        else:
            return np.array(list(stats.values()))
       

                