import numpy as np
import networkx as nx

def connectivity_matrix_to_G(W: np.ndarray):
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

    ## TODO - 1. How do handle directed graphs?
    """
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
    def __init__(self, statistic_names=[], custom_statistics={}):
        """
        Initialize a NetworkStatistics object with a set of statistics.
        
        Parameters
        ----------
        statistic_names : list
            A list of names of statistics to calculate. 
            Currently supported statistics are - 
                [num_edges, num_triangles]
        
        custom_statistics : dict
            A custom statistic is registered via a key, value - 
                key : str
                    The name of the statistic.
                value : function
                    A function that takes a graph object and returns the value of the statistic
                
            For example - 
                {"sqrt_of_edges": lambda G: G.number_of_edges() ** 0.5}

        """
        self._SUPPORTED_STATS = ["num_edges", "num_triangles"]

        self.statistic_names = statistic_names
        self._validate_predefined_statistics()

        self.custom_statistics = custom_statistics
        self._validate_custom_statistics()      

        self._register_statistics()        
    
    def _validate_predefined_statistics(self):
        for stat_name in self.statistic_names:
            if stat_name not in self._SUPPORTED_STATS:
                raise ValueError(f"Statistic {stat_name} is not supported.")

    def _validate_custom_statistics(self):
        for stat_name, stat_func in self.custom_statistics:
            if not callable(stat_func):
                raise ValueError(f"Custom statistic {stat_name} is not a function.")
    
    def _register_statistics(self):
        self.statistics_functions = []
        for stat_name in self.statistic_names:
            if stat_name == "num_edges":
                self.statistics_functions.append(lambda G: len(G.edges()))
            elif stat_name == "num_triangles":
                self.statistics_functions.append(lambda G: sum(nx.triangles(G)))
    
    def calculate_statistics(self, W: np.ndarray):
        """
        Calculate the statistics of a graph. 
        This is equivalent to calculating what is known in the ERGM literature as g(y) for a given graph y.
        
        Parameters
        ----------
        W : np.ndarray
            A connectivity matrix.
        
        Returns
        -------
        stats : np.ndarray
            An array of statistics.
        """
        G = connectivity_matrix_to_G(W)
        
        stats = [f(G) for f in self.statistics_functions]
        
        return np.array(stats)

                