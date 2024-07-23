from abc import ABC, abstractmethod

import numpy as np
import networkx as nx

from utils import *

class Metric(ABC):
    def __init__(self, metric_name, requires_graph=False):
        self.metric_name = metric_name
        self.requires_graph = requires_graph

    @abstractmethod
    def calculate(self, input):
        pass
    
class NumberOfEdges(Metric):
    def __init__(self):
        super().__init__(metric_name="num_edges", requires_graph=False)
    
    def calculate(self, W, is_directed):
        if is_directed:
            return np.sum(W)
        else:
            return np.sum(W) // 2

class NumberOfTriangles(Metric):
    def __init__(self):
        super().__init__(metric_name="num_triangles", requires_graph=True)
    
    def calculate(self, G, is_directed=False):
        if is_directed or isinstance(G, nx.DiGraph):
            raise ValueError("NumOfTriangles not implemented for directed graphs")
        
        return sum(nx.triangles(G).values()) // 3 # nx.triangles counts each triangle 3 times, for every one of its nodes

class MetricsCollection():
    def __init__(self, metrics, is_directed):
        self.metrics = tuple(metrics)
        self.requires_graph = any([x.requires_graph for x in self.metrics])
        self.is_directed = is_directed

    def get_num_of_statistics(self):
        """
        Get the number of statistics that are registered.
        
        Returns
        -------
        n_stats : int
            The number of statistics.
        """
        return len(self.metrics)    

    def calculate_statistics(self, W: np.ndarray):
        """
        Calculate the statistics of a graph, formally written as g(y).

        Parameters
        ----------
        W : np.ndarray
            A connectivity matrix.
        
        Returns
        -------
        stats : np.ndarray or dict 
            An array of statistics
        """
        if self.requires_graph:
            G = connectivity_matrix_to_G(W, directed=self.is_directed)
        
        statistics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            if metric.requires_graph:
                input = G
            else:
                input = W

            statistics[i] = metric.calculate(input, self.is_directed)

        return statistics
       

                

    

# class NetworkStatistics():
#     def __init__(self, metric_names=[], custom_metrics={}, directed=False):
#         """
#         Initialize a NetworkStatistics object with a set of metrics to be calculated.
        
#         Parameters
#         ----------
#         metric_names : list
#             A list of names of metrics to calculate. 
#             Currently supported metrics are - 
#                 [num_edges, num_triangles]
        
#         custom_metrics : dict
#             A custom metric is registered via a key, value - 
#                 key : str
#                     The name of the metric.
#                 value : function
#                     A function that takes a graph object and returns the value of the metric (i.e. the statistic value).
                
#             For example - 
#                 {"sqrt_of_edges": lambda G: G.number_of_edges() ** 0.5}
        
#         directed : bool
#             Indicate whether the graph is directed or not.

#         """
#         self._SUPPORTED_METRICS = ["num_edges", "num_triangles"]

#         self._is_directed = directed

#         self.metric_names = metric_names
#         self._validate_predefined_metrics()

#         self.custom_metrics = custom_metrics
#         self._validate_custom_metrics()      

#         self._register_metrics()        
    
#     def _validate_predefined_metrics(self):
#         if len(self.metric_names) == 0:
#             raise ValueError("At least one metric must be specified.")
#         for metric_name in self.metric_names:
#             if metric_name not in self._SUPPORTED_METRICS:
#                 raise ValueError(f"Metric {metric_name} is not supported.")

#             if metric_name == "num_triangles" and self._is_directed:
#                 raise ValueError("The 'num_triangles' metric is not supported for directed graphs.")

#     def _validate_custom_metrics(self):
#         for metric_name, stat_func in self.custom_metrics:
#             if not callable(stat_func):
#                 raise ValueError(f"Custom Metric {metric_name} is not a function.")
    
#     def _register_metrics(self):
#         self.statistics_functions = {}

#         for metric_name in self.metric_names:
#             if metric_name == "num_edges":
#                 func = lambda G: len(G.edges())
#             elif metric_name == "num_triangles":
#                 func = lambda G: sum(nx.triangles(G).values()) // 3 # nx.triangles counts each triangle 3 times, for every node in it

#             self.statistics_functions[metric_name] = func
    
#     def get_num_of_statistics(self):
#         """
#         Get the number of statistics that are registered.
        
#         Returns
#         -------
#         n_stats : int
#             The number of statistics.
#         """
#         return len(self.statistics_functions)
    
#     def calculate_statistics(self, W: np.ndarray, verbose=False):
#         """
#         Calculate the statistics of a graph. 
#         This is equivalent to calculating what is known in the ERGM literature as g(y) for a given graph y.
        
#         Parameters
#         ----------
#         W : np.ndarray
#             A connectivity matrix.
        
#         Returns
#         -------
#         stats : np.ndarray or dict 
#             An array of statistics, represented as a numpy array of values, or a dict if verbose=True
#         """
#         G = connectivity_matrix_to_G(W, directed=self._is_directed)
        
#         stats = {name: func(G) for name, func in self.statistics_functions.items()}
#         if verbose:
#             return stats
#         else:
#             return np.array(list(stats.values()))
       

                