from abc import ABC, abstractmethod

import numpy as np
import networkx as nx

from utils import *

## TODO - Metrics needs to implement a diff() function

class Metric(ABC):
    def __init__(self, metric_name, requires_graph=False):
        self.metric_name = metric_name
        self.requires_graph = requires_graph

    @abstractmethod
    def calculate(self, input):
        pass

## TODO - Rethink the is_directed flag in the Metrics objects. Should we have a separate class for directed metrics?    
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
        self.num_of_metrics = len(self.metrics)

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
