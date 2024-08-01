from abc import ABC, abstractmethod
import math
from typing import List

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

    def get_effective_feature_count(self, n):
        """
        How many features does this metric produce. Defaults to 1
        """
        return 1

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

class Reciprocity(Metric):
    """
    The Reciprocity metric takes the connectivity matrix of a directed graph, and returns a vector
    of size n-choose-2 indicating whether nodes i,j are connected. i.e. $ y_{i, j} \cdot y_{j, i} $
    for every possible pair of nodes   
    """
    def __init__(self):
        super().__init__(metric_name="reciprocity", requires_graph=False)
    
    def calculate(self, W, is_directed):
        if not is_directed:
            raise ValueError("Reciprocity effect cannot be calculated for undirected networks")

        n = W.shape[0]
        y_ij = W[np.triu_indices(n, k=1)]
        y_ji = W.T[np.triu_indices(n, k=1)]

        return np.multiply(y_ij, y_ji)

    def get_effective_feature_count(self, n):
        return math.comb(n, 2)
    
class TotalReciprocity(Metric):
    """
    Calculates how many reciprocal connections exist in a network  
    """
    def __init__(self):
        super().__init__(metric_name="total_reciprocity", requires_graph=False)
    
    def calculate(self, W, is_directed):
        if not is_directed:
            raise ValueError("Total Reciprocity effect cannot be calculated for undirected networks")

        n = W.shape[0]
        y_ij = W[np.triu_indices(n, k=1)]
        y_ji = W.T[np.triu_indices(n, k=1)]

        return sum(np.multiply(y_ij, y_ji))  

class MetricsCollection():
    def __init__(self, metrics: List[Metric], is_directed: bool):
        self.metrics = tuple(metrics)
        self.requires_graph = any([x.requires_graph for x in self.metrics])
        self.is_directed = is_directed
        self.num_of_metrics = len(self.metrics)

    def get_num_of_features(self, n: int):
        """
        Returns the number of features that are being calculated. Since a single metric might
        return more than one feature, the length of the statistics vector might be larger than 
        the amount of metrics. Since it also depends on the network size, n is a mandatory parameters.

        Parameters
        ----------
        n : int
            Number of nodes in network
        
        Returns
        -------
        num_of_features : int
            How many features will be calculated for a network of size n
        """
        num_of_features = sum([metric.get_effective_feature_count(n) for metric in self.metrics])
        return num_of_features

    def calculate_statistics(self, W: np.ndarray):
        """
        Calculate the statistics of a graph, formally written as g(y).

        Parameters
        ----------
        W : np.ndarray
            A connectivity matrix.
        
        Returns
        -------
        statistics : np.ndarray
            An array of statistics
        """
        if self.requires_graph:
            G = connectivity_matrix_to_G(W, directed=self.is_directed)
        
        n_nodes = W.shape[0]
        statistics = np.zeros(self.get_num_of_features(n_nodes))

        feature_idx = 0
        for metric in self.metrics:
            if metric.requires_graph:
                input = G
            else:
                input = W

            n_features_from_metric = metric.get_effective_feature_count(n_nodes)
            statistics[feature_idx:feature_idx+n_features_from_metric] = metric.calculate(input, self.is_directed)
            feature_idx += n_features_from_metric
            

        return statistics
