from abc import ABC, abstractmethod
from typing import Collection
from copy import deepcopy

import numpy as np
import networkx as nx
from numba import njit

from utils import *


class Metric(ABC):
    def __init__(self, requires_graph=False):
        self.requires_graph = requires_graph
        # Each metric either expects directed or undirected graphs. This field should be initialized in the constructor
        # and should not change.
        self._is_directed = None

    @abstractmethod
    def calculate(self, input: np.ndarray | nx.Graph):
        pass

    def get_effective_feature_count(self, n):
        """
        How many features does this metric produce. Defaults to 1.
        """
        return 1

    def calc_change_score(self, current_network: np.ndarray | nx.Graph, indices: tuple):
        """
        The default naive way to calculate the change score (namely, the difference in statistics) of a pair of
        networks.

        The newly proposed network is created by flipping the edge denoted by `indices`

        Returns
        -------
        statistic of proposed_network minus statistic of current_network.
        """
        i, j = indices
        if self.requires_graph:
            proposed_network = current_network.copy()
            if proposed_network.has_edge(i, j):
                proposed_network.remove_edge(i, j)
            else:
                proposed_network.add_edge(i, j)
        else:
            proposed_network = current_network.copy()
            proposed_network[i, j] = 1 - proposed_network[i, j]

            if not self.network_stats_calculator.is_directed:
                proposed_network[j, i] = 1 - proposed_network[j, i]

        proposed_network_stat = self.calculate(proposed_network)
        current_network_stat = self.calculate(current_network)
        return proposed_network_stat - current_network_stat

    def calculate_for_sample(self, networks_sample: np.ndarray | Collection[nx.Graph]):
        if self.requires_graph:
            n = networks_sample[0].number_of_nodes()
        else:
            n = networks_sample.shape[0]

        num_of_samples = networks_sample.shape[2]

        result = np.zeros((self.get_effective_feature_count(n), num_of_samples))
        for i in range(num_of_samples):
            network = networks_sample[i] if self.requires_graph else networks_sample[:, :, i]
            result[:, i] = self.calculate(network)
        return result


class NumberOfEdgesUndirected(Metric):
    def __str__(self):
        return "num_edges_undirected"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = False

    def calculate(self, W: np.ndarray):
        return np.sum(W) // 2

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        return -1 if current_network[indices[0], indices[1]] else 1

    def calculate_for_sample(self, networks_sample: np.ndarray):
        """
        Sum each matrix over all matrices in sample
        """
        return networks_sample.sum(axis=(0, 1)) // 2


class NumberOfEdgesDirected(Metric):
    def __str__(self):
        return "num_edges_directed"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = True

    def calculate(self, W: np.ndarray):
        return np.sum(W)

    @staticmethod
    @njit
    def calc_change_score(current_network: np.ndarray, indices: tuple):
        return -1 if current_network[indices[0], indices[1]] else 1

    @staticmethod
    @njit
    def calculate_for_sample(networks_sample: np.ndarray):
        """
        Sum each matrix over all matrices in sample
        """
        n = networks_sample.shape[0]
        reshaped_networks_sample = networks_sample.reshape(n ** 2, networks_sample.shape[2])
        return np.sum(reshaped_networks_sample, axis=0)


# TODO: change the name of this one to undirected and implement also a directed version?
class NumberOfTriangles(Metric):
    def __str__(self):
        return "num_triangles"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = False

    def calculate(self, W: np.ndarray):
        if not np.all(W.T == W):
            raise ValueError("NumOfTriangles not implemented for directed graphs")
        # the (i,j)-th entry of W^3 counts the number of 3-length paths from node i to node j. Thus, the i-th element on
        # the diagonal counts the number of triangles that node 1 is part of (3-length paths from i to itself). As the
        # graph is undirected, we get that each path is counted twice ("forward and backwards"), thus the division by 2.
        # Additionally, each triangle is counted 3 times by diagonal elements (once for each node that takes part in
        # forming it), thus the division by 3.
        return (np.linalg.matrix_power(W, 3)).diagonal().sum() // (3 * 2)

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        # The triangles that are affected by the edge toggling are those that involve it, namely, if the (i,j)-th edge
        # is toggled, the change in absolute value equals to the number of nodes k for which the edges (i,k) and (j,k)
        # exist. This is equivalent to the number of 2-length paths from i to j, which is the (i,j)-th entry of W^2.
        # If the edge is turned on, the change is positive, and otherwise negative.

        sign = -1 if current_network[indices[0], indices[1]] else 1
        return sign * np.dot(current_network[indices[0]], current_network[:, indices[1]])


class BaseDegreeVector(Metric):
    """
    A base class for calculating a degree vector for a network.
    To avoid multicollinearity with other features, an optional parameter `base_idx` can be used to specify
    which index the calculation starts from.
    """

    def __init__(self, requires_graph: bool, is_directed: bool, base_idx=0):
        super().__init__(requires_graph=requires_graph)
        self._is_directed = is_directed
        self.base_idx = base_idx

    def get_effective_feature_count(self, n):
        return n - self.base_idx


class InDegree(BaseDegreeVector):
    """
    Calculate the in-degree of each node in a directed graph.
    """

    def __str__(self):
        return "indegree"

    def __init__(self, base_idx=0):
        super().__init__(requires_graph=False, is_directed=True, base_idx=base_idx)

    def calculate(self, W: np.ndarray):
        return W.sum(axis=0)[self.base_idx:]

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        n = current_network.shape[0]
        diff = np.zeros(n)
        i, j = indices

        sign = -1 if current_network[i, j] else 1

        diff[j] = sign
        return diff[self.base_idx:]

    def calculate_for_sample(self, networks_sample: np.ndarray):
        return networks_sample.sum(axis=0)[self.base_idx:]


class OutDegree(BaseDegreeVector):
    """
    Calculate the out-degree of each node in a directed graph.
    """

    def __str__(self):
        return "outdegree"

    def __init__(self, base_idx=0):
        super().__init__(requires_graph=False, is_directed=True, base_idx=base_idx)

    def calculate(self, W: np.ndarray):
        return W.sum(axis=1)[self.base_idx:]

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        n = current_network.shape[0]
        diff = np.zeros(n)
        i, j = indices

        sign = -1 if current_network[i, j] else 1

        diff[i] = sign
        return diff[self.base_idx:]

    def calculate_for_sample(self, networks_sample: np.ndarray):
        return networks_sample.sum(axis=1)[self.base_idx:]


class UndirectedDegree(BaseDegreeVector):
    """
    Calculate the degree of each node in an undirected graph.
    """

    def __str__(self):
        return "undirected_degree"

    def __init__(self, base_idx=0):
        super().__init__(requires_graph=False, is_directed=False, base_idx=base_idx)

    def calculate(self, W: np.ndarray):
        return W.sum(axis=0)[self.base_idx:]


class Reciprocity(Metric):
    """
    The Reciprocity metric takes the connectivity matrix of a directed graph, and returns a vector
    of size n-choose-2 indicating whether nodes i,j are connected. i.e. $ y_{i, j} \cdot y_{j, i} $
    for every possible pair of nodes   
    """

    def __str__(self):
        return "reciprocity"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = True

    def calculate(self, W: np.ndarray):
        return (W * W.T)[np.triu_indices(W.shape[0], 1)]

    def get_effective_feature_count(self, n):
        # n choose 2
        return n * (n - 1) // 2

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        # Note: we intentionally initialize the whole matrix and return np.triu_indices() by the end (rather than
        # initializing an array of zeros of size n choose 2) to ensure compliance with the indexing returned by
        # the calculate method.
        i, j = indices
        all_changes = np.zeros(current_network.shape)
        min_idx = min(indices)
        max_idx = max(indices)

        if current_network[j, i] and not current_network[i, j]:
            all_changes[min_idx, max_idx] = 1
        elif current_network[j, i] and current_network[i, j]:
            all_changes[min_idx, max_idx] = -1
        return all_changes[np.triu_indices(all_changes.shape[0], 1)]


class TotalReciprocity(Metric):
    """
    Calculates how many reciprocal connections exist in a network  
    """

    def __str__(self):
        return "total_reciprocity"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = True

    def calculate(self, W: np.ndarray):
        return (W * W.T).sum() / 2

    @staticmethod
    @njit
    def calc_change_score(current_network: np.ndarray, indices: tuple):
        i, j = indices

        if current_network[j, i] and not current_network[i, j]:
            return 1
        elif current_network[j, i] and current_network[i, j]:
            return -1
        else:
            return 0

    @staticmethod
    # @njit
    def calculate_for_sample(networks_sample: np.ndarray):
        return np.einsum("ijk,jik->k", networks_sample, networks_sample) / 2


class ExWeightNumEdges(Metric):
    """
    Weighted sum of the number of edges, based on exogenous attributes.
    """

    # TODO: Collection doesn't necessarily support __getitem__, find a typing hint of a sized Iterable that does.
    def __init__(self, exogenous_attr: Collection):
        super().__init__(requires_graph=False)
        self.exogenous_attr = exogenous_attr
        self.num_weight_mats = self._get_num_weight_mats()
        self.edge_weights = None
        self._calc_edge_weights()

    @abstractmethod
    def _calc_edge_weights(self):
        ...

    @abstractmethod
    def _get_num_weight_mats(self):
        ...

    def get_effective_feature_count(self, n):
        return self.num_weight_mats

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        sign = -1 if current_network[indices[0], indices[1]] else 1
        return sign * self.edge_weights[:, indices[0], indices[1]]

    def calculate(self, input: np.ndarray):
        res = np.einsum('ij,kij->k', input, self.edge_weights)
        if not self._is_directed:
            res = res / 2
        return res

    def calculate_for_sample(self, networks_sample: np.ndarray):
        res = np.einsum('ijk,mij->mk', networks_sample, self.edge_weights)
        if not self._is_directed:
            res = res / 2
        return res


class NumberOfEdgesTypesDirected(ExWeightNumEdges):
    def __init__(self, exogenous_attr: Collection):
        super().__init__(exogenous_attr)
        self._is_directed = True
        
    def _calc_edge_weights(self):
        num_nodes = len(self.exogenous_attr)
        unique_types = sorted(set(self.exogenous_attr))
        self.edge_weights = np.zeros((self.num_weight_mats, num_nodes, num_nodes))
        weight_mat_idx = 0
        for pre_type in unique_types:
            for post_type in unique_types:
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i == j:
                            continue
                        if self.exogenous_attr[i] == pre_type and self.exogenous_attr[j] == post_type:
                            self.edge_weights[weight_mat_idx, i, j] = 1
                weight_mat_idx += 1

    def _get_num_weight_mats(self):
        return len(set(self.exogenous_attr)) ** 2


class NodeAttrSum(ExWeightNumEdges):
    def _calc_edge_weights(self):
        num_nodes = len(self.exogenous_attr)
        self.edge_weights = np.zeros((self.num_weight_mats, num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                self.edge_weights[0, i, j] = self.exogenous_attr[i] + self.exogenous_attr[j]

    def _get_num_weight_mats(self):
        return 1


class NodeAttrSumOut(ExWeightNumEdges):
    def __init__(self, exogenous_attr: Collection):
        super().__init__(exogenous_attr)
        self._is_directed = True

    def _calc_edge_weights(self):
        num_nodes = len(self.exogenous_attr)
        self.edge_weights = np.zeros((self.num_weight_mats, num_nodes, num_nodes))
        for i in range(num_nodes):
            self.edge_weights[0, i, :] = self.exogenous_attr[i] * np.ones(num_nodes)
            self.edge_weights[0, i, i] = 0

    def _get_num_weight_mats(self):
        return 1


class NodeAttrSumIn(ExWeightNumEdges):
    def __init__(self, exogenous_attr: Collection):
        super().__init__(exogenous_attr)
        self._is_directed = True

    def _calc_edge_weights(self):
        num_nodes = len(self.exogenous_attr)
        self.edge_weights = np.zeros((self.num_weight_mats, num_nodes, num_nodes))
        for j in range(num_nodes):
            self.edge_weights[0, :, j] = self.exogenous_attr[j] * np.ones(num_nodes)
            self.edge_weights[0, j, j] = 0

    def _get_num_weight_mats(self):
        return 1

class MetricsCollection:
    def __init__(self, metrics: Collection[Metric], is_directed: bool, n_nodes: int, fix_collinearity=True):
        self.metrics = tuple([deepcopy(metric) for metric in metrics])
        self.metric_names = tuple([str(metric) for metric in self.metrics])

        self.requires_graph = any([x.requires_graph for x in self.metrics])
        self.is_directed = is_directed
        for x in self.metrics:
            if x._is_directed != self.is_directed:
                model_is_directed_str = "a directed" if self.is_directed else "an undirected"
                metric_is_directed_str = "a directed" if x._is_directed else "an undirected"
                raise ValueError(f"Trying to initialize {model_is_directed_str} model with {metric_is_directed_str} "
                                 f"metric `{str(x)}`!")
        self.num_of_metrics = len(self.metrics)

        self._fix_collinearity = fix_collinearity
        if self._fix_collinearity:
            self.collinearity_fixer()

        self.n_nodes = n_nodes

        # Returns the number of features that are being calculated. Since a single metric might return more than one feature, the length of the statistics vector might be larger than 
        # the amount of metrics. Since it also depends on the network size, n is a mandatory parameters. That's why we're using the get_effective_feature_count function
        self.num_of_features = sum([metric.get_effective_feature_count(self.n_nodes) for metric in self.metrics])

    def get_metric(self, metric_name: str) -> Metric:
        """
        Get a metric instance
        """
        return self.metrics[self.metric_names.index(metric_name)]

    def collinearity_fixer(self):
        """
        Find collinearity between metrics in the collection.

        Currently this is a naive version that only handles the very simple cases. 
        TODO - Implement a smarter solution
        """
        num_edges_name = str(NumberOfEdgesDirected())
        indegree_name = str(InDegree())
        outdegree_name = str(OutDegree())

        if num_edges_name in self.metric_names and indegree_name in self.metric_names:
            self.get_metric(indegree_name).base_idx = 1

        if num_edges_name in self.metric_names and outdegree_name in self.metric_names:
            self.get_metric(outdegree_name).base_idx = 1

        if indegree_name in self.metric_names and outdegree_name in self.metric_names:
            self.get_metric(indegree_name).base_idx = 1

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

        statistics = np.zeros(self.num_of_features)

        feature_idx = 0
        for metric in self.metrics:
            if metric.requires_graph:
                input = G
            else:
                input = W

            n_features_from_metric = metric.get_effective_feature_count(self.n_nodes)
            statistics[feature_idx:feature_idx + n_features_from_metric] = metric.calculate(input)
            feature_idx += n_features_from_metric

        return statistics

    def calc_change_scores(self, current_network: np.ndarray, indices: tuple):
        """
        Calculates the vector of change scores, namely g(net_2) - g(net_1)
        """
        if self.requires_graph:
            G1 = connectivity_matrix_to_G(current_network, directed=self.is_directed)

        change_scores = np.zeros(self.num_of_features)

        feature_idx = 0
        for metric in self.metrics:
            if metric.requires_graph:
                input = G1
            else:
                input = current_network

            n_features_from_metric = metric.get_effective_feature_count(self.n_nodes)
            change_scores[feature_idx:feature_idx + n_features_from_metric] = metric.calc_change_score(input, indices)
            feature_idx += n_features_from_metric

        return change_scores

    def calculate_sample_statistics(self, networks_sample: np.ndarray) -> np.ndarray:
        """
        Calculate the statistics over a sample of networks

        Parameters
        ----------
        networks_sample
            The networks sample - an array of n X n X sample_size
        Returns
        -------
        an array of the statistics vector per sample (num_features X sample_size)
        """
        num_of_samples = networks_sample.shape[2]
        features_of_net_samples = np.zeros((self.num_of_features, num_of_samples))

        if self.requires_graph:
            networks_as_graphs = [connectivity_matrix_to_G(W, self.is_directed) for W in networks_sample]

        feature_idx = 0
        for metric in self.metrics:
            n_features_from_metric = metric.get_effective_feature_count(self.n_nodes)

            if metric.requires_graph:
                networks = networks_as_graphs
            else:
                networks = networks_sample

            features_of_net_samples[feature_idx:feature_idx + n_features_from_metric] = metric.calculate_for_sample(
                networks)
            feature_idx += n_features_from_metric

        return features_of_net_samples
