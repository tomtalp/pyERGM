from abc import ABC, abstractmethod
from typing import Collection
from copy import deepcopy

import numpy as np
import torch
import networkx as nx
from numba import njit

from utils import *


class Metric(ABC):
    def __init__(self, requires_graph=False):
        self.requires_graph = requires_graph
        # Each metric either expects directed or undirected graphs. This field should be initialized in the constructor
        # and should not change.
        self._is_directed = None
        self._is_dyadic_independent = True
        self._n_nodes = None

    @abstractmethod
    def calculate(self, input: np.ndarray | nx.Graph):
        pass

    def _get_effective_feature_count(self):
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

            if not self._is_directed:
                proposed_network[j, i] = 1 - proposed_network[j, i]

        proposed_network_stat = self.calculate(proposed_network)
        current_network_stat = self.calculate(current_network)
        return proposed_network_stat - current_network_stat

    def calculate_for_sample(self, networks_sample: np.ndarray | Collection[nx.Graph]):
        num_of_samples = networks_sample.shape[2]

        result = np.zeros((self._get_effective_feature_count(), num_of_samples))
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
        self._is_dyadic_independent = True

    def calculate(self, W: np.ndarray):
        return np.sum(W) // 2

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        return -1 if current_network[indices[0], indices[1]] else 1

    def calculate_for_sample(self, networks_sample: np.ndarray | torch.Tensor):
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
        self._is_dyadic_independent = True

    def calculate(self, W: np.ndarray):
        return np.sum(W)

    @staticmethod
    @njit
    def calc_change_score(current_network: np.ndarray, indices: tuple):
        return -1 if current_network[indices[0], indices[1]] else 1

    @staticmethod
    @calc_for_sample_njit()
    def calculate_for_sample(networks_sample: np.ndarray | torch.Tensor):
        """
        Sum each matrix over all matrices in sample
        """
        return networks_sample.sum(axis=0).sum(axis=0)


# TODO: change the name of this one to undirected and implement also a directed version?
class NumberOfTriangles(Metric):
    def __str__(self):
        return "num_triangles"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = False
        self._is_dyadic_independent = False

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
    To avoid multicollinearity with other features, an optional parameter `indices_to_ignore` can be used to specify
    which indices the calculation ignores.
    """

    def __init__(self, requires_graph: bool, is_directed: bool, indices_to_ignore=None):
        super().__init__(requires_graph=requires_graph)
        self._is_directed = is_directed
        if indices_to_ignore is None:
            self._indices_to_ignore = []
        else:
            self._indices_to_ignore = deepcopy(indices_to_ignore)

    def _get_effective_feature_count(self):
        return self._n_nodes - len(self._indices_to_ignore)


class InDegree(BaseDegreeVector):
    """
    Calculate the in-degree of each node in a directed graph.
    """

    def __str__(self):
        return "indegree"

    def __init__(self, indices_to_ignore=None):
        super().__init__(requires_graph=False, is_directed=True, indices_to_ignore=indices_to_ignore)
        self._is_dyadic_independent = True

    def calculate(self, W: np.ndarray):
        return np.delete(W.sum(axis=0), self._indices_to_ignore)

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        n = current_network.shape[0]
        diff = np.zeros(n)
        i, j = indices

        sign = -1 if current_network[i, j] else 1

        diff[j] = sign
        return np.delete(diff, self._indices_to_ignore)

    def calculate_for_sample(self, networks_sample: np.ndarray | torch.Tensor):
        summed_tensor = networks_sample.sum(axis=0)

        if isinstance(networks_sample, torch.Tensor) and networks_sample.is_sparse:
            n_nodes = networks_sample.shape[0]
            n_samples = networks_sample.shape[2]

            indices_to_keep = [i for i in range(self._n_nodes) if i not in self._indices_to_ignore]
            indices = summed_tensor.indices()[:, indices_to_keep]
            values = summed_tensor.values()[indices_to_keep]
            return torch.sparse_coo_tensor(indices, values, (n_nodes, n_samples))
        else:
            return np.delete(summed_tensor, self._indices_to_ignore, axis=0)


class OutDegree(BaseDegreeVector):
    """
    Calculate the out-degree of each node in a directed graph.
    """

    def __str__(self):
        return "outdegree"

    def __init__(self, indices_to_ignore=None):
        super().__init__(requires_graph=False, is_directed=True, indices_to_ignore=indices_to_ignore)
        self._is_dyadic_independent = True

    def calculate(self, W: np.ndarray):
        return np.delete(W.sum(axis=1), self._indices_to_ignore)

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        n = current_network.shape[0]
        diff = np.zeros(n)
        i, j = indices

        sign = -1 if current_network[i, j] else 1

        diff[i] = sign
        return np.delete(diff, self._indices_to_ignore)

    def calculate_for_sample(self, networks_sample: np.ndarray | torch.Tensor):
        summed_tensor = networks_sample.sum(axis=1)

        if isinstance(networks_sample, torch.Tensor) and networks_sample.is_sparse:
            n_nodes = networks_sample.shape[0]
            n_samples = networks_sample.shape[2]

            indices_to_keep = [i for i in range(self._n_nodes) if i not in self._indices_to_ignore]
            indices = summed_tensor.indices()[:, indices_to_keep]
            values = summed_tensor.values()[indices_to_keep]
            return torch.sparse_coo_tensor(indices, values, (n_nodes, n_samples))
        else:
            return np.delete(summed_tensor, self._indices_to_ignore, axis=0)


class UndirectedDegree(BaseDegreeVector):
    """
    Calculate the degree of each node in an undirected graph.
    """

    def __str__(self):
        return "undirected_degree"

    def __init__(self, indices_to_ignore=None):
        super().__init__(requires_graph=False, is_directed=False, indices_to_ignore=indices_to_ignore)
        self._is_dyadic_independent = True

    def calculate(self, W: np.ndarray):
        return np.delete(W.sum(axis=0), self._indices_to_ignore)


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
        self._is_dyadic_independent = False

    def calculate(self, W: np.ndarray):
        return (W * W.T)[np.triu_indices(W.shape[0], 1)]

    def _get_effective_feature_count(self):
        # n choose 2
        return self._n_nodes * (self._n_nodes - 1) // 2

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
        self._is_dyadic_independent = False

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
    # @njit # Not supporting neither np.einsum nor sparse torch
    def calculate_for_sample(networks_sample: np.ndarray | torch.Tensor):
        if isinstance(networks_sample, torch.Tensor) and networks_sample.is_sparse:
            transposed_sparse_tensor = transpose_sparse_sample_matrices(networks_sample)
            return torch.sum(networks_sample * transposed_sparse_tensor, axis=(0, 1)) / 2
        elif isinstance(networks_sample, np.ndarray):
            return np.einsum("ijk,jik->k", networks_sample, networks_sample) / 2
        else:
            raise ValueError(f"Unsupported type of sample: {type(networks_sample)}! Supported types are np.ndarray and "
                             f"torch.Tensor with is_sparse=True")


class ExWeightNumEdges(Metric):
    """
    Weighted sum of the number of edges, based on exogenous attributes.
    """

    # TODO: Collection doesn't necessarily support __getitem__, find a typing hint of a sized Iterable that does.
    def __init__(self, exogenous_attr: Collection):
        super().__init__(requires_graph=False)
        self.exogenous_attr = exogenous_attr
        self.edge_weights = None
        self._calc_edge_weights()

    @abstractmethod
    def _calc_edge_weights(self):
        ...

    @abstractmethod
    def _get_num_weight_mats(self):
        ...

    def _get_effective_feature_count(self):
        return self._get_num_weight_mats()

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
    def __init__(self, exogenous_attr: Collection, indices_to_ignore=None):
        if indices_to_ignore is None:
            self._indices_to_ignore = []
        else:
            self._indices_to_ignore = deepcopy(indices_to_ignore)
        super().__init__(exogenous_attr)
        self._is_directed = True

    def _calc_edge_weights(self):
        num_nodes = len(self.exogenous_attr)
        unique_types = sorted(set(self.exogenous_attr))
        self.edge_weights = np.zeros((self._get_num_weight_mats(), num_nodes, num_nodes))
        weight_mat_idx = 0
        num_skipped_indices = 0
        for pre_type in unique_types:
            for post_type in unique_types:
                if weight_mat_idx not in self._indices_to_ignore:
                    for i in range(num_nodes):
                        for j in range(num_nodes):
                            if i == j:
                                continue
                            if self.exogenous_attr[i] == pre_type and self.exogenous_attr[j] == post_type:
                                self.edge_weights[weight_mat_idx - num_skipped_indices, i, j] = 1
                else:
                    num_skipped_indices += 1
                weight_mat_idx += 1

    def _get_num_weight_mats(self):
        return len(set(self.exogenous_attr)) ** 2 - len(self._indices_to_ignore)

    def calculate_for_sample(self, networks_sample: np.ndarray):
        return np.delete(super().calculate_for_sample(networks_sample), self._indices_to_ignore, axis=0)

    def __str__(self):
        return "num_edges_between_types_directed"


class NodeAttrSum(ExWeightNumEdges):
    def __init__(self, exogenous_attr: Collection, is_directed: bool):
        super().__init__(exogenous_attr)
        self._is_directed = is_directed

    def _calc_edge_weights(self):
        num_nodes = len(self.exogenous_attr)
        self.edge_weights = np.zeros((self._get_num_weight_mats(), num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                self.edge_weights[0, i, j] = self.exogenous_attr[i] + self.exogenous_attr[j]

    def _get_num_weight_mats(self):
        return 1

    def __str__(self):
        return "node_attribute_sum"


class NodeAttrSumOut(ExWeightNumEdges):
    def __init__(self, exogenous_attr: Collection):
        super().__init__(exogenous_attr)
        self._is_directed = True

    def _calc_edge_weights(self):
        num_nodes = len(self.exogenous_attr)
        self.edge_weights = np.zeros((self._get_num_weight_mats(), num_nodes, num_nodes))
        for i in range(num_nodes):
            self.edge_weights[0, i, :] = self.exogenous_attr[i] * np.ones(num_nodes)
            self.edge_weights[0, i, i] = 0

    def _get_num_weight_mats(self):
        return 1

    def __str__(self):
        return "node_attribute_sum_out"


class NodeAttrSumIn(ExWeightNumEdges):
    def __init__(self, exogenous_attr: Collection):
        super().__init__(exogenous_attr)
        self._is_directed = True

    def _calc_edge_weights(self):
        num_nodes = len(self.exogenous_attr)
        self.edge_weights = np.zeros((self._get_num_weight_mats(), num_nodes, num_nodes))
        for j in range(num_nodes):
            self.edge_weights[0, :, j] = self.exogenous_attr[j] * np.ones(num_nodes)
            self.edge_weights[0, j, j] = 0

    def _get_num_weight_mats(self):
        return 1

    def __str__(self):
        return "node_attribute_in"


class MetricsCollection:

    def __init__(self,
                 metrics: Collection[Metric],
                 is_directed: bool,
                 n_nodes: int,
                 fix_collinearity=True,
                 use_sparse_matrix=False,
                 # TODO: For tests only, find a better solution
                 do_copy_metrics=True):
        if not do_copy_metrics:
            self.metrics = tuple([metric for metric in metrics])
        else:
            self.metrics = tuple([deepcopy(metric) for metric in metrics])
        for m in self.metrics:
            m._n_nodes = n_nodes
            if hasattr(m, "_indices_to_ignore"):
                if m._indices_to_ignore:
                    cur_num_features = m._get_effective_feature_count()
                    if max(m._indices_to_ignore) >= cur_num_features or min(m._indices_to_ignore) <= -cur_num_features:
                        raise ValueError(
                            f"{str(m)} got indices to ignore {m._indices_to_ignore} which are out of bound for "
                            f"{cur_num_features} features it has")

        self.is_directed = is_directed
        for x in self.metrics:
            if x._is_directed != self.is_directed:
                model_is_directed_str = "a directed" if self.is_directed else "an undirected"
                metric_is_directed_str = "a directed" if x._is_directed else "an undirected"
                raise ValueError(f"Trying to initialize {model_is_directed_str} model with {metric_is_directed_str} "
                                 f"metric `{str(x)}`!")

        self.n_nodes = n_nodes

        self.use_sparse_matrix = use_sparse_matrix
        self.requires_graph = any([x.requires_graph for x in self.metrics])

        self._fix_collinearity = fix_collinearity
        if self._fix_collinearity:
            self.collinearity_fixer()

        # Returns the number of features that are being calculated. Since a single metric might return more than one
        # feature, the length of the statistics vector might be larger than the amount of metrics. Since it also depends
        # on the network size, n is a mandatory parameters. That's why we're using the get_effective_feature_count
        # function
        self.num_of_features = self.calc_num_of_features()

        self.num_of_metrics = len(self.metrics)
        self.metric_names = tuple([str(metric) for metric in self.metrics])
        self._has_dyadic_dependent_metrics = any([not x._is_dyadic_independent for x in self.metrics])

    def _delete_metric(self, metric: Metric):
        self.metrics = tuple([m for m in self.metrics if m != metric])
        self.requires_graph = any([x.requires_graph for x in self.metrics])

    def calc_num_of_features(self):
        return sum([metric._get_effective_feature_count() for metric in self.metrics])

    def get_metric(self, metric_name: str) -> Metric:
        """
        Get a metric instance
        """
        return self.metrics[self.metric_names.index(metric_name)]

    def get_metric_by_feat_idx(self, idx: int):
        cum_sum_num_feats = 0
        for m in self.metrics:
            cum_sum_num_feats += m._get_effective_feature_count()
            if cum_sum_num_feats > idx:
                return m

    def get_feature_idx_within_metric(self, idx: int):
        cum_sum_num_feats = 0
        for m_idx in range(len(self.metrics)):
            next_met_num_feats = self.metrics[m_idx]._get_effective_feature_count()
            if cum_sum_num_feats + next_met_num_feats > idx:
                # We want to return the index in the "full" array, namely regardless of ignored features. So, in case
                # there are indices that are ignored, we must take the returned index from the array of non-ignored
                # indices, to compensate for the ones ignored by the indexing of MetricCollection (which holds only
                # effective features, after ignoring).
                # For example - if we ignore the degree of the first node, and now want to ignore the degree of the
                # fifth node as well, we must return 4 for the metric to correctly ignore it. But it is the fourth
                # feature corresponding to the metric in the MetricCollection vector of features (not fifth, because the
                # first is missing, not returned by the Metric as it's ignored). By returning the fourth element from
                # the list of non-ignored indices, which is [1, 2, ..., n], we solve the problem.

                effective_idx_within_metric = idx - cum_sum_num_feats
                if hasattr(self.metrics[m_idx], "_indices_to_ignore"):
                    # The number of ignored indices + the number of used indices is the total number (without ignoring).
                    total_num_indices = len(self.metrics[m_idx]._indices_to_ignore) + next_met_num_feats
                    non_ignored_indices = [i for i in range(total_num_indices) if
                                           i not in self.metrics[m_idx]._indices_to_ignore]
                    # Return the index of the feature with relation to the whole set of features (without ignoring).
                    return non_ignored_indices[effective_idx_within_metric]
                else:
                    return effective_idx_within_metric
            else:
                cum_sum_num_feats += next_met_num_feats

    def collinearity_fixer(self, sample_size=1000, thr=10 ** -5):
        """
        Find collinearity between metrics in the collection.

        Currently this is a naive version that only handles the very simple cases.
        TODO: revisit the threshold and sample size
        """
        is_linearly_dependent = True
        while is_linearly_dependent:
            self.num_of_features = self.calc_num_of_features()

            # Sample networks from a maximum entropy distribution, for avoiding edge cases (such as a feature is 0 for
            # all networks in the sample).
            sample = np.random.binomial(n=1, p=0.5, size=(self.n_nodes, self.n_nodes, sample_size))

            # Symmetrize samples if not directed
            if not self.is_directed:
                sample = np.round((sample + sample.transpose(1, 0, 2)) / 2)

            # Make sure the main diagonal is 0
            for i in range(self.n_nodes):
                for k in range(sample_size):
                    sample[i, i, k] = 0

            # Calculate the features of the sample
            sample_features = self.calculate_sample_statistics(sample)
            features_cov_mat = sample_features @ sample_features.T

            # Determine whether the matrix of features is invertible. If not - this means there is a non-trivial vector,
            # that when multiplied by the matrix gives the 0 vector. Namely, there is a single set of coefficients that
            # defines a non-trivial linear combination that equals 0, for *all* the sampled feature vectors. This means
            # the features are linearly dependent.
            eigen_vals, eigen_vecs = np.linalg.eigh(features_cov_mat)
            small_eigen_vals_indices = np.where(np.abs(eigen_vals) < thr)[0]
            if small_eigen_vals_indices.size == 0:
                is_linearly_dependent = False
            else:
                # For each linear dependency (corresponding to an eigen vector with a low value), mark the indices of
                # features that are involved (identified by a non-zero coefficient in the eigen vector).
                dependent_features_flags = np.zeros((small_eigen_vals_indices.size, self.num_of_features))
                for i in range(small_eigen_vals_indices.size):
                    dependent_features_flags[
                        i, np.where(np.abs(eigen_vecs[:, small_eigen_vals_indices[i]]) > thr)[0]] = 1

                # Calculate the fraction of dependencies each feature is involved in.
                fraction_of_dependencies_involved = dependent_features_flags.mean(axis=0)

                # Sort the features (their indices) by the fraction of dependencies they are involved in (remove first
                # features that are involved in more dependencies). Break ties by the original order of the features in
                # the array (for the consistency of sorting. E.g, if we need to get rid of degree features, always
                # remove them by the order of nodes).
                removal_order = np.lexsort((np.arange(self.num_of_features), -fraction_of_dependencies_involved))

                # Iterate the metrics to find one with multiple features, namely effective number of features that is
                # larger than 1 ('trimmable'). We prefer to trim metrics rather than totally eliminate them from the
                # collection.
                i = 0
                cur_metric = self.get_metric_by_feat_idx(removal_order[i])
                is_trimmable = cur_metric._get_effective_feature_count() > 1
                while (not is_trimmable and i < removal_order.size - 1 and fraction_of_dependencies_involved[
                    removal_order[i]] > 0):
                    i += 1
                    cur_metric = self.get_metric_by_feat_idx(removal_order[i])
                    is_trimmable = cur_metric._get_effective_feature_count() > 1

                # If a trimmable metric was not found (i.e., all features that are involved in the dependency are of
                # metrics with an effective number of features of 1), totally remove the metric that is involved in most
                # dependencies.
                if not is_trimmable:
                    first_metric = self.get_metric_by_feat_idx(removal_order[0])
                    print(f"Removing the metric {str(first_metric)} from the collection to fix multi-collinearity")
                    self._delete_metric(metric=first_metric)
                else:
                    idx_to_delete = self.get_feature_idx_within_metric(removal_order[i])
                    print(f"Removing the {idx_to_delete} feature of {str(cur_metric)} to fix multi-collinearity")
                    cur_metric._indices_to_ignore.append(idx_to_delete)

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

            n_features_from_metric = metric._get_effective_feature_count()
            statistics[feature_idx:feature_idx + n_features_from_metric] = metric.calculate(input)
            feature_idx += n_features_from_metric

        return statistics

    def calc_change_scores(self, current_network: np.ndarray, indices: tuple):
        """
        Calculates the vector of change scores, namely g(net_2) - g(net_1)

        NOTE - this function assumes that the size current_network and self.n_nodes are the same, and doesn't validate
        it, due to runtime considerations. Currently, the only use of this function is within ERGM and
        NaiveMetropolisHastings, so this is fine.
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

            n_features_from_metric = metric._get_effective_feature_count()
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
        if networks_sample.shape[0] != self.n_nodes:
            raise ValueError(
                f"Got a sample of networks of size {networks_sample.shape[0]}, but Metrics expect size {self.n_nodes}")

        num_of_samples = networks_sample.shape[2]
        features_of_net_samples = np.zeros((self.num_of_features, num_of_samples))

        if self.requires_graph:
            networks_as_graphs = [connectivity_matrix_to_G(W, self.is_directed) for W in networks_sample]

        if self.use_sparse_matrix:
            networks_as_sparse_tensor = np_tensor_to_sparse_tensor(networks_sample)

        feature_idx = 0
        for metric in self.metrics:
            n_features_from_metric = metric._get_effective_feature_count()

            if metric.requires_graph:
                networks = networks_as_graphs
            elif self.use_sparse_matrix:
                networks = networks_as_sparse_tensor
            else:
                networks = networks_sample

            features = metric.calculate_for_sample(networks)

            if isinstance(features, torch.Tensor):
                if features.is_sparse:
                    features = features.to_dense()
                features = features.numpy()

            features_of_net_samples[feature_idx:feature_idx + n_features_from_metric] = features
            feature_idx += n_features_from_metric

        return features_of_net_samples
