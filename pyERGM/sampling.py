from copy import deepcopy
import numpy as np
from pyERGM.utils import *
from pyERGM.metrics import MetricsCollection
import time
from scipy.special import softmax


class Sampler():
    def __init__(self, thetas, metrics_collection: MetricsCollection):
        self.thetas = deepcopy(thetas)
        self.metrics_collection = deepcopy(metrics_collection)

    def sample(self, initial_state, n_iterations):
        pass


class NaiveMetropolisHastings(Sampler):
    def __init__(self, thetas, metrics_collection: MetricsCollection):
        """
        An implementation for the symmetric proposal Metropolis-Hastings algorithm for ERGMS, using the logit
        of the acceptance rate. See docs for more details.
        Throughout this implementation, networks are represented as adjacency matrices.

        Parameters
        ----------
        thetas : np.ndarray
            Coefficients of the ERGM
            # TODO: should be a field of the class or passed to sample?
        
        metrics_collection : MetricsCollection
            A MetricsCollection object that can calculate statistics of a network.
        """
        super().__init__(thetas, metrics_collection)

        self._edge_proposal_dists = {}

    def set_thetas(self, thetas):
        self.thetas = deepcopy(thetas)

    def _calculate_weighted_change_score(self, current_network, edge_flip_info: dict, node_flip_info={}):
        """
        Calculate g(proposed_network)-g(current_network) and then inner product with thetas.
        """
        change_score = self.metrics_collection.calc_change_scores(current_network, edge_flip_info, node_flip_info)
        return np.dot(self.thetas, change_score)

    def _flip_network_edge(self, current_network, i, j):
        """
        Flip the edge between nodes i, j. If it's an undirected network, we flip entries W_i,j and W_j,i.
        NOTE! This function changes the network that is passed by reference
        """
        current_network[i, j] = 1 - current_network[i, j]
        if not self.metrics_collection.is_directed:
            current_network[j, i] = 1 - current_network[j, i]

    def _update_node_features(self, current_network, i, f, c):
        """
        Update node i's feature f (feature index) to be of category c.
        NOTE! This function changes the network that is passed by reference
        """
        num_nodes = current_network.shape[0]
        current_network[i, num_nodes + f] = c

    def _calc_edge_influence_on_features(self, net_for_change_scores: np.ndarray):
        """
        Calculates the influence of each edge on each one of the features in the MetricsCollection.
        The influence of an edge on a feature is defined as the ratio between the change in the feature that flipping
        the edge would cause divided by the average change over all edges.
        If the average influence is 0 (the feature does not change by flipping any edge), the influence of all edges is
        0.
        TODO: maybe we need to handle these all-0 cases separately (e.g., on the empty network, no edge flip will change
         the reciprocity, but obviously all edges have influence on the reciprocity). Or maybe this is not a good
         definition for the influence of an edge.
        For example, each edge has an influence of 1 on NumberOfEdges (no matter what edge we flip, it changes by 1).
        Similarly, an edge (i,j) has an influence of n on OutDegree of node i, and 0 on OutDegree of other nodes (the
        out degree of node 1 would change by 1 as a result of flipping the edge, and there are exactly n-1 such edges,
        out of the n(n-1) edges in the matrix, so the average influence is 1/n).
        Parameters
        ----------
        net_for_change_scores : np.ndarray
            The network using which the change scores for all edges is calculated.
        Returns
        -------
        A n(n-1) vector with the sum of influences of each edge over all features (total influence of each edge).
        """
        change_scores = self.metrics_collection.prepare_mple_regressors(net_for_change_scores)
        mean_importance_per_feature = change_scores.mean(axis=0)
        change_scores[:, mean_importance_per_feature != 0] /= mean_importance_per_feature[
            None, mean_importance_per_feature != 0]
        return change_scores.sum(axis=1)

    def _calc_proposal_dist_features_influence__sum(self, net_for_change_scores: np.ndarray):
        """
        Calculate a proposal distribution over edges such that edges with high influence will get higher probabilities.
        The transformation of total influences into a distribution is done by normalized using the sum of total
        influences.
        Parameters
        ----------
        net_for_change_scores : np.ndarray
            The network using which the change scores for all edges is calculated.
        """
        edge_influence = self._calc_edge_influence_on_features(net_for_change_scores)
        self._edge_proposal_dists['features_influence__sum'] = edge_influence / edge_influence.sum()

    def _calc_proposal_dist_features_influence__softmax(self, net_for_change_scores: np.ndarray):
        """
        Calculate a proposal distribution over edges such that edges with high influence will get higher probabilities.
        The transformation of total influences into a distribution is done by softmax.
        Parameters
        ----------
        net_for_change_scores : np.ndarray
            The network using which the change scores for all edges is calculated.
        """
        edge_influence = self._calc_edge_influence_on_features(net_for_change_scores)
        self._edge_proposal_dists['features_influence__softmax'] = softmax(edge_influence)

    def sample(self,
               initial_state,
               num_of_nets,
               replace=True,
               edge_proposal_method="uniform",
               # TODO - these two params need to be dependent on the network size
               burn_in=1000,
               steps_per_sample=10,
               node_feature_names={},
               node_features_n_categories={},
               edge_node_flip_ratio=None):
        """
        Sample networks using the Metropolis-Hastings algorithm.
        
        Parameters
        ----------
        initial_state : np.ndarray
            The initial network to start the Markov Chain from
        
        num_of_nets : int
            The number of networks to sample

        burn_in : int
            Optional. The number of burn-in steps for the sampler (number of steps in the chain that are discarded
            before the sampler starts to take samples). *Defaults to 1000*.

        steps_per_sample : int
            Optional. The number of steps to advance the chain between samples. *Defaults to 10*.
        
        replace : bool
            A boolean flag indicating whether we sample with our without replacement. replace=True means networks can be
            duplicated.

        edge_proposal_method : str
            Optional. The method for the MCMC proposal distribution. This is defined as a distribution over the edges
            of the network, which implies how to choose a proposed graph out of all graphs that are 1-edge-away from the
            current graph. *Defaults to "uniform"*.
        """
        current_network = initial_state.copy()

        net_size = current_network.shape[0]
        num_of_node_features = current_network.shape[1] - current_network.shape[0]
        inverse_node_feature_names = {}
        for feature_name, feature_indices in node_feature_names.items():
            for i in feature_indices:
                inverse_node_feature_names[i] = feature_name

        sampled_networks = np.zeros((net_size, net_size + num_of_node_features, num_of_nets))
        sampled_node_features = np.zeros((net_size, num_of_node_features, num_of_nets))
        if edge_node_flip_ratio is None:
            if num_of_node_features > 0:
                edge_node_flip_ratio = max(1, int(net_size / num_of_node_features))
            else:
                edge_node_flip_ratio = net_size

        num_flips = burn_in + (num_of_nets * steps_per_sample)
        if edge_proposal_method == 'uniform':
            edges_to_flip = get_uniform_random_edges_to_flip(net_size, num_flips)
        elif edge_proposal_method == 'features_influence__sum':
            if edge_proposal_method not in self._edge_proposal_dists.keys():
                self._calc_proposal_dist_features_influence__sum(current_network)
            edges_to_flip = get_custom_distribution_random_edges_to_flip(num_flips, self._edge_proposal_dists[
                'features_influence__sum'])
        elif edge_proposal_method == 'features_influence__softmax':
            if edge_proposal_method not in self._edge_proposal_dists.keys():
                self._calc_proposal_dist_features_influence__softmax(current_network)
            edges_to_flip = get_custom_distribution_random_edges_to_flip(num_flips, self._edge_proposal_dists[
                'features_influence__softmax'])
        else:
            raise ValueError(f"Got an unsupported edge proposal method {edge_proposal_method}")

        # for now, nodes_to_flip supports only uniform sampling
        if num_of_node_features != 0:
            nodes_to_flip = get_uniform_random_nodes_to_flip(net_size, num_flips)
            node_features_to_flip = np.random.choice(num_of_node_features, size=num_flips)
            node_features_inds_to_n_categories = {feature_ind: node_features_n_categories[feature_name] for feature_ind, feature_name in inverse_node_feature_names.items()}
            new_node_feature_categories = get_uniform_random_new_node_feature_categories(node_features_to_flip, node_features_inds_to_n_categories)

        random_nums_for_change_acceptance = np.random.rand(num_flips)

        networks_count = 0
        mcmc_iter_count = 0
        feature_flip_counts = {feature_ind: 0 for feature_ind in range(num_of_node_features)}

        t1 = time.time()
        while networks_count != num_of_nets:
            # Edge flip:
            random_edge_entry = edges_to_flip[:, mcmc_iter_count % edges_to_flip.shape[1]] # the number in axis 1 is always just mcmc_iter_count?
            edge_flip_info = {
                'edge': random_edge_entry,
            } # This will be convenient when we'll start using non-binary edges

            # Node flip
            if (num_of_node_features != 0) and (mcmc_iter_count % edge_node_flip_ratio == 0):
                random_node_entry = nodes_to_flip[mcmc_iter_count % nodes_to_flip.shape[0]]
                random_feature_to_flip = node_features_to_flip[mcmc_iter_count % nodes_to_flip.shape[0]]
                current_category = current_network[random_node_entry, random_feature_to_flip]
                # random_feature_name_to_flip = inverse_node_feature_names[random_feature_to_flip]
                random_new_category = new_node_feature_categories[random_feature_to_flip][current_category][feature_flip_counts[random_feature_to_flip]]
                feature_flip_counts[random_feature_to_flip] += 1

            else:
                random_node_entry = None
                random_feature_to_flip = None
                current_category = None
                random_feature_name_to_flip = None
                random_new_category = None
            node_flip_info = {
                'node': random_node_entry,
                'feature': random_feature_to_flip,
                'new_category': random_new_category,
            }

            change_score = self._calculate_weighted_change_score(current_network, edge_flip_info, node_flip_info)

            rand_num = random_nums_for_change_acceptance[mcmc_iter_count % edges_to_flip.shape[1]]
            perform_change = change_score >= 1 or rand_num <= min(1, np.exp(change_score))
            if perform_change:
                self._flip_network_edge(current_network, edge_flip_info['edge'][0], edge_flip_info['edge'][1])
                if random_new_category is not None: # a node flip was suggested
                    self._update_node_features(current_network,
                                               node_flip_info['node'],
                                               node_flip_info['feature'],
                                               node_flip_info['new_category'])

            if mcmc_iter_count >= burn_in and (mcmc_iter_count - burn_in) % steps_per_sample == 0:
                sampled_networks[:, :, networks_count] = current_network

                if not replace:
                    # TODO - Since we're flipping coins in the beginning, we're reusing them in case of the non-replacement (since we might need more coin flips that num of networks).
                    #  Consider reflipping coins to get better "pseudorandom properties", in the case that we run out of pre-flipped coins
                    if np.unique(sampled_networks[:, :, :networks_count + 1], axis=2).shape[2] == networks_count + 1:
                        networks_count += 1
                    else:
                        sampled_networks[:, :, networks_count] = np.zeros((net_size, net_size))
                else:
                    networks_count += 1
                    t2 = time.time()
                    if networks_count % 100 == 0:
                        print(f"Sampled {networks_count}/{num_of_nets} networks, time taken: {t2-t1}")

            mcmc_iter_count += 1
        return sampled_networks
