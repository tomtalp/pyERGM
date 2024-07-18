import numpy as np
import networkx as nx
from utils import connectivity_matrix_to_G

class ERGM():
    """
    A representation of an ERGM model. 

    The model is initialized by registering `feature functions` of a graph.
    
    ## TODO - Thetas are currently set to 1 while in dev. We need to fit them!
    """
    def __init__(self, n_nodes):
        self._n_nodes = n_nodes
        self._theta = []
        self._feature_functions = []
        self._normalization_factor = None ## TODO - Partition function is first calculated in fit. What should we initialize it to be?
    
    def register_feature_function(self, feature_function, theta):
        """
        ## TODO - 
            1. When do we call this? At init? 
            2. When are thetas set? Do we register features without setting thetas first?
            
            But in general - I think model should be initialized with a list of functions, so that the user
            never has to manually call `register_feature_function()`.
        """
        self._feature_functions.append(feature_function)
        self._theta.append(theta)

    def _calculate_weight(self, W: np.ndarray):
        G = connectivity_matrix_to_G(W)

        features = [f(G) for f in self._feature_functions]

        weight = np.exp(np.dot(self._theta, features))

        return weight
    
    def _calculate_normalization_factor(self):
        # networks_for_sample = get_sample_networks()
        networks_for_sample = []
        
        self._normalization_factor = 0

        for network in networks_for_sample:
            weight = self._calculate_weight(network)
            self._normalization_factor += weight


    def fit(self):
        """
        TODO - This is just a mock implementation. 
        Currently just calculating the normalization factor. 
        """
        self._calculate_normalization_factor()

    def calculate_probability(self, W: np.ndarray):
        """
        Calculate the probability of a graph under the ERGM model.
        
        Parameters
        ----------
        W : np.ndarray
            A connectivity matrix.
        
        Returns
        -------
        prob : float
            The probability of the graph under the ERGM model.
        """

        if self._normalization_factor is None:
            raise ValueError("Normalization factor not set, fit the model before calculating probability.")

        weight = self._calculate_weight(W)
        prob = weight / self._normalization_factor
        
        return prob

    def sample_network(self, seed_network=None, steps=500, burn_in=100):
        """
        Sample a network from the ERGM model.
        
        Parameters
        ----------
        seed_network : np.ndarray
            A seed connectivity matrix to start the MCMC sampler from.
            ## TODO - Maybe this should just be a boolean flag/string and not the actual object. and sample_network() will decide with some logic
        steps : int
            The number of steps to run the MCMC sampler.
        burn_in : int
            The number of steps to run before collecting samples.
        
        Returns
        -------
        W : np.ndarray
            The sampled connectivity matrix.
        """
        if self._normalization_factor is None:
            raise ValueError("Normalization factor not set, fit the model before sampling.")

        if not seed_network:
            W = np.random.randint(0, 2, (self._n_nodes, self._n_nodes))
        
        



            # # Propose a new graph
            # W_proposed = self.propose(W)
            # # Calculate the acceptance probability
            # acceptance_prob = self.calculate_acceptance_prob(W_proposed)
            # # Accept or reject the proposal
            # if acceptance_prob > np.random.rand():
            #     W = W_proposed
        

