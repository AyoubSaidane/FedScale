from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.fllibs import *
from overrides import overrides
import numpy as np
import copy
from scipy.stats import trim_mean

class CGEggregator(Aggregator):
    """Updates the aggregation with the new results using Comparative gradient elimination."""
    
    def __init__(self, args):
        super().__init__(args)
        self.client_updates = [] # accumulate client updates
        self.byzantine_proportion = 1/4  # This should be equal to the fraction f/n, for now we assume f/n = 0.25
        


    def update_weight_aggregation(self, results):
        """
        :param results: the results collected from a client.
        """
        update_weights = results["update_weight"]
        if type(update_weights) is dict:
            update_weights = [x for x in update_weights.values()]

        self.client_updates.append(update_weights)

        if self._is_last_result_in_round():
            self.model_weights = self.comparative_gradient_elimination(self.client_updates)
            self.model_wrapper.set_weights(
                copy.deepcopy(self.model_weights),
                client_training_results=self.client_training_results,
            )
            self.client_updates = []
    
    def comparative_gradient_elimination(self, client_updates):
        """
        CGE algorithm
        """
        CGE_weights = []
        for i in range(len(client_updates[0])):
            sorted_weight_accross_clients = sorted([client[i] for client in client_updates], key=np.linalg.norm)
            CGE_weights.append(np.mean(sorted_weight_accross_clients[:int(len(client_updates)*(1-self.byzantine_proportion))], axis=0))
        return CGE_weights
    

if __name__ == "__main__":
    aggregator = CGEggregator(parser.args)
    aggregator.run()