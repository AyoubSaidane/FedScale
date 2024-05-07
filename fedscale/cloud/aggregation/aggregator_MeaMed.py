from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.fllibs import *
from overrides import overrides
import numpy as np
import copy
import torch

class MeaMedAggregator(Aggregator):
    """Updates the aggregation with the new results using Coordinate-wise median."""

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
            self.model_weights = self.mean_around_median(self.client_updates)
            self.model_wrapper.set_weights(
                copy.deepcopy(self.model_weights),
                client_training_results=self.client_training_results,
            )
            self.client_updates = []
    
    def mean_around_median(self, client_updates):
        """
        calculate the mean of the closest weights to the median weight
        """
        meamed_weights = []
        for i in range(len(client_updates[0])):    
            weight_accross_clients = np.stack([client[i] for client in client_updates])
            sorted_indices = np.argsort(np.abs((weight_accross_clients- np.median(weight_accross_clients, axis=0))), axis=0)
            sorted_weight_accross_clients = np.take_along_axis(weight_accross_clients, sorted_indices, axis=0)
            meamed_weights.append(np.mean(sorted_weight_accross_clients[:int(len(client_updates)*(1-self.byzantine_proportion))], axis=0))
        return meamed_weights
        

if __name__ == "__main__":
    aggregator = MeaMedAggregator(parser.args)
    aggregator.run()