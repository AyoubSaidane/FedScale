from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.fllibs import *
from overrides import overrides
import numpy as np
import copy
import torch

class CwMedAggregator(Aggregator):
    """Updates the aggregation with the new results using Coordinate-wise median."""

    def __init__(self, args):
        super().__init__(args)
        self.client_updates = [] # accumulate client updates

    def update_weight_aggregation(self, results):
        """
        :param results: the results collected from a client.
        """
        update_weights = results["update_weight"]
        if type(update_weights) is dict:
            update_weights = [x for x in update_weights.values()]

        self.client_updates.append(update_weights)

        if self._is_last_result_in_round():
            self.model_weights  = self.median_accross_client(self.client_updates)
            self.model_wrapper.set_weights(
                copy.deepcopy(self.model_weights),
                client_training_results=self.client_training_results,
            )
            self.client_updates = []
    
    def median_accross_client(self, client_updates):
        """
        calculate the median for all weights accross all client updates
        """
        median_weights = []
        for i in range(len(client_updates[0])):
            median_weights.append(torch.median(torch.stack([torch.tensor(client[i]) for client in client_updates]), dim=0)[0].numpy())
        return median_weights



if __name__ == "__main__":
    aggregator = CwMedAggregator(parser.args)
    aggregator.run()