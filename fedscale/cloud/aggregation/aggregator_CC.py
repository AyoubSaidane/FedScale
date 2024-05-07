from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.fllibs import *
from overrides import overrides
import numpy as np
import copy
from scipy.stats import trim_mean
import torch

class CCAggregator(Aggregator):
    """Updates the aggregation with the new results using Coordinate Clipping."""
    
    def __init__(self, args):
        super().__init__(args)
        self.client_updates = [] # accumulate client updates
        self.clipping_parameter = 1 # This is an arbitrary value that should be changed later to match theoretical results
        self.iteration_number = 10 # This is an arbitrary value that should be changed later to match theoretical results
        

    def update_weight_aggregation(self, results):
        """
        :param results: the results collected from a client.
        """
        update_weights = results["update_weight"]
        if type(update_weights) is dict:
            update_weights = [x for x in update_weights.values()]

        self.client_updates.append(update_weights)

        if self._is_last_result_in_round():
            self.model_weights = self.centered_clipping(self.client_updates)
            self.model_wrapper.set_weights(
                copy.deepcopy(self.model_weights),
                client_training_results=self.client_training_results,
            )
            self.client_updates = []

    def centered_clipping(self, client_updates):
        """
        centered clipping algorithm
        """
        aggregate = self.median_accross_client(client_updates)
        for _ in range(self.iteration_number):
            for client in client_updates:
                for i in range(len(client)):
                    aggregate[i] += self.clipp(client[i]-aggregate[i], self.clipping_parameter)/len(client_updates)
        return aggregate

    def median_accross_client(self, client_updates):
        """
        calculate the median for all weights accross all client updates
        """
        median_weights = []
        for i in range(len(client_updates[0])):
            median_weights.append(torch.median(torch.stack([torch.tensor(client[i]) for client in client_updates]), dim=0)[0].numpy())
        return median_weights
    
    def clipp(z, clipping_parameter):
        """
        clipping function for Centered Clipping
        """
        return z * min(1,clipping_parameter/np.linalg.norm(np.array(z)))


if __name__ == "__main__":
    aggregator = CCAggregator(parser.args)
    aggregator.run()