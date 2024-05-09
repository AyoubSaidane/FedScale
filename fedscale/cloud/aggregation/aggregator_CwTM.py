from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.fllibs import *
from overrides import overrides
import numpy as np
import copy
from scipy.stats import trim_mean

class CwTMAggregator(Aggregator):
    """Updates the aggregation with the new results using Coordinate-wise trimmed mean."""
    
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
            self.model_weights = self.trimmed_mean(self.client_updates)
            self.model_wrapper.set_weights(
                copy.deepcopy(self.model_weights),
                client_training_results=self.client_training_results,
            )
            self.client_updates = []

    def trimmed_mean(self, client_updates):
        """
        calculate the trimmed mean 
        """
        trimmed_mean_weights = []
        for i in range(len(client_updates[0])):
            trimmed_mean_weights.append(np.apply_along_axis(trim_mean, axis=0, arr=np.stack([client[i] for client in client_updates]), proportiontocut=self.byzantine_proportion))
        return trimmed_mean_weights


if __name__ == "__main__":
    aggregator = CwTMAggregator(parser.args)
    aggregator.run()
