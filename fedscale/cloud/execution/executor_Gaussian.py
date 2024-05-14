from fedscale.cloud.execution.executor import Executor
from fedscale.cloud.fllibs import *
import numpy as np
import copy


class GaussianExecutor(Executor):
    def __init__(self, config):
        super().__init__(config)
        self.byzantine_proportion = 1/4     # Probability of returning the opposite sign
        self.noise_level = 2    # This is the standard deviation of the gaussian noise


    def Train(self, config):
        client_id, train_res = super().Train(config)
        # Randomly decide whether to return the opposite sign
        if np.random.random() < self.byzantine_proportion:
            return client_id, self.gaussian_noise(train_res)
        return client_id, train_res
    
    def gaussian_noise(self, results):
        Byz_results = copy.deepcopy(results)
        update_weights = Byz_results["update_weight"]
        assert isinstance(update_weights, dict), "update_weights must be a dictionary"
        for key, value in update_weights.items():
            Byz_results['update_weight'][key] = value + np.random.normal(0, self.noise_level, value.shape)
        return Byz_results


if __name__ == "__main__":
    executor = GaussianExecutor(parser.args)
    executor.run()
