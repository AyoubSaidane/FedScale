from fedscale.cloud.execution.executor import Executor
from fedscale.cloud.fllibs import *
import numpy as np
import copy


class ByzExecutor(Executor):
    def __init__(self, config):
        super().__init__(config)
        self.byzantine_proportion = 1/4  # Probability of returning the opposite sign

    def Train(self, config):
        client_id, train_res = super().Train(config)
        # Randomly decide whether to return the opposite sign
        if np.random.random() < self.byzantine_proportion:
            return client_id, self.sign_flipping(train_res)
        return client_id, train_res
    
    def sign_flipping(self, results):
        Byz_results = copy.deepcopy(results)
        update_weights = Byz_results["update_weight"]
        assert isinstance(update_weights, dict), "update_weights must be a dictionary"
        for key, value in update_weights.items():
            Byz_results['update_weight'][key] = -value
        return Byz_results


if __name__ == "__main__":
    executor = ByzExecutor(parser.args)
    executor.run()
