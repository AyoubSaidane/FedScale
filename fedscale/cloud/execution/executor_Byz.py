from fedscale.cloud.execution.executor import Executor
from fedscale.cloud.fllibs import *
import numpy as np


class ByzExecutor(Executor):
    def __init__(self, config):
        super().__init__(config)
        self.byzantine_proportion = 1/4  # Probability of returning the opposite sign

    def Train(self, config):
        client_id, train_res = super().Train(config)
        # Randomly decide whether to return the opposite sign
        if np.random.random() < self.byzantine_proportion:
            return client_id, -train_res
        return client_id, train_res

if __name__ == "__main__":
    executor = ByzExecutor(parser.args)
    executor.run()
