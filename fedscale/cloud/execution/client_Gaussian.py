import logging
import math
import os
import sys
import copy

import numpy as np
import torch

from torch.autograd import Variable
from fractions import Fraction

from fedscale.cloud.execution.torch_client import TorchClient


class GaussianClient(TorchClient):
    """Gaussian client component"""

    def __init__(self, args):
        super().__init__(args)
        self.noise_level = 2


    def train(self, client_data, model, conf):
        results = super().train(client_data, model, conf)

        client_id = conf.client_id
        """1 out of malicious_factor client is malicious"""
        fraction = Fraction(conf.malicious_factor).limit_denominator()
        is_malicious = ((client_id+1) % fraction.denominator <= fraction.numerator - 1)

        if is_malicious:
            return self.gaussian_noise(results)
        return results
    
    def gaussian_noise(self, results):
        Byz_results = copy.deepcopy(results)
        update_weights = Byz_results["update_weight"]
        assert isinstance(update_weights, dict), "update_weights must be a dictionary"
        for key, value in update_weights.items():
            Byz_results['update_weight'][key] = value + np.random.normal(0, self.noise_level, value.shape)
        return Byz_results

