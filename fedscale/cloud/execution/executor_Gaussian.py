# -*- coding: utf-8 -*-

import os
import sys


from client_Gaussian import GaussianClient

import fedscale.cloud.config_parser as parser
from fedscale.cloud.execution.executor import Executor


"""In this example, we only need to change the TorchClient Component we need to import"""

class GaussianExecutor(Executor):
    """
       Each run simulates the execution of an individual client as a Gaussian attacker
    """
        

    def get_client_trainer(self, conf):
        return GaussianClient(conf)

if __name__ == "__main__":
    executor = GaussianExecutor(parser.args)
    executor.run()

