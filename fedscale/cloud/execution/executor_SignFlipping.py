# -*- coding: utf-8 -*-

import os
import sys


from client_SignFlipping import SignFlippingClient

import fedscale.cloud.config_parser as parser
from fedscale.cloud.execution.executor import Executor


"""In this example, we only need to change the TorchClient Component we need to import"""

class SignFlippingExecutor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""
        

    def get_client_trainer(self, conf):
        return SignFlippingClient(conf)

if __name__ == "__main__":
    executor = SignFlippingExecutor(parser.args)
    executor.run()

