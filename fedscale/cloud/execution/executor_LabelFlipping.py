# -*- coding: utf-8 -*-

import os
import sys

from client_LabelFlipping import LabelFlippingClient

import fedscale.cloud.config_parser as parser
from fedscale.cloud.execution.executor import Executor


"""In this example, we only need to change the TorchClient Component we need to import"""

class LabelFlippingExecutor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""

    def __init__(self, args):
        super().__init__(args)

    def get_client_trainer(self, conf):
        return LabelFlippingClient(conf)

if __name__ == "__main__":
    executor = LabelFlippingExecutor(parser.args)
    executor.run()

