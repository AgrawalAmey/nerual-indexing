from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse


class Parser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Neural Indexing')

    def setup_base_parser(self):
        self.parser.add_argument(
            "--train_epochs", "-te", type=int, default=100,
            help="[default: %(default)s] The number of epochs used to train."
        )

        self.parser.add_argument(
            "--epochs_between_evals", "-ebe", type=int, default=4,
            help="[default: %(default)s] The number of training epochs to run "
            "between evaluations."
        )

        self.parser.add_argument(
            "--batch_size", "-bs", type=int, default=32,
            help="[default: %(default)s] Batch size for training and evaluation."
        )
    
    def setup_performance_parser(self):
        self.parser.add_argument(
            "--num_parallel_calls", "-npc",
            type=int, default=5,
            help="[default: %(default)s] The number of records that are "
            "processed in parallel  during input processing. This can be "
            "optimized per data set but for generally homogeneous data "
            "sets, should be approximately the number of available CPU "
            "cores."
        )

        self.parser.add_argument(
            "--inter_op_parallelism_threads", "-inter",
            type=int, default=0,
            help="[default: %(default)s Number of inter_op_parallelism_threads "
            "to use for CPU. See TensorFlow config.proto for details."
        )

        self.parser.add_argument(
            "--intra_op_parallelism_threads", "-intra",
            type=int, default=0,
            help="[default: %(default)s Number of intra_op_parallelism_threads "
            "to use for CPU. See TensorFlow config.proto for details."
        )

    def setup_resnet_parser(self):
        self.parser.add_argument(
            '--version', '-v', type=int, choices=[1, 2],
            default=2,
            help='Version of ResNet. (1 or 2) See README.md for details.'
        )

        self.parser.add_argument(
            '--resnet_size', '-rs', type=int, default=50,
            choices=[18, 34, 50, 101, 152, 200],
            help='[default: %(default)s] The size of the ResNet model to use.'
        )

        self.parser.add_argument(
            "--data_format", "-df",
            default=None,
            choices=["channels_first", "channels_last"],
            help="A flag to override the data format used in the model. "
            "channels_first provides a performance boost on GPU but is not "
            "always compatible with CPU. If left unspecified, the data "
            "format will be chosen automatically based on whether TensorFlow"
            "was built for CPU or GPU.",
        )

    def setup_parser(self):
        self.setup_base_parser()
        self.setup_performance_parser()
        self.setup_resnet_parser()

    def parse(self):
        args = self.parser.parse_args()
        return args
