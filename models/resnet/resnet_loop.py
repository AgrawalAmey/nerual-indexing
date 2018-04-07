"""Contains utility and supporting functions for ResNet.

    This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from neural_indexing.models.resnet import resnet_model

################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################


def resnet_main(flags, model_function, input_function, 
                train_file_pattern, eval_file_pattern, log_dir):
    """Shared main loop for ResNet Models.

    Args:
        flags: FLAGS object that contains the params for running. See
        ArgParser for created flags.
        model_function: the function that instantiates the Model and builds the
            ops for train/eval. This will be passed directly into the estimator.
        input_function: the function that processes the dataset and returns a
            dataset that the estimator can train on. This will be wrapped with
            all the relevant flags for running and passed to estimator.
    """

    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Create session config based on values of inter_op_parallelism_threads and
    # intra_op_parallelism_threads. Note that we default to having
    # allow_soft_placement = True, which is required for multi-GPU and not
    # harmful for other modes.
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=flags.inter_op_parallelism_threads,
        intra_op_parallelism_threads=flags.intra_op_parallelism_threads,
        allow_soft_placement=True)

    # Set up a RunConfig to save checkpoint and set session config.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                                  session_config=session_config)

    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=log_dir, config=run_config,
        params={
            'resnet_size': flags.resnet_size,
            'data_format': flags.data_format,
            'batch_size': flags.batch_size,
            'version': flags.version,
        })

    for _ in range(flags.train_epochs // flags.epochs_between_evals):

        print('Starting a training cycle.')

        def input_fn_train():
            return input_function(train_file_pattern, flags.batch_size,
                                  flags.epochs_between_evals,
                                  flags.num_parallel_calls)

        classifier.train(input_fn=input_fn_train)

        print('Starting to evaluate.')
        # Evaluate the model and print results

        def input_fn_eval():
            return input_function(eval_file_pattern, flags.batch_size,
                                  1, flags.num_parallel_calls, flags.multi_gpu)

        eval_results = classifier.evaluate(input_fn=input_fn_eval)
        
        tf.summary.scalar('eval_loss', eval_results)
        
        print(eval_results)
