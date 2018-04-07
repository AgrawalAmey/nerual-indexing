from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, version, loss_filter_fn=None, triplet_loss_margin=1.0):
    """Shared functionality for different resnet model_fns.

    Initializes the ResnetModel representing the model layers
    and uses that model to build the necessary EstimatorSpecs for
    the `mode` in question. For training, this means building losses,
    the optimizer, and the train op that get passed into the EstimatorSpec.
    For evaluation and prediction, the EstimatorSpec is returned without
    a train op, but with the necessary parameters for the given mode.

    Args:
        features: tensor representing input images
        labels: tensor representing class labels for all input images
        mode: current estimator mode; should be one of
            `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
        model_class: a class representing a TensorFlow model that has a __call__
            function. We assume here that this is a subclass of ResnetModel.
        resnet_size: A single integer for the size of the ResNet model.
        weight_decay: weight decay loss rate used to regularize learned variables.
        learning_rate_fn: function that returns the current learning rate given
            the current global_step
        momentum: momentum term used for optimization
        data_format: Input format ('channels_last', 'channels_first', or None).
            If set to None, the format is dependent on whether a GPU is available.
        version: Integer representing which version of the ResNet network to use.
            See README for details. Valid values: [1, 2]
        loss_filter_fn: function that takes a string variable name and returns
            True if the var should be included in loss calculation, and False
            otherwise. If None, batch_normalization variables will be excluded
            from the loss.

    Returns:
        EstimatorSpec parameterized according to the input params and the
        current mode.
    """

    # Generate a summary node for the images
    tf.summary.image('images', features, max_outputs=6)

    model = model_class(resnet_size, data_format, version=version)
    embeddings = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        'embeddings': embeddings
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    triplet_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
        labels, embeddings, triplet_loss_margin)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(triplet_loss, name='triplet_loss')
    tf.summary.scalar('triplet_loss', triplet_loss)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name
    loss_filter_fn = loss_filter_fn or exclude_batch_norm

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = triplet_loss + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
    else:
        train_op = None

    metrics = {'triplet_loss': triplet_loss}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)
