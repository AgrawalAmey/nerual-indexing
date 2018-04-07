from .parser import Parser
from .storage_handler import StorageHandler
from .models.resnet.resnet_loop import resnet_main
from .models.resnet.resnet_model import Model
from .models.resnet.resnet_model_fn import resnet_model_fn
from .utils.data import DataUtils
from .utils.lr import learning_rate_with_decay

TRAIN_PATTERN = '../data/nd-iris/tfrecords/nd-iris-train-*.tfrecords'
EVAL_PATTERN = '../data/nd-iris/tfrecords/nd-iris-val-*.tfrecords'
NUM_TRAIN_IMAGES = 104000

###############################################################################
# Running the model
###############################################################################


class IrisModel(Model):
	"""Model class with appropriate defaults for Imagenet data."""

	def __init__(self, resnet_size, data_format=None, embedding_size=300,
              version=1):
		"""These are the parameters that work for Imagenet data.
		Args:
			resnet_size: The number of convolutional layers needed in the model.
			data_format: Either 'channels_first' or 'channels_last', specifying which
				data format to use when setting up the model.
			embedding_size: The number of output classes needed from the model. This
				enables users to extend the same model to their own datasets.
			version: Integer representing which version of the ResNet network to use.
				See README for details. Valid values: [1, 2]
		"""

		# For bigger models, we want to use "bottleneck" layers
		if resnet_size < 50:
			bottleneck = False
			final_size = 512
		else:
			bottleneck = True
			final_size = 2048

		super(IrisModel, self).__init__(
                    resnet_size=resnet_size,
                    bottleneck=bottleneck,
                    embedding_size=embedding_size,
                    num_filters=64,
                    kernel_size=7,
                    conv_stride=2,
                    first_pool_size=3,
                    first_pool_stride=2,
                    second_pool_size=7,
                    second_pool_stride=1,
                    block_sizes=_get_block_sizes(resnet_size),
                    block_strides=[1, 2, 2, 2],
                    final_size=final_size,
                    version=version,
                    data_format=data_format)


def _get_block_sizes(resnet_size):
	"""Retrieve the size of each block_layer in the ResNet model.
	The number of block layers used for the Resnet model varies according
	to the size of the model. This helper grabs the layer set we want, throwing
	an error if a non-standard size has been selected.
	Args:
		resnet_size: The number of convolutional layers needed in the model.
	Returns:
		A list of block sizes to use in building the model.
	Raises:
		KeyError: if invalid resnet_size is received.
	"""
	choices = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
            200: [3, 24, 36, 3]
	}

	try:
		return choices[resnet_size]
	except KeyError:
		err = ('Could not find layers for selected Resnet size.\n'
                    'Size received: {}; sizes allowed: {}.'.format(
                        resnet_size, choices.keys()))
		raise ValueError(err)


def iris_model_fn(features, labels, mode, params):
	"""Our model_fn for ResNet to be used with our Estimator."""
	learning_rate_fn = learning_rate_with_decay(
            batch_size=params['batch_size'], batch_denom=256,
            num_images=NUM_TRAIN_IMAGES, boundary_epochs=[30, 60, 80, 90],
            decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])

	return resnet_model_fn(features, labels, mode, IrisModel,
                                        resnet_size=params['resnet_size'],
                                        weight_decay=1e-4,
                                        learning_rate_fn=learning_rate_fn,
                                        momentum=0.9,
                                        data_format=params['data_format'],
                                        version=params['version'],
                                        loss_filter_fn=None)

if __name__ == "__main__":
    parser = Parser()
    parser.setup_parser()
    flags = parser.parse()
    
    # Store args
    storage_handler = StorageHandler()
    storage_handler.store_args(flags)
    log_dir = storage_handler.get_log_dir()

    # Train
    resnet_main(flags, iris_model_fn, DataUtils.get_iterator,
                TRAIN_PATTERN, EVAL_PATTERN, log_dir)
