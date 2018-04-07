import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow as tf

class DataUtils(object):
    @classmethod
    def _decode(self, serialized_example):
        """Parses an image and label from the given `serialized_example`."""
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string),
            })

        # Convert from a scalar string tensor (whose single string has
        # length 250 * 250) to a float64 tensor with shape
        # [250 * 250].
        image = tf.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, [1, 250, 250])

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int64)

        return image, label

    @classmethod
    def get_iterator(cls, file_pattern, batch_size,
                     num_epochs, num_parallel_batches=8):
        """Reads input data num_epochs times.
        Args:
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
            train forever.
        Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, 250, 250]
            in the range [-0.5, 0.5].
        * labels is an int64 tensor with shape [batch_size] with the true label.

        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
        """
        if not num_epochs:
            num_epochs = None

        with tf.name_scope('input'):
            # Load multiple files by pattern
            files = tf.data.Dataset.list_files(file_pattern)

            dataset = files.apply(tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=4,
                block_length=4,
                buffer_output_elements=batch_size))

            # Shuffle 500 elements at a time
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(3 * batch_size, num_epochs))

            # Create batch and decode
            dataset = dataset.apply(
                tf.contrib.data.map_and_batch(cls._decode, batch_size,
                                              num_parallel_batches=num_parallel_batches))

            # Prefetch the next batch
            dataset = dataset.prefetch(batch_size)

            iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    @classmethod
    def show_samples(cls, file_pattern):
        """Show sample images from a TFRecords file."""
        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # Input images and labels.
            image_batch, label_batch = cls.get_iterator(
                file_pattern=file_pattern, batch_size=32, num_epochs=1)

            # The op for initializing the variables.
            init_op = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())

            # Create a session for running operations in the Graph.
            with tf.Session() as sess:
                # Initialize the variables (the trained variables and the
                # epoch counter).
                sess.run(init_op)
                image, label = sess.run([image_batch, label_batch])
                fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20, 10))
                for j in range(32):
                    row = j // 8
                    col = j % 8
                    axes[row, col].axis("off")
                    axes[row, col].imshow(image[j], cmap=cm.Greys_r)
                    axes[row, col].set_title(label[j])
                plt.show()
