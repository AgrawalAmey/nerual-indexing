import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def decode(serialized_example):
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
    image = tf.reshape(image, [64, 512, 1])

    # Remove NaN
    image = tf.where(tf.is_nan(image), tf.zeros_like(image), image)

    # Normalize image
    image = (image + 0.49) * 500 + 4.5

    # split into images
    images = split_images(image)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int64)

    return images, label


split_images =  lambda image: tf.split(image, num_or_size_splits=8, axis=1)

def translate(image, label, num_samples=5, num_quantums=16):
    transformed_images = []
    translate_vectors = []

    for _ in range(num_samples):
        translation_index = np.random.randint(0, num_quantums)
        translation = translation_index * (512 // num_quantums)
        transformed_images.append(
            tf.concat([image[:, translation:],
                       image[:, :translation]], 1))
        translate_vector = np.zeros(num_quantums, np.uint16)
        translate_vector[translation_index] = 1
        translate_vector = tf.convert_to_tensor(translate_vector)
        translate_vectors.append(translate_vector)

    images = tf.stack([image] * num_samples)
    labels = tf.stack([label] * num_samples)
    transformed_images = tf.stack(transformed_images)
    translate_vectors = tf.stack(translate_vectors)

    return images, labels, transformed_images, translate_vectors


translate_lambda = lambda image, label: translate(image, label)


def inputs(sess, file_pattern, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, 350, 250]
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
#                          .map(translate, num_parallel_calls=4)\
#                          .apply(tf.contrib.data.unbatch())\
        dataset = (dataset.map(decode, num_parallel_calls=4)
                   .shuffle(batch_size * 5)
                   .batch(batch_size)
                   .prefetch(batch_size))

        iterator = dataset.make_one_shot_iterator()

        while True:
            #             image, label, transformed_image, translate_vector = iterator.get_next()
            
            next_element = iterator.get_next()

            while True:
                try:
                    images, _ = sess.run(next_element)
                    # image, label, transformed_image, translate_vector = sess.run(next_element)
                    # yield ([image, translate_vector], transformed_image)
                    yield (images, images.copy())
                except tf.errors.OutOfRangeError:
                    print("End of dataset.")
                    break


def show_samples_from_tfr(file_pattern):
    """Show sample images from a TFRecords file."""
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Create a session for running operations in the Graph.
        with tf.Session() as sess:
            # Initialize the variables (the trained variables and the
            # epoch counter).
            sess.run(init_op)

            gen = inputs(sess, file_pattern=file_pattern,
                         batch_size=32, num_epochs=1)

            for images, _ in gen:
                print(images, images[0].shape)

            # for image, label, transformed, translation_vector in gen:
            #     fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(20, 10))
            #     print(transformed.mean(), transformed.std(), transformed.shape)
            #     for row in range(8):
            #         axes[row, 0].axis("off")
            #         axes[row, 1].axis("off")
            #         axes[row, 0].imshow(image[row].reshape(64, 512))
            #         axes[row, 0].set_title(label[row])
            #         axes[row, 1].imshow(transformed[row].reshape(64, 512))
            #         axes[row, 1].set_title(translation_vector[row].argmax())
            #     break
