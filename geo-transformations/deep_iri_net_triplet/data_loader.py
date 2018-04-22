import numpy as np
import tensorflow as tf


class DataLoader(object):
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

    def decode(self, serialized_example):
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

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int64)

        return image, label
        
    def translate(self, image, label, num_samples=5, num_quantums=16):
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


    def translate_lambda(self, image, label): 
        return self.translate(image, label)

    def inputs(self, sess, file_pattern, batch_size, num_epochs):
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
            dataset = dataset.map(self.decode, num_parallel_calls=4)\
                            .shuffle(batch_size * 5)\
                            .batch(batch_size)\
                            .prefetch(batch_size)

            iterator = dataset.make_one_shot_iterator()

            while True:
                images, labels = iterator.get_next()
                images, labels = sess.run([images, labels])
                lables = np.repeat(labels, self.embedding_size)\
                           .reshape(-1, self.embedding_size)
                yield (images, labels)
