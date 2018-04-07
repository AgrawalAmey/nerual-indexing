import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

record_iterator = tf.python_io.tf_record_iterator(
    path='../data/nd-iris/tfrecords/nd-iris-train-1.tfrecords')

i = 0

for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    features = example.features.feature
    image = tf.decode_raw(features['image'].bytes_list.value[0], tf.float32)
    image = tf.reshape(image, [250, 250])
    if i == 90:
        plt.imshow(image, cmap=cm.Greys_r)
    if i > 100:
        break
    i += 1
