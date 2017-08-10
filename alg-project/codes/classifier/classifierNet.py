from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from embedding.facenet import TripletLossNetwork
import tensorflow as tf
from utils import data_utils
import tensorflow.contrib.slim as slim
from embedding import facenet
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from utils import image_utils


class ClassifierNet(object):
    def __init__(self, model_def, image_size, embedding_size, nrof_classes, train_set=None, batch_size=1, epoch_size=1,
                 weight_decay=0.0, top_k=5, center_loss_factor=0.0, center_loss_alfa=0, learning_rate_decay_epochs=0,
                 learning_rate_decay_factor=0, nrof_preprocess_threads=4, random_crop=True,
                 random_flip=True, random_contrast=True, random_rotate=True, is_training=True):
        if is_training:
            self.global_step = tf.Variable(0, trainable=False)

            # Get a list of image paths and their labels
            self.image_list, self.label_list = data_utils.get_image_paths_and_labels(train_set)
            assert len(self.image_list) > 0, 'The dataset should not be empty'
            # Create a queue that produces indices into the image_list and label_list
            labels = ops.convert_to_tensor(self.label_list, dtype=tf.int32)
            range_size = array_ops.shape(labels)[0]
            index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                                        shuffle=True, seed=None, capacity=32)

            self.index_dequeue_op = index_queue.dequeue_many(batch_size * epoch_size, 'index_dequeue')

            self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

            self.batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

            self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            self.image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')

            self.labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels')

            self.input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                                       dtypes=[tf.string, tf.int64],
                                                       shapes=[(1,), (1,)],
                                                       shared_name=None, name=None)
            self.enqueue_op = self.input_queue.enqueue_many([self.image_paths_placeholder, self.labels_placeholder],
                                                            name='enqueue_op')

            images_and_labels = []
            for _ in range(nrof_preprocess_threads):
                filenames, label = self.input_queue.dequeue()
                images = []
                for filename in tf.unstack(filenames):
                    file_contents = tf.read_file(filename)
                    image = tf.image.decode_image(file_contents, channels=3)
                    if random_rotate:
                        image = tf.py_func(image_utils.random_rotate_image, [image], tf.uint8)
                    if random_crop:
                        image = tf.random_crop(image, [image_size, image_size, 3])
                    else:
                        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
                    if random_flip:
                        image = tf.image.random_flip_left_right(image)
                    if random_contrast:
                        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)

                    # pylint: disable=no-member
                    image.set_shape((image_size, image_size, 3))
                    images.append(tf.image.per_image_standardization(image))
                images_and_labels.append([images, label])

            image_batch, self.label_batch = tf.train.batch_join(
                images_and_labels, batch_size=self.batch_size_placeholder,
                shapes=[(image_size, image_size, 3), ()], enqueue_many=True,
                capacity=4 * nrof_preprocess_threads * batch_size,
                allow_smaller_final_batch=True)
            image_batch = tf.identity(image_batch, 'image_batch')
            image_batch = tf.identity(image_batch, 'input')
            self.label_batch = tf.identity(self.label_batch, 'label_batch')
        else:
            image_batch = tf.placeholder(tf.float32, [None, image_size, image_size, 3], name='input')

        TripletLossNetwork(model_def=model_def, image_size=image_size,
                           embedding_size=embedding_size, is_training=False, is_classifier=True,
                           image_batch=image_batch)

        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        logits = slim.fully_connected(self.embeddings, nrof_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(weight_decay), scope='Logits',
                                      reuse=False)

        prod = tf.nn.softmax(logits, name='prod')

        # Here we create a tensor to hold the index of top_k result
        top_k_res = tf.nn.top_k(prod, k=top_k, name='top_k')

        if is_training:
            if center_loss_factor > 0.0:
                prelogits_center_loss, _ = facenet.center_loss(self.embeddings, self.label_batch, center_loss_alfa,
                                                               nrof_classes)
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                     prelogits_center_loss * center_loss_factor)

            self.learning_rate = tf.train.exponential_decay(self.learning_rate_placeholder, self.global_step,
                                                            learning_rate_decay_epochs * epoch_size,
                                                            learning_rate_decay_factor, staircase=True)
            tf.summary.scalar('learning_rate', self.learning_rate)

            # Calculate the average cross entropy loss across the batch
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_batch, logits=logits, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)

            # Calculate the total losses
            self.regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.total_loss = tf.add_n([cross_entropy_mean] + self.regularization_losses, name='total_loss')
