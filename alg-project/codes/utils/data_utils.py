from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import codecs
import random
from datetime import datetime
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from utils import ckpt_revision_utils
# import sys
# import argparse
import numpy as np
from scipy import misc

# from tensorflow.python.framework import ops

from utils import image_utils


def get_dataset(paths, with_dateset_id=True):
    dataset = []
    for path in paths.split(','):
        path_val = path
        if with_dateset_id:
            path_exp, path_val = path.split('|')
        classes = os.listdir(path_val)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_val, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir, img) for img in images]
                if with_dateset_id:
                    dataset.append(image_utils.ImageClass(path_exp + '_' + class_name, image_paths))
                else:
                    dataset.append(image_utils.ImageClass(class_name, image_paths))
    return dataset


def read_and_augument_data(batch_size_placeholder, image_paths_placeholder, labels_placeholder, batch_size,
                           image_size,
                           random_crop, random_flip, random_contrast, random_rotate, nrof_preprocess_threads, image_channels=3):
    input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                          dtypes=[tf.string, tf.int64],
                                          shapes=[(3,), (3,)],
                                          shared_name=None, name=None)
    enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
        filenames, label = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, channels=image_channels)

            if random_crop:
                image = tf.random_crop(image, [image_size, image_size, 3])
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
            if random_flip:
                image = tf.image.random_flip_left_right(image)
            if random_contrast:
                image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            if random_rotate:
                image = tf.py_func(image_utils.random_rotate_image, [image], tf.uint8)

                # pylint: disable=no-member
            image.set_shape((image_size, image_size, 3))
            images.append(tf.image.per_image_standardization(image))
        images_and_labels.append([images, label])

    image_batch, labels_batch = tf.train.batch_join(
        images_and_labels, batch_size=batch_size_placeholder,
        shapes=[(image_size, image_size, 3), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * batch_size,
        allow_smaller_final_batch=True)
    return enqueue_op, input_queue, image_batch, labels_batch


def generate_validtion_set(data_paths, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    runtime = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    output_file = os.path.join(output_dir, 'validtion_set_%s.txt' % runtime)
    if os.path.exists(output_file):
        os.remove(output_file)
    output_f = codecs.open(output_file, 'w', 'utf-8')
    dataset = get_dataset(data_paths)
    num_of_classes = len(dataset)
    for i in range(num_of_classes):
        print('generating: %s/%s' % (i + 1, num_of_classes))
        for j in range(i, num_of_classes):
            flag = 0
            if i == j:
                flag = 1
            print(random.choice(dataset[i].image_paths), random.choice(dataset[j].image_paths), flag, file=output_f)
    output_f.close()
    print('generate validtion set success: %s' % output_file)
    return output_file


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


# def shuffle_examples(image_paths, labels):
#     shuffle_list = list(zip(image_paths, labels))
#     random.shuffle(shuffle_list)
#     image_paths_shuff, labels_shuff = zip(*shuffle_list)
#     return image_paths_shuff, labels_shuff


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label


# def random_rotate_image(image):
#     angle = np.random.uniform(low=-10.0, high=10.0)
#     return misc.imrotate(image, angle, 'bicubic')


def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = image_utils.to_rgb(img)
        if do_prewhiten:
            img = image_utils.prewhiten(img)
        img = image_utils.crop(img, do_random_crop, image_size)
        img = image_utils.flip(img, do_random_flip)
        images[i, :, :, :] = img
    return images


def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        ckpt_file, meta_file = ckpt_revision_utils.get_ckpt_and_metagraph(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(tf.get_default_session(), ckpt_file)

# def split_dataset(dataset, split_ratio, mode):
#     if mode == 'SPLIT_CLASSES':
#         nrof_classes = len(dataset)
#         class_indices = np.arange(nrof_classes)
#         np.random.shuffle(class_indices)
#         split = int(round(nrof_classes * split_ratio))
#         train_set = [dataset[i] for i in class_indices[0:split]]
#         test_set = [dataset[i] for i in class_indices[split:-1]]
#     elif mode == 'SPLIT_IMAGES':
#         train_set = []
#         test_set = []
#         min_nrof_images = 2
#         for cls in dataset:
#             paths = cls.image_paths
#             np.random.shuffle(paths)
#             split = int(round(len(paths) * split_ratio))
#             if split < min_nrof_images:
#                 continue  # Not enough images for test set. Skip class...
#             train_set.append(ImageClass(cls.name, paths[0:split]))
#             test_set.append(ImageClass(cls.name, paths[split:-1]))
#     else:
#         raise ValueError('Invalid train/test split mode "%s"' % mode)
#     return train_set, test_set


# def get_label_batch(label_data, batch_size, batch_index):
#     nrof_examples = np.size(label_data, 0)
#     j = batch_index * batch_size % nrof_examples
#     if j + batch_size <= nrof_examples:
#         batch = label_data[j:j + batch_size]
#     else:
#         x1 = label_data[j:nrof_examples]
#         x2 = label_data[0:nrof_examples - j]
#         batch = np.vstack([x1, x2])
#     batch_int = batch.astype(np.int64)
#     return batch_int


# def get_batch(image_data, batch_size, batch_index):
#     nrof_examples = np.size(image_data, 0)
#     j = batch_index * batch_size % nrof_examples
#     if j + batch_size <= nrof_examples:
#         batch = image_data[j:j + batch_size, :, :, :]
#     else:
#         x1 = image_data[j:nrof_examples, :, :, :]
#         x2 = image_data[0:nrof_examples - j, :, :, :]
#         batch = np.vstack([x1, x2])
#     batch_float = batch.astype(np.float32)
#     return batch_float


# def get_triplet_batch(triplets, batch_index, batch_size):
#     ax, px, nx = triplets
#     a = get_batch(ax, int(batch_size / 3), batch_index)
#     p = get_batch(px, int(batch_size / 3), batch_index)
#     n = get_batch(nx, int(batch_size / 3), batch_index)
#     batch = np.vstack([a, p, n])
#     return batch
