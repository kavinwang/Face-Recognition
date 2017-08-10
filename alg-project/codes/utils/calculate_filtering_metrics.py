# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
from utils import data_utils
import sys
import time
import h5py
import math
from tensorflow.python.framework import ops
from utils import image_utils
from embedding.facenet import TripletLossNetwork
from utils import ckpt_revision_utils
from utils import db_utils


def main(args):
    dataset = data_utils.get_dataset(args.dataset_dir)
    infos = _get_facenet_model_info(args.model_version)
    with tf.Graph().as_default():

        # Get a list of image paths and their labels
        image_list, label_list = data_utils.get_image_paths_and_labels(dataset)
        nrof_images = len(image_list)
        image_indices = range(nrof_images)
        image_indices = [i for i in image_indices]

        image_batch, label_batch = read_and_augment_data(image_list, image_indices, args.image_size, args.batch_size,
                                                         None, False, False, False, nrof_preprocess_threads=4,
                                                         shuffle=False)

        with tf.Session() as sess:
            # Load the model
            print('Loading model from directory: %s' % args.model_dir)
            ckpt_file, meta_file = ckpt_revision_utils.get_ckpt_and_metagraph(args.model_dir)
            print('Checkpoint file: %s' % ckpt_file)
            TripletLossNetwork(model_def=infos['model_def'], image_size=int(infos['image_size']),
                               embedding_size=int(infos['embedding_size']), is_training=False, is_classifier=True,
                               image_batch=image_batch)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)

            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            embedding_size = int(embeddings.get_shape()[1])
            nrof_batches = int(math.ceil(nrof_images / args.batch_size))
            nrof_classes = len(dataset)
            label_array = np.array(label_list)
            class_names = [cls.name for cls in dataset]
            nrof_examples_per_class = [len(cls.image_paths) for cls in dataset]
            class_variance = np.zeros((nrof_classes,))
            class_center = np.zeros((nrof_classes, embedding_size))
            distance_to_center = np.ones((len(label_list),)) * np.NaN
            emb_array = np.zeros((0, embedding_size))
            idx_array = np.zeros((0,), dtype=np.int32)
            lab_array = np.zeros((0,), dtype=np.int32)
            index_arr = np.append(0, np.cumsum(nrof_examples_per_class))
            for i in range(nrof_batches):
                t = time.time()
                emb, idx = sess.run([embeddings, label_batch])
                emb_array = np.append(emb_array, emb, axis=0)
                idx_array = np.append(idx_array, idx, axis=0)
                lab_array = np.append(lab_array, label_array[idx], axis=0)
                for cls in set(lab_array):
                    cls_idx = np.where(lab_array == cls)[0]
                    if cls_idx.shape[0] == nrof_examples_per_class[cls]:
                        # We have calculated all the embeddings for this class
                        i2 = np.argsort(idx_array[cls_idx])
                        emb_class = emb_array[cls_idx, :]
                        emb_sort = emb_class[i2, :]
                        center = np.mean(emb_sort, axis=0)
                        diffs = emb_sort - center
                        dists_sqr = np.sum(np.square(diffs), axis=1)
                        class_variance[cls] = np.mean(dists_sqr)
                        class_center[cls, :] = center
                        distance_to_center[index_arr[cls]:index_arr[cls + 1]] = np.sqrt(dists_sqr)
                        emb_array = np.delete(emb_array, cls_idx, axis=0)
                        idx_array = np.delete(idx_array, cls_idx, axis=0)
                        lab_array = np.delete(lab_array, cls_idx, axis=0)

                print('Batch [%d | %d] in %.3f seconds' % (i, nrof_batches, time.time() - t))

        print('Writing filtering data to %s' % args.data_file_name)
        mdict = {'class_names': class_names, 'image_list': image_list, 'label_list': label_list,
                 'distance_to_center': distance_to_center}
        with h5py.File(args.data_file_name, 'w') as f:
            for key, value in mdict.items():
                value = [i.encode('utf-8') for i in value if isinstance(i, str)]
                f.create_dataset(key, data=value)


def read_and_augment_data(image_list, label_list, image_size, batch_size, max_nrof_epochs,
                          random_crop, random_flip, random_rotate, nrof_preprocess_threads, shuffle=True):
    images = ops.convert_to_tensor(image_list, dtype=tf.string)
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=max_nrof_epochs, shuffle=shuffle)

    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
        image, label = data_utils.read_images_from_disk(input_queue)
        if random_rotate:
            image = tf.py_func(image_utils.random_rotate_image, [image], tf.uint8)
        if random_crop:
            image = tf.random_crop(image, [image_size, image_size, 3])
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        if random_flip:
            image = tf.image.random_flip_left_right(image)
        # pylint: disable=no-member
        image.set_shape((image_size, image_size, 3))
        image = tf.image.per_image_standardization(image)
        images_and_labels.append([image, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels, batch_size=batch_size,
        capacity=4 * nrof_preprocess_threads * batch_size,
        allow_smaller_final_batch=True)

    return image_batch, label_batch


def _get_facenet_model_info(version):
    conn = db_utils.open_connection()
    infos = db_utils.get_model_info(conn, 'facenet', version)[0]
    db_utils.close_connection(conn)
    infos = infos.split('|')
    infosMap = dict()
    for info in infos:
        arr = info.split(':')
        infosMap[arr[0]] = arr[1]
    return infosMap


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str,
                        help='Path to the directory containing aligned dataset.')
    parser.add_argument('--model_dir', type=str,
                        help='Directory containing the meta_file and ckpt_file')
    parser.add_argument('--model_version', type=int,
                        help='version number of the model')
    parser.add_argument('--data_file_name', type=str,
                        help='The name of the file to store filtering data in.')
    parser.add_argument('--image_size', type=int,
                        help='Image size.', default=160)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
