"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
import argparse
from embedding import embedding_train
import os
import sys
from utils import data_utils
from utils import image_utils
from embedding import facenet
from utils import ckpt_revision_utils
from datetime import datetime
from embedding import embedding_eval
from classifier.classifierNet import ClassifierNet
from utils import db_utils
from classifier import evaluation


def main(args):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    runtime_dir = os.path.join(os.path.expanduser(args.runtime_dir), subdir)

    # this folder is unused, Do we need it?
    classifier_log_dir = os.path.join(runtime_dir, "classifier", "logs")
    if not os.path.isdir(classifier_log_dir):
        os.makedirs(classifier_log_dir)
    classifier_model_dir = os.path.join(runtime_dir, "classifier", "models")
    if not os.path.isdir(classifier_model_dir):
        os.makedirs(classifier_model_dir)

    whole_log_dir = os.path.join(runtime_dir, "whole", "logs")
    if not os.path.isdir(whole_log_dir):
        os.makedirs(whole_log_dir)
    whole_model_dir = os.path.join(runtime_dir, "whole", "models")
    if not os.path.isdir(whole_model_dir):
        os.makedirs(whole_model_dir)

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    ckpt_revision_utils.store_revision_info(src_path, whole_log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    train_set = data_utils.get_dataset(args.data_dir, with_dateset_id=False)
    persistent_candidate_info(train_set)
    print('Complete save classifier_info into database')
    nrof_classes = len(train_set)
    print('Classifier Model directory: %s' % classifier_model_dir)
    print('Classifier Log directory: %s' % classifier_log_dir)

    print('Whole Model directory: %s' % whole_model_dir)
    print('Whole Log directory: %s' % whole_log_dir)
    if args.pretrained_classifier:
        pretrained_classifier = tf.train.latest_checkpoint(args.pretrained_classifier)
        print('Pre-trained model: %s' % os.path.expanduser(pretrained_classifier))

    print('Network pretrained model: %s' % tf.train.latest_checkpoint(args.emb_ckpt_dir))

    if args.eval_dir:
        print('Evaluate data directory: %s' % args.eval_dir)
        # Read the file containing the pairs used for testing
        # pairs = embedding_eval.read_pairs(os.path.expanduser(args.eval_pairs))
        # Get the paths for the corresponding images
        eval_paths, actual_issame = evaluation.generate_pairs(os.path.expanduser(args.eval_dir),
                                                              args.eval_same_num,
                                                              args.eval_diff_num)
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        network = ClassifierNet(model_def=args.model_def, image_size=args.image_size,
                                embedding_size=args.embedding_size, nrof_classes=nrof_classes, train_set=train_set,
                                batch_size=args.batch_size,
                                epoch_size=args.epoch_size, weight_decay=args.weight_decay, top_k=args.top_k,
                                center_loss_factor=args.center_loss_factor, center_loss_alfa=args.center_loss_alfa,
                                learning_rate_decay_epochs=args.learning_rate_decay_epochs,
                                learning_rate_decay_factor=args.learning_rate_decay_factor,
                                nrof_preprocess_threads=args.nrof_preprocess_threads,
                                random_crop=args.random_crop,
                                random_flip=args.random_flip, random_contrast=args.random_contrast,
                                random_rotate=args.random_rotate)

        persistent_model_info(args.version, args.model_def, args.image_size, args.embedding_size, nrof_classes)

        train_op = facenet.train(network.total_loss, network.global_step, args.optimizer,
                                 network.learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver
        saver_whole = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(whole_log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            all_vars = tf.trainable_variables()
            net_var = []
            new_var = []
            for k in all_vars:
                if k.name.startswith('InceptionResnetV2'):
                    net_var.append(k)
                else:
                    new_var.append(k)

            print('Restoring network parameters from: %s' % tf.train.latest_checkpoint(args.emb_ckpt_dir))
            tf.train.Saver(net_var).restore(sess, tf.train.latest_checkpoint(args.emb_ckpt_dir))
            print('Restore successfully!')

            saver = tf.train.Saver(new_var, max_to_keep=3)

            if args.pretrained_classifier:
                pretrained_classifier = tf.train.latest_checkpoint(args.pretrained_classifier)
                print('Restoring pretrained model: %s' % pretrained_classifier)
                saver.restore(sess, pretrained_classifier)

            # Training and validation loop
            print('Running training')
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(network.global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, epoch, network.image_list, network.label_list, network.index_dequeue_op,
                      network.enqueue_op, network.image_paths_placeholder,
                      network.labels_placeholder,
                      network.learning_rate_placeholder, network.phase_train_placeholder,
                      network.batch_size_placeholder, network.global_step,
                      network.total_loss, train_op, summary_op, summary_writer, network.regularization_losses,
                      args.learning_rate_schedule_file)

                # Save variables and the metagraph if it doesn't exist already
                # save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
                classifier_ckpt_path = os.path.join(classifier_model_dir, 'model-%s.ckpt' % subdir)
                saver.save(sess, classifier_ckpt_path, global_step=step, write_meta_graph=False)
                save_variables_and_metagraph(sess, saver_whole, summary_writer, whole_model_dir, subdir, step)

                # # Evaluate on CSAIR
                if args.eval_dir:
                    evaluate(sess, eval_paths, network.embeddings, network.label_batch,
                             network.image_paths_placeholder, network.labels_placeholder,
                             network.batch_size_placeholder, network.learning_rate_placeholder,
                             network.phase_train_placeholder, network.enqueue_op,
                             actual_issame, args.batch_size,
                             args.eval_nrof_folds, whole_log_dir, step, summary_writer, args.embedding_size)
    return whole_model_dir


def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = embedding_train.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True,
                     batch_size_placeholder: args.batch_size}
        if batch_number % 100 == 0:
            err, _, step, reg_loss, summary_str = sess.run(
                [loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses],
                                              feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (epoch, batch_number + 1, args.epoch_size, duration, err, np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step


def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame,
             batch_size,
             nrof_folds, whole_log_dir, step, summary_writer, embedding_size):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on evaluate images: ', end='')

    nrof_images = len(actual_issame) * 2
    assert (len(image_paths) == nrof_images)
    # print(nrof_images)
    assert (nrof_images % 3 == 0), "the number of images must can be divided by 3"
    labels_array = np.reshape(np.arange(nrof_images), (-1, 3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        batch_size = min(nrof_images - i * batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                   learning_rate_placeholder: 0.0,
                                                                   phase_train_placeholder: False})
        emb_array[lab, :] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time() - start_time))

    assert (np.all(label_check_array == 1))

    _, _, accuracy, val, val_std, far = evaluation.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    eval_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='eval/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='eval/val_rate', simple_value=val)
    summary.value.add(tag='time/eval', simple_value=eval_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(whole_log_dir, 'eval_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))


def save_variables_and_metagraph(sess, saver, summary_writer, whole_model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(whole_model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(whole_model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(image_utils.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(image_utils.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set


def persistent_model_info(version, model_def, image_size, embedding_size, nrof_classes):
    conn = db_utils.open_connection()
    db_utils.clean_model_info(conn, 'classifier', version)
    params = 'model_def:%s|image_size:%s|embedding_size:%s|nrof_classes:%s' % (
        model_def, image_size, embedding_size, nrof_classes)
    db_utils.insert_model_info(conn, 'classifier', version, params)
    db_utils.close_connection(conn)


def persistent_candidate_info(train_set):
    conn = db_utils.open_connection()
    params = []
    db_utils.clean_model_info(conn, 'candidate', None)
    for i in range(len(train_set)):
        params.append((i, train_set[i].name))
    db_utils.insert_candidate_info(conn, params)
    db_utils.close_connection(conn)


def create_classifier(sess, model_def, image_size, embedding_size, nrof_classes, ckpt_dir):
    ClassifierNet(model_def=model_def, image_size=image_size, embedding_size=embedding_size, nrof_classes=nrof_classes,
                  is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
    input_x = tf.get_default_graph().get_tensor_by_name('input:0')
    value = tf.get_default_graph().get_tensor_by_name('top_k:0')
    index = tf.get_default_graph().get_tensor_by_name('top_k:1')

    recognize = lambda img: sess.run([value, index], feed_dict={input_x: img})

    return recognize


def classify(img, recognize):
    value, index = recognize(img)
    return value, index


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.',
                        default='/run/media/csairmind/e45852df-3e1b-4f5b-9e92-14026ad1f9db/workdatas/Face/CSAIR/crop')
    parser.add_argument('--emb_ckpt_dir', type=str,
                        help='Load a ckpt_file from the Embedding training network.',
                        default='/run/media/csairmind/db8d9379-90c2-4fdd-9369-d40e5f9a1d95/projects/face-recognition-project/v3/runtime/embedding/20170717-160610/models')
    parser.add_argument('--version', type=int,
                        help='The model version', default=0)
    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.',
                        action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--runtime_dir', type=str,
                        help='Directory where to write event logs and trained models and checkpoints')
    parser.add_argument('--pretrained_classifier', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing',
                        default=10)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.0)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--center_loss_factor', type=float,
                        help='Center loss factor factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
                        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.inception_resnet_v2')
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_contrast',
                        help='Performs random contrast change of training images.', action='store_true')
    parser.add_argument('--random_rotate',
                        help='Performs random rotation change of training images.', action='store_true')
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of threads to preprocess train images.',
                        default='8')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='../../properties/learning_rate_train_classifier.txt')
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=100)
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=1000)
    parser.add_argument('--top_k', type=int,
                        help='The number of result we attain.', default=5)

    # Parameters for validation on evaluate
    parser.add_argument('--eval_dir', type=str,
                        help='Path to the data directory containing evaluate face patches.', default='')
    parser.add_argument('--eval_same_num', type=int,
                        help='Number of photo from one persion for evaluate.', default=6)
    parser.add_argument('--eval_diff_num', type=int,
                        help='Number of photo from diffetent perople for evaluate.', default=4)
    parser.add_argument('--eval_batch_size', type=int,
                        help='Number of images to process in a batch in the evaluate set.', default=120)
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
