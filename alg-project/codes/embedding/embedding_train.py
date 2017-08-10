"""
Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import itertools
import argparse

import numpy as np
import tensorflow as tf
from embedding import facenet
from embedding.facenet import TripletLossNetwork
from embedding import embedding_eval
from utils import data_utils
from utils import ckpt_revision_utils
from utils import db_utils


def main(args):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    runtime_dir = os.path.join(os.path.expanduser(args.runtime_dir), subdir)
    log_dir = os.path.join(runtime_dir, "logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    model_dir = os.path.join(runtime_dir, "models")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    ckpt_revision_utils.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    train_set = data_utils.get_dataset(args.data_dir)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        pretrained_model = tf.train.latest_checkpoint(args.pretrained_model)
        print('Pre-trained model: %s' % os.path.expanduser(pretrained_model))

    if args.eval_dir:
        print('Evaluate data directory: %s' % args.eval_dir)
        # Read the file containing the pairs used for testing
        pairs = embedding_eval.read_pairs(os.path.expanduser(args.eval_pairs))
        # Get the paths for the corresponding images
        eval_paths, actual_issame = embedding_eval.get_paths(os.path.expanduser(args.eval_dir), pairs,
                                                             args.eval_file_ext)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)

        network = TripletLossNetwork(
            model_def=args.model_def, image_size=args.image_size, embedding_size=args.embedding_size, alpha=args.alpha,
            epoch_size=args.epoch_size, batch_size=args.batch_size, keep_probability=args.keep_probability,
            weight_decay=args.weight_decay, learning_rate_decay_epochs=args.learning_rate_decay_epochs,
            learning_rate_decay_factor=args.learning_rate_decay_factor,
            random_crop=args.random_crop, random_flip=args.random_flip, random_contrast=args.random_contrast,
            random_rotate=args.random_rotate, nrof_preprocess_threads=args.nrof_preprocess_threads)

        persistent_model_info(args.version, args.model_def, args.image_size, args.embedding_size)

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(network.total_loss, network.global_step, args.optimizer,
                                 network.learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={network.phase_train_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={network.phase_train_placeholder: True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if args.pretrained_model:
                pretrained_model = tf.train.latest_checkpoint(args.pretrained_model)
                print('Restoring the latest pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(network.global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, train_set, epoch, network.image_paths_placeholder, network.labels_placeholder,
                      network.labels_batch, network.batch_size_placeholder, network.learning_rate_placeholder,
                      network.phase_train_placeholder, network.enqueue_op, network.input_queue, network.global_step,
                      network.embeddings, network.total_loss, train_op, summary_op, summary_writer,
                      args.learning_rate_schedule_file, args.embedding_size, network.anchor, network.positive,
                      network.negative, network.triplet_loss)

                # Save variables and the metagraph if it doesn't exist already
                ckpt_revision_utils.save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # Evaluate
                if args.eval_dir:
                    evaluate(sess, eval_paths, network.embeddings, network.labels_batch,
                             network.image_paths_placeholder, network.labels_placeholder,
                             network.batch_size_placeholder, network.learning_rate_placeholder,
                             network.phase_train_placeholder, network.enqueue_op,
                             actual_issame, args.batch_size,
                             args.eval_nrof_folds, log_dir, step, summary_writer, args.embedding_size)

    sess.close()
    return model_dir


def train(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue,
          global_step, embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.images_per_person)

        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = args.people_per_batch * args.images_per_person
        labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))

        sess.run(enqueue_op, feed_dict={image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
        emb_array = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        for i in range(nrof_batches):
            batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                       learning_rate_placeholder: lr,
                                                                       phase_train_placeholder: True})
            emb_array[lab, :] = emb
        print('%.3f' % (time.time() - start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
                                                                    image_paths, args.people_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
              (nrof_random_negs, nrof_triplets, selection_time))

        # Perform training on the selected triplets
        nrof_batches = int(np.ceil(nrof_triplets * 3 / args.batch_size))
        triplet_paths = list(itertools.chain(*triplets))
        assert (len(triplet_paths) % 3 == 0), "triplet_paths must be divided by 3"
        labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr,
                         phase_train_placeholder: True}
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch],
                                              feed_dict=feed_dict)
            emb_array[lab, :] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            current_time = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            # print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
            #       (epoch, batch_number + 1, nrof_batches, duration, err))
            print('%s: Epoch [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (current_time, epoch, i, nrof_batches, duration, err))
            batch_number += 1
            i += 1
            train_time += duration

        # Add validation loss and accuracy to summary
        summary = tf.Summary()
        # pylint: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return step


def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """
    Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in range(j, nrof_images):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame,
             batch_size,
             nrof_folds, log_dir, step, summary_writer, embedding_size):
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

    _, _, accuracy, val, val_std, far = embedding_eval.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)

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
    with open(os.path.join(log_dir, 'eval_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


def persistent_model_info(version, model_def, image_size, embedding_size):
    conn = db_utils.open_connection()
    db_utils.clean_model_info(conn, 'facenet', version)
    params = 'model_def:%s|image_size:%s|embedding_size:%s' % (model_def, image_size, embedding_size)
    db_utils.insert_model_info(conn, 'facenet', version, params)
    db_utils.close_connection(conn)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int,
                        help='The model version', default=0)
    parser.add_argument('--runtime_dir', type=str,
                        help='Directory where to write event logs and trained models and checkpoints')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.inception_resnet_v2')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=1000)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--people_per_batch', type=int,
                        help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
                        help='Number of images per person.', default=40)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--alpha', type=float,
                        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
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
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
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
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='../../properties/learning_rate_train_embedding.txt')
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of threads to preprocess train images.',
                        default='8')

    # Parameters for validation
    parser.add_argument('--eval_pairs', type=str,
                        help='The file containing the pairs to use for validation.',
                        default='../../properties/evaluate_pairs.txt')
    parser.add_argument('--eval_file_ext', type=str,
                        help='The file extension for the evaluate dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--eval_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='/sdb/workdatas/Face/LFW/crop/')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
