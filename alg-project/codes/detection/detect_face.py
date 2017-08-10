"""
Performs face alignment and stores face thumbnails in the output directory.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import random
import numpy as np
from scipy import misc
import tensorflow as tf
from detection import mtcnn
from utils import data_utils
from utils import image_utils


def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset = data_utils.get_dataset(args.input_dir, with_dateset_id=False)

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = mtcnn.create_mtcnn(sess, args.caffe_model_dir)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = image_utils.to_rgb(img)
                        img = img[:, :, 0:3]

                        bounding_boxes, _ = mtcnn.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                              factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces > 0:
                            # retain only the the "good" face
                            bounding_boxes = [i for i in bounding_boxes if i[4] > args.faceprob]
                            if len(bounding_boxes) == 0:
                                print('image "%s" has find an unsure face' % image_path)
                                continue
                            # retain only the face that large than args.size
                            bounding_boxes = np.array(
                                [i for i in bounding_boxes if min(abs(i[0] - i[2]), abs(i[1] - i[3])) > args.facesize])
                            if len(bounding_boxes) == 0:
                                print('Face from image "%s" is too small to find' % image_path)
                                continue
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces > 1:
                                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                img_center = img_size / 2
                                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                     (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                index = np.argmax(
                                    bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                det = det[index, :]
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                            bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                            bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
                            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                            nrof_successfully_aligned += 1
                            misc.imsave(output_filename, cropped)
                            text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--caffe_model_dir', type=str, help='Directory with caffe mtcnn models.')
    parser.add_argument('--input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('--output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--margin', type=int, default=0,
                        help='Margin for the crop around the bounding box (height, width) in pixels.')
    parser.add_argument('--faceprob', type=float, default=0,
                        help='Save the face when the probability it is classified as a face above faceprob.')
    parser.add_argument('--facesize', type=float, default=80,
                        help='Save the face only the size of the window large than facesize.')
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float, default=1.0,
                        help='Upper bound on the amount of GPU memory that will be used by the process.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
