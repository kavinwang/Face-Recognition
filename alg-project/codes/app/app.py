import numpy as np
import cv2
import tensorflow as tf
from detection import mtcnn
from classifier import csair_classifier
import argparse
import sys
from utils import db_utils


def main(args):
    infos = _get_classifier_model_info(args.model_version)
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = mtcnn.create_mtcnn(sess, args.caffe_model_dir)
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            recognize = csair_classifier.create_classifier(sess, model_def=infos['model_def'],
                                                           image_size=int(infos['image_size']),
                                                           embedding_size=int(infos['embedding_size']),
                                                           nrof_classes=int(infos['nrof_classes']),
                                                           ckpt_dir=args.ckpt_dir)
    conn = db_utils.open_connection()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        bounding_boxes, points = mtcnn.detect_face(frame, 20, pnet, rnet, onet, args.threshold, args.factor)
        if len(bounding_boxes) > 0:
            for i in range(len(bounding_boxes)):
                box = bounding_boxes[i].astype(int)
                # mark = np.reshape(points[:, i].astype(int), (2, 5)).T
                crop = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
                crop = np.expand_dims(crop, 0)
                value, index = csair_classifier.classify(crop, recognize)

                font = cv2.FONT_HERSHEY_TRIPLEX
                name = db_utils.get_candidate_info(conn, int(index[0][0]))[0]
                text = 'person: ' + name + ' probability: ' + str(value[0][0])
                # print('text: ', text)
                cv2.putText(frame, text, (box[0], box[1]), font, 0.42, (255, 255, 0))
                # for p in mark:
                #     cv2.circle(frame, (p[0], p[1]), 3, (0, 0, 255))

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    db_utils.close_connection(conn)


def _get_classifier_model_info(version):
    conn = db_utils.open_connection()
    infos = db_utils.get_model_info(conn, 'classifier', version)[0]
    db_utils.close_connection(conn)
    infos = infos.split('|')
    infosMap = dict()
    for info in infos:
        arr = info.split(':')
        infosMap[arr[0]] = arr[1]
    return infosMap


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_version', type=int, help='version number of the model')
    parser.add_argument('--caffe_model_dir', type=str, help='Directory with caffe mtcnn models.')
    parser.add_argument('--ckpt_dir', type=str, help='whole network checkpoint directory.')
    parser.add_argument('--threshold', type=list, help='three steps\'s threshold.', default=[0.6, 0.7, 0.7])
    parser.add_argument('--factor', type=float, help='scale factor.', default=0.709)
    # parser.add_argument('--minsize', type=int, help='minimum size of face.', default=20)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
