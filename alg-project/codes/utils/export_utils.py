import os
import sys
import argparse
import tensorflow as tf
from embedding.facenet import TripletLossNetwork
from detection import mtcnn
from tensorflow.python.saved_model import builder as saved_model_builder
from utils import ckpt_revision_utils
from utils import db_utils
from classifier.classifierNet import ClassifierNet


def main(args):
    if args.model_type == 'facenet':
        export_path = os.path.join(args.export_path, 'facenet')
        export_facenet(args.model_dir, args.model_version, export_path, args.export_version)
    elif args.model_type == 'classifier':
        export_path = os.path.join(args.export_path, 'classifier')
        export_classifier(args.model_dir, args.model_version, export_path, args.export_version)


def export_facenet(model_dir, model_version, export_path, export_version):
    infos = _get_facenet_model_info(model_version)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            print('Loading model from directory: %s' % model_dir)
            ckpt_file, meta_file = ckpt_revision_utils.get_ckpt_and_metagraph(model_dir)
            print('Checkpoint file: %s' % ckpt_file)

            TripletLossNetwork(model_def=infos['model_def'], image_size=int(infos['image_size']),
                               embedding_size=int(infos['embedding_size']), is_training=False)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)

            # Export model
            model_export_path = os.path.join(
                tf.compat.as_bytes(export_path),
                tf.compat.as_bytes(str(export_version)))
            print('Exporting trained model to: %s' % model_export_path)

            builder = saved_model_builder.SavedModelBuilder(model_export_path)

            images_placeholder = tf.get_default_graph().get_tensor_by_name('inputs:0')
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')

            images_input = tf.saved_model.utils.build_tensor_info(images_placeholder)
            embeddings_info = tf.saved_model.utils.build_tensor_info(embeddings)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input': images_input},
                    outputs={'embeddings': embeddings_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'facenet': prediction_signature
                },
                legacy_init_op=legacy_init_op)
            builder.save()
            print('Export Facenet serving model success, version %d' % export_version)

            # Export model graph
            graph_export_path = os.path.join(export_path, str(export_version), 'graph')
            print('Exporting Facenet model graph')
            train_writer = tf.summary.FileWriter(graph_export_path, graph=sess.graph)
            train_writer.close()
            print('Export Facenet model graph successfully!')
    print('Done exporting!')


def export_classifier(model_dir, model_version, export_path, export_version):
    infos = _get_classifier_model_info(model_version)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            print('Loading model from directory: %s' % model_dir)
            ckpt_file, meta_file = ckpt_revision_utils.get_ckpt_and_metagraph(model_dir)
            print('Checkpoint file: %s' % ckpt_file)

            print('Building ClassifierNet')
            ClassifierNet(model_def=infos['model_def'], image_size=int(infos['image_size']),
                          embedding_size=int(infos['embedding_size']), nrof_classes=int(infos['nrof_classes']),
                          is_training=False)
            print('Successfully Built ClassifierNet')

            print('Restoring parameters from: %s' % ckpt_file)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)
            # Export model
            model_export_path = os.path.join(
                tf.compat.as_bytes(export_path),
                tf.compat.as_bytes(str(export_version)))
            print('Exporting trained model to: %s' % model_export_path)

            # input_x = tf.get_default_graph().get_tensor_by_name('input_x:0')
            # scores = tf.get_default_graph().get_tensor_by_name('scores:0')
            # predict = tf.get_default_graph().get_tensor_by_name('predict:0')
            input_x = tf.get_default_graph().get_tensor_by_name('input:0')
            logits = tf.get_default_graph().get_tensor_by_name('logits:0')
            top_k = tf.get_default_graph().get_tensor_by_name('top_k:0')

            builder = saved_model_builder.SavedModelBuilder(model_export_path)

            images_input = tf.saved_model.utils.build_tensor_info(input_x)
            predict = tf.saved_model.utils.build_tensor_info(logits)
            top_k_info = tf.saved_model.utils.build_tensor_info(top_k)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input': images_input},
                    outputs={'logits': predict, 'top_k_res': top_k_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'classifier': prediction_signature
                },
                legacy_init_op=legacy_init_op)
            builder.save()
            print('Export Whole Network serving model success, version %d' % export_version)

            # saver = tf.train.Saver(sharded=True)
            # model_exporter = exporter.Exporter(saver)
            # model_exporter.init(
            #     sess.graph.as_graph_def(),
            #     named_graph_signatures={
            #         'inputs': exporter.generic_signature({'input': input_x}),
            #         'outputs': exporter.generic_signature({'scores': scores,
            #                                                'predict': predict})})
            # model_exporter.export(export_path, tf.constant(export_version), sess)

            # Export model graph
            graph_export_path = os.path.join(export_path, str(export_version), 'graph')
            print('Exporting Facenet model graph')
            train_writer = tf.summary.FileWriter(graph_export_path, graph=sess.graph)
            train_writer.close()
            print('Export Whole Network model graph successfully!')
        print('Done exporting!')


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

    parser.add_argument('--model_type', type=str,
                        choices=['facenet', 'classifier'],
                        help='Which model to export')
    parser.add_argument('--model_version', type=int,
                        help='version number of the model')
    parser.add_argument('--model_dir', type=str,
                        help='Directory containing the meta_file and ckpt_file')
    parser.add_argument('--export_path', type=str,
                        help='Directory for model to export')
    parser.add_argument('--export_version', type=int,
                        help='version number of the export model.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
