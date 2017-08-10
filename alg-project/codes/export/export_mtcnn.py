import os
import sys
import argparse
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder

from detection import mtcnn


def convert_model_from_caffe_to_tensorflow(caffe_model_dir, output_dir):
    print('Converting mtcnn model from caffe to tensorflow')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            _, _, _ = mtcnn.create_mtcnn(sess, caffe_model_dir)
            saver = tf.train.Saver()
            save_path = saver.save(sess, os.path.join(output_dir, 'model.ckpt'))
            print('save model to  %s' % save_path)


def export_serving_model(model_dir, export_path, export_version):
    print('Begin export mtcnn model to %s' % export_path)
    export_path = os.path.join(
        tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(export_version)))
    __export_serving_mtcnn_pnet(model_dir, export_path, export_version)
    __export_serving_mtcnn_rnet(model_dir, export_path, export_version)
    __export_serving_mtcnn_onet(model_dir, export_path, export_version)

    # Export model graph
    model_graph_path = bytes.decode(os.path.join(export_path, tf.compat.as_bytes('graph')))
    print('Exporting model graph to %s' % model_graph_path)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print('Loading model from directory: %s' % model_dir)
            _, _, _ = mtcnn.create_mtcnn(sess, model_dir)
            train_writer = tf.summary.FileWriter(model_graph_path, graph=sess.graph)
            train_writer.close()
            print('Done model graph export!')
    print('Done export!')


def __export_serving_mtcnn_pnet(model_dir, export_path, export_version):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            print('Loading model from directory: %s' % model_dir)
            _, _, _ = mtcnn.create_mtcnn(sess, model_dir)
            export_path = os.path.join(
                tf.compat.as_bytes(export_path),
                tf.compat.as_bytes('pnet'))
            # print('Exporting trained model to %s' % export_path)
            print('Exporting Proposal Net model to: %s' % export_path)

            builder = saved_model_builder.SavedModelBuilder(export_path)

            pnet_input = tf.get_default_graph().get_tensor_by_name('pnet/input:0')
            pnet_prob = tf.get_default_graph().get_tensor_by_name('pnet/prob1:0')
            pnet_biasadd = tf.get_default_graph().get_tensor_by_name('pnet/conv4-2/BiasAdd:0')

            pnet_info_inputs = tf.saved_model.utils.build_tensor_info(pnet_input)
            pnet_prob_info = tf.saved_model.utils.build_tensor_info(pnet_prob)
            pnet_biasadd_info = tf.saved_model.utils.build_tensor_info(pnet_biasadd)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input': pnet_info_inputs},
                    outputs={'prob': pnet_prob_info, 'biasadd': pnet_biasadd_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'mtcnn_pnet': prediction_signature
                },
                legacy_init_op=legacy_init_op)
            builder.save()
    print('Export Proposal Net serving model success, version %d' % export_version)


def __export_serving_mtcnn_rnet(model_dir, export_path, export_version):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            print('Loading model from directory: %s' % model_dir)
            _, _, _ = mtcnn.create_mtcnn(sess, model_dir)
            export_path = os.path.join(
                tf.compat.as_bytes(export_path),
                tf.compat.as_bytes('rnet'))
            # rnet_saver = tf.train.Saver(sharded=True)
            print('Exporting Refine Net model to: %s' % export_path)

            builder = saved_model_builder.SavedModelBuilder(export_path)

            rnet_input = tf.get_default_graph().get_tensor_by_name('rnet/input:0')
            rnet_prob = tf.get_default_graph().get_tensor_by_name('rnet/prob1:0')
            rnet_conv52 = tf.get_default_graph().get_tensor_by_name('rnet/conv5-2/conv5-2:0')

            rnet_info_inputs = tf.saved_model.utils.build_tensor_info(rnet_input)
            rnet_prob_info = tf.saved_model.utils.build_tensor_info(rnet_prob)
            rnet_conv52_info = tf.saved_model.utils.build_tensor_info(rnet_conv52)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input': rnet_info_inputs},
                    outputs={'prob': rnet_prob_info, 'conv52': rnet_conv52_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'mtcnn_rnet': prediction_signature
                },
                legacy_init_op=legacy_init_op)
            builder.save()
            # print('Done Refine Net export!')
            print('Export Refine Net serving model success, version %d' % export_version)


def __export_serving_mtcnn_onet(model_dir, export_path, export_version):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            print('Loading model from directory: %s' % model_dir)
            _, _, _ = mtcnn.create_mtcnn(sess, model_dir)
            export_path = os.path.join(
                tf.compat.as_bytes(export_path),
                tf.compat.as_bytes('onet'))
            # onet_saver = tf.train.Saver(sharded=True)
            print('Exporting Output Net model to: %s' % export_path)

            builder = saved_model_builder.SavedModelBuilder(export_path)

            onet_input = tf.get_default_graph().get_tensor_by_name('onet/input:0')
            onet_prob = tf.get_default_graph().get_tensor_by_name('onet/prob1:0')
            onet_conv62 = tf.get_default_graph().get_tensor_by_name('onet/conv6-2/conv6-2:0')
            onet_conv63 = tf.get_default_graph().get_tensor_by_name('onet/conv6-3/conv6-3:0')

            onet_info_inputs = tf.saved_model.utils.build_tensor_info(onet_input)
            onet_prob_info = tf.saved_model.utils.build_tensor_info(onet_prob)
            onet_conv62_info = tf.saved_model.utils.build_tensor_info(onet_conv62)
            onet_conv63_info = tf.saved_model.utils.build_tensor_info(onet_conv63)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input': onet_info_inputs},
                    outputs={'prob': onet_prob_info, 'conv62': onet_conv62_info, 'conv63': onet_conv63_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'mtcnn_onet': prediction_signature
                },
                legacy_init_op=legacy_init_op)
            builder.save()
            print('Export Refine Net serving model success, version %d' % export_version)


def export_binary_model(caffe_model_dir, export_path):
    print('Converting mtcnn model from caffe to binary')
    if not os.path.isdir(export_path):
        os.makedirs(export_path)
    model_file = os.path.join(export_path, 'mtcnn.pb')
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            _, _, _ = mtcnn.create_mtcnn(sess, caffe_model_dir)
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names=['pnet/prob1',
                                                                                               'pnet/conv4-2/BiasAdd',
                                                                                               'rnet/prob1',
                                                                                               'rnet/conv5-2/conv5-2',
                                                                                               'onet/prob1',
                                                                                               'onet/conv6-2/conv6-2',
                                                                                               'onet/conv6-3/conv6-3'])
            with tf.gfile.FastGFile(model_file, mode='wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('save model to  %s' % model_file)


def main(args):
    if args.type == 'convert':
        convert_model_from_caffe_to_tensorflow(args.model_dir, args.export_path)
    elif args.type == 'binary':
        export_binary_model(args.model_dir, args.export_path)
    elif args.type == 'serving':
        export_serving_model(args.model_dir, args.export_path, args.export_version)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str,
                        choices=['convert', 'binary', 'serving'],
                        help='Which type for model to export')
    parser.add_argument('--model_dir', type=str,
                        help='Directory containing caffe model(.npy)')
    parser.add_argument('--export_path', type=str,
                        help='Directory for model to export')
    parser.add_argument('--export_version', type=int,
                        help='version number of the export model.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
