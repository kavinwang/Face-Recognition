from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import codecs
import argparse
import numpy as np
from subprocess import Popen, PIPE
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def store_revision_info(src_path, output_dir, arg_string):
    # Get git hash
    gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_hash = stdout.strip()

    # Get local changes
    gitproc = Popen(['git', 'diff', 'HEAD'], stdout=PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_diff = stdout.strip()

    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
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


def get_ckpt_and_metagraph(model_dir):
    ckpt_file = tf.train.latest_checkpoint(model_dir)
    meta_files = [s for s in os.listdir(model_dir) if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory %s' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory %s' % model_dir)
    meta_file = meta_files[0]
    meta_file = os.path.join(model_dir, meta_file)
    return ckpt_file, meta_file


def print_tensors_in_checkpoint_file(ckpt_file, all_tensors=True, tensor_name=[], output_file=None):
    """Prints tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:
      file_name: Name of the checkpoint file.
      tensor_name: Name of the tensor in the checkpoint file to print.
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
        if output_file is not None:
            output_f = codecs.open(output_file, 'w+', 'utf-8')
        if all_tensors:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                if output_file is not None:
                    print('tensor_name: ', key, ', shape: ', np.shape(reader.get_tensor(key)), file=output_f)
                    # print(reader.get_tensor(key))
                else:
                    print('tensor_name: ', key, ', shape: ', np.shape(reader.get_tensor(key)))
        elif not tensor_name:
            if output_file is not None:
                print(reader.debug_string().decode('utf-8'), file=output_f)
            else:
                print(reader.debug_string().decode('utf-8'))
        else:
            if output_file is not None:
                print('tensor_name: ', tensor_name, file=output_f)
                print(reader.get_tensor(tensor_name), file=output_f)
            else:
                print('tensor_name: ', tensor_name)
                print(reader.get_tensor(tensor_name))
        if output_file is not None:
            output_f.close()
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if 'corrupted compressed block contents' in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")


def main(args):
    if args.action == 'check_ckpt':
        print_tensors_in_checkpoint_file(args.ckpt_file, output_file=args.output_file)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--action', type=str,
                        choices=['check_ckpt'],
                        help='Which action to do.')
    parser.add_argument('--ckpt_file', type=str,
                        help='Directory checkpoint file to check.')
    parser.add_argument('--output_file', type=str,
                        help='File to write infos')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
