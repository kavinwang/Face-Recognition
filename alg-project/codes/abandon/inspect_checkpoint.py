# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple script for inspect checkpoint files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import pywrap_tensorflow


def print_tensors_in_checkpoint_file(file_name, all_tensors=True, tensor_name=[]):
    """Prints tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:
      file_name: Name of the checkpoint file.
      tensor_name: Name of the tensor in the checkpoint file to print.
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        f = open("/home/csairmind/Desktop/inspect_checkpoint.txt", 'w+')
        if all_tensors:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                print("tensor_name: ", key, ", shape: ", np.shape(reader.get_tensor(key)), file=f)
                # print(reader.get_tensor(key), file=f)
        elif not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            print(reader.get_tensor(tensor_name))
        f.close()
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")


if __name__ == "__main__":
    file_name = '/run/media/csairmind/DataDevices/projects/face-recognition-project/v3/runtime/20170717-160610/models/model-20170717-160610.ckpt-1488'
    file_name = '/run/media/csairmind/DataDevices/projects/face-recognition-project/v3/runtime/classifier/20170726-102701/models/model-20170726-102701.ckpt-200'
    print_tensors_in_checkpoint_file(file_name)
