#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
#
"""unit tests of variable
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import math
import time

import numpy as np
import os
import six
import tempfile

from merlin_kv import tensorflow as mkv
from merlin_kv.utils.check_platform import is_macos, is_arm64

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)
except:
  pass


# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
def _type_converter(tf_type):
  mapper = {
      dtypes.int32: np.int32,
      dtypes.int64: np.int64,
      dtypes.float32: np.float,
      dtypes.float64: np.float64,
      dtypes.string: np.str,
      dtypes.half: np.float16,
      dtypes.int8: np.int8,
      dtypes.bool: np.bool,
  }
  return mapper[tf_type]


g_start = 0


def _create_dynamic_shape_continous_tensor(
    start=100000000,
    length=8192,
    dtype=np.int64,
):
  global g_start
  g_start = start

  def _func():
    global g_start
    tensor = np.arange(g_start, g_start + length, dtype=dtype)
    g_start += length
    return tensor

  return _func


def _get_devices():
  return ["/gpu:0" if test_util.is_gpu_available() else "/cpu:0"]


def _check_device(op, expexted_device="gpu"):
  return expexted_device.upper() in op.device


def data_fn(shape, maxval):
  return random_ops.random_uniform(shape, maxval=maxval, dtype=dtypes.int64)


def Murmur3Hash(key):

  def uint64_right_shift(key, bits=33):
    k = np.int64(key)
    k = np.right_shift(k, bits)
    k = k & 0x7FFFFFFF
    return np.int64(k)

  def uint64_xor(a, b):
    _a = np.int64(a)
    _b = np.int64(b)
    _a_abs = np.abs(a)
    _b_abs = np.abs(b)
    py_int = int(_a_abs) ^ int(_b_abs)
    return py_int

  k = np.int64(key)
  k ^= uint64_right_shift(k, 33)
  k = np.ulonglong(k)
  k *= 0xff51afd7ed558ccd
  xk = uint64_right_shift(k, 33)
  k = uint64_xor(k, xk)
  k = np.ulonglong(k)
  k *= 0xc4ceb9fe1a85ec53
  xk = uint64_right_shift(k, 33)
  k = uint64_xor(k, xk)
  return k


def ids_and_weights_2d(embed_dim=4):
  # Each row demonstrates a test case:
  #   Row 0: multiple valid ids, 1 invalid id, weighted mean
  #   Row 1: all ids are invalid (leaving no valid ids after pruning)
  #   Row 2: no ids to begin with
  #   Row 3: single id
  #   Row 4: all ids have <=0 weight
  indices = [[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [4, 0], [4, 1]]
  ids = [0, 1, -1, -1, 2, 0, 1]
  weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
  shape = [5, embed_dim]

  sparse_ids = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(ids, dtypes.int64),
      constant_op.constant(shape, dtypes.int64),
  )

  sparse_weights = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(weights, dtypes.float32),
      constant_op.constant(shape, dtypes.int64),
  )

  return sparse_ids, sparse_weights


def ids_and_weights_3d(embed_dim=4):
  # Each (2-D) index demonstrates a test case:
  #   Index 0, 0: multiple valid ids, 1 invalid id, weighted mean
  #   Index 0, 1: all ids are invalid (leaving no valid ids after pruning)
  #   Index 0, 2: no ids to begin with
  #   Index 1, 0: single id
  #   Index 1, 1: all ids have <=0 weight
  #   Index 1, 2: no ids to begin with
  indices = [
      [0, 0, 0],
      [0, 0, 1],
      [0, 0, 2],
      [0, 1, 0],
      [1, 0, 0],
      [1, 1, 0],
      [1, 1, 1],
  ]
  ids = [0, 1, -1, -1, 2, 0, 1]
  weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
  shape = [2, 3, embed_dim]

  sparse_ids = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(ids, dtypes.int64),
      constant_op.constant(shape, dtypes.int64),
  )

  sparse_weights = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(weights, dtypes.float32),
      constant_op.constant(shape, dtypes.int64),
  )

  return sparse_ids, sparse_weights


def _get_meta_file(ckpt_dir):
  for fname in os.listdir(ckpt_dir):
    if fname.endswith(".meta"):
      return os.path.join(ckpt_dir, fname)
  else:
    raise ValueError("No meta file found in {}.".format(ckpt_dir))


default_config = config_pb2.ConfigProto(
    allow_soft_placement=True,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


@test_util.run_all_in_graph_and_eager_modes
class VariableTest(test.TestCase):

  def test_variable(self):
    id = 0
    dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
    kv_list = [[dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int32],
               [dtypes.int64, dtypes.half], [dtypes.int64, dtypes.int8],
               [dtypes.int64, dtypes.int64]]

    def _convert(v, t):
      return np.array(v).astype(_type_converter(t))

    for (key_dtype, value_dtype), dim in itertools.product(kv_list, dim_list):
      id += 1
      # Skip float16 tests if the platform is macOS arm64 architecture
      if is_macos() and is_arm64():
        if value_dtype == dtypes.half:
          continue
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        keys = constant_op.constant(
            np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
            key_dtype)
        values = constant_op.constant(
            _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
            value_dtype)
        table = mkv.get_variable('t1-' + str(id),
                                 key_dtype=key_dtype,
                                 value_dtype=value_dtype,
                                 initializer=np.array([-1]).astype(
                                     _type_converter(value_dtype)),
                                 dim=dim)
        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(table.upsert(keys, values))
        self.assertAllEqual(4, self.evaluate(table.size()))

        remove_keys = constant_op.constant(_convert([1, 5], key_dtype),
                                           key_dtype)
        self.evaluate(table.remove(remove_keys))
        self.assertAllEqual(3, self.evaluate(table.size()))

        remove_keys = constant_op.constant(_convert([0, 1, 5], key_dtype),
                                           key_dtype)
        output = table.lookup(remove_keys)
        self.assertAllEqual([3, dim], output.get_shape())

        result = self.evaluate(output)
        self.assertAllEqual(
            _convert([[0] * dim, [-1] * dim, [-1] * dim], value_dtype),
            _convert(result, value_dtype))

        exported_keys, exported_values = table.export()

        # exported data is in the order of the internal map, i.e. undefined
        sorted_keys = np.sort(self.evaluate(exported_keys))
        sorted_values = np.sort(self.evaluate(exported_values), axis=0)
        self.assertAllEqual(_convert([0, 2, 3], key_dtype),
                            _convert(sorted_keys, key_dtype))
        self.assertAllEqual(
            _convert([[0] * dim, [2] * dim, [3] * dim], value_dtype),
            _convert(sorted_values, value_dtype))

        del table

  def test_variable_find_with_exists_and_accum(self):
    id = 0
    dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
    kv_list = [[dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int32],
               [dtypes.int64, dtypes.half], [dtypes.int64, dtypes.int8]]

    def _convert(v, t):
      return np.array(v).astype(_type_converter(t))

    for (key_dtype, value_dtype), dim in itertools.product(kv_list, dim_list):
      id += 1
      # Skip float16 tests if the platform is macOS arm64 archtecture
      if is_macos() and is_arm64():
        if value_dtype == dtypes.half:
          continue
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        base_keys = constant_op.constant(
            np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
            key_dtype)
        base_values = constant_op.constant(
            _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
            value_dtype)

        simulate_other_process_add_keys = constant_op.constant(
            np.array([100]).astype(_type_converter(key_dtype)), key_dtype)
        simulate_other_process_add_vals = constant_op.constant(
            _convert([
                [99] * dim,
            ], value_dtype), value_dtype)

        simulate_other_process_remove_keys = constant_op.constant(
            np.array([1]).astype(_type_converter(key_dtype)), key_dtype)
        accum_keys = constant_op.constant(
            np.array([0, 1, 100, 3]).astype(_type_converter(key_dtype)),
            key_dtype)
        old_values = constant_op.constant(
            _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
            value_dtype)
        new_values = constant_op.constant(
            _convert([[10] * dim, [11] * dim, [100] * dim, [13] * dim],
                     value_dtype), value_dtype)
        exported_exists = constant_op.constant([True, True, False, True],
                                               dtype=dtypes.bool)

        table = mkv.get_variable('taccum1-' + str(id),
                                 key_dtype=key_dtype,
                                 value_dtype=value_dtype,
                                 initializer=np.array([-1]).astype(
                                     _type_converter(value_dtype)),
                                 dim=dim)

        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(table.upsert(base_keys, base_values))
        _, exists = table.lookup(accum_keys, return_exists=True)
        self.assertAllEqual(self.evaluate(exported_exists),
                            self.evaluate(exists))
        # Simulate multi-process situation that other process operated table,
        # between lookup and accum in this process.
        self.evaluate(
            table.upsert(simulate_other_process_add_keys,
                         simulate_other_process_add_vals))
        self.evaluate(table.remove(simulate_other_process_remove_keys))
        self.assertAllEqual(4, self.evaluate(table.size()))
        self.evaluate(
            table.accum(accum_keys, old_values, new_values, exported_exists))

        exported_keys, exported_values = table.export()

        # exported data is in the order of the internal map, i.e. undefined
        sorted_keys = np.sort(self.evaluate(exported_keys), axis=0)
        sorted_values = np.sort(self.evaluate(exported_values), axis=0)
        self.assertAllEqual(
            np.sort(_convert([0, 2, 3, 100], key_dtype), axis=0),
            _convert(sorted_keys, key_dtype))
        self.assertAllEqual(
            _convert([[2] * dim, [10] * dim, [13] * dim, [99] * dim],
                     value_dtype), _convert(sorted_values, value_dtype))

        del table

  def test_variable_initializer(self):
    id = 0
    for initializer, target_mean, target_stddev in [
        (-1.0, -1.0, 0.0),
        (init_ops.random_normal_initializer(0.0, 0.01, seed=2), 0.0, 0.01),
    ]:
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()):
        id += 1
        keys = constant_op.constant(list(range(2**17)), dtypes.int64)
        table = mkv.get_variable(
            "t1" + str(id),
            key_dtype=dtypes.int64,
            value_dtype=dtypes.float32,
            initializer=initializer,
            dim=10,
        )
        vals_op = table.lookup(keys)
        mean = self.evaluate(math_ops.reduce_mean(vals_op))
        stddev = self.evaluate(math_ops.reduce_std(vals_op))
        rtol = 2e-5
        atol = rtol
        self.assertAllClose(target_mean, mean, rtol, atol)
        self.assertAllClose(target_stddev, stddev, rtol, atol)

  def test_save_restore(self):
    if context.executing_eagerly():
      self.skipTest('skip eager test when using legacy Saver.')
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(config=default_config, graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0.0], [1.0], [2.0]], dtypes.float32)
      table = mkv.Variable(
          key_dtype=dtypes.int64,
          value_dtype=dtypes.float32,
          initializer=-1.0,
          name="t1",
          dim=1,
      )

      save = saver.Saver(var_list=[v0, v1, table])
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

      del table

    with self.session(config=default_config, graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      table = mkv.Variable(
          name="t1",
          key_dtype=dtypes.int64,
          value_dtype=dtypes.float32,
          initializer=-1.0,
          dim=1,
          checkpoint=True,
      )
      self.evaluate(
          table.upsert(
              constant_op.constant([0, 1], dtypes.int64),
              constant_op.constant([[12.0], [24.0]], dtypes.float32),
          ))
      size_op = table.size()
      self.assertAllEqual(2, self.evaluate(size_op))

      save = saver.Saver(var_list=[v0, v1, table])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual([10.0], self.evaluate(v0))
      self.assertEqual([20.0], self.evaluate(v1))

      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([5, 0, 1, 2, 6], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([[-1.0], [0.0], [1.0], [2.0], [-1.0]],
                          self.evaluate(output))

      del table

  def test_save_restore_only_table(self):
    if context.executing_eagerly():
      self.skipTest('skip eager test when using legacy Saver.')
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(
        config=default_config,
        graph=ops.Graph(),
        use_gpu=test_util.is_gpu_available(),
    ) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = mkv.Variable(
          dtypes.int64,
          dtypes.int32,
          name="t1",
          initializer=default_val,
          checkpoint=True,
      )

      save = saver.Saver(table.tables)
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)
      del table

    with self.session(
        config=default_config,
        graph=ops.Graph(),
        use_gpu=test_util.is_gpu_available(),
    ) as sess:
      default_val = -1
      table = mkv.Variable(
          dtypes.int64,
          dtypes.int32,
          name="t1",
          initializer=default_val,
          checkpoint=True,
      )
      self.evaluate(
          table.upsert(
              constant_op.constant([0, 2], dtypes.int64),
              constant_op.constant([[12], [24]], dtypes.int32),
          ))
      self.assertAllEqual(2, self.evaluate(table.size()))

      save = saver.Saver([table._tables[0]])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.

      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 2, 3, 4], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([[0], [1], [2], [-1], [-1]], self.evaluate(output))

      del table

  def test_get_variable(self):
    with self.session(
        config=default_config,
        graph=ops.Graph(),
        use_gpu=test_util.is_gpu_available(),
    ):
      default_val = -1
      with variable_scope.variable_scope("embedding", reuse=True):
        table1 = mkv.get_variable("t1",
                                  dtypes.int64,
                                  dtypes.int32,
                                  initializer=default_val,
                                  dim=2)
        table2 = mkv.get_variable("t1",
                                  dtypes.int64,
                                  dtypes.int32,
                                  initializer=default_val,
                                  dim=2)
        table3 = mkv.get_variable("t2",
                                  dtypes.int64,
                                  dtypes.int32,
                                  initializer=default_val,
                                  dim=2)

      self.assertAllEqual(table1, table2)
      self.assertNotEqual(table1, table3)

  def test_get_variable_reuse_error(self):
    ops.disable_eager_execution()
    with self.session(
        config=default_config,
        graph=ops.Graph(),
        use_gpu=test_util.is_gpu_available(),
    ):
      with variable_scope.variable_scope("embedding", reuse=False):
        _ = mkv.get_variable("t900", initializer=-1, dim=2)
        with self.assertRaisesRegexp(ValueError,
                                     "Variable embedding/t900 already exists"):
          _ = mkv.get_variable("t900", initializer=-1, dim=2)

  @test_util.run_v1_only("Multiple sessions")
  def test_sharing_between_multi_sessions(self):
    ops.disable_eager_execution()
    # Start a server to store the table state
    server = server_lib.Server({"local0": ["localhost:0"]},
                               protocol="grpc",
                               start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target, config=default_config)
    session2 = session.Session(server.target, config=default_config)

    table = mkv.get_variable("tx100",
                             dtypes.int64,
                             dtypes.int32,
                             initializer=0,
                             dim=1)

    # Populate the table in the first session
    with session1:
      with ops.device(_get_devices()[0]):
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(variables.local_variables_initializer())
        self.assertAllEqual(0, table.size().eval())

        keys = constant_op.constant([11, 12], dtypes.int64)
        values = constant_op.constant([[11], [12]], dtypes.int32)
        table.upsert(keys, values).run()
        self.assertAllEqual(2, table.size().eval())

        output = table.lookup(constant_op.constant([11, 12, 13], dtypes.int64))
        self.assertAllEqual([[11], [12], [0]], output.eval())

    # Verify that we can access the shared data from the second session
    with session2:
      with ops.device(_get_devices()[0]):
        self.assertAllEqual(2, table.size().eval())

        output = table.lookup(constant_op.constant([10, 11, 12], dtypes.int64))
        self.assertAllEqual([[0], [11], [12]], output.eval())

  def test_merlin_kv_variable(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = constant_op.constant([-1, -2], dtypes.int64)
      keys = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      values = constant_op.constant([[0, 1], [2, 3], [4, 5], [6, 7]],
                                    dtypes.int32)
      table = mkv.get_variable("t10",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val,
                               dim=2)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_keys = constant_op.constant([3, 4], dtypes.int64)
      self.evaluate(table.remove(remove_keys))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 4], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([3, 2], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([[0, 1], [2, 3], [-1, -2]], result)

      exported_keys, exported_values = table.export()
      # exported data is in the order of the internal map, i.e. undefined
      sorted_keys = np.sort(self.evaluate(exported_keys))
      sorted_values = np.sort(self.evaluate(exported_values), axis=0)
      self.assertAllEqual([0, 1, 2], sorted_keys)
      sorted_expected_values = np.sort([[4, 5], [2, 3], [0, 1]], axis=0)
      self.assertAllEqual(sorted_expected_values, sorted_values)

      del table

  def test_merlin_kv_variable_export_insert(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int32)
      table1 = mkv.get_variable("t101",
                                dtypes.int64,
                                dtypes.int32,
                                initializer=default_val,
                                dim=2)
      self.assertAllEqual(0, self.evaluate(table1.size()))
      self.evaluate(table1.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table1.size()))

      input_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      expected_output = [[0, 1], [2, 3], [-1, -1]]
      output1 = table1.lookup(input_keys)
      self.assertAllEqual(expected_output, self.evaluate(output1))

      exported_keys, exported_values = table1.export()
      self.assertAllEqual(3, self.evaluate(exported_keys).size)
      self.assertAllEqual(6, self.evaluate(exported_values).size)

      # Populate a second table from the exported data
      table2 = mkv.get_variable("t102",
                                dtypes.int64,
                                dtypes.int32,
                                initializer=default_val,
                                dim=2)
      self.assertAllEqual(0, self.evaluate(table2.size()))
      self.evaluate(table2.upsert(exported_keys, exported_values))
      self.assertAllEqual(3, self.evaluate(table2.size()))

      # Verify lookup result is still the same
      output2 = table2.lookup(input_keys)
      self.assertAllEqual(expected_output, self.evaluate(output2))

  def test_merlin_kv_variable_invalid_shape(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      table = mkv.get_variable("t110",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val,
                               dim=2)

      # Shape [6] instead of [3, 2]
      values = constant_op.constant([0, 1, 2, 3, 4, 5], dtypes.int32)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.upsert(keys, values))

      # Shape [2,3] instead of [3, 2]
      values = constant_op.constant([[0, 1, 2], [3, 4, 5]], dtypes.int32)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.upsert(keys, values))

      # Shape [2, 2] instead of [3, 2]
      values = constant_op.constant([[0, 1], [2, 3]], dtypes.int32)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.upsert(keys, values))

      # Shape [3, 1] instead of [3, 2]
      values = constant_op.constant([[0], [2], [4]], dtypes.int32)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.upsert(keys, values))

      # Valid Insert
      values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int32)
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

  def test_merlin_kv_variable_duplicate_insert(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([0, 1, 2, 2], dtypes.int64)
      values = constant_op.constant([[0.0], [1.0], [2.0], [3.0]],
                                    dtypes.float32)
      table = mkv.get_variable("t130",
                               dtypes.int64,
                               dtypes.float32,
                               initializer=default_val)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_keys = constant_op.constant([0, 1, 2], dtypes.int64)
      output = table.lookup(input_keys)

      result = self.evaluate(output)
      self.assertTrue(
          list(result) in [[[0.0], [1.0], [3.0]], [[0.0], [1.0], [2.0]]])

  def test_merlin_kv_variable_find_high_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = mkv.get_variable("t140",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_keys = constant_op.constant([[0, 1], [2, 4]], dtypes.int64)
      output = table.lookup(input_keys)
      self.assertAllEqual([2, 2, 1], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([[[0], [1]], [[2], [-1]]], result)

  def test_merlin_kv_variable_insert_low_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      values = constant_op.constant([[[0], [1]], [[2], [3]]], dtypes.int32)
      table = mkv.get_variable("t150",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3, 4], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllEqual([[0], [1], [3], [-1]], result)

  def test_merlin_kv_variable_remove_low_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      values = constant_op.constant([[[0], [1]], [[2], [3]]], dtypes.int32)
      table = mkv.get_variable("t160",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_keys = constant_op.constant([1, 4], dtypes.int64)
      self.evaluate(table.remove(remove_keys))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3, 4], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllEqual([[0], [-1], [3], [-1]], result)

  def test_merlin_kv_variable_insert_high_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = constant_op.constant([-1, -1, -1], dtypes.int32)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                    dtypes.int32)
      table = mkv.get_variable("t170",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val,
                               dim=3)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([[0, 1], [3, 4]], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([2, 2, 3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual(
          [[[0, 1, 2], [2, 3, 4]], [[-1, -1, -1], [-1, -1, -1]]], result)

  def test_merlin_kv_variable_remove_high_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = constant_op.constant([-1, -1, -1], dtypes.int32)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                    dtypes.int32)
      table = mkv.get_variable("t180",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val,
                               dim=3)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([[0, 3]], dtypes.int64)
      self.evaluate(table.remove(remove_keys))
      self.assertAllEqual(2, self.evaluate(table.size()))

      remove_keys = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([2, 2, 3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual(
          [[[-1, -1, -1], [2, 3, 4]], [[4, 5, 6], [-1, -1, -1]]], result)

  def test_merlin_kv_variables(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)

      table1 = mkv.get_variable("t191",
                                dtypes.int64,
                                dtypes.int32,
                                initializer=default_val)
      table2 = mkv.get_variable("t192",
                                dtypes.int64,
                                dtypes.int32,
                                initializer=default_val)
      table3 = mkv.get_variable("t193",
                                dtypes.int64,
                                dtypes.int32,
                                initializer=default_val)
      self.evaluate(table1.upsert(keys, values))
      self.evaluate(table2.upsert(keys, values))
      self.evaluate(table3.upsert(keys, values))

      self.assertAllEqual(3, self.evaluate(table1.size()))
      self.assertAllEqual(3, self.evaluate(table2.size()))
      self.assertAllEqual(3, self.evaluate(table3.size()))

      remove_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      output1 = table1.lookup(remove_keys)
      output2 = table2.lookup(remove_keys)
      output3 = table3.lookup(remove_keys)

      out1, out2, out3 = self.evaluate([output1, output2, output3])
      self.assertAllEqual([[0], [1], [-1]], out1)
      self.assertAllEqual([[0], [1], [-1]], out2)
      self.assertAllEqual([[0], [1], [-1]], out3)

  def test_merlin_kv_variable_with_tensor_default(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = constant_op.constant(-1, dtypes.int32)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = mkv.get_variable("t200",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllEqual([[0], [1], [-1]], result)

  def test_signature_mismatch(self):
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with self.session(config=config, use_gpu=test_util.is_gpu_available()):
      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = mkv.get_variable("t210",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val)

      # upsert with keys of the wrong type
      with self.assertRaises(ValueError):
        self.evaluate(
            table.upsert(constant_op.constant([4.0, 5.0, 6.0], dtypes.float32),
                         values))

      # upsert with values of the wrong type
      with self.assertRaises(ValueError):
        self.evaluate(table.upsert(keys, constant_op.constant(["a", "b", "c"])))

      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys_ref = variables.Variable(0, dtype=dtypes.int64)
      input_int64_ref = variables.Variable([-1], dtype=dtypes.int32)
      self.evaluate(variables.global_variables_initializer())

      # Ref types do not produce an upsert signature mismatch.
      self.evaluate(table.upsert(remove_keys_ref, input_int64_ref))
      self.assertAllEqual(3, self.evaluate(table.size()))

      # Ref types do not produce a lookup signature mismatch.
      self.assertEqual([-1], self.evaluate(table.lookup(remove_keys_ref)))

      # lookup with keys of the wrong type
      remove_keys = constant_op.constant([1, 2, 3], dtypes.int32)
      with self.assertRaises(ValueError):
        self.evaluate(table.lookup(remove_keys))

  def test_merlin_kv_variable_int_float(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = -1.0
      keys = constant_op.constant([3, 7, 0], dtypes.int64)
      values = constant_op.constant([[7.5], [-1.2], [9.9]], dtypes.float32)
      table = mkv.get_variable("t220",
                               dtypes.int64,
                               dtypes.float32,
                               initializer=default_val)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([7, 0, 11], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllClose([[-1.2], [9.9], [default_val]], result)

  def test_merlin_kv_variable_with_random_init(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0.0], [1.0], [2.0]], dtypes.float32)
      default_val = init_ops.random_uniform_initializer()
      table = mkv.get_variable("t230",
                               dtypes.int64,
                               dtypes.float32,
                               initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertNotEqual([-1.0], result[2])

  def test_merlin_kv_variable_insert_lookup_with_default_metas(self):
    for allow_duplicated_keys in [False]:
      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config):
        DIM = 2
        default_val = [0.1] * DIM
        default_buckets_size = 128
        key_num_for_base = default_buckets_size
        key_num_for_test = 2

        def create_keys_in_one_bucket(
            num=128,
            min=0,
            max=0x7FFFFFFFFFFFFFFF,
            bucket_num=2,
            target_bucket=0,
        ):
          keys = set()
          while len(keys) < num:
            key = np.random.randint(min, max, size=1, dtype=np.int64)[0]
            hashed_key = Murmur3Hash(key)
            if hashed_key % bucket_num == target_bucket:
              keys.add(key)
          return list(keys)

        raw_base_keys = create_keys_in_one_bucket(key_num_for_base,
                                                  min=0,
                                                  max=0x3FFFFFFFFFFFFFFF)

        raw_test_keys = create_keys_in_one_bucket(
            key_num_for_test, min=0x3FFFFFFFFFFFFFFF,
            max=0x7FFFFFFFFFFFFFFF) + raw_base_keys[72:74]
        key_num_for_test = 4

        base_keys = constant_op.constant(raw_base_keys, dtypes.int64)
        base_values = constant_op.constant(
            [[i * 0.1] * DIM for i in range(key_num_for_base)], dtypes.float32)

        test_keys = constant_op.constant(raw_test_keys, dtypes.int64)
        test_values = constant_op.constant(
            [[i * 100.0] * DIM for i in range(key_num_for_test)],
            dtypes.float32)
        table = mkv.get_variable("y001" + str(allow_duplicated_keys),
                                 dtypes.int64,
                                 dtypes.float32,
                                 dim=DIM,
                                 init_size=256,
                                 initializer=default_val)
        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(
            table.upsert(base_keys,
                         base_values,
                         allow_duplicated_keys=allow_duplicated_keys))
        self.assertAllEqual(min(default_buckets_size, key_num_for_base),
                            self.evaluate(table.size()))

        export_keys, export_values = table.export()
        export_keys_np = self.evaluate(export_keys)

        lookup_values, lookup_metas = table.lookup(base_keys, return_metas=True)

        lookup_values_np = self.evaluate(lookup_values)
        lookup_metas_np = self.evaluate(lookup_metas)

        all_keys_np = self.evaluate(base_keys)
        all_values_np = self.evaluate(base_values)
        expected_values = all_values_np[np.where(
            np.in1d(all_keys_np, export_keys_np))[0]]

        sorted_metas = np.sort(lookup_metas_np, axis=0)
        self.assertAllEqual(list(range(1, key_num_for_base + 1)), sorted_metas)
        self.assertAllCloseAccordingToType(expected_values, lookup_values_np)

        # simulate upsert when the buckets are full.
        self.evaluate(
            table.upsert(test_keys,
                         test_values,
                         allow_duplicated_keys=allow_duplicated_keys))
        self.assertAllEqual(default_buckets_size, self.evaluate(table.size()))

        export_keys, export_values = table.export()
        export_keys_np = self.evaluate(export_keys)

        lookup_values, lookup_metas = table.lookup(test_keys, return_metas=True)

        lookup_values_np = self.evaluate(lookup_values)
        lookup_metas_np = self.evaluate(lookup_metas)

        all_keys_np = self.evaluate(test_keys)
        all_values_np = self.evaluate(test_values)
        expected_values = all_values_np[np.where(
            np.in1d(all_keys_np, export_keys_np))[0]]

        sorted_metas = np.sort(lookup_metas_np, axis=0)
        self.assertAllEqual(
            list(
                range(key_num_for_base + 1,
                      key_num_for_base + key_num_for_test + 1)), sorted_metas)
        self.assertAllCloseAccordingToType(expected_values, lookup_values_np)

  def test_merlin_kv_variable_with_customized_metas_regular_test(self):
    for allow_duplicated_keys in [True, False]:
      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config):
        DIM = 64
        default_val = [0.1] * DIM
        default_buckets_size = 128
        key_num_for_base = default_buckets_size
        key_num_for_test = 64

        def create_keys_in_one_bucket(
            num=128,
            min=0,
            max=0x7FFFFFFFFFFFFFFF,
            bucket_num=2,
            target_bucket=0,
        ):
          keys = set()
          while len(keys) < num:
            key = np.random.randint(min, max, size=1, dtype=np.int64)[0]
            hashed_key = Murmur3Hash(key)
            if hashed_key % bucket_num == target_bucket:
              keys.add(key)
          return list(keys)

        raw_base_keys = create_keys_in_one_bucket(key_num_for_base,
                                                  min=0,
                                                  max=0x3FFFFFFFFFFFFFFF)
        raw_test_keys = create_keys_in_one_bucket(
            key_num_for_test, min=0x3FFFFFFFFFFFFFFF,
            max=0x7FFFFFFFFFFFFFFF) + raw_base_keys[64:]
        key_num_for_test += (key_num_for_base - 64)

        base_meta_start = 1000
        base_meta_end = base_meta_start + key_num_for_base
        base_keys = constant_op.constant(raw_base_keys, dtypes.int64)
        base_values = constant_op.constant(
            [[i * 0.1] * DIM for i in range(key_num_for_base)], dtypes.float32)
        base_metas = constant_op.constant(
            [i for i in range(base_meta_start, base_meta_end)], dtypes.int64)

        test_meta_start = base_meta_end
        test_meta_end = test_meta_start + key_num_for_test
        test_keys = constant_op.constant(raw_test_keys, dtypes.int64)
        test_values = constant_op.constant(
            [[i * 1.0] * DIM for i in range(key_num_for_test)], dtypes.float32)

        raw_test_metas = [i for i in range(test_meta_start, test_meta_end)]
        test_metas = constant_op.constant(raw_test_metas, dtypes.int64)

        table = mkv.get_variable("y002" + str(allow_duplicated_keys),
                                 dtypes.int64,
                                 dtypes.float32,
                                 dim=DIM,
                                 init_size=256,
                                 initializer=default_val)
        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(
            table.upsert(base_keys,
                         base_values,
                         base_metas,
                         allow_duplicated_keys=allow_duplicated_keys))
        self.assertAllEqual(min(default_buckets_size, key_num_for_base),
                            self.evaluate(table.size()))

        export_keys, export_values = table.export()
        export_keys_np = self.evaluate(export_keys)

        lookup_values, lookup_metas = table.lookup(base_keys, return_metas=True)

        lookup_values_np = self.evaluate(lookup_values)
        lookup_metas_np = self.evaluate(lookup_metas)

        all_keys_np = self.evaluate(base_keys)
        all_values_np = self.evaluate(base_values)
        expected_values = all_values_np[np.where(
            np.in1d(all_keys_np, export_keys_np))[0]]

        sorted_metas = np.sort(lookup_metas_np, axis=0)
        self.assertAllEqual(list(range(base_meta_start, base_meta_end)),
                            sorted_metas)
        self.assertAllCloseAccordingToType(expected_values, lookup_values_np)

        # simulate upsert when the buckets are full.
        self.evaluate(
            table.upsert(test_keys,
                         test_values,
                         test_metas,
                         allow_duplicated_keys=allow_duplicated_keys))
        self.assertAllEqual(default_buckets_size, self.evaluate(table.size()))

        export_keys, export_values = table.export()
        export_keys_np = self.evaluate(export_keys)

        lookup_values, lookup_metas = table.lookup(test_keys, return_metas=True)

        lookup_values_np = self.evaluate(lookup_values)
        lookup_metas_np = self.evaluate(lookup_metas)

        all_keys_np = self.evaluate(test_keys)
        all_values_np = self.evaluate(test_values)
        expected_values = all_values_np[np.where(
            np.in1d(all_keys_np, export_keys_np))[0]]

        sorted_metas = np.sort(lookup_metas_np, axis=0)
        self.assertAllEqual(list(range(test_meta_start, test_meta_end)),
                            sorted_metas)
        self.assertAllCloseAccordingToType(expected_values, lookup_values_np)

  def test_merlin_kv_variable_with_customized_metas_special_test(self):
    for allow_duplicated_keys in [True, False]:
      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config):
        DIM = 2
        default_val = [0.1] * DIM
        default_buckets_size = 128
        key_num_for_base = default_buckets_size
        key_num_for_test = 4

        def create_keys_in_one_bucket(
            num=128,
            min=0,
            max=0x7FFFFFFFFFFFFFFF,
            bucket_num=2,
            target_bucket=0,
        ):
          keys = set()
          while len(keys) < num:
            key = np.random.randint(min, max, size=1, dtype=np.int64)[0]
            hashed_key = Murmur3Hash(key)
            if hashed_key % bucket_num == target_bucket:
              keys.add(key)
          return list(keys)

        raw_base_keys = create_keys_in_one_bucket(key_num_for_base,
                                                  min=0,
                                                  max=0x3FFFFFFFFFFFFFFF)
        raw_test_keys = create_keys_in_one_bucket(
            key_num_for_test, min=0x3FFFFFFFFFFFFFFF,
            max=0x7FFFFFFFFFFFFFFF) + raw_base_keys[72:76]
        key_num_for_test += 4
        raw_test_values = [[i * 1.0] * DIM for i in range(key_num_for_test)]

        base_meta_start = 1000
        base_meta_end = base_meta_start + key_num_for_base
        base_keys = constant_op.constant(raw_base_keys, dtypes.int64)
        base_values = constant_op.constant(
            [[i * 0.1] * DIM for i in range(key_num_for_base)], dtypes.float32)
        base_metas = constant_op.constant(
            [i for i in range(base_meta_start, base_meta_end)], dtypes.int64)

        test_meta_start = base_meta_end
        test_meta_end = test_meta_start + key_num_for_test
        test_keys = constant_op.constant(raw_test_keys, dtypes.int64)
        test_values = constant_op.constant(raw_test_values, dtypes.float32)

        raw_test_metas = [i for i in range(test_meta_start, test_meta_end)]
        # replace three new keys to lower metas, would not be inserted.
        raw_test_metas[0] = 200
        raw_test_metas[1] = 78
        raw_test_metas[2] = 101

        # replace three exist keys to lower metas, just refresh the meta for them.
        raw_test_metas[4] = 99
        raw_test_metas[5] = 98
        raw_test_metas[6] = 100

        test_metas = constant_op.constant(raw_test_metas, dtypes.int64)
        test_expected_metas = [
            0, 0, 0, raw_test_metas[3], 99, 98, 100, raw_test_metas[7]
        ]
        test_expected_values = [default_val, default_val, default_val
                               ] + raw_test_values[3:]

        table = mkv.get_variable("y004" + str(allow_duplicated_keys),
                                 dtypes.int64,
                                 dtypes.float32,
                                 dim=DIM,
                                 init_size=256,
                                 initializer=default_val)
        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(
            table.upsert(base_keys,
                         base_values,
                         base_metas,
                         allow_duplicated_keys=allow_duplicated_keys))
        self.assertAllEqual(min(default_buckets_size, key_num_for_base),
                            self.evaluate(table.size()))

        export_keys, export_values = table.export()
        export_keys_np = self.evaluate(export_keys)

        lookup_values, lookup_metas = table.lookup(base_keys, return_metas=True)

        lookup_values_np = self.evaluate(lookup_values)
        lookup_metas_np = self.evaluate(lookup_metas)

        all_keys_np = self.evaluate(base_keys)
        all_values_np = self.evaluate(base_values)
        expected_values = all_values_np[np.where(
            np.in1d(all_keys_np, export_keys_np))[0]]

        sorted_metas = np.sort(lookup_metas_np, axis=0)
        self.assertAllEqual(list(range(base_meta_start, base_meta_end)),
                            sorted_metas)
        self.assertAllCloseAccordingToType(expected_values, lookup_values_np)

        # simulate upsert when the buckets are full.
        self.evaluate(
            table.upsert(test_keys,
                         test_values,
                         test_metas,
                         allow_duplicated_keys=allow_duplicated_keys))
        self.assertAllEqual(default_buckets_size, self.evaluate(table.size()))

        lookup_values, lookup_metas = table.lookup(test_keys, return_metas=True)

        lookup_values_np = self.evaluate(lookup_values)
        lookup_metas_np = self.evaluate(lookup_metas)

        self.assertAllCloseAccordingToType(test_expected_values,
                                           lookup_values_np)
        self.assertAllEqual(test_expected_metas, lookup_metas_np)

  def test_merlin_kv_variable_customized_metas_on_big_table(self):
    if context.executing_eagerly():
      self.skipTest('skip eager test when using legacy Saver.')

    print(
        '\033[93m' +
        "[Warning]: the case of 'test_merlin_kv_variable_customized_metas_on_big_table' could take several minutes!"
        + '\033[0m')
    for allow_duplicated_keys in [False]:
      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config):
        DIM = 2
        default_val = [0.8] * DIM
        batch_size = 1048576
        capacity = batch_size * 128  # capacity = 134,217,728
        steps = 128
        start = 1000000000
        rounds = 3
        expected_correct_rate = 0.964

        keys = script_ops.py_func(
            _create_dynamic_shape_continous_tensor(start=start,
                                                   length=batch_size),
            inp=[],
            Tout=dtypes.int64,
            stateful=True,
        )
        values = math_ops.cast(
            array_ops.repeat(array_ops.reshape(keys, [-1, 1]),
                             repeats=DIM,
                             axis=1), dtypes.float32) / (1.0 * start)
        metas = keys

        table = mkv.get_variable("y006" + str(allow_duplicated_keys),
                                 dtypes.int64,
                                 dtypes.float32,
                                 dim=DIM,
                                 init_size=capacity,
                                 initializer=default_val)
        self.assertAllEqual(0, self.evaluate(table.size()))
        np.set_printoptions(suppress=True)

        upsert_op = table.upsert(keys,
                                 values,
                                 metas,
                                 allow_duplicated_keys=allow_duplicated_keys)
        size_op = table.size()
        export_keys, export_values = table.export()
        lookup_values, lookup_metas = table.lookup(export_keys,
                                                   return_metas=True)
        for r in range(rounds):
          expected_min_key = start + capacity * r
          expected_max_key = start + capacity * (r + 1) - 1
          expected_table_size = int(expected_correct_rate *
                                    capacity) if r == 0 else capacity
          for s in range(steps):
            self.evaluate(upsert_op)
          self.assertAllGreaterEqual(self.evaluate(size_op),
                                     expected_table_size)

          export_keys_np, lookup_values_np, lookup_metas_np = self.evaluate(
              [export_keys, lookup_values, lookup_metas])
          expeted_values_np = np.repeat(np.reshape(export_keys_np,
                                                   [-1, 1]).astype(float),
                                        repeats=DIM,
                                        axis=1) / (1.0 * start)

          self.assertAllCloseAccordingToType(export_keys_np, lookup_metas_np)
          self.assertAllCloseAccordingToType(expeted_values_np,
                                             lookup_values_np,
                                             rtol=1e-03,
                                             atol=1e-03)

          correct_rate = (lookup_metas_np >= expected_min_key).sum() / capacity

          self.assertAllGreaterEqual(correct_rate, expected_correct_rate)
          max_key_np = np.max(export_keys_np)
          self.assertAllEqual(expected_max_key, max_key_np)
          print('\033[92m' +
                "[Round {}] correct_rate={:.4f}".format(r, correct_rate) +
                '\033[0m')


if __name__ == "__main__":
  test.main()
