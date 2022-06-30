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
# lint-as: python3

from abc import ABCMeta
from tensorflow.python.eager import context
from tensorflow.python.ops import gen_parsing_ops
from merlin_kv import tensorflow as mkv


class KVCreator(object, metaclass=ABCMeta):
  """
  A generic KV table creator.

    KV table instance will be created by the create function with config.
  And also a config class for specific table instance backend should be
  inited before callling the creator function.
    And then, the KVCreator class instance will be passed to the Variable
  class for creating the real KV table backend(TF resource).

    Example usage:

    ```python
    from merlin_kv import tensorflow as mkv

    mkv_config=mkv.MerlinKVConfig(
        merlin_kv_config_abs_dir="xx/yy.json")
    mkv_creator=mkv.MerlinKVCreator(mkv_config)
    ```
  """

  def __init__(self, config=None):
    self.config = config

  def create(self,
             key_dtype=None,
             value_dtype=None,
             default_value=None,
             name=None,
             checkpoint=None,
             init_size=None,
             config=None):

    raise NotImplementedError('create function must be implemented')


class MerlinKVConfig(object):

  def __init__(self):
    """ MerlinKVConfig include nothing for parameter default satisfied.
    """
    pass


class MerlinKVCreator(KVCreator):

  def create(
      self,
      key_dtype=None,
      value_dtype=None,
      default_value=None,
      name=None,
      checkpoint=None,
      init_size=None,
      config=None,
  ):
    self.key_dtype = key_dtype
    self.value_dtype = value_dtype
    self.default_value = default_value
    self.name = name
    self.checkpoint = checkpoint
    self.init_size = init_size
    self.config = config

    return mkv.MerlinKV(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        default_value=default_value,
        name=name,
        checkpoint=checkpoint,
        init_size=init_size,
        config=config,
    )

  def get_config(self):
    if not context.executing_eagerly():
      raise RuntimeError(
          'Unsupported to serialize python object of MerlinKVCreator.')

    config = {
        'key_dtype': self.key_dtype,
        'value_dtype': self.value_dtype,
        'default_value': self.default_value.numpy(),
        'name': self.name,
        'checkpoint': self.checkpoint,
        'init_size': self.init_size,
        'config': self.config,
    }
    return config
