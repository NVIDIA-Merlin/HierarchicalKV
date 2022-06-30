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
"""NVIDIA Merlin-KV is a hierarchical Key-Value storage library for building recommender system models.

 - A hierarchical key-value storage library designed for the large models in recommenders systems
 - Storing the key-value (embedding) on the HBM and Host memory (support SSD/NVMe in the future)
 - The performance is close to those implementations running on pure HBM thru innovative design
 - Can also be used as a generic key-value storage

"""
__all__ = ['tensorflow']  #, 'pytorch']

from merlin_kv.utils.ensure_tf_install import _check_tf_version
from merlin_kv.version import __version__

_check_tf_version()

from merlin_kv import tensorflow
# from merlin_kv import pytorch
from merlin_kv.register import register_all
