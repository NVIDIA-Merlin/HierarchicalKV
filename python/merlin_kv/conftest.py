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

import numpy as np
import pytest
import tensorflow as tf

from merlin_kv import tensorflow as mkv

from merlin_kv.utils.test_utils import (  # noqa: F401
    maybe_run_functions_eagerly, only_run_functions_eagerly,
    run_with_mixed_precision_policy, pytest_make_parametrize_id, data_format,
    set_seeds, pytest_addoption, set_global_variables, pytest_configure, device,
    pytest_generate_tests, pytest_collection_modifyitems,
)

# fixtures present in this file will be available
# when running tests and can be referenced with strings
# https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions


@pytest.fixture(autouse=True)
def add_doctest_namespace(doctest_namespace):
  doctest_namespace["np"] = np
  doctest_namespace["tf"] = tf
  doctest_namespace["mkv"] = mkv
