/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "types.cuh"
#include "utils.cuh"

namespace nv {
namespace merlin {
namespace embedding {

template <class K, class V, class M, size_t DIM>
void create_slot(Table<K, V, M, DIM>* primary_table,
                 Table<K, V, M, DIM>** slot_table, int vector_offset = 0) {
  cudaMallocManaged((void**)slot_table, sizeof(Table<K, V, M, DIM>));
  cudaMemcpy(slot_table, primary_table, sizeof(Table<K, V, M, DIM>),
             cudaMemcpyDeviceToDevice);
  (*slot_table)->primary_table = false;
  (*slot_table)->vector_offset = vector_offset;
}

}  // namespace embedding
}  // namespace merlin
}  // namespace nv