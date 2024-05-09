/*
* Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cstdint>
#include <gtest/gtest.h>
#include "merlin/types.cuh"
#include "test_util.cuh"
#include "merlin/utils.cuh"


using namespace nv::merlin;

__global__ void testReservedKeysKernel(uint64_t* keys, bool* results, size_t numKeys) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < numKeys) {
    results[idx] = IS_RESERVED_KEY(keys[idx]);
  }
}

void testCustomMemsetAsync() {
  size_t numElements = 4;
  uint64_t value = 0xFFFFFFFFFFFFFFF1;
  uint64_t* devPtr;
  uint64_t* hostData = new uint64_t[numElements];

  cudaMalloc((void**)&devPtr, numElements * sizeof(uint64_t));
  memset64Async(devPtr, value, numElements);
  cudaMemcpy(hostData, devPtr, numElements * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < numElements; i++) {
    assert(hostData[i] == value);
  }

  std::cout << "All values were set correctly!" << std::endl;

  cudaFree(devPtr);
  delete[] hostData;
}

void testReservedKeys(uint64_t* testKeys, bool* expectedResults, size_t numKeys) {
  uint64_t* d_keys;
  bool* d_results;
  bool* h_results = new bool[numKeys];

  cudaMalloc(&d_keys, numKeys * sizeof(uint64_t));
  cudaMalloc(&d_results, numKeys * sizeof(bool));

  cudaMemcpy(d_keys, testKeys, numKeys * sizeof(uint64_t), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (numKeys + blockSize - 1) / blockSize;

  testReservedKeysKernel<<<numBlocks, blockSize>>>(d_keys, d_results, numKeys);
  cudaDeviceSynchronize();

  cudaMemcpy(h_results, d_results, numKeys * sizeof(bool), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < numKeys; i++) {
    assert(h_results[i] == expectedResults[i]);
  }

  cudaFree(d_keys);
  cudaFree(d_results);
  delete[] h_results;
  CudaCheckError();
  std::cout << "All tests passed." << std::endl;
}

void testKeyOptions() {
  for (int i = 0; i <= MAX_RESERVED_KEY_BIT; i++) {
    CUDA_CHECK(init_reserved_keys(i));
    uint64_t host_reclaim_key, host_locked_key;
    cudaMemcpyFromSymbol(&host_reclaim_key, RECLAIM_KEY, sizeof(uint64_t));
    cudaMemcpyFromSymbol(&host_locked_key, LOCKED_KEY, sizeof(uint64_t));

    uint64_t testKeys[6] = {
        EMPTY_KEY_CPU, host_reclaim_key, host_locked_key,
        UINT64_C(0x0), UINT64_C(0x10),
        DEFAULT_EMPTY_KEY
    };
    bool expectedResults[6] = {
        true, true, true, false, false,
        (i == 0)? true : false
    };
    testReservedKeys(testKeys, expectedResults, 4);
  }
}

TEST(ReservedKeysTest, testKeyOptions) {
  testKeyOptions();
  testCustomMemsetAsync();
}