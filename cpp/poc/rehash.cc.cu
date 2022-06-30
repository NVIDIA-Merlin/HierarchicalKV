/* Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <assert.h>
#include <cuda.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

typedef std::chrono::nanoseconds ReportingDuration;

template <typename T>
class ExtenablePtr {
 public:
  ExtenablePtr(CUcontext ctx) : allocator_(ctx) {}

  CUresult reserve(size_t num) { return allocator_.reserve(num * sizeof(T)); }

  CUresult grow(size_t num) { return allocator_.grow(num * sizeof(T)); }

  T* get_ptr() const { return (T*)allocator_.get_ptr(); }

  size_t get_size() const { return allocator_.get_size(); }

 private:
  MemMapAllocator allocator_;
};

class MemMapAllocator {
 private:
  CUdeviceptr d_ptr;
  CUmemAllocationProp prop;
  CUmemAccessDesc accessDesc;
  struct Range {
    CUdeviceptr start;
    size_t sz;
  };
  std::vector<Range> va_ranges;
  std::vector<CUmemGenericAllocationHandle> handles;
  std::vector<size_t> handle_sizes;
  size_t allocated_size;
  size_t reserved_size;
  size_t chunk_size;

 public:
  MemMapAllocator(CUcontext context);
  ~MemMapAllocator();

  CUdeviceptr get_ptr() const { return d_ptr; }

  size_t get_size() const { return allocated_size; }

  CUresult reserve(size_t new_size);

  CUresult grow(size_t new_size);
};

MemMapAllocator::MemMapAllocator(CUcontext context)
    : d_ptr(0ULL),
      prop(),
      handles(),
      allocated_size(0ULL),
      reserved_size(0ULL),
      chunk_size(0ULL) {
  CUdevice device;
  CUcontext prev_ctx;
  CUresult status = CUDA_SUCCESS;
  (void)status;

  status = cuCtxGetCurrent(&prev_ctx);
  assert(status == CUDA_SUCCESS);
  if (cuCtxSetCurrent(context) == CUDA_SUCCESS) {
    status = cuCtxGetDevice(&device);
    assert(status == CUDA_SUCCESS);
    status = cuCtxSetCurrent(prev_ctx);
    assert(status == CUDA_SUCCESS);
  }

  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = (int)device;
  prop.win32HandleMetaData = NULL;

  accessDesc.location = prop.location;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  status = cuMemGetAllocationGranularity(&chunk_size, &prop,
                                         CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  assert(status == CUDA_SUCCESS);
}

MemMapAllocator::~MemMapAllocator() {
  CUresult status = CUDA_SUCCESS;
  (void)status;
  if (d_ptr != 0ULL) {
    status = cuMemUnmap(d_ptr, allocated_size);
    assert(status == CUDA_SUCCESS);
    for (size_t i = 0ULL; i < va_ranges.size(); i++) {
      status = cuMemAddressFree(va_ranges[i].start, va_ranges[i].sz);
      assert(status == CUDA_SUCCESS);
    }
    for (size_t i = 0ULL; i < handles.size(); i++) {
      status = cuMemRelease(handles[i]);
      assert(status == CUDA_SUCCESS);
    }
  }
}

CUresult MemMapAllocator::reserve(size_t new_size) {
  CUresult status = CUDA_SUCCESS;
  CUdeviceptr new_ptr = 0ULL;

  if (new_size <= reserved_size) {
    return CUDA_SUCCESS;
  }

  const size_t aligned_size =
      ((new_size + chunk_size - 1) / chunk_size) * chunk_size;

  status = cuMemAddressReserve(&new_ptr, (aligned_size - reserved_size), 0ULL,
                               d_ptr + reserved_size, 0ULL);

  // Try to reserve an address just after what we already have reserved
  if (status != CUDA_SUCCESS || (new_ptr != d_ptr + reserved_size)) {
    if (new_ptr != 0ULL) {
      (void)cuMemAddressFree(new_ptr, (aligned_size - reserved_size));
    }
    // Slow path - try to find a new address reservation big enough for us
    status = cuMemAddressReserve(&new_ptr, aligned_size, 0ULL, 0U, 0);
    if (status == CUDA_SUCCESS && d_ptr != 0ULL) {
      CUdeviceptr ptr = new_ptr;
      // Found one, now unmap our previous allocations
      status = cuMemUnmap(d_ptr, allocated_size);
      assert(status == CUDA_SUCCESS);
      for (size_t i = 0ULL; i < handles.size(); i++) {
        const size_t handle_size = handle_sizes[i];
        // And remap them, enabling their access
        if ((status = cuMemMap(ptr, handle_size, 0ULL, handles[i], 0ULL)) !=
            CUDA_SUCCESS)
          break;
        if ((status = cuMemSetAccess(ptr, handle_size, &accessDesc, 1ULL)) !=
            CUDA_SUCCESS)
          break;
        ptr += handle_size;
      }
      if (status != CUDA_SUCCESS) {
        // Failed the mapping somehow... clean up!
        status = cuMemUnmap(new_ptr, aligned_size);
        assert(status == CUDA_SUCCESS);
        status = cuMemAddressFree(new_ptr, aligned_size);
        assert(status == CUDA_SUCCESS);
      } else {
        // Clean up our old VA reservations!
        for (size_t i = 0ULL; i < va_ranges.size(); i++) {
          (void)cuMemAddressFree(va_ranges[i].start, va_ranges[i].sz);
        }
        va_ranges.clear();
      }
    }
    // Assuming everything went well, update everything
    if (status == CUDA_SUCCESS) {
      Range r;
      d_ptr = new_ptr;
      reserved_size = aligned_size;
      r.start = new_ptr;
      r.sz = aligned_size;
      va_ranges.push_back(r);
    }
  } else {
    Range r;
    r.start = new_ptr;
    r.sz = aligned_size - reserved_size;
    va_ranges.push_back(r);
    if (d_ptr == 0ULL) {
      d_ptr = new_ptr;
    }
    reserved_size = aligned_size;
  }

  return status;
}

CUresult MemMapAllocator::grow(size_t new_size) {
  CUresult status = CUDA_SUCCESS;
  CUmemGenericAllocationHandle handle;
  if (new_size <= allocated_size) {
    return CUDA_SUCCESS;
  }

  const size_t size_diff = new_size - allocated_size;
  // Round up to the next chunk size
  const size_t sz = ((size_diff + chunk_size - 1) / chunk_size) * chunk_size;

  if ((status = reserve(allocated_size + sz)) != CUDA_SUCCESS) {
    return status;
  }

  if ((status = cuMemCreate(&handle, sz, &prop, 0)) == CUDA_SUCCESS) {
    if ((status = cuMemMap(d_ptr + allocated_size, sz, 0ULL, handle, 0ULL)) ==
        CUDA_SUCCESS) {
      if ((status = cuMemSetAccess(d_ptr + allocated_size, sz, &accessDesc,
                                   1ULL)) == CUDA_SUCCESS) {
        handles.push_back(handle);
        handle_sizes.push_back(sz);
        allocated_size += sz;
      }
      if (status != CUDA_SUCCESS) {
        (void)cuMemUnmap(d_ptr + allocated_size, sz);
      }
    }
    if (status != CUDA_SUCCESS) {
      (void)cuMemRelease(handle);
    }
  }

  return status;
}

static inline void checkDrvError(CUresult res, const char* tok,
                                 const char* file, unsigned line) {
  if (res != CUDA_SUCCESS) {
    const char* errStr = NULL;
    (void)cuGetErrorString(res, &errStr);
    std::cerr << file << ':' << line << ' ' << tok << "failed ("
              << (unsigned)res << "): " << errStr << std::endl;
  }
}

#define CUDA_CHECK(x) checkDrvError(x, #x, __FILE__, __LINE__);

template <typename V>
void measureGrow(V& v, size_t minN, size_t maxN,
                 std::vector<ReportingDuration>& durations) {
  for (size_t n = minN; n <= maxN; n <<= 1) {
    typedef std::chrono::time_point<std::chrono::steady_clock> time_point;

    time_point start = std::chrono::steady_clock::now();
    CUresult status = v.grow(n);
    time_point end = std::chrono::steady_clock::now();

    durations.push_back(
        std::chrono::duration_cast<ReportingDuration>(end - start));
    // In non-release, verify the memory is accessible and everything worked
    // properly
    assert(CUDA_SUCCESS == status);
    assert(CUDA_SUCCESS ==
           cuMemsetD8((CUdeviceptr)v.get_ptr(), 0, v.get_size()));
    assert(CUDA_SUCCESS == cuCtxSynchronize());
  }
}

template <typename Allocator, typename Elem>
void runVectorPerfTest(CUcontext ctx, size_t minN, size_t maxN,
                       std::vector<ReportingDuration>& noReserveDurations,
                       std::vector<ReportingDuration>& reserveDurations) {
  typedef ExtenablePtr<Elem, Allocator> VectorDUT;

  if (false) {
    // Warm-up
    VectorDUT dut(ctx);
    if (!dut.grow(maxN)) {
      std::cerr << "Failed to grow to max elements, test invalid!\n"
                << std::endl;
      return;
    }
  }

  // Wait for the OS to settle it's GPU pages from past perf runs
  std::this_thread::sleep_for(std::chrono::seconds(2));
  {
    // Measure without reserving
    VectorDUT dut(ctx);
    measureGrow(dut, minN, maxN, noReserveDurations);
  }

  // Wait for the OS to settle it's GPU pages from past perf runs
  std::this_thread::sleep_for(std::chrono::seconds(2));
  {
    size_t free = 0ULL;
    VectorDUT dut(ctx);

    dut.reserve(maxN);
    CUDA_CHECK(cuMemGetInfo(&free, NULL));
    std::cout << "\tReserved " << maxN << " elements..." << std::endl
              << "\tFree Memory: " << (float)free / std::giga::num << "GB"
              << std::endl;

    measureGrow(dut, minN, maxN, reserveDurations);
  }
}

int main() {
  size_t free;
  typedef unsigned char ElemType;
  CUcontext ctx;
  CUdevice dev;
  int supportsVMM = 0;

  CUDA_CHECK(cuInit(0));
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, 0));
  CUDA_CHECK(cuCtxSetCurrent(ctx));
  CUDA_CHECK(cuCtxGetDevice(&dev));

  std::vector<std::vector<ReportingDuration> > durations(2);

  CUDA_CHECK(cuMemGetInfo(&free, NULL));

  std::cout << "Total Free Memory: " << (float)free / std::giga::num << "GB"
            << std::endl;

  // Skip the smaller cases
  const size_t minN =
      (2ULL * 1024ULL * 1024ULL + sizeof(ElemType) - 1ULL) / sizeof(ElemType);
  // Use at max about 50% of all vidmem for perf testing
  // Also, some vector allocators like MemAlloc cannot handle more than this,
  // as they would run out of memory during the grow algorithm
  const size_t maxN = 3ULL * free / (4ULL * sizeof(ElemType));

  CUDA_CHECK(cuDeviceGetAttribute(
      &supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
      dev));
  if (!supportsVMM) {
    std::cout << "This device doesn't supports VMM!" << std::endl;
    return 0;
  }

  std::cout << "====== cuMemMap ElemSz=" << sizeof(ElemType)
            << " ======" << std::endl;
  runVectorPerfTest<MemMapAllocator, ElemType>(ctx, minN, maxN, durations[0],
                                               durations[1]);
  // Quick and dirty table of results
  std::cout << "Size(bytes)    | "
            << "cuMemMap(us)   | "
            << "cuMemMapRes(us)| ";

  std::cout << std::endl;

  for (size_t i = 0; i < durations[0].size(); i++) {
    std::cout << std::left << std::setw(15) << std::setfill(' ') << (minN << i)
              << "| ";
    for (size_t j = 0; j < durations.size(); j++) {
      std::cout << std::left << std::setw(15) << std::setfill(' ')
                << std::setprecision(2) << std::fixed
                << std::chrono::duration_cast<
                       std::chrono::duration<float, std::micro> >(
                       durations[j][i])
                       .count()
                << "| ";
    }
    std::cout << std::endl;
  }

  CUDA_CHECK(cuDevicePrimaryCtxRelease(0));

  return 0;
}
