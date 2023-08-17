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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <iostream>
#include "merlin/allocator.cuh"
#include "merlin/memory_pool.cuh"

using namespace nv::merlin;

/**
 * Wrapper around another allocator that prints debug messages.
 */
template <class Allocator>
struct DebugAllocator final
    : AllocatorBase<typename Allocator::type, DebugAllocator<Allocator>> {
  using type = typename Allocator::type;

  static constexpr const char* name{"DebugAllocator"};

  inline static type* alloc(size_t n, BaseAllocator* allocator,
                            cudaStream_t stream = 0) {
    type* ptr{Allocator::alloc(n, allocator, stream)};
    std::cout << Allocator::name << "[type_name = " << typeid(type).name()
              << "]: " << static_cast<void*>(ptr) << " allocated = " << n
              << " x " << sizeof(type) << " bytes, stream = " << stream << '\n';
    return ptr;
  }

  inline static void free(type* ptr, BaseAllocator* allocator,
                          cudaStream_t stream = 0) {
    Allocator::free(ptr, allocator, stream);
    std::cout << Allocator::name << "[type_name = " << typeid(type).name()
              << "]: " << static_cast<void*>(ptr)
              << " freed, stream = " << stream << '\n';
  }
};

void print_divider() {
  for (size_t i{0}; i < 80; ++i) std::cout << '-';
  std::cout << '\n';
}

void print_pool_options(const MemoryPoolOptions& opt) {
  print_divider();
  std::cout << "Memory Pool Configuration\n";
  print_divider();
  std::cout << "opt.max_stock   : " << opt.max_stock << " buffers\n";
  std::cout << "opt.max_pending : " << opt.max_pending << " buffers\n";
  print_divider();
  std::cout.flush();
}

MemoryPoolOptions opt{
    3,  //< max_stock
    5,  //< max_pending
};

struct SomeType {
  int a;
  float b;

  friend std::ostream& operator<<(std::ostream&, const SomeType&);
};

std::ostream& operator<<(std::ostream& os, const SomeType& obj) {
  cudaPointerAttributes attr;
  CUDA_CHECK(cudaPointerGetAttributes(&attr, &obj));

  SomeType tmp;
  if (attr.type == cudaMemoryTypeDevice) {
    CUDA_CHECK(
        cudaMemcpy(&tmp, &obj, sizeof(SomeType), cudaMemcpyDeviceToHost));
  } else {
    tmp = obj;
  }

  os << "a = " << tmp.a << ", b = " << tmp.b;
  return os;
}

void test_standard_allocator() {
  using Allocator = DebugAllocator<StandardAllocator<SomeType>>;
  std::shared_ptr<DefaultAllocator> default_allocator(new DefaultAllocator());

  {
    auto ptr{Allocator::make_unique(1, default_allocator.get())};
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "Sync UPtr after alloc: " << *ptr << std::endl;
    ptr->a = 47;
    ptr->b = 11;
    std::cout << "Sync UPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr{Allocator::make_unique(1, default_allocator.get(), nullptr)};
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "Async UPtr after alloc: " << *ptr << std::endl;
    ptr->a = 47;
    ptr->b = 11;
    std::cout << "Async UPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr{Allocator::make_shared(1, default_allocator.get())};
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "SPtr after alloc: " << *ptr << std::endl;
    ptr->a = 47;
    ptr->b = 11;
    std::cout << "SPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }
}

void test_host_allocator() {
  using Allocator = DebugAllocator<HostAllocator<SomeType>>;
  std::shared_ptr<DefaultAllocator> default_allocator(new DefaultAllocator());

  {
    auto ptr{Allocator::make_unique(1, default_allocator.get())};
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "Sync UPtr after alloc: " << *ptr << std::endl;
    ptr->a = 47;
    ptr->b = 11;
    std::cout << "Sync UPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr{Allocator::make_unique(1, default_allocator.get(), nullptr)};
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "Async UPtr after alloc: " << *ptr << std::endl;
    ptr->a = 47;
    ptr->b = 11;
    std::cout << "Async UPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr{Allocator::make_shared(1, default_allocator.get())};
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "SPtr after alloc: " << *ptr << std::endl;
    ptr->a = 47;
    ptr->b = 11;
    std::cout << "SPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }
}

void test_device_allocator() {
  using Allocator = DebugAllocator<DeviceAllocator<SomeType>>;
  std::shared_ptr<DefaultAllocator> default_allocator(new DefaultAllocator());

  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  MERLIN_CHECK(num_devices > 0,
               "Need at least one CUDA capable device for running this test.");

  CUDA_CHECK(cudaSetDevice(num_devices - 1));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  {
    auto ptr{Allocator::make_unique(1, default_allocator.get())};
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "Sync UPtr after alloc: " << *ptr << std::endl;
    const SomeType tmp{47, 11};

    std::cout << "Sync UPtr after alloc get ptr: " << ptr.get() << std::endl;
    CUDA_CHECK(cudaMemset(ptr.get(), 0, sizeof(SomeType)));
    CUDA_CHECK(
        cudaMemcpy(ptr.get(), &tmp, sizeof(SomeType), cudaMemcpyHostToDevice));
    std::cout << "Sync UPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr{Allocator::make_unique(1, default_allocator.get(), stream)};
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "Async UPtr after alloc: " << *ptr << std::endl;
    const SomeType tmp{47, 11};
    CUDA_CHECK(
        cudaMemcpy(ptr.get(), &tmp, sizeof(SomeType), cudaMemcpyHostToDevice));
    std::cout << "Async UPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr{Allocator::make_shared(1, default_allocator.get(), stream)};
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "SPtr after alloc: " << *ptr << std::endl;
    const SomeType tmp{47, 11};
    CUDA_CHECK(
        cudaMemcpy(ptr.get(), &tmp, sizeof(SomeType), cudaMemcpyHostToDevice));
    std::cout << "SPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
}

void test_borrow_return_no_context() {
  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  MERLIN_CHECK(num_devices > 0,
               "Need at least one CUDA capable device for running this test.");
  CUDA_CHECK(cudaSetDevice(0));

  std::shared_ptr<DefaultAllocator> default_allocator(new DefaultAllocator());
  {
    MemoryPool<DebugAllocator<DeviceAllocator<SomeType>>> pool{
        opt, default_allocator.get()};
    const size_t buffer_size{256L * 1024};

    // Initial status.
    std::cout << ".:: Initial state ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow and return one buffer (unique ptr).
    {
      auto buffer{pool.get_unique(buffer_size)};
      std::cout << ".:: Borrow 1 (unique) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 1 (unique) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow and return one buffer (shared ptr).
    {
      auto buffer{pool.get_shared(buffer_size)};
      std::cout << ".:: Borrow 1 (shared) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 1 (shared) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow static workspace with less than `max_stock` buffers.
    {
      auto ws{pool.get_workspace<2>(buffer_size)};
      std::cout << ".:: Borrow 2 (static) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 2 (static) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 2);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow dynamic workspace with less than `max_stock` buffers.
    {
      auto ws{pool.get_workspace(2, buffer_size)};
      std::cout << ".:: Borrow 2 (dynamic) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }

    std::cout << ".:: Return 2 (dynamic) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 2);
    ASSERT_EQ(pool.num_pending(), 0);

    // Await unfinished GPU work (shouldn't change anything).
    pool.await_pending();
    std::cout << ".:: Await pending (shouldn't change anything) ::.\n"
              << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 2);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow workspace that exceeds base pool size.
    {
      auto ws{pool.get_workspace<6>(buffer_size)};
      std::cout << ".:: Borrow 6 (static) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 6 (static) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), opt.max_stock);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow a buffer that is smaller than the current buffer size.
    {
      auto ws{pool.get_unique(buffer_size / 2)};
      std::cout << ".:: Borrow 1 (smaller) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), opt.max_stock - 1);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 1 (smaller) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), opt.max_stock);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow a buffer that is bigger than the current buffer size.
    {
      auto ws{pool.get_unique(buffer_size + 37)};
      std::cout << ".:: Borrow 1 (bigger) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 1 (smaller) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);
  }
}

void test_borrow_return_with_context() {
  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  MERLIN_CHECK(num_devices > 0,
               "Need at least one CUDA capable device for running this test.");
  CUDA_CHECK(cudaSetDevice(0));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::shared_ptr<DefaultAllocator> default_allocator(new DefaultAllocator());
  {
    MemoryPool<DebugAllocator<DeviceAllocator<SomeType>>> pool(
        opt, default_allocator.get());
    const size_t buffer_size{256L * 1024};

    // Initial status.
    std::cout << ".:: Initial state ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow and return one buffer (unique ptr).
    {
      auto buffer{pool.get_unique(buffer_size, stream)};
      std::cout << ".:: Borrow 1 (unique) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 1 (unique) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 1);

    // Await unfinished GPU work (ensure stable situation).
    pool.await_pending(stream);
    std::cout << ".:: Await pending ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow and return one buffer (shared ptr).
    {
      auto buffer{pool.get_shared(buffer_size, stream)};
      std::cout << ".:: Borrow 1 (shared) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 1 (shared) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 1);

    // Await unfinished GPU work (ensure stable situation).
    pool.await_pending(stream);
    std::cout << ".:: Await pending ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow static workspace with less than `max_stock` buffers.
    {
      auto ws{pool.get_workspace<2>(buffer_size, stream)};
      std::cout << ".:: Borrow 2 (static) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 2 (static) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 2);

    // Await unfinished GPU work (ensure stable situation).
    pool.await_pending(stream);
    std::cout << ".:: Await pending ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 2);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow workspace that exceeds base pool size. Possible results:
    // 1. If this thread is slower than the driver.
    //    Upon return we will see a partial deallocation before inserting the
    //    last buffer into the pending queue.
    // 2. If this the driver is slower than this thread queuing/querying events.
    //    Either 0-3 buffers in stock partial dallocation
    //    1-5 buffers pending. Hence there is no good way to check.
    {
      auto ws{pool.get_workspace<6>(buffer_size, stream)};
      std::cout << ".:: Borrow 6 (static) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 6 (static) ::.\n" << pool << std::endl;
    ASSERT_GE(pool.num_pending(), 1);

    // Await unfinished GPU work (ensure stable situation).
    pool.await_pending(stream);
    std::cout << ".:: Await pending ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 3);
    ASSERT_EQ(pool.num_pending(), 0);

    // Pin 1 and deplete stock.
    {
      auto ws{pool.get_workspace<1>(buffer_size, stream)};
      pool.deplete_stock();
      std::cout << ".:: Deplete stock ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Deplete stock ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 1);

    // Await unfinished GPU work (ensure stable situation).
    pool.await_pending(stream);
    std::cout << ".:: Await pending ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);

    // Increase stock to 3 buffers.
    { auto ws{pool.get_workspace<3>(buffer_size, stream)}; }
    pool.await_pending(stream);
    ASSERT_EQ(pool.current_stock(), 3);
    ASSERT_EQ(pool.num_pending(), 0);

    // Pin 1 of the 3 buffers and release it to make it pending.
    { auto ws{pool.get_workspace<1>(buffer_size, stream)}; }
    ASSERT_EQ(pool.current_stock(), 2);
    ASSERT_EQ(pool.num_pending(), 1);
    std::cout << ".:: Ensure 2 stock + 1 pending situation ::.\n"
              << pool << std::endl;

    // Borrow a buffer that is smaller than the current buffer size.
    {
      auto ws{pool.get_unique(buffer_size / 2, stream)};
      std::cout << ".:: Borrow 1 (smaller) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 1);
      ASSERT_EQ(pool.num_pending(), 1);
    }
    std::cout << ".:: Return 1 (smaller) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 2);

    // Borrow a buffer that is bigger than the current buffer size. This will
    // evict the stock buffers which are smaller, but will not concern the
    // buffers that are still pending.
    {
      auto ws{pool.get_unique(buffer_size + 37, stream)};
      std::cout << ".:: Borrow 1 (bigger) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 2);
    }
    std::cout << ".:: Return 1 (bigger) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 3);

    // Because there are now pending buffers that are too small, they will be
    // cleared once the associated work has been completed.
    pool.await_pending(stream);
    std::cout << ".:: Await pending ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);

    CUDA_CHECK(cudaStreamDestroy(stream));
  }
}

TEST(MemoryPoolTest, standard_allocator) { test_standard_allocator(); }
TEST(MemoryPoolTest, host_allocator) { test_host_allocator(); }
TEST(MemoryPoolTest, device_allocator) { test_device_allocator(); }

TEST(MemoryPoolTest, borrow_return_no_context) {
  test_borrow_return_no_context();
}
TEST(MemoryPoolTest, borrow_return_with_context) {
  test_borrow_return_with_context();
}