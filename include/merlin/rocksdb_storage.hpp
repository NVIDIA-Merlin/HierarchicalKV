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

#include <rocksdb/db.h>
#include <iostream>
#include <merlin/debug.hpp>
#include <merlin/external_storage.hpp>
#include <merlin/memory_pool.cuh>
#include <sstream>
#include <string>

#ifdef ROCKSDB_CHECK
#error Unexpected redfinition of ROCKSDB_CHECK! Something is wrong.
#endif

#define ROCKSDB_CHECK(status)                                        \
  do {                                                               \
    if (!status.ok()) {                                              \
      std::cerr << __FILE__ << ':' << __LINE__ << ": RocksDB error " \
                << status.ToString() << '\n';                        \
      std::abort();                                                  \
    }                                                                \
  } while (0)

namespace nv {
namespace merlin {

struct RocksDBStorageOptions {
  std::string path;
  std::string column_name{rocksdb::kDefaultColumnFamilyName};
  bool read_only{};
  MemoryPoolOptions host_mem_pool;
};

std::ostream& operator<<(std::ostream& os, const RocksDBStorageOptions& opts) {
  return os << std::setw(15) << "path"
            << ": " << opts.path << '\n'
            << std::setw(15) << "column_name"
            << ": " << opts.column_name << '\n'
            << std::setw(15) << "read_only"
            << ": " << opts.read_only;
}

template <class Key, class Value>
class RocksDBStorage : public ExternalStorage<Key, Value> {
 public:
  using base_type = ExternalStorage<Key, Value>;

  using size_type = typename base_type::size_type;
  using key_type = typename base_type::key_type;
  using value_type = typename base_type::value_type;

  using host_mem_pool_type = MemoryPool<HostAllocator<char>>;

  RocksDBStorage(const RocksDBStorageOptions& opts) {
    MERLIN_CHECK(
        !opts.path.empty(),
        "Must provide where the database files are / should be stored!");
    MERLIN_CHECK(!opts.column_name.empty(),
                 "Must specify a RocksDB column group!");

    // Basic behavior.
    rocksdb::Options rdb_opts;
    rdb_opts.create_if_missing = true;
    rdb_opts.manual_wal_flush = true;
    rdb_opts.OptimizeForPointLookup(8);
    rdb_opts.OptimizeLevelStyleCompaction();
    rdb_opts.IncreaseParallelism(32);

    // Configure various behaviors and options used in later operations.
    rocksdb::ColumnFamilyOptions col_fam_opts;
    col_fam_opts.OptimizeForPointLookup(8);
    col_fam_opts.OptimizeLevelStyleCompaction();

    read_opts_.readahead_size = 2 * 1024 * 1024;
    read_opts_.verify_checksums = false;

    write_opts_.sync = false;
    write_opts_.disableWAL = false;
    write_opts_.no_slowdown = false;

    // Connect to DB with all column families.
    {
      // Enumerate column families and link to our preferred options.
      std::vector<rocksdb::ColumnFamilyDescriptor> col_descs;
      {
        std::vector<std::string> col_names;
        if (!rocksdb::DB::ListColumnFamilies(rdb_opts, opts.path, &col_names)
                 .ok()) {
          col_names.clear();
        }

        bool has_default{};
        for (const std::string& cn : col_names) {
          has_default |= cn == rocksdb::kDefaultColumnFamilyName;
        }
        if (!has_default) {
          col_names.push_back(rocksdb::kDefaultColumnFamilyName);
        }

        for (const std::string& cn : col_names) {
          col_descs.emplace_back(cn, col_fam_opts);
        }
      }

      // Connect to database.
      rocksdb::DB* db;
      if (opts.read_only) {
        ROCKSDB_CHECK(rocksdb::DB::OpenForReadOnly(
            rdb_opts, opts.path, col_descs, &col_handles_, &db));
      } else {
        ROCKSDB_CHECK(rocksdb::DB::Open(rdb_opts, opts.path, col_descs,
                                        &col_handles_, &db));
      }
      db_.reset(db);
    }

    // Create column family for this storage, if it doesn't exist yet.
    for (rocksdb::ColumnFamilyHandle* const ch : col_handles_) {
      if (ch->GetName() == opts.column_name) {
        col_handle_ = ch;
      }
    }
    if (!col_handle_) {
      ROCKSDB_CHECK(db_->CreateColumnFamily(col_fam_opts, opts.column_name,
                                            &col_handle_));
      col_handles_.emplace_back(col_handle_);
    }

    // Create memory pools.
    host_mem_pool_ = std::make_unique<host_mem_pool_type>(opts.host_mem_pool);
  }

  virtual ~RocksDBStorage() {
    // Destroy memory pool.
    host_mem_pool_.reset();

    // Synchronize and close database.
    ROCKSDB_CHECK(db_->SyncWAL());
    for (auto& ch : col_handles_) {
      ROCKSDB_CHECK(db_->DestroyColumnFamilyHandle(ch));
    }
    col_handles_.clear();

    ROCKSDB_CHECK(db_->Close());
    db_.reset();
  }

  virtual void insert_or_assign(
      const size_type n,
      const key_type* const d_keys,      // (n)
      const value_type* const d_values,  // (n * value_dims)
      const size_type value_dims, cudaStream_t stream) override {
    const size_t ws_size{(sizeof(key_type) + sizeof(value_type) * value_dims) *
                         n};
    auto ws{host_mem_pool_->get_workspace<1>(ws_size, stream)};

    // Copy keys & values to host.
    auto h_keys{ws.get<key_type*>(0)};
    auto h_values{reinterpret_cast<value_type*>(h_keys + n)};

    CUDA_CHECK(cudaMemcpyAsync(h_keys, d_keys, sizeof(key_type) * n,
                               cudaMemcpyDeviceToHost, stream));
    const size_type value_size{sizeof(value_type) * value_dims};
    CUDA_CHECK(cudaMemcpyAsync(h_values, d_values, value_size * n,
                               cudaMemcpyDeviceToHost, stream));

    // Create some structures that we will need.
    rocksdb::WriteBatch batch(12 +  // rocksdb::WriteBatchInternal::kHeader
                              n * (sizeof(char) +
                                   sizeof(uint32_t) +  // column_id
                                   sizeof(uint32_t) + sizeof(key_type) +  // key
                                   sizeof(uint32_t) + value_size  // value
                                   ));

    rocksdb::ColumnFamilyHandle* const col_handle{col_handle_};
    rocksdb::Slice k_view{nullptr, sizeof(key_type)};
    rocksdb::Slice v_view{nullptr, value_size};

    // Ensure copy operation is complete.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_type i{}; i != n; ++i) {
      k_view.data_ = reinterpret_cast<const char*>(&h_keys[i]);
      v_view.data_ = reinterpret_cast<const char*>(&h_values[i * value_dims]);
      ROCKSDB_CHECK(batch.Put(col_handle, k_view, v_view));
    }
    ROCKSDB_CHECK(db_->Write(write_opts_, &batch));
    ROCKSDB_CHECK(db_->FlushWAL(true));
  }

  virtual size_type find(const size_type n,
                         const key_type* const d_keys,  // (n)
                         value_type* const d_values,    // (n * value_dims)
                         const size_type value_dims, bool* const d_founds,
                         cudaStream_t stream) const override {
    const size_t ws_size{(sizeof(key_type) + sizeof(bool)) * n};
    auto ws{host_mem_pool_->get_workspace<1>(ws_size, stream)};

    auto h_keys{ws.get<key_type*>(0)};
    auto h_founds{reinterpret_cast<bool*>(h_keys + n)};

    // Copy keys and founds to host.
    CUDA_CHECK(cudaMemcpyAsync(h_keys, d_keys, sizeof(key_type) * n,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_founds, d_founds, sizeof(bool) * n,
                               cudaMemcpyDeviceToHost, stream));
    const size_type value_size{sizeof(value_type) * value_dims};

    std::vector<rocksdb::ColumnFamilyHandle*> col_handles(n, col_handle_);
    std::vector<rocksdb::Slice> k_views;
    k_views.reserve(n);
    std::vector<std::string> v_views;
    v_views.reserve(n);

    // Ensure copy operation is complete.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_type i{}; i != n; ++i) {
      k_views.emplace_back(reinterpret_cast<const char*>(&h_keys[i]),
                           sizeof(key_type));
    }

    const std::vector<rocksdb::Status> statuses{
        db_->MultiGet(read_opts_, col_handles, k_views, &v_views)};

    size_type miss_count{};
    for (size_type i{}; i != n; ++i) {
      const rocksdb::Status& s{statuses[i]};
      if (s.ok()) {
        auto& v_view{v_views[i]};
        MERLIN_CHECK(v_view.size() == value_size, "Value size mismatch!");
        CUDA_CHECK(cudaMemcpyAsync(&d_values[i * value_dims], v_view.data(),
                                   value_size, cudaMemcpyHostToDevice, stream));
        h_founds[i] = true;
      } else if (s.IsNotFound()) {
        ++miss_count;
      } else {
        ROCKSDB_CHECK(s);
      }
    }

    // Copy founds back and ensure we finished copying.
    CUDA_CHECK(cudaMemcpyAsync(d_founds, h_founds, sizeof(bool) * n,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return n - miss_count;
  }

  virtual void erase(size_type n, const key_type* d_keys,
                     cudaStream_t stream) override {
    const size_t ws_size{sizeof(key_type) * n};
    auto ws{host_mem_pool_->get_workspace<1>(ws_size, stream)};

    auto h_keys{ws.get<key_type*>(0)};

    // Copy keys to host.
    CUDA_CHECK(cudaMemcpyAsync(h_keys, d_keys, sizeof(key_type) * n,
                               cudaMemcpyDeviceToHost, stream));

    // Create some structures that we will need.
    rocksdb::WriteBatch batch(12 +  // rocksdb::WriteBatchInternal::kHeader
                              n * (sizeof(char) +
                                   sizeof(uint32_t) +  // column_id
                                   sizeof(uint32_t) + sizeof(key_type)  // key
                                   ));

    rocksdb::ColumnFamilyHandle* const col_handle{col_handle_};
    rocksdb::Slice k_view{nullptr, sizeof(key_type)};

    // Ensure copy operation is complete.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_type i{}; i != n; ++i) {
      k_view.data_ = reinterpret_cast<const char*>(&h_keys[i]);
      ROCKSDB_CHECK(batch.Delete(col_handle, k_view));
    }
    ROCKSDB_CHECK(db_->Write(write_opts_, &batch));
    ROCKSDB_CHECK(db_->FlushWAL(true));
  }

 private:
  rocksdb::ReadOptions read_opts_;
  rocksdb::WriteOptions write_opts_;

  std::unique_ptr<rocksdb::DB> db_;
  std::vector<rocksdb::ColumnFamilyHandle*> col_handles_;
  rocksdb::ColumnFamilyHandle* col_handle_{};

  std::unique_ptr<host_mem_pool_type> host_mem_pool_;
};

}  // namespace merlin
}  // namespace nv