/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if GOOGLE_CUDA

#include "merlin_kv/tensorflow/core/kernels/merlin_kv_op_gpu.h"
// Never
#include "merlin_kv/tensorflow/core/kernels/lookup_impl/lookup_table_op_gpu.h"
// CaseInsensitive

#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include <stdlib.h>

#include <cstdlib>
#include <iomanip>
#include <type_traits>
#include <utility>

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {
namespace merlin_kv {
namespace lookup {

using tensorflow::OpKernelContext;
using tensorflow::lookup::LookupInterface;

template <class K, class V, class M = uint64_t>
class MerlinKVOfTensorsGpu final : public LookupInterface {
 public:
  MerlinKVOfTensorsGpu(OpKernelContext* ctx, OpKernel* kernel) {
    int64 init_size = 0;

    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "init_size", &init_size));

    if (init_size == 0) {
      int64 env_var = 0;
      Status status = ReadInt64FromEnvVar("TF_HASHTABLE_INIT_SIZE",
                                          1024 * 8,  // 8192 KV pairs by default
                                          &env_var);
      min_size_ = (size_t)env_var;
      max_size_ = (size_t)env_var;
    } else {
      min_size_ = init_size;
      max_size_ = init_size;
    }

    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(value_shape_),
        errors::InvalidArgument("Default value must be a vector, got shape ",
                                value_shape_.DebugString()));
    runtime_dim_ = value_shape_.dim_size(0);
    OP_REQUIRES(ctx, (runtime_dim_ <= 200),
                errors::InvalidArgument("The dim of HashTable on GPU should be "
                                        "less than or equal to 200, got ",
                                        runtime_dim_));
    this->CreateTable(max_size_, &table_);
    OP_REQUIRES(ctx, (table_ != nullptr),
                errors::InvalidArgument("HashTable on GPU is created failed!"));

    LOG(INFO) << "HashTable on GPU is created successfully:"
              << " K=" << std::type_index(typeid(K)).name()
              << ", V=" << std::type_index(typeid(V)).name()
              << ", max_size=" << max_size_ << ", min_size=" << min_size_;
  }

  ~MerlinKVOfTensorsGpu() { delete table_; }

  void CreateTable(size_t max_size, gpu::TableWrapperBase<K, V>** pptable) {
    if (runtime_dim_ <= 50) {
      gpu::CreateTable0(max_size, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 100) {
      gpu::CreateTable1(max_size, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 150) {
      gpu::CreateTable2(max_size, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 200) {
      gpu::CreateTable3(max_size, runtime_dim_, pptable);
    } else {
      *pptable = nullptr;
    }
  }

  size_t size() const override {
    tf_shared_lock l(mu_);

    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));
    size_t retv = table_->get_size(_stream);
    CUDA_CHECK(cudaStreamSynchronize(_stream));
    CUDA_CHECK(cudaStreamDestroy(_stream));
    return retv;
  }

  Status Find(OpKernelContext* ctx, const Tensor& d_keys, Tensor* value,
              const Tensor& default_value) override {
    size_t len = d_keys.flat<K>().size();
    bool* d_status;

    auto value_flat = value->flat_inner_dims<V, 2>();
    const auto default_flat = default_value.flat<V>();
    int64 total = value_flat.size();
    int64 default_total = default_flat.size();
    bool is_full_default = (total == default_total);

    cudaStream_t _stream;

    if (len > 0) {
      size_t default_value_num =
          is_full_default ? default_value.shape().dim_size(0) : 1;
      CUDA_CHECK(cudaStreamCreate(&_stream));
      CUDA_CHECK(cudaMallocManaged((void**)&d_status, sizeof(bool) * len));
      {
        tf_shared_lock l(mu_);
        table_->get((const K*)d_keys.tensor_data().data(),
                    (gpu::ValueArrayBase<V>*)value->tensor_data().data(),
                    d_status, len,
                    (gpu::ValueArrayBase<V>*)default_value.tensor_data().data(),
                    is_full_default, _stream);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaFree(d_status));
      CUDA_CHECK(cudaStreamDestroy(_stream));
    }
    return Status::OK();
  }

  Status FindWithMetas(OpKernelContext* ctx, const Tensor& d_keys,
                       Tensor* value, Tensor* metas,
                       const Tensor& default_value) {
    size_t len = d_keys.flat<K>().size();
    bool* d_status;

    auto value_flat = value->flat_inner_dims<V, 2>();
    const auto default_flat = default_value.flat<V>();
    auto d_metas = metas->tensor_data().data();
    int64 total = value_flat.size();
    int64 default_total = default_flat.size();
    bool is_full_default = (total == default_total);

    cudaStream_t _stream;

    if (len > 0) {
      size_t default_value_num =
          is_full_default ? default_value.shape().dim_size(0) : 1;
      CUDA_CHECK(cudaStreamCreate(&_stream));
      CUDA_CHECK(cudaMallocManaged((void**)&d_status, sizeof(bool) * len));
      CUDA_CHECK(cudaMemset((void*)d_metas, 0, sizeof(M) * len));
      {
        tf_shared_lock l(mu_);
        table_->get((const K*)d_keys.tensor_data().data(),
                    (gpu::ValueArrayBase<V>*)value->tensor_data().data(),
                    (M*)metas->tensor_data().data(), d_status, len,
                    (gpu::ValueArrayBase<V>*)default_value.tensor_data().data(),
                    is_full_default, _stream);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaFree(d_status));
      CUDA_CHECK(cudaStreamDestroy(_stream));
    }
    return Status::OK();
  }

  Status FindWithExists(OpKernelContext* ctx, const Tensor& d_keys,
                        Tensor* value, const Tensor& default_value,
                        Tensor* exists) {
    size_t len = d_keys.flat<K>().size();

    auto value_flat = value->flat_inner_dims<V, 2>();
    const auto default_flat = default_value.flat<V>();
    int64 total = value_flat.size();
    int64 default_total = default_flat.size();
    bool is_full_default = (total == default_total);

    cudaStream_t _stream;

    if (len > 0) {
      size_t default_value_num =
          is_full_default ? default_value.shape().dim_size(0) : 1;
      CUDA_CHECK(cudaStreamCreate(&_stream));
      {
        tf_shared_lock l(mu_);
        table_->get((const K*)d_keys.tensor_data().data(),
                    (gpu::ValueArrayBase<V>*)value->tensor_data().data(),
                    (bool*)exists->tensor_data().data(), len,
                    (gpu::ValueArrayBase<V>*)default_value.tensor_data().data(),
                    is_full_default, _stream);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaStreamDestroy(_stream));
    }
    return Status::OK();
  }

  void RehashIfNeeded(cudaStream_t stream) {
    return;
    /*
    K* d_keys;
    gpu::ValueArrayBase<V>* d_values;
    size_t* d_dump_counter;
    size_t new_max_size = max_size_;

    size_t total_size = table_->get_size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (total_size >= 0.75 * max_size_) {
      new_max_size = max_size_ * 2;
    }
    if (total_size < 0.25 * max_size_ && max_size_ > min_size_) {
      new_max_size = max_size_ / 2;
    }
    if (new_max_size != max_size_) {  // rehash manually.
      size_t capacity = table_->get_capacity();
      size_t h_dump_counter = 0;
      CUDA_CHECK(cudaMallocManaged((void**)&d_dump_counter, sizeof(size_t)));
      CUDA_CHECK(cudaMallocManaged((void**)&d_keys, sizeof(K) * capacity));
      CUDA_CHECK(cudaMallocManaged((void**)&d_values,
                                   sizeof(V) * runtime_dim_ * capacity));
      table_->dump(d_keys, (gpu::ValueArrayBase<V>*)d_values, 0, capacity,
                   d_dump_counter, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      delete table_;
      table_ = NULL;
      CreateTable(new_max_size, &table_);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMemcpy((size_t*)&h_dump_counter, (size_t*)d_dump_counter,
                            sizeof(size_t), cudaMemcpyDefault));
      table_->upsert((const K*)d_keys, (const gpu::ValueArrayBase<V>*)d_values,
                     h_dump_counter, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(d_keys));
      CUDA_CHECK(cudaFree(d_values));
      CUDA_CHECK(cudaFree(d_dump_counter));
      max_size_ = new_max_size;
      LOG(INFO) << "HashTable on GPU changes to new status: [size="
                << total_size << ", max_size=" << max_size_
                << ", load factor=" << std::setprecision(2)
                << (float)total_size / (float)max_size_ << "].";
    }
    */
  }

  Status Insert(OpKernelContext* ctx, const Tensor& keys, const Tensor& values,
                bool allow_duplicated_keys) {
    size_t len = keys.flat<K>().size();
    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));
    {
      mutex_lock l(mu_);
      RehashIfNeeded(_stream);
      table_->upsert((const K*)keys.tensor_data().data(),
                     (const gpu::ValueArrayBase<V>*)values.tensor_data().data(),
                     len, allow_duplicated_keys, _stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    };
    CUDA_CHECK(cudaStreamDestroy(_stream));

    return Status::OK();
  }

  Status Insert(OpKernelContext* ctx, const Tensor& keys,
                const Tensor& values) override {
    size_t len = keys.flat<K>().size();
    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));
    {
      mutex_lock l(mu_);
      RehashIfNeeded(_stream);
      table_->upsert((const K*)keys.tensor_data().data(),
                     (const gpu::ValueArrayBase<V>*)values.tensor_data().data(),
                     len, true, _stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    };
    CUDA_CHECK(cudaStreamDestroy(_stream));

    return Status::OK();
  }

  Status InsertWithMetas(OpKernelContext* ctx, const Tensor& keys,
                         const Tensor& values, const Tensor& metas,
                         bool allow_duplicated_keys) {
    size_t len = keys.flat<K>().size();
    cudaStream_t _stream;

    CUDA_CHECK(cudaStreamCreate(&_stream));
    {
      mutex_lock l(mu_);
      RehashIfNeeded(_stream);
      table_->upsert((const K*)keys.tensor_data().data(),
                     (const gpu::ValueArrayBase<V>*)values.tensor_data().data(),
                     (const M*)metas.tensor_data().data(), len,
                     allow_duplicated_keys, _stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    };
    CUDA_CHECK(cudaStreamDestroy(_stream));

    return Status::OK();
  }

  Status Accum(OpKernelContext* ctx, const Tensor& keys,
               const Tensor& values_or_deltas, const Tensor& exists) {
    size_t len = keys.flat<K>().size();
    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));
    {
      mutex_lock l(mu_);
      RehashIfNeeded(_stream);
      table_->accum(
          (const K*)keys.tensor_data().data(),
          (const gpu::ValueArrayBase<V>*)values_or_deltas.tensor_data().data(),
          (const bool*)exists.tensor_data().data(), len, _stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    };
    CUDA_CHECK(cudaStreamDestroy(_stream));

    return Status::OK();
  }

  Status Remove(OpKernelContext* ctx, const Tensor& keys) override {
    size_t len = keys.flat<K>().size();
    K* d_keys;
    cudaStream_t _stream;

    CUDA_CHECK(cudaStreamCreate(&_stream));
    if (len > 0) {
      CUDA_CHECK(cudaMallocManaged((void**)&d_keys, sizeof(K) * len));
      CUDA_CHECK(cudaMemcpy((void*)d_keys, (void*)keys.tensor_data().data(),
                            sizeof(K) * len, cudaMemcpyDefault));
      {
        mutex_lock l(mu_);
        table_->remove((const K*)d_keys, len, _stream);
        RehashIfNeeded(_stream);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaStreamDestroy(_stream));
      CUDA_CHECK(cudaFree(d_keys));
    }
    return Status::OK();
  }

  Status Clear(OpKernelContext* ctx) {
    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));
    {
      mutex_lock l(mu_);
      table_->clear(_stream);
      RehashIfNeeded(_stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    }
    CUDA_CHECK(cudaStreamDestroy(_stream));
    return Status::OK();
  }

  Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values) override {
    size_t len = keys.flat<K>().size();
    K* d_keys;
    gpu::ValueArrayBase<V>* d_values;
    bool allow_duplicated_keys = true;

    if (len > 0) {
      cudaStream_t _stream;
      CUDA_CHECK(cudaStreamCreate(&_stream));
      CUDA_CHECK(cudaMallocManaged((void**)&d_keys, sizeof(K) * len));
      CUDA_CHECK(
          cudaMallocManaged((void**)&d_values, sizeof(V) * runtime_dim_ * len));
      CUDA_CHECK(cudaMemcpy((void*)d_keys, (void*)keys.tensor_data().data(),
                            sizeof(K) * len, cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy((void*)d_values, (void*)values.tensor_data().data(),
                            sizeof(V) * runtime_dim_ * len, cudaMemcpyDefault));
      {
        mutex_lock l(mu_);
        table_->clear(_stream);
        table_->upsert((const K*)d_keys,
                       (const gpu::ValueArrayBase<V>*)d_values, len,
                       allow_duplicated_keys, _stream);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaStreamDestroy(_stream));
      CUDA_CHECK(cudaFree(d_keys));
      CUDA_CHECK(cudaFree(d_values));
    }
    return Status::OK();
  }

  Status ExportValues(OpKernelContext* ctx) override {
    size_t len = 0;
    int64 size = 0;

    const size_t offset = 0;

    Tensor* keys;
    Tensor* values;

    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));

    {
      tf_shared_lock l(mu_);
      len = table_->get_capacity();
      size = (int64)table_->get_size(_stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    }

    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    attr.set_on_host(false);

    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({(size)}), &keys, attr));
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values", TensorShape({size, (int64)runtime_dim_}), &values, attr));
    if (size) {
      tf_shared_lock l(mu_);
      table_->dump((K*)keys->flat<K>().data(),
                   (gpu::ValueArrayBase<V>*)values->matrix<V>().data(), offset,
                   len, _stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    }
    CUDA_CHECK(cudaStreamDestroy(_stream));
    return Status::OK();
  }

  Status ExportValuesToFile(OpKernelContext* ctx, const string filepath,
                            const size_t buffer_size) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    {
      tf_shared_lock l(mu_);
      table_->dump_to_file(ctx, filepath, runtime_dim_, stream, buffer_size);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return Status::OK();
  }

  Status ImportValuesFromFile(OpKernelContext* ctx, const string filepath,
                              const size_t buffer_size) {
    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));

    {
      tf_shared_lock l(mu_);

      string keyfile = filepath + ".keys";
      FILE* tmpfd = fopen(keyfile.c_str(), "rb");
      if (tmpfd == nullptr) {
        return errors::NotFound("Failed to read key file", keyfile);
      }
      fseek(tmpfd, 0, SEEK_END);
      long int filesize = ftell(tmpfd);
      if (filesize <= 0) {
        fclose(tmpfd);
        return errors::NotFound("Empty key file.");
      }
      size_t size = static_cast<size_t>(filesize) / sizeof(K);
      fseek(tmpfd, 0, SEEK_SET);
      fclose(tmpfd);

      table_->clear(_stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
	  // TODO: Expand the capacity when load_from_file.
      //RehashIfNeeded(_stream, size);
      table_->load_from_file(ctx, filepath, size, runtime_dim_, _stream,
                             buffer_size);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    }
    CUDA_CHECK(cudaStreamDestroy(_stream));
    return Status::OK();
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }
  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }
  TensorShape key_shape() const final { return TensorShape(); }
  TensorShape value_shape() const override { return value_shape_; }

 private:
  TensorShape value_shape_;
  size_t max_size_;
  size_t min_size_;
  size_t runtime_dim_;
  mutable mutex mu_;
  gpu::TableWrapperBase<K, V, M>* table_ = nullptr GUARDED_BY(mu_);
};

}  // namespace lookup

Status CheckKeyMetasShape(const Tensor& keys, const Tensor& metas) {
  if (!(keys.shape() == metas.shape())) {
    return errors::InvalidArgument("Input key shape ", keys.shape(),
                                   "and metas shape ", metas.shape(),
                                   " must the same shape!");
  }
  return Status::OK();
}

// Table lookup op. Perform the lookup operation on the given table.
class HashTableFindGpuOp : public OpKernel {
 public:
  explicit HashTableFindGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    // Input 0 could be a STRING_REF or a RESOURCE
    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& keys = ctx->input(1);
    const Tensor& default_values = ctx->input(2);

    TensorShape output_shape = keys.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor* out;
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("values", output_shape, &out, attr));

    OP_REQUIRES_OK(ctx, table->Find(ctx, keys, out, default_values));
  }
};

REGISTER_KERNEL_BUILDER(Name("MerlinKVFind").Device(DEVICE_GPU),
                        HashTableFindGpuOp);

// Table lookup op with return metas.
template <class K, class V, class M = uint64_t>
class HashTableFindWithMetasGpuOp : public OpKernel {
 public:
  explicit HashTableFindWithMetasGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    lookup::MerlinKVOfTensorsGpu<K, V>* table_merlin =
        (lookup::MerlinKVOfTensorsGpu<K, V>*)table;

    // Input 0 could be a STRING_REF or a RESOURCE
    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype(), DT_INT64};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& keys = ctx->input(1);
    const Tensor& default_values = ctx->input(2);

    TensorShape values_shape = keys.shape();
    TensorShape metas_shape = keys.shape();

    values_shape.RemoveLastDims(table->key_shape().dims());
    values_shape.AppendShape(table->value_shape());
    Tensor* values;
    Tensor* metas;
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("values", values_shape, &values, attr));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("metas", metas_shape, &metas, attr));

    OP_REQUIRES_OK(ctx, table_merlin->FindWithMetas(ctx, keys, values, metas,
                                                    default_values));
  }
};

// Table lookup op. Perform the lookup operation on the given table.

template <class K, class V>
class HashTableFindWithExistsGpuOp : public OpKernel {
 public:
  explicit HashTableFindWithExistsGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    lookup::MerlinKVOfTensorsGpu<K, V>* table_merlin =
        (lookup::MerlinKVOfTensorsGpu<K, V>*)table;

    // Input 0 could be a STRING_REF or a RESOURCE
    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype(), DT_BOOL};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& keys = ctx->input(1);
    const Tensor& default_values = ctx->input(2);

    TensorShape output_shape = keys.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor* values;
    Tensor* exists;
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("values", output_shape, &values, attr));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("exists", keys.shape(), &exists, attr));

    OP_REQUIRES_OK(ctx, table_merlin->FindWithExists(ctx, keys, values,
                                                     default_values, exists));
  }
};

// Table insert op.
template <class K, class V>
class HashTableInsertGpuOp : public OpKernel {
 public:
  explicit HashTableInsertGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("allow_duplicated_keys", &allow_duplicated_keys_));
  }

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    lookup::MerlinKVOfTensorsGpu<K, V>* table_merlin =
        (lookup::MerlinKVOfTensorsGpu<K, V>*)table;

    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES_OK(
        ctx, table_merlin->CheckKeyAndValueTensorsForInsert(keys, values));
    OP_REQUIRES_OK(
        ctx, table_merlin->Insert(ctx, keys, values, allow_duplicated_keys_));
  }

 private:
  bool allow_duplicated_keys_;
};

// Table insert with metas op.
template <class K, class V, class M = uint64_t>
class HashTableInsertWithMetasGpuOp : public OpKernel {
 public:
  explicit HashTableInsertWithMetasGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("allow_duplicated_keys", &allow_duplicated_keys_));
  }

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    lookup::MerlinKVOfTensorsGpu<K, V>* table_merlin =
        (lookup::MerlinKVOfTensorsGpu<K, V>*)table;

    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype(), DT_INT64};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    const Tensor& metas = ctx->input(3);

    OP_REQUIRES_OK(
        ctx, table_merlin->CheckKeyAndValueTensorsForInsert(keys, values));
    OP_REQUIRES_OK(ctx, CheckKeyMetasShape(keys, metas));
    OP_REQUIRES_OK(ctx, table_merlin->InsertWithMetas(ctx, keys, values, metas,
                                                      allow_duplicated_keys_));
  }

 private:
  bool allow_duplicated_keys_;
};

// Table accum op.
template <class K, class V>
class HashTableAccumGpuOp : public OpKernel {
 public:
  explicit HashTableAccumGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);
    lookup::MerlinKVOfTensorsGpu<K, V>* table_merlin =
        (lookup::MerlinKVOfTensorsGpu<K, V>*)table;

    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype(),
                                      DataTypeToEnum<bool>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values_or_deltas = ctx->input(2);
    const Tensor& exists = ctx->input(3);
    OP_REQUIRES_OK(
        ctx, table->CheckKeyAndValueTensorsForInsert(keys, values_or_deltas));
    OP_REQUIRES_OK(ctx,
                   table_merlin->Accum(ctx, keys, values_or_deltas, exists));
  }
};

// Table remove op.
class HashTableRemoveGpuOp : public OpKernel {
 public:
  explicit HashTableRemoveGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& key = ctx->input(1);
    OP_REQUIRES_OK(ctx, table->CheckKeyTensorForRemove(key));
    OP_REQUIRES_OK(ctx, table->Remove(ctx, key));
  }
};

REGISTER_KERNEL_BUILDER(Name("MerlinKVRemove").Device(DEVICE_GPU),
                        HashTableRemoveGpuOp);

// Table clear op.
template <class K, class V>
class HashTableClearGpuOp : public OpKernel {
 public:
  explicit HashTableClearGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);
    lookup::MerlinKVOfTensorsGpu<K, V>* table_merlin =
        (lookup::MerlinKVOfTensorsGpu<K, V>*)table;
    OP_REQUIRES_OK(ctx, table_merlin->Clear(ctx));
  }
};

// Op that returns the size of the given table.
class HashTableSizeGpuOp : public OpKernel {
 public:
  explicit HashTableSizeGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    Tensor* out;
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_on_host(false);

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("size", TensorShape({}), &out, attr));

    size_t size = table->size();
    const int64* p_size = (const int64*)out->flat<int64>().data();
    CUDA_CHECK(cudaMemcpy((void*)out->tensor_data().data(), (void*)&size,
                          sizeof(size_t), cudaMemcpyDefault));
  }
};

REGISTER_KERNEL_BUILDER(Name("MerlinKVSize").Device(DEVICE_GPU),
                        HashTableSizeGpuOp);

// Op that outputs tensors of all keys and all values.
class HashTableExportGpuOp : public OpKernel {
 public:
  explicit HashTableExportGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, table->ExportValues(ctx));
  }
};

REGISTER_KERNEL_BUILDER(Name("MerlinKVExport").Device(DEVICE_GPU),
                        HashTableExportGpuOp);

// Clear the table and insert data.
class HashTableImportGpuOp : public OpKernel {
 public:
  explicit HashTableImportGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForImport(keys, values));
    OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values));
  }
};

REGISTER_KERNEL_BUILDER(Name("MerlinKVImport").Device(DEVICE_GPU),
                        HashTableImportGpuOp);

template <class K, class V, class M = uint64_t>
class HashTableExportToFileGpuOp : public OpKernel {
 public:
  explicit HashTableExportToFileGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    int64 signed_buffer_size = 0;
    ctx->GetAttr("buffer_size", &signed_buffer_size);
    buffer_size_ = static_cast<size_t>(signed_buffer_size);
  }

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    const Tensor& ftensor = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ftensor.shape()),
                errors::InvalidArgument("filepath must be scalar."));
    string filepath = string(ftensor.scalar<tstring>()().data());
    lookup::MerlinKVOfTensorsGpu<K, V, M>* table_merlin =
        (lookup::MerlinKVOfTensorsGpu<K, V, M>*)table;
    OP_REQUIRES_OK(
        ctx, table_merlin->ExportValuesToFile(ctx, filepath, buffer_size_));
  }

 private:
  size_t buffer_size_;
};

// Op that import from file.
template <class K, class V, class M = uint64_t>
class HashTableImportFromFileGpuOp : public OpKernel {
 public:
  explicit HashTableImportFromFileGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    int64 signed_buffer_size = 0;
    ctx->GetAttr("buffer_size", &signed_buffer_size);
    buffer_size_ = static_cast<size_t>(signed_buffer_size);
  }

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    const Tensor& ftensor = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ftensor.shape()),
                errors::InvalidArgument("filepath must be scalar."));
    string filepath = string(ftensor.scalar<tstring>()().data());
    lookup::MerlinKVOfTensorsGpu<K, V, M>* table_merlin =
	    (lookup::MerlinKVOfTensorsGpu<K, V, M>*)table;
    OP_REQUIRES_OK(
        ctx, table_merlin->ImportValuesFromFile(ctx, filepath, buffer_size_));
  }

 private:
  size_t buffer_size_;
};

// Register the MerlinKVOfTensors op.

#define REGISTER_KERNEL(key_dtype, value_dtype)                                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MerlinKVOfTensors")                                                \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      HashTableGpuOp<lookup::MerlinKVOfTensorsGpu<key_dtype, value_dtype>,     \
                     key_dtype, value_dtype>)                                 \
  REGISTER_KERNEL_BUILDER(Name("MerlinKVClear")                                \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<key_dtype>("key_dtype")          \
                              .TypeConstraint<value_dtype>("value_dtype"),     \
                          HashTableClearGpuOp<key_dtype, value_dtype>)         \
  REGISTER_KERNEL_BUILDER(Name("MerlinKVAccum")                                \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<key_dtype>("key_dtype")          \
                              .TypeConstraint<value_dtype>("value_dtype"),     \
                          HashTableAccumGpuOp<key_dtype, value_dtype>)         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MerlinKVFindWithExists")                                           \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<key_dtype>("Tin")                                    \
          .TypeConstraint<value_dtype>("Tout"),                                \
      HashTableFindWithExistsGpuOp<key_dtype, value_dtype>)                    \
  REGISTER_KERNEL_BUILDER(Name("MerlinKVFindWithMetas")                        \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<key_dtype>("Tin")                \
                              .TypeConstraint<value_dtype>("Tout"),            \
                          HashTableFindWithMetasGpuOp<key_dtype, value_dtype>) \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MerlinKVInsertWithMetas")                                          \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<key_dtype>("Tin")                                    \
          .TypeConstraint<value_dtype>("Tout"),                                \
      HashTableInsertWithMetasGpuOp<key_dtype, value_dtype>)                   \
  REGISTER_KERNEL_BUILDER(Name("MerlinKVInsert")                               \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<key_dtype>("Tin")                \
                              .TypeConstraint<value_dtype>("Tout"),            \
                          HashTableInsertGpuOp<key_dtype, value_dtype>)        \
  REGISTER_KERNEL_BUILDER(Name("MerlinKVExportToFile")                         \
	                      .Device(DEVICE_GPU)                             \
                              .TypeConstraint<key_dtype>("key_dtype")           \
                              .TypeConstraint<value_dtype>("value_dtype"),      \
	                      HashTableExportToFileGpuOp<key_dtype, value_dtype>)  \
  REGISTER_KERNEL_BUILDER(Name("MerlinKVImportFromFile")                       \
	                      .Device(DEVICE_GPU)                             \
                              .TypeConstraint<key_dtype>("key_dtype")           \
                              .TypeConstraint<value_dtype>("value_dtype"),      \
	                      HashTableImportFromFileGpuOp<key_dtype, value_dtype>);

REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(int64, Eigen::half);
REGISTER_KERNEL(int64, int64);
REGISTER_KERNEL(int64, int32);
REGISTER_KERNEL(int64, int8);

#undef REGISTER_KERNEL

}  // namespace merlin_kv
}  // namespace tensorflow
#endif
