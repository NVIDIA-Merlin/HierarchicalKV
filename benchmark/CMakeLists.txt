# Optional standalone benchmark targets. Existing library/test targets stay in
# the repository root; ablation definitions are scoped to their own targets.
set(HKV_REPRO_SOURCES
  batch_size_benchmark
  bucket_size_benchmark
  concurrency_benchmark
  digest_ablation_benchmark
  e8_hkv_baseline
  e9_eviction_comparison
  e10_dual_bucket_analysis
  e17_single_vs_dual
  e19_cache_quality_sweep
  error_bars_benchmark
  hash_sensitivity_benchmark
  hkv_adversarial_performance_benchmark
  hkv_correctness_stress_benchmark
  hkv_hash_sensitivity_benchmark
  hkv_lookup_modes_benchmark
  latency_benchmark
  p25_admission_control
  score_strategy_benchmark
  tbe_like_cache_baseline
)

foreach(name IN LISTS HKV_REPRO_SOURCES)
  add_executable(${name} ${name}.cc.cu)
  target_compile_features(${name} PRIVATE cxx_std_17)
  set_target_properties(${name} PROPERTIES
    CUDA_ARCHITECTURES OFF RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach()

# A source-level sensitivity setting, not a claim that every hash variant has
# the same eviction quality or that the original measurements used this value.
set(HKV_REPRO_HASH_VARIANT 0 CACHE STRING "Hash variant for sensitivity target")
set(HKV_REPRO_HASH_OUTCOMES_AVAILABLE 0 CACHE STRING "Expose per-key hash outcome audit")
target_compile_definitions(hkv_hash_sensitivity_benchmark PRIVATE
  HKV_HASH_VARIANT=${HKV_REPRO_HASH_VARIANT}
  HKV_HASH_OUTCOMES_AVAILABLE=${HKV_REPRO_HASH_OUTCOMES_AVAILABLE})
target_compile_definitions(hash_sensitivity_benchmark PRIVATE
  HKV_HASH_VARIANT=${HKV_REPRO_HASH_VARIANT})

set(HKV_REPRO_LOOKUP_DIM 32 CACHE STRING "Lookup value dimension")
set(HKV_REPRO_LOOKUP_CAPACITY 134217728 CACHE STRING "Lookup table capacity")
set(HKV_REPRO_LOOKUP_BATCH_SIZE 1048576 CACHE STRING "Lookup batch size")
set(HKV_REPRO_LOOKUP_WARMUP 3 CACHE STRING "Lookup warm-up iterations")
set(HKV_REPRO_LOOKUP_RUNS 5 CACHE STRING "Lookup measured iterations")
set(HKV_REPRO_LOOKUP_HBM_GB 16 CACHE STRING "Lookup value HBM allowance in GiB")
set(HKV_REPRO_LOOKUP_BLOCK_SIZE 128 CACHE STRING "Lookup CUDA block size")
set(HKV_REPRO_LOOKUP_API_LOCK 1 CACHE STRING "Lookup public API lock")
set(HKV_REPRO_CONTAINS_LOW_LF_TLP 1 CACHE STRING "Enable low-LF contains TLP path")
set(HKV_REPRO_CONTAINS_TLP_LF_THRESHOLD 0.75 CACHE STRING "Contains LF threshold")
target_compile_definitions(hkv_lookup_modes_benchmark PRIVATE
  HKV_LOOKUP_DIM=${HKV_REPRO_LOOKUP_DIM}
  HKV_LOOKUP_CAPACITY=${HKV_REPRO_LOOKUP_CAPACITY}
  HKV_LOOKUP_BATCH_SIZE=${HKV_REPRO_LOOKUP_BATCH_SIZE}
  HKV_LOOKUP_WARMUP=${HKV_REPRO_LOOKUP_WARMUP}
  HKV_LOOKUP_RUNS=${HKV_REPRO_LOOKUP_RUNS}
  HKV_LOOKUP_HBM_GB=${HKV_REPRO_LOOKUP_HBM_GB}
  HKV_LOOKUP_BLOCK_SIZE=${HKV_REPRO_LOOKUP_BLOCK_SIZE}
  HKV_LOOKUP_API_LOCK=${HKV_REPRO_LOOKUP_API_LOCK}
  HKV_CONTAINS_LOW_LF_TLP=${HKV_REPRO_CONTAINS_LOW_LF_TLP}
  HKV_CONTAINS_TLP_LF_THRESHOLD=${HKV_REPRO_CONTAINS_TLP_LF_THRESHOLD}f)

# Two explicit binaries avoid contaminating all other targets with ablation
# macros or changing a shared CMake cache option between runs.
add_executable(digest_ablation_no_digest digest_ablation_benchmark.cc.cu)
target_compile_features(digest_ablation_no_digest PRIVATE cxx_std_17)
target_compile_definitions(digest_ablation_no_digest PRIVATE DISABLE_DIGEST)
set_target_properties(digest_ablation_no_digest PROPERTIES
  CUDA_ARCHITECTURES OFF RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(concurrency_rw_lock concurrency_benchmark.cc.cu)
target_compile_features(concurrency_rw_lock PRIVATE cxx_std_17)
target_compile_definitions(concurrency_rw_lock PRIVATE USE_RW_LOCK)
set_target_properties(concurrency_rw_lock PROPERTIES
  CUDA_ARCHITECTURES OFF RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# The root currently sets CMAKE_BUILD_TYPE=Debug while declaring test targets.
# Keep optional benchmark host code optimized without changing the tests' flags.
foreach(name IN ITEMS ${HKV_REPRO_SOURCES}
    digest_ablation_no_digest concurrency_rw_lock)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endforeach()
