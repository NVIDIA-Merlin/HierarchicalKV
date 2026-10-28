cmake_minimum_required(VERSION 3.18)
project(hkv_external_baselines LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Configure the architecture explicitly for the GPU used in a new measurement.
set(GPU_ARCH "80" CACHE STRING "CUDA architecture")
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})
option(ENABLE_CUCO_BENCH "Build cuCollections benchmark target" ON)
set(BASELINE_DIM "32" CACHE STRING "Baseline audit value dimension")
set(BASELINE_CAPACITY "134217728" CACHE STRING "Baseline audit table capacity")
set(BASELINE_BATCH_SIZE "1048576" CACHE STRING "Baseline audit batch/query size")
set(BASELINE_WARMUP "3" CACHE STRING "Baseline audit warmup runs")
set(BASELINE_RUNS "5" CACHE STRING "Baseline audit measured runs")
add_compile_definitions(
  BASELINE_DIM=${BASELINE_DIM}
  BASELINE_CAPACITY=${BASELINE_CAPACITY}
  BASELINE_BATCH_SIZE=${BASELINE_BATCH_SIZE}
  BASELINE_WARMUP=${BASELINE_WARMUP}
  BASELINE_RUNS=${BASELINE_RUNS}
)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA compile flags needed by dependencies
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

include(FetchContent)

# Disable tests/examples/benchmarks from all dependencies
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BGHT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CUCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CUCO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CUCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# These immutable commits were resolved from each project's default HEAD on
# 2026-10-27. They define new runs, not the historical paper's dependency
# identity; do not retroactively attach old performance numbers to them.
# FetchContent's SOURCE_DIR overrides allow an existing local source checkout.
# ─── WarpCore ───
FetchContent_Declare(warpcore
  GIT_REPOSITORY https://github.com/sleeepyjack/warpcore.git
  GIT_TAG        1a2fe03d438e8dfa1dbd8d9da73f183bd2d72051
)

# ─── BGHT (header-only, just download) ───
FetchContent_Declare(bght
  GIT_REPOSITORY https://github.com/owensgroup/BGHT.git
  GIT_TAG        7584547fcba6e980e8c80beb1694e52310ade16c
)

if(ENABLE_CUCO_BENCH)
  # ─── cuCollections ───
  FetchContent_Declare(cuco
    GIT_REPOSITORY https://github.com/NVIDIA/cuCollections.git
    GIT_TAG        3f7ff1043ff538a9941123075a67edaed637c572
  )
endif()

# ─── WarpCore benchmark ───
message(STATUS "Fetching WarpCore ...")
FetchContent_MakeAvailable(warpcore)
add_executable(warpcore_bench warpcore_bench.cu)
target_include_directories(warpcore_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(warpcore_bench PRIVATE warpcore)
set_target_properties(warpcore_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_executable(warpcore_lookup_modes_bench warpcore_lookup_modes_bench.cu)
target_include_directories(warpcore_lookup_modes_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(warpcore_lookup_modes_bench PRIVATE warpcore)
set_target_properties(warpcore_lookup_modes_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_executable(warpcore_warpspeed_value_size_bench warpcore_warpspeed_value_size_bench.cu)
target_include_directories(warpcore_warpspeed_value_size_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(warpcore_warpspeed_value_size_bench PRIVATE warpcore)
set_target_properties(warpcore_warpspeed_value_size_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# ─── BGHT benchmark (header-only: populate manually, skip its CMake) ───
message(STATUS "Fetching BGHT ...")
FetchContent_GetProperties(bght)
if(NOT bght_POPULATED)
  FetchContent_Populate(bght)
endif()
add_executable(bght_bench bght_bench.cu)
target_include_directories(bght_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${bght_SOURCE_DIR}/include
)
set_target_properties(bght_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# ─── P2BHT benchmark (uses BGHT library, p2bht variant) ───
add_executable(p2bht_bench p2bht_bench.cu)
target_include_directories(p2bht_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${bght_SOURCE_DIR}/include
)
set_target_properties(p2bht_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

if(ENABLE_CUCO_BENCH)
  # ─── cuCollections benchmark ───
  message(STATUS "Fetching cuCollections ...")
  FetchContent_MakeAvailable(cuco)
  add_executable(cuco_bench cuco_bench.cu)
  target_include_directories(cuco_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(cuco_bench PRIVATE cuco::cuco)
  set_target_properties(cuco_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()
