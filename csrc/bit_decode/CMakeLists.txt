cmake_minimum_required(VERSION 3.22.1)
project(bitdecoding CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Completely disable automatic CUDA architecture detection
set(CMAKE_CUDA_ARCHITECTURES OFF)

# Stop Torch from adding its own arches
set(ENV{TORCH_CUDA_ARCH_LIST} "")
set(TORCH_CUDA_ARCH_LIST "" CACHE STRING "" FORCE)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../../libs/cutlass/include)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Override any CUDA flags that Torch may have added
# We ONLY want sm_120a, not sm_120
set(CMAKE_CUDA_FLAGS "")

message(STATUS "Compile benchmarking kernel.")
add_executable(bench_single_packdecode 
  ${PROJECT_SOURCE_DIR}/src/bench_single_packdecode.cu
  ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_hdim128_fp16_sm80.cu
  ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_split_hdim128_fp16_sm80_4bit.cu
)

# CRITICAL: Disable automatic architecture flag generation for this target
set_property(TARGET bench_single_packdecode PROPERTY CUDA_ARCHITECTURES OFF)

target_link_libraries(bench_single_packdecode "${TORCH_LIBRARIES}")
target_include_directories(bench_single_packdecode PRIVATE ${INCLUDE_DIR})

# Add ONLY our manual sm_120a flags - explicitly disable default gencode
target_compile_options(bench_single_packdecode PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-code=arch=compute_120a,code=sm_120a
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -maxrregcount=255
    -DCUTLASS_ARCH_MMA_SM120_ENABLED
    -w
  >
)