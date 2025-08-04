cmake_minimum_required(VERSION 3.22.1)
project(flashdecoding CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)

set(INCLUDE_DIR /home/zhichen/dayou/BitAttn/3rdparty/cutlass/include)

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message(STATUS "Compile testing packdecode kernel.")
add_executable(test_single_decode 
    ${PROJECT_SOURCE_DIR}/src/test_single_decode.cu
    ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_hdim128_fp16_split_sm90.cu
    ${PROJECT_SOURCE_DIR}/src/genfile/flash_qpack_hdim128_fp16_sm80_4bit.cu
    ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_combine.cu
)
target_link_libraries(test_single_decode "${TORCH_LIBRARIES}")
target_include_directories(test_single_decode PRIVATE ${INCLUDE_DIR})
target_compile_options(test_single_decode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=255 -gencode arch=compute_90a,code=sm_90a -w>)

message(STATUS "Compile benchmark single decode kernel.")
add_executable(bench_single_decode 
    ${PROJECT_SOURCE_DIR}/src/bench_single_decode.cu
    ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_hdim128_fp16_split_sm90.cu
    ${PROJECT_SOURCE_DIR}/src/genfile/flash_qpack_hdim128_fp16_sm80_4bit.cu
    ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_combine.cu
)
target_link_libraries(bench_single_decode "${TORCH_LIBRARIES}")
target_include_directories(bench_single_decode PRIVATE ${INCLUDE_DIR})
target_compile_options(bench_single_decode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=255 -gencode arch=compute_90a,code=sm_90a -w -O3 -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DNDEBUG>)
