# Builds libtensorrt_wan_plugins.so — every plugin in this directory, statically registered via
# REGISTER_TENSORRT_PLUGIN (see common/plugin_creator_base.h). tensorrt_wan.plugins.registry loads
# this one shared library and never links against individual plugin object files.
#
# Not built as part of this repository's normal install (see scripts/build_plugins.sh) — requires
# a CUDA toolkit + TensorRT SDK, which this development phase does not assume are present.
cmake_minimum_required(VERSION 3.24)
project(tensorrt_wan_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED TENSORRT_ROOT)
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "Path to a TensorRT SDK install (containing include/ and lib/)")
endif()
if(NOT TENSORRT_ROOT)
  message(FATAL_ERROR "Set -DTENSORRT_ROOT=/path/to/TensorRT or the TENSORRT_ROOT env var")
endif()

add_library(tensorrt_wan_plugins SHARED
  common/plugin_base.h
  common/plugin_creator_base.h
  rotary_embedding/plugin.h
  rotary_embedding/plugin.cpp
  rotary_embedding/kernel.cu
  adalayernorm/plugin.h
  adalayernorm/plugin.cpp
  adalayernorm/kernel.cu
  custom_attention/plugin.h
  custom_attention/plugin.cpp
  custom_attention/kernel_dispatch.cpp
  patch_embed/plugin.h
  patch_embed/plugin.cpp
  patch_embed/kernel.cu
  patch_reconstruct/plugin.h
  patch_reconstruct/plugin.cpp
  patch_reconstruct/kernel.cu
  time_embedding/plugin.h
  time_embedding/plugin.cpp
  time_embedding/kernel.cu
  video_ops/plugin.h
  video_ops/plugin.cpp
  video_ops/kernel.cu
  activation/plugin.h
  activation/plugin.cpp
  activation/kernel.cu
)

target_include_directories(tensorrt_wan_plugins PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TENSORRT_ROOT}/include
)
target_link_directories(tensorrt_wan_plugins PRIVATE ${TENSORRT_ROOT}/lib)
target_link_libraries(tensorrt_wan_plugins PRIVATE CUDA::cudart nvinfer)
set_target_properties(tensorrt_wan_plugins PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
