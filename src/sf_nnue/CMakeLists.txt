cmake_minimum_required(VERSION 3.24)

add_library(sf_nnue STATIC
  misc.cpp
  nnue.cpp
)

target_include_directories(sf_nnue
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SKAKS_ROOT}/include
)

target_compile_features(sf_nnue PUBLIC cxx_std_23)
