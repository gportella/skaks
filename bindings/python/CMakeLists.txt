cmake_minimum_required(VERSION 3.24)

project(skaks_eval LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

# Locate repo root and generated headers
get_filename_component(SKAKS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

include(FetchContent)
list(APPEND CMAKE_MODULE_PATH "${SKAKS_ROOT}/cmake")
include(CompilerWarnings)
include(Sanitizers)

set(SKAKS_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY "${SKAKS_GENERATED_INCLUDE_DIR}/chess")
configure_file(
  "${SKAKS_ROOT}/include/chess/build_config.hpp.in"
  "${SKAKS_GENERATED_INCLUDE_DIR}/chess/build_config.hpp"
  @ONLY
)


# Dependency: yaml-cpp (mirrors root project setup)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)


FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_MakeAvailable(yaml-cpp)

# set(YAML_CPP_URL "https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.8.0.tar.gz")
# if(DEFINED ENV{YAML_CPP_TARBALL} AND EXISTS $ENV{YAML_CPP_TARBALL})
#   set(YAML_CPP_URL $ENV{YAML_CPP_TARBALL})
# endif()

# FetchContent_Declare(
#   yaml-cpp
#   URL ${YAML_CPP_URL}
# )
# FetchContent_MakeAvailable(yaml-cpp)

set(YAML_CPP_INCLUDE_DIR "${yaml-cpp_SOURCE_DIR}/include")
set(YAML_CPP_LIB_DIR "${yaml-cpp_BINARY_DIR}")

# Build core library inside the wheel build
add_subdirectory("${SKAKS_ROOT}/src" core_build)

add_library(skaks_eval MODULE
  src/eval_module.cpp
)

target_link_libraries(skaks_eval PRIVATE pybind11::module chess_engine_core)

target_include_directories(skaks_eval
  PRIVATE
    ${SKAKS_ROOT}/include
    ${SKAKS_GENERATED_INCLUDE_DIR}
)

target_compile_features(skaks_eval PRIVATE cxx_std_23)

set_target_properties(skaks_eval PROPERTIES
  CXX_STANDARD 23
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  POSITION_INDEPENDENT_CODE ON
)

pybind11_extension(skaks_eval)
pybind11_strip(skaks_eval)

install(TARGETS skaks_eval
  LIBRARY DESTINATION "."
)