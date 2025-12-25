cmake_minimum_required(VERSION 3.24)

project(chess_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CHESS_BUILD_TESTS "Build unit tests" ON)
option(CHESS_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(CHESS_ENABLE_IPO "Enable interprocedural optimization (LTO)" OFF)
option(CHESS_ENABLE_SANITIZERS "Enable address/undefined sanitizers in supported builds" OFF)
option(CHESS_BUILD_PYTHON "Build Python bindings" OFF)

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_MakeAvailable(yaml-cpp)


set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)

set(YAML_CPP_URL "https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.8.0.tar.gz")
if(DEFINED ENV{YAML_CPP_TARBALL} AND EXISTS $ENV{YAML_CPP_TARBALL})
  set(YAML_CPP_URL $ENV{YAML_CPP_TARBALL})
endif()

FetchContent_Declare(
  yaml-cpp
  URL ${YAML_CPP_URL}
)

FetchContent_MakeAvailable(yaml-cpp)
set(YAML_CPP_INCLUDE_DIR "${yaml-cpp_SOURCE_DIR}/include")
set(YAML_CPP_LIB_DIR "${yaml-cpp_BINARY_DIR}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CompilerWarnings)
include(Sanitizers)
include(GNUInstallDirs)

set(SKAKS_POLYGLOT_BOOK_RELATIVE_PATH "book/baron30.bin")
set(SKAKS_POLYGLOT_BOOK_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SKAKS_POLYGLOT_BOOK_RELATIVE_PATH}")
set(SKAKS_POLYGLOT_BOOK_DATADIR_RELATIVE "skaks")
set(SKAKS_POLYGLOT_BOOK_INSTALL_PATH "${CMAKE_INSTALL_FULL_DATADIR}/${SKAKS_POLYGLOT_BOOK_DATADIR_RELATIVE}/${SKAKS_POLYGLOT_BOOK_RELATIVE_PATH}")

set(SKAKS_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY "${SKAKS_GENERATED_INCLUDE_DIR}/chess")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/chess/build_config.hpp.in"
  "${SKAKS_GENERATED_INCLUDE_DIR}/chess/build_config.hpp"
  @ONLY
)

install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/${SKAKS_POLYGLOT_BOOK_RELATIVE_PATH}"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/${SKAKS_POLYGLOT_BOOK_DATADIR_RELATIVE}/book"
)

set(SKAKS_INSTALL_BINDIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH "Install directory for the skaks executable")

add_subdirectory(src)

if(CHESS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CHESS_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()

if(CHESS_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
  if(ipo_supported)
    foreach(target_name IN ITEMS chess_engine_core chess_engine chess_engine_tests)
      if(TARGET ${target_name})
        set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
      endif()
    endforeach()
  else()
    message(WARNING "IPO is not supported: ${ipo_error}")
  endif()
endif()
