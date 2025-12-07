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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CompilerWarnings)
include(Sanitizers)

add_subdirectory(src)

if(CHESS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
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
