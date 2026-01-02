include(FetchContent)

find_package(Python3 COMPONENTS Interpreter)

set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)

FetchContent_MakeAvailable(googletest)

add_executable(chess_engine_tests
  board_tests.cpp
  engine_tests.cpp
  incremental_tests.cpp
  move_tests.cpp
  move_ordering_tests.cpp
  scoring_tests.cpp
  attack_tests.cpp
  exchange_tests.cpp
  pin_tests.cpp
  polyglot_tests.cpp
  cli_tests.cpp
  time_manager_tests.cpp
  zobrist_tests.cpp
  params_loader_tests.cpp
  nnue_features_tests.cpp
  nnue_sf_tests.cpp
  sf_eval_tests.cpp
)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/sf_nnue/nnue.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/sf_nnue/misc.cpp
  PROPERTIES
  COMPILE_OPTIONS "-Wno-old-style-cast;-Wno-sign-conversion;-Wno-implicit-int-conversion;-Wno-vla-cxx-extension"
  COMPILE_DEFINITIONS "USE_NEON;IS_64BIT=1"
)

target_link_libraries(chess_engine_tests
  PRIVATE
    chess_engine_core
    GTest::gtest_main
)

target_link_directories(chess_engine_tests PRIVATE
  ${YAML_CPP_LIB_DIR}
)

target_include_directories(chess_engine_tests
  PRIVATE
    ${SKAKS_GENERATED_INCLUDE_DIR}
)

target_include_directories(chess_engine_tests SYSTEM PUBLIC
  ${YAML_CPP_INCLUDE_DIR}
)

chess_set_default_warnings(chess_engine_tests)
chess_enable_sanitizers(chess_engine_tests)

include(GoogleTest)
gtest_discover_tests(chess_engine_tests)

if(Python3_Interpreter_FOUND)
  add_test(
    NAME fight_against_engine_cli
    COMMAND ${Python3_EXECUTABLE} -m unittest validation_moves.tests.test_fight_cli
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
endif()
