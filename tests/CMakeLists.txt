include(FetchContent)

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
  move_tests.cpp
  scoring_tests.cpp
  attack_tests.cpp
  pin_tests.cpp
  polyglot_tests.cpp
)

target_link_libraries(chess_engine_tests
  PRIVATE
    chess_engine_core
    GTest::gtest_main
)

target_include_directories(chess_engine_tests
  PRIVATE
    ${SKAKS_GENERATED_INCLUDE_DIR}
)

chess_set_default_warnings(chess_engine_tests)
chess_enable_sanitizers(chess_engine_tests)

include(GoogleTest)
gtest_discover_tests(chess_engine_tests)
