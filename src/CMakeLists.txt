add_library(chess_engine_core STATIC
  board.cpp
  pins.cpp
  engine.cpp
  zobrist.cpp
  moves.cpp
  demo_debug.cpp
  attack_masks.cpp
)

target_include_directories(chess_engine_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_features(chess_engine_core PUBLIC cxx_std_23)

chess_set_default_warnings(chess_engine_core)
chess_enable_sanitizers(chess_engine_core)

add_executable(chess_engine
  main.cpp
)

target_link_libraries(chess_engine
  PRIVATE
    chess_engine_core
)

target_compile_features(chess_engine PUBLIC cxx_std_23)

chess_set_default_warnings(chess_engine)
chess_enable_sanitizers(chess_engine)
