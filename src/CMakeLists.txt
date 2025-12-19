add_library(chess_engine_core STATIC
  attack_masks.cpp
  board.cpp
  demo_debug.cpp
  engine.cpp
  moves.cpp
  pins.cpp
  perf.cpp
  polyglot.cpp
  quiescence.cpp
  scoring_rules.cpp
  search.cpp
  zobrist.cpp
)

target_include_directories(chess_engine_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${SKAKS_GENERATED_INCLUDE_DIR}
)

target_compile_features(chess_engine_core PUBLIC cxx_std_23)

chess_set_default_warnings(chess_engine_core)
chess_enable_sanitizers(chess_engine_core)

add_executable(skaks
  cli.cpp
  main.cpp
  uci.cpp
)

target_include_directories(skaks
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SKAKS_GENERATED_INCLUDE_DIR}
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(skaks
  PRIVATE
    chess_engine_core
)

target_compile_features(skaks PUBLIC cxx_std_23)

chess_set_default_warnings(skaks)
chess_enable_sanitizers(skaks)

install(TARGETS skaks
  RUNTIME DESTINATION "${SKAKS_INSTALL_BINDIR}"
)
