add_library(chess_engine_core STATIC
  attack_masks.cpp
  board.cpp
  complexity.cpp
  demo_debug.cpp
  engine.cpp
  exchange.cpp
  cli.cpp
  moves.cpp
    move_ordering.cpp
  pins.cpp
  perf.cpp
  polyglot.cpp
  quiescence.cpp
  params_loader.cpp
  scoring_rules.cpp
  search.cpp
  time_manager.cpp
  zobrist.cpp
  uci.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(chess_engine_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(chess_engine_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${SKAKS_GENERATED_INCLUDE_DIR}
)

target_include_directories(chess_engine_core SYSTEM PUBLIC
  ${YAML_CPP_INCLUDE_DIR}
)

target_compile_features(chess_engine_core PUBLIC cxx_std_23)

chess_set_default_warnings(chess_engine_core)
chess_enable_sanitizers(chess_engine_core)

target_link_libraries(chess_engine_core
  PRIVATE
    yaml-cpp
)

target_link_directories(chess_engine_core PRIVATE
  ${YAML_CPP_LIB_DIR}
)

add_executable(skaks
  main.cpp
)

target_include_directories(skaks
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SKAKS_GENERATED_INCLUDE_DIR}
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_include_directories(skaks SYSTEM PUBLIC
  ${YAML_CPP_INCLUDE_DIR}
)

target_link_libraries(skaks
  PRIVATE
    chess_engine_core
    yaml-cpp
)

target_link_directories(skaks PRIVATE
  ${YAML_CPP_LIB_DIR}
)

target_compile_features(skaks PUBLIC cxx_std_23)

chess_set_default_warnings(skaks)
chess_enable_sanitizers(skaks)

install(TARGETS skaks
  RUNTIME DESTINATION "${SKAKS_INSTALL_BINDIR}"
)
