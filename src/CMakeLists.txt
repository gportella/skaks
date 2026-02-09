find_package(Threads REQUIRED)

get_filename_component(SKAKS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

add_library(chess_engine_core STATIC
  attack_masks.cpp
  board.cpp
  complexity.cpp
  engine.cpp
  exchange.cpp
  cli.cpp
  magic_bitboards.cpp
  move_picker.cpp
  nnue_sf.cpp
  moves.cpp
    move_ordering.cpp
  pins.cpp
  perf.cpp
  polyglot.cpp
  quiescence.cpp
  params_loader.cpp
  search.cpp
  syzygy.cpp
  transposition_table.cpp
  time_manager.cpp
  zobrist.cpp
  uci.cpp
  ${SKAKS_ROOT}/fathom/src/tbprobe.c
)

set_source_files_properties(
  ${SKAKS_ROOT}/fathom/src/tbprobe.c
  PROPERTIES
    COMPILE_FLAGS "-Wno-error -Wno-sign-conversion -Wno-sign-compare -Wno-conversion"
)

set_source_files_properties(
  ${SKAKS_ROOT}/fathom/src/tbchess.c
  PROPERTIES
    HEADER_FILE_ONLY TRUE
)

add_library(sf_nnue_probe STATIC
  ${SKAKS_ROOT}/sf_nnue/bitboard.cpp
  ${SKAKS_ROOT}/sf_nnue/evaluate.cpp
  ${SKAKS_ROOT}/sf_nnue/misc.cpp
  ${SKAKS_ROOT}/sf_nnue/position.cpp
  ${SKAKS_ROOT}/sf_nnue/probe.cpp
  ${SKAKS_ROOT}/sf_nnue/nnue_adapter.cpp
  ${SKAKS_ROOT}/sf_nnue/nnue/evaluate_nnue.cpp
  ${SKAKS_ROOT}/sf_nnue/nnue/features/half_ka_v2_hm.cpp
)

target_include_directories(sf_nnue_probe
  PUBLIC
    ${SKAKS_ROOT}/sf_nnue
    ${SKAKS_ROOT}/sf_nnue/nnue
    ${SKAKS_ROOT}/sf_nnue/nnue/features
    ${SKAKS_ROOT}/sf_nnue/nnue/layers
    ${SKAKS_ROOT}/include
)

target_compile_features(sf_nnue_probe PUBLIC cxx_std_23)

chess_set_default_warnings(sf_nnue_probe)
chess_enable_sanitizers(sf_nnue_probe)
target_compile_options(sf_nnue_probe PRIVATE
  -Wno-error
  -Wno-old-style-cast
  -Wno-sign-conversion
  -Wno-implicit-int-conversion
  -Wno-unused-const-variable
)
target_compile_definitions(sf_nnue_probe PRIVATE
  EvalFileDefaultNameBig="${SKAKS_ROOT}/sf_nnue/nn-b1a57edbea57.nnue"
  EvalFileDefaultNameSmall="${SKAKS_ROOT}/sf_nnue/nn-baff1ede1f90.nnue"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(chess_engine_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(chess_engine_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${SKAKS_GENERATED_INCLUDE_DIR}
  PRIVATE
    ${SKAKS_ROOT}/sf_nnue
    ${SKAKS_ROOT}/fathom/src
)

target_include_directories(chess_engine_core SYSTEM PUBLIC
  ${YAML_CPP_INCLUDE_DIR}
)


target_compile_features(chess_engine_core PUBLIC cxx_std_23)

chess_set_default_warnings(chess_engine_core)
chess_enable_sanitizers(chess_engine_core)

target_link_libraries(chess_engine_core
  PUBLIC
    Threads::Threads
    yaml-cpp
  PRIVATE
    sf_nnue_probe
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
