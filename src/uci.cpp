// SPDX-License-Identifier: GLP-3.0-or-later
#include "chess/uci.hpp"

#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/moves.hpp"
#include "chess/polyglot.hpp"
#include "chess/score.hpp"
#include "chess/search_params.hpp"
#include "chess/types_io.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace chess {
namespace {

std::ofstream& uci_log_stream() {
  static std::ofstream stream;
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    const char* path = std::getenv("SKAKS_UCI_LOG");
    if (path == nullptr || *path == '\0') {
      path = "/tmp/skaks-uci.log";
    }
    stream.open(path, std::ios::app);
  }
  return stream;
}

void log_uci(std::string_view direction, std::string_view payload) {
  auto& stream = uci_log_stream();
  if (!stream.is_open()) {
    return;
  }
  stream << direction << ": " << payload << '\n';
  stream.flush();
}

bool info_strings_enabled() {
  static bool initialized = false;
  static bool enabled = true;
  if (!initialized) {
    initialized = true;
    const char* flag = std::getenv("SKAKS_SUPPRESS_INFO_STRINGS");
    if (flag != nullptr && *flag != '\0') {
      std::string value(flag);
      std::transform(
          value.begin(), value.end(), value.begin(),
          [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (value == "1" || value == "true" || value == "yes" || value == "on") {
        enabled = false;
      }
    }
  }
  return enabled;
}

void emit_spin_option(std::string_view name, int value, int min_value,
                      int max_value) {
  std::ostringstream line;
  line << "option name " << name << " type spin default " << value << " min "
       << min_value << " max " << max_value;
  const std::string line_out = line.str();
  log_uci("out", line_out);
  std::cout << line_out << '\n';
}

void emit_string_option(std::string_view name, double value) {
  std::ostringstream line;
  line << "option name " << name << " type string default " << std::fixed
       << std::setprecision(3) << value;
  const std::string line_out = line.str();
  log_uci("out", line_out);
  std::cout << line_out << '\n';
}

void emit_search_param_options() {
  const auto& sparams = search_params();
  emit_spin_option("aspiration_window_initial",
                   sparams.aspiration_window_initial, 50, 2000);
  emit_spin_option("aspiration_window_max", sparams.aspiration_window_max, 200,
                   8000);
  emit_spin_option("quiescence_delta_margin", sparams.quiescence_delta_margin, 0,
                   1000);
  emit_spin_option("quiescence_max_ply", sparams.quiescence_max_ply, 0, 32);
  emit_spin_option("quiescence_max_noisy_moves",
                   sparams.quiescence_max_noisy_moves, 0, 64);
  emit_spin_option("quiescence_zero_gain_skip_index",
                   sparams.quiescence_zero_gain_skip_index, 0, 16);
  emit_spin_option("quiescence_max_quiet_checks",
                   sparams.quiescence_max_quiet_checks, 0, 16);
  emit_spin_option("null_move_reduction", sparams.null_move_reduction, 0, 6);
  emit_spin_option("null_move_reduction_divisor",
                   sparams.null_move_reduction_divisor, 1, 12);
  emit_spin_option("null_move_min_depth", sparams.null_move_min_depth, 0, 12);
  emit_string_option("lmr_intercept", sparams.lmr_intercept);
  emit_string_option("lmr_divisor", sparams.lmr_divisor);
  emit_string_option("lmr_history_divisor", sparams.lmr_history_divisor);
  emit_string_option("lmr_pv_offset", sparams.lmr_pv_offset);
}

void apply_search_param_option(std::string_view lowered_name,
                               std::string_view value) {
  if (value.empty()) {
    return;
  }
  auto current = search_params();
  try {
    if (lowered_name == "aspiration_window_initial") {
      current.aspiration_window_initial = std::stoi(std::string(value));
    } else if (lowered_name == "aspiration_window_max") {
      current.aspiration_window_max = std::stoi(std::string(value));
    } else if (lowered_name == "quiescence_delta_margin") {
      current.quiescence_delta_margin = std::stoi(std::string(value));
    } else if (lowered_name == "quiescence_max_ply") {
      current.quiescence_max_ply = std::stoi(std::string(value));
    } else if (lowered_name == "quiescence_max_noisy_moves") {
      current.quiescence_max_noisy_moves = std::stoi(std::string(value));
    } else if (lowered_name == "quiescence_zero_gain_skip_index") {
      current.quiescence_zero_gain_skip_index = std::stoi(std::string(value));
    } else if (lowered_name == "quiescence_max_quiet_checks") {
      current.quiescence_max_quiet_checks = std::stoi(std::string(value));
    } else if (lowered_name == "null_move_reduction") {
      current.null_move_reduction = std::stoi(std::string(value));
    } else if (lowered_name == "null_move_reduction_divisor") {
      current.null_move_reduction_divisor = std::stoi(std::string(value));
    } else if (lowered_name == "null_move_min_depth") {
      current.null_move_min_depth = std::stoi(std::string(value));
    } else if (lowered_name == "lmr_intercept") {
      current.lmr_intercept = std::stod(std::string(value));
    } else if (lowered_name == "lmr_divisor") {
      current.lmr_divisor = std::stod(std::string(value));
    } else if (lowered_name == "lmr_history_divisor") {
      current.lmr_history_divisor = std::stod(std::string(value));
    } else if (lowered_name == "lmr_pv_offset") {
      current.lmr_pv_offset = std::stod(std::string(value));
    } else {
      return;
    }
  } catch (const std::exception&) {
    return;
  }
  set_search_params(current);
}

[[noreturn]] void fatal_uci_violation(const Board& board,
                                      std::string_view detail) {
  const std::string fen = board_to_fen(board);
  std::ostringstream oss;
  oss << "fatal uci violation: " << detail << " | fen=" << fen;
  const std::string message = oss.str();

  log_uci("err", message);

  std::cerr << message << '\n';
  std::cerr.flush();

  if (info_strings_enabled()) {
    std::cout << "info string " << message << '\n';
    std::cout.flush();
  }

  std::abort();
}

int parse_square(std::string_view token) {
  if (token.size() != 2)
    return -1;
  const char file =
      static_cast<char>(std::tolower(static_cast<unsigned char>(token[0])));
  const char rank = token[1];
  if (file < 'a' || file > 'h')
    return -1;
  if (rank < '1' || rank > '8')
    return -1;
  return (rank - '1') * 8 + (file - 'a');
}

char promotion_to_char(OccupancyType promo) {
  switch (promo) {
  case OccupancyType::wQ:
  case OccupancyType::bQ:
    return 'q';
  case OccupancyType::wR:
  case OccupancyType::bR:
    return 'r';
  case OccupancyType::wB:
  case OccupancyType::bB:
    return 'b';
  case OccupancyType::wN:
  case OccupancyType::bN:
    return 'n';
  default:
    return '\0';
  }
}

OccupancyType promotion_from_char(SideToMove stm, char symbol) {
  const char lower =
      static_cast<char>(std::tolower(static_cast<unsigned char>(symbol)));
  if (stm == SideToMove::White) {
    switch (lower) {
    case 'q':
      return OccupancyType::wQ;
    case 'r':
      return OccupancyType::wR;
    case 'b':
      return OccupancyType::wB;
    case 'n':
      return OccupancyType::wN;
    default:
      return OccupancyType::empty;
    }
  }
  switch (lower) {
  case 'q':
    return OccupancyType::bQ;
  case 'r':
    return OccupancyType::bR;
  case 'b':
    return OccupancyType::bB;
  case 'n':
    return OccupancyType::bN;
  default:
    return OccupancyType::empty;
  }
}

std::string move_to_uci(const Move& move) {
  std::string result;
  result.reserve(5);
  result += square_to_string(move.from);
  result += square_to_string(move.to);
  const char promo = promotion_to_char(move.promo_pc);
  if (promo != '\0') {
    result.push_back(promo);
  }
  return result;
}

bool apply_uci_move(Board& board, Engine& engine, std::string_view token) {
  if (token.size() < 4)
    return false;

  const int from_sq = parse_square(token.substr(0, 2));
  const int to_sq = parse_square(token.substr(2, 2));
  if (from_sq < 0 || to_sq < 0)
    return false;

  const char promo_symbol = (token.size() >= 5) ? token[4] : '\0';
  uint16_t move_count = 0;
  auto moves = generate_legal_moves(board, board.side_to_move, move_count);

  for (uint16_t i = 0; i < move_count; ++i) {
    const Move move = decode_move(moves[i]);
    if (move.from != static_cast<uint16_t>(from_sq) ||
        move.to != static_cast<uint16_t>(to_sq))
      continue;
    if (promo_symbol != '\0') {
      const OccupancyType expected =
          promotion_from_char(board.side_to_move, promo_symbol);
      if (move.promo_pc != expected)
        continue;
    } else if (move.promo_pc != OccupancyType::empty) {
      continue;
    }

    [[maybe_unused]] auto undo = make_move(board, move);
    engine.record_position(board.position_key, move_is_irreversible(move));
    return true;
  }
  std::ostringstream detail;
  detail << "failed to apply move token='" << token
         << "' stm=" << (board.side_to_move == SideToMove::White ? 'w' : 'b');
  fatal_uci_violation(board, detail.str());
  return false;
}

void handle_position(Board& board, Engine& engine, std::string_view args) {
  auto stream = std::istringstream{std::string(args)};
  std::string token;
  Board new_board{};
  bool have_board = false;

  if (!(stream >> token)) {
    return;
  }

  if (token == "startpos") {
    new_board = initial_board(kStartFEN);
    have_board = true;
  } else if (token == "fen") {
    std::array<std::string, 6> fen_parts{};
    for (std::size_t i = 0; i < fen_parts.size(); ++i) {
      if (!(stream >> fen_parts[i])) {
        return;
      }
    }
    std::ostringstream fen;
    for (std::size_t i = 0; i < fen_parts.size(); ++i) {
      if (i != 0)
        fen << ' ';
      fen << fen_parts[i];
    }
    new_board = initial_board(fen.str());
    have_board = true;
  } else {
    return;
  }

  if (!have_board) {
    return;
  }

  board = new_board;
  engine.reset_history(board);

  if (!(stream >> token)) {
    return;
  }
  if (token != "moves") {
    return;
  }

  while (stream >> token) {
    apply_uci_move(board, engine, token);
  }
}

struct GoParameters {
  int depth = 0;
  chess::SearchLimits limits;
  bool ponder = false;
  bool depth_explicit = false;
  bool infinite = false;
};

std::atomic<int> g_info_depth_cap{0};

/// Parses UCI "go" command arguments into a GoParameters structure.
///
/// Processes depth, time controls (movetime, wtime/btime, increments,
/// movestogo), node limits, and ponder mode, falling back to the provided depth
/// when needed.
///
/// @param args Space-delimited UCI "go" arguments.
/// @param fallback_depth Default depth if no explicit depth or time controls are
/// set.
/// @return Parsed GoParameters with limits and flags populated.
GoParameters parse_go_arguments(std::string_view args, int fallback_depth) {
  GoParameters parsed{};
  parsed.depth = fallback_depth;

  auto stream = std::istringstream{std::string(args)};
  std::string token;

  bool have_depth = false;
  bool have_wtime = false;
  bool have_btime = false;
  bool have_increment = false;
  bool have_movetime = false;
  bool ponder = false;
  bool infinite = false;

  while (stream >> token) {
    if (token == "depth") {
      int depth_override = fallback_depth;
      if (stream >> depth_override) {
        parsed.depth = depth_override;
        have_depth = true;
        parsed.depth_explicit = true;
      }
    } else if (token == "movetime") {
      std::uint64_t movetime_ms = 0;
      if (stream >> movetime_ms) {
        parsed.limits.use_time = true;
        parsed.limits.per_move = true;
        parsed.limits.move_time_ms = movetime_ms;
        have_movetime = true;
      }
    } else if (token == "wtime") {
      std::uint64_t wtime = 0;
      if (stream >> wtime) {
        parsed.limits.white_time_ms = wtime;
        have_wtime = true;
      }
    } else if (token == "btime") {
      std::uint64_t btime = 0;
      if (stream >> btime) {
        parsed.limits.black_time_ms = btime;
        have_btime = true;
      }
    } else if (token == "winc") {
      std::uint64_t winc = 0;
      if (stream >> winc) {
        parsed.limits.white_increment_ms = winc;
        have_increment = true;
      }
    } else if (token == "binc") {
      std::uint64_t binc = 0;
      if (stream >> binc) {
        parsed.limits.black_increment_ms = binc;
        have_increment = true;
      }
    } else if (token == "movestogo") {
      std::uint32_t mtg = 0;
      if (stream >> mtg) {
        parsed.limits.moves_to_go = mtg;
      }
    } else if (token == "infinite") {
      infinite = true;
      parsed.infinite = true;
    } else if (token == "ponder") {
      ponder = true;
    } else if (token == "nodes") {
      std::uint64_t node_limit = 0;
      if (stream >> node_limit) {
        parsed.limits.use_nodes = true;
        parsed.limits.node_limit = node_limit;
      }
    }
  }

  if (infinite) {
    parsed.limits = chess::SearchLimits{};
  } else if (have_movetime) {
    parsed.limits.use_time = true;
  } else if (have_wtime || have_btime || have_increment ||
             parsed.limits.moves_to_go > 0) {
    parsed.limits.use_time = true;
    parsed.limits.per_move = false;
  }

  if (!have_depth && !parsed.limits.use_time && !parsed.limits.use_nodes &&
      !infinite) {
    parsed.depth = fallback_depth;
  }

  parsed.ponder = ponder;

  return parsed;
}

struct AsyncSearchState {
  std::thread worker;
  std::atomic<bool> abort_flag{false};
  std::mutex mutex;
  bool active = false;
  bool pondering = false;
  bool result_ready = false;
  std::unique_ptr<SearchResult> result;
  std::unique_ptr<Board> board;
};

void emit_book_move(const Move& move) {
  const std::string bestmove = move_to_uci(move);
  std::ostringstream info_line;
  info_line << "info depth 0 seldepth 0 score cp 0 time 0 nodes 0 nps 0 pv "
            << bestmove << " (book)";
  const auto info_str = info_line.str();
  log_uci("out", info_str);
  std::cout << info_str << '\n';
  std::cout.flush();

  const std::string response = "bestmove " + bestmove;
  log_uci("out", response);
  std::cout << response << '\n';
  std::cout.flush();
}

/// Returns true if the current board state is drawable by the 50-move rule.
inline bool rule50_draw_reached(const Board& board) {
  return board.fifty_move_counter >= 100;
}

void emit_search_info_line(Board& board, const SearchResult& result) {
  const std::uint64_t elapsed_ms = std::max<std::uint64_t>(1, result.elapsed_ms);
  const std::uint64_t nodes = result.nodes;
  const std::uint64_t nps = (nodes * 1000ULL) / elapsed_ms;

  const int reported_depth = std::max(result.searched_depth, 0);
  const int reported_seldepth = std::max(result.selective_depth, reported_depth);
  int printed_depth = std::max(reported_depth, 1);
  int printed_sel_depth = std::max(reported_seldepth, printed_depth);
  const int depth_cap = g_info_depth_cap.load(std::memory_order_relaxed);
  if (depth_cap > 0) {
    printed_depth = std::min(printed_depth, depth_cap);
    printed_sel_depth = std::min(printed_sel_depth, depth_cap);
  }

  std::ostringstream info_line;
  info_line << "info depth " << printed_depth << " seldepth "
            << printed_sel_depth << " score ";
  if (is_mate_score(result.score)) {
    const int mate_moves = mate_moves_from_score(result.score);
    info_line << "mate ";
    if (result.score < 0) {
      info_line << '-';
    }
    info_line << mate_moves;
  } else {
    info_line << "cp " << result.score;
  }
  info_line << " time " << elapsed_ms << " nodes " << nodes << " nps " << nps;

  if (result.pv_length > 0 && !result.principal_variation.empty()) {
    std::string pv_line;
    Board pv_board = board;
    SideToMove pv_side = board.side_to_move;
    const int pv_len = std::min(
        result.pv_length, static_cast<int>(result.principal_variation.size()));
    for (int idx = 0; idx < pv_len; ++idx) {
      const Move& pv_move =
          result.principal_variation[static_cast<std::size_t>(idx)];
      if (pv_move.moving_pc == OccupancyType::empty) {
        break;
      }
      if (!pv_line.empty()) {
        pv_line.push_back(' ');
      }
      pv_line += move_to_uci(pv_move);

      [[maybe_unused]] auto pv_undo = make_move(pv_board, pv_move);
      pv_side = flip_side(pv_side);
      if (rule50_draw_reached(pv_board)) {
        break;
      }
    }
    if (!pv_line.empty()) {
      info_line << " pv " << pv_line;
    }
  }

  const auto info_str = info_line.str();
  log_uci("out", info_str);
  std::cout << info_str << '\n';
  std::cout.flush();
}

void emit_search_result(Board& board, const SearchResult& result) {
  bool has_move =
      result.best_move.moving_pc != OccupancyType::empty && !result.aborted;
  Move chosen_move = result.best_move;
  uint16_t legal_count = 0;
  auto legal_moves =
      generate_legal_moves(board, board.side_to_move, legal_count);

  auto encode_move_if_present = [](const Move& m) {
    if (m.moving_pc == OccupancyType::empty) {
      return uint32_t{0};
    }
    return encode_move(m.from, m.to, m.moving_pc, m.captured_pc, m.promo_pc,
                       m.flags);
  };

  uint32_t encoded_best = encode_move_if_present(chosen_move);

  if (!has_move && legal_count > 0) {
    const Move fallback = decode_move(legal_moves[0]);
    chosen_move = fallback;
    encoded_best = legal_moves[0];
    has_move = true;
    std::ostringstream warn;
    warn << "warn: search returned no move; using fallback '"
         << move_to_uci(fallback) << "' legal_count=" << legal_count;
    log_uci("err", warn.str());
    std::cerr << warn.str() << '\n';
  }

  bool best_found = false;
  if (has_move) {
    for (uint16_t idx = 0; idx < legal_count; ++idx) {
      if (legal_moves[idx] == encoded_best) {
        best_found = true;
        break;
      }
    }
  }

  if (has_move && !best_found && legal_count > 0) {
    const Move fallback = decode_move(legal_moves[0]);
    std::ostringstream warn;
    warn << "warn: search produced illegal bestmove '"
         << move_to_uci(result.best_move) << "'; using fallback '"
         << move_to_uci(fallback) << "' legal_count=" << legal_count;
    log_uci("err", warn.str());
    std::cerr << warn.str() << '\n';
    chosen_move = fallback;
    encoded_best = legal_moves[0];
    best_found = true;
  }

  const std::uint64_t elapsed_ms = std::max<std::uint64_t>(1, result.elapsed_ms);
  const std::uint64_t nodes = result.nodes;
  const std::uint64_t nps = (nodes * 1000ULL) / elapsed_ms;

  const int reported_depth = std::max(result.searched_depth, 0);
  const int reported_seldepth = std::max(result.selective_depth, reported_depth);
  const int printed_depth =
      has_move ? std::max(reported_depth, 1) : reported_depth;
  const int printed_sel_depth =
      has_move ? std::max(reported_seldepth, printed_depth) : reported_seldepth;

  std::string bestmove = has_move ? move_to_uci(chosen_move) : "0000";
  std::string ponder_move;
  std::string pv_line;

  if (has_move && result.pv_length > 0) {
    Board pv_board = board;
    SideToMove pv_side = board.side_to_move;
    bool pv_valid = true;
    const int pv_len = std::min(
        result.pv_length, static_cast<int>(result.principal_variation.size()));
    for (int idx = 0; idx < pv_len; ++idx) {
      const Move& pv_move =
          result.principal_variation[static_cast<std::size_t>(idx)];
      if (pv_move.moving_pc == OccupancyType::empty) {
        break;
      }

      uint16_t pv_legal_count = 0;
      const auto pv_legals =
          generate_legal_moves(pv_board, pv_side, pv_legal_count);
      const uint32_t* pv_legals_data = pv_legals.data();
      const uint32_t pv_code =
          encode_move(pv_move.from, pv_move.to, pv_move.moving_pc,
                      pv_move.captured_pc, pv_move.promo_pc, pv_move.flags);
      if (idx == 0 && has_move && pv_code != encoded_best) {
        pv_valid = false;
        break;
      }
      bool pv_found = false;
      for (uint16_t pv_idx = 0; pv_idx < pv_legal_count; ++pv_idx) {
        if (pv_legals_data[pv_idx] == pv_code) {
          pv_found = true;
          break;
        }
      }
      if (!pv_found) {
        pv_valid = false;
        break;
      }

      const std::string move_str = move_to_uci(pv_move);
      if (!pv_line.empty()) {
        pv_line.push_back(' ');
      }
      pv_line += move_str;
      if (idx == 0) {
        bestmove = move_str;
      } else if (idx == 1) {
        ponder_move = move_str;
      }

      [[maybe_unused]] auto pv_undo = make_move(pv_board, pv_move);
      pv_side = flip_side(pv_side);
      if (rule50_draw_reached(pv_board)) {
        break;
      }
    }

    if (!pv_valid) {
      pv_line.clear();
      ponder_move.clear();
    }
  }

  std::ostringstream info_line;
  info_line << "info depth " << printed_depth << " seldepth "
            << printed_sel_depth << " score ";
  if (is_mate_score(result.score)) {
    const int mate_moves = mate_moves_from_score(result.score);
    info_line << "mate ";
    if (result.score < 0) {
      info_line << '-';
    }
    info_line << mate_moves;
  } else {
    info_line << "cp " << result.score;
  }
  info_line << " time " << elapsed_ms << " nodes " << nodes << " nps " << nps;
  if (has_move) {
    const std::string& pv_output = pv_line.empty() ? bestmove : pv_line;
    info_line << " pv " << pv_output;
  }
  const auto info_str = info_line.str();
  log_uci("out", info_str);
  std::cout << info_str << '\n';
  std::cout.flush();

  std::string response = "bestmove " + bestmove;
  if (!ponder_move.empty()) {
    response += " ponder " + ponder_move;
  }
  log_uci("out", response);
  std::cout << response << '\n';
  std::cout.flush();
}

} // namespace

/**
 * @brief Runs the UCI command processing loop for the engine.
 *
 * Initializes engine/board state, manages search threads, and handles UCI
 * commands such as "uci", "isready", "position", "go", "stop", "quit",
 * "setoption", "staticeval", and "ponderhit". Supports optional Polyglot
 * opening book usage and configures threading and ponder settings.
 *
 * @param engine Engine instance used to perform searches and evaluations.
 * @param default_depth Fallback search depth when none is specified in "go".
 * @param polyglot_ctx Optional Polyglot context for opening book support.
 */
void run_uci_loop(Engine& engine, int default_depth,
                  const std::optional<UciPolyglotContext>& polyglot_ctx) {
  // Disable buffering on stdout
  std::cout.setf(std::ios::unitbuf);
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  Board board = initial_board(kStartFEN);
  engine.reset_history(board);

  auto search_state = std::make_shared<AsyncSearchState>();
  bool ponder_enabled = true;
  constexpr int kMinThreads = 1;
  const unsigned detected_threads = std::thread::hardware_concurrency();
  const int max_threads =
      static_cast<int>(std::clamp(detected_threads, 1u, 256u));
  int thread_count = std::clamp(engine.thread_count(), kMinThreads, max_threads);
  engine.set_thread_count(thread_count);

  auto join_worker = [&]() {
    if (search_state->worker.joinable()) {
      search_state->worker.join();
    }
  };

  auto reset_state = [&](bool clear_result) {
    std::lock_guard<std::mutex> lock(search_state->mutex);
    search_state->active = false;
    if (clear_result) {
      search_state->result_ready = false;
      search_state->result.reset();
    }
    search_state->pondering = false;
    search_state->board.reset();
  };

  auto stop_search = [&](bool clear_result) {
    search_state->abort_flag.store(true, std::memory_order_relaxed);
    join_worker();
    reset_state(clear_result);
    search_state->abort_flag.store(false, std::memory_order_relaxed);
  };

  auto start_search = [&](SearchParameters params, bool ponder) {
    stop_search(true);
    params.abort_flag = &search_state->abort_flag;
    {
      std::lock_guard<std::mutex> lock(search_state->mutex);
      search_state->board = std::make_unique<Board>(board);
      search_state->active = true;
      search_state->pondering = ponder;
      search_state->result_ready = false;
      search_state->result.reset();
    }
    search_state->abort_flag.store(false, std::memory_order_relaxed);
    search_state->worker = std::thread(
        [state_ptr = search_state, engine_ptr = &engine, params]() mutable {
          Board* worker_board = nullptr;
          {
            std::lock_guard<std::mutex> lock(state_ptr->mutex);
            worker_board = state_ptr->board.get();
            if (!worker_board) {
              state_ptr->active = false;
              state_ptr->result_ready = false;
              state_ptr->pondering = false;
              state_ptr->result.reset();
            }
          }
          if (!worker_board) {
            return;
          }

          params.info_callback = [worker_board](const SearchResult& res) {
            if (worker_board) {
              emit_search_info_line(*worker_board, res);
            }
          };
          SearchResult res = engine_ptr->search(*worker_board, params);

          std::unique_ptr<Board> board_for_output;
          bool deliver_now = true;
          {
            std::lock_guard<std::mutex> lock(state_ptr->mutex);
            state_ptr->result = std::make_unique<SearchResult>(res);
            state_ptr->active = false;
            if (res.aborted) {
              state_ptr->result_ready = false;
              state_ptr->board.reset();
              state_ptr->pondering = false;
            } else if (state_ptr->pondering) {
              state_ptr->result_ready = true;
              deliver_now = false; // wait for ponderhit
            } else {
              board_for_output = std::move(state_ptr->board);
              state_ptr->result_ready = false;
              state_ptr->pondering = false;
            }
          }

          if (!res.aborted && deliver_now && board_for_output) {
            emit_search_result(*board_for_output, res);
          }

          std::lock_guard<std::mutex> lock(state_ptr->mutex);
          if (!state_ptr->result_ready) {
            state_ptr->result.reset();
          }
        });
  };

  const polyglot::Book* opening_book = nullptr;
  std::filesystem::path opening_book_path;
  bool use_weighted_book = true;
  if (polyglot_ctx && polyglot_ctx->book != nullptr) {
    opening_book = polyglot_ctx->book;
    opening_book_path = polyglot_ctx->book_path;
    use_weighted_book = polyglot_ctx->use_weighted;
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty())
      continue;

    log_uci("in", line);

    std::istringstream cmd(line);
    std::string keyword;
    cmd >> keyword;
    if (keyword == "uci") {
      log_uci("out", "id name Skaks");
      std::cout << "id name Skaks" << '\n';
      log_uci("out", "id author G Portella -- NNUE by Stockfish developers");
      std::cout << "id author G Portella -- NNUE by Stockfish developers"
                << '\n';
      log_uci("out", "option name Ponder type check default true");
      std::cout << "option name Ponder type check default true" << '\n';
      {
        std::ostringstream threads_line;
        threads_line << "option name Threads type spin default " << thread_count
                     << " min " << kMinThreads << " max " << max_threads;
        const std::string line_out = threads_line.str();
        log_uci("out", line_out);
        std::cout << line_out << '\n';
      }
      {
        const char* eval_label =
            (engine.evaluation_mode() == EvaluationMode::Stockfish) ? "nnue"
                                                                    : "native";
        std::ostringstream info_line;
        info_line << "info string eval_mode=" << eval_label;
        const std::string line_out = info_line.str();
        log_uci("out", line_out);
        std::cout << line_out << '\n';
      }
      emit_search_param_options();
      log_uci("out", "uciok");
      std::cout << "uciok" << '\n';
      std::cout.flush();
    } else if (keyword == "isready") {
      log_uci("out", "readyok");
      std::cout << "readyok" << '\n';
      std::cout.flush();
    } else if (keyword == "ucinewgame") {
      stop_search(true);
      board = initial_board(kStartFEN);
      engine.clear_transposition_table();
      engine.clear_history();
      engine.reset_history(board);
    } else if (keyword == "position") {
      stop_search(true);
      std::string remainder;
      std::getline(cmd, remainder);
      // trim leading spaces
      const auto first = remainder.find_first_not_of(' ');
      if (first != std::string::npos) {
        remainder.erase(0, first);
      } else {
        remainder.clear();
      }
      handle_position(board, engine, remainder);
    } else if (keyword == "go") {
      std::string remainder;
      std::getline(cmd, remainder);
      const auto first = remainder.find_first_not_of(' ');
      if (first != std::string::npos) {
        remainder.erase(0, first);
      } else {
        remainder.clear();
      }
      const int fallback_depth = default_depth > 0 ? default_depth : 4;
      const GoParameters go_params =
          parse_go_arguments(remainder, fallback_depth);
      if (go_params.limits.use_nodes && !go_params.depth_explicit) {
        g_info_depth_cap.store(64, std::memory_order_relaxed);
      } else {
        g_info_depth_cap.store(0, std::memory_order_relaxed);
      }
      std::optional<uint32_t> book_move;
      if (opening_book != nullptr) {
        book_move = polyglot::choose_move(
            *opening_book, board, use_weighted_book,
            static_cast<uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count()));
      }
      if (book_move) {
        stop_search(true);
        const Move move = decode_move(*book_move);
        emit_book_move(move);
        continue;
      }
      SearchParameters params{};
      if (go_params.infinite) {
        params.depth = static_cast<int>(MAX_PLY) - 1;
      } else if (go_params.limits.use_time || go_params.limits.use_nodes) {
        if (!go_params.depth_explicit) {
          params.depth = static_cast<int>(MAX_PLY) - 1;
        } else {
          params.depth = std::min(std::max(go_params.depth, 1),
                                  static_cast<int>(MAX_PLY) - 1);
        }
      } else {
        params.depth = std::max(go_params.depth, 1);
      }
      params.alpha = -INF;
      params.beta = INF;
      params.limits = go_params.limits;
      start_search(params, go_params.ponder && ponder_enabled);
    } else if (keyword == "quit") {
      stop_search(true);
      break;
    } else if (keyword == "stop") {
      stop_search(true);
    } else if (keyword == "setoption") {
      std::string remainder;
      std::getline(cmd, remainder);
      std::istringstream option_stream(remainder);
      std::string token;
      std::string name;
      std::string value;
      bool reading_name = false;
      bool reading_value = false;
      while (option_stream >> token) {
        if (token == "name") {
          reading_name = true;
          reading_value = false;
          name.clear();
          continue;
        }
        if (token == "value") {
          reading_value = true;
          reading_name = false;
          value.clear();
          continue;
        }
        if (reading_name) {
          if (!name.empty()) {
            name.push_back(' ');
          }
          name += token;
        } else if (reading_value) {
          if (!value.empty()) {
            value.push_back(' ');
          }
          value += token;
        }
      }
      std::string lowered_name = name;
      std::transform(
          lowered_name.begin(), lowered_name.end(), lowered_name.begin(),
          [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

      if (lowered_name == "ponder") {
        std::string lowered_value = value;
        std::transform(
            lowered_value.begin(), lowered_value.end(), lowered_value.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lowered_value == "false" || lowered_value == "0" ||
            lowered_value == "off") {
          ponder_enabled = false;
        } else {
          ponder_enabled = true;
        }
      } else if (lowered_name == "threads") {
        int requested = thread_count;
        if (!value.empty()) {
          try {
            requested = std::stoi(value);
          } catch (const std::exception&) {
            requested = thread_count;
          }
        }
        const int clamped = std::clamp(requested, kMinThreads, max_threads);
        thread_count = clamped;
        engine.set_thread_count(thread_count);
        if (clamped != requested) {
          const std::string info = "info string Threads option is limited to " +
                                   std::to_string(max_threads);
          log_uci("out", info);
          if (info_strings_enabled()) {
            std::cout << info << '\n';
            std::cout.flush();
          }
        }
      } else {
        apply_search_param_option(lowered_name, value);
      }
    } else if (keyword == "staticeval") {
      stop_search(true);
      std::unique_ptr<NnueAdapter> nnue_adapter =
          std::make_unique<NnueAdapter>(board);
      const int white_eval =
          engine.evaluate(board, engine.evaluation_mode(), nnue_adapter.get());
      const int stm_eval =
          (board.side_to_move == SideToMove::White) ? white_eval : -white_eval;
      std::ostringstream oss;
      oss << "info string static_eval_white " << white_eval;
      log_uci("out", oss.str());
      if (info_strings_enabled()) {
        std::cout << oss.str() << '\n';
        std::cout.flush();
      }
      std::ostringstream oss2;
      oss2 << "info score cp " << stm_eval;
      log_uci("out", oss2.str());
      std::cout << oss2.str() << '\n';
      std::cout.flush();
    } else if (keyword == "ponderhit") {
      std::unique_ptr<SearchResult> ready_result;
      std::unique_ptr<Board> ready_board;
      {
        std::lock_guard<std::mutex> lock(search_state->mutex);
        if (search_state->pondering && search_state->result_ready &&
            search_state->result && search_state->board) {
          ready_result = std::move(search_state->result);
          ready_board = std::move(search_state->board);
          search_state->result_ready = false;
          search_state->pondering = false;
        } else {
          search_state->pondering = false;
        }
      }
      if (ready_result && ready_board) {
        emit_search_result(*ready_board, *ready_result);
      }
    }
  }

  stop_search(true);
}

} // namespace chess
