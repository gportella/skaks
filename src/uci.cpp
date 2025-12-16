#include "chess/uci.hpp"

#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/moves.hpp"
#include "chess/types_io.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

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

int parse_square(std::string_view token) {
  if (token.size() != 2)
    return -1;
  const char file = static_cast<char>(std::tolower(static_cast<unsigned char>(token[0])));
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
  const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(symbol)));
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
    if (move.from != static_cast<uint16_t>(from_sq) || move.to != static_cast<uint16_t>(to_sq))
      continue;
    if (promo_symbol != '\0') {
      const OccupancyType expected = promotion_from_char(board.side_to_move, promo_symbol);
      if (move.promo_pc != expected)
        continue;
    } else if (move.promo_pc != OccupancyType::empty) {
      continue;
    }

    [[maybe_unused]] auto undo = make_move(board, move);
    engine.record_position(board.position_key, move_is_irreversible(move));
    return true;
  }
  const char stm = board.side_to_move == SideToMove::White ? 'w' : 'b';
  log_uci("warn", std::string("failed to apply move (") + stm + ") " + std::string(token));
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
    if (!apply_uci_move(board, engine, token)) {
      break;
    }
  }
}

int extract_go_depth(std::string_view args, int fallback) {
  auto stream = std::istringstream{std::string(args)};
  std::string token;
  int depth = fallback;
  while (stream >> token) {
    if (token == "depth") {
      if (stream >> depth) {
        continue;
      }
      depth = fallback;
    }
  }
  return depth;
}

} // namespace

void run_uci_loop(Engine& engine, int default_depth) {
  // Disable buffering on stdout
  std::cout.setf(std::ios::unitbuf);
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  Board board = initial_board(kStartFEN);
  engine.reset_history(board);

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
      log_uci("out", "id author G Portella");
      std::cout << "id author G Portella" << '\n';
      log_uci("out", "uciok");
      std::cout << "uciok" << '\n';
      std::cout.flush();
    } else if (keyword == "isready") {
      log_uci("out", "readyok");
      std::cout << "readyok" << '\n';
      std::cout.flush();
    } else if (keyword == "ucinewgame") {
      board = initial_board(kStartFEN);
      engine.clear_transposition_table();
      engine.clear_history();
      engine.reset_history(board);
    } else if (keyword == "position") {
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
      const int depth = extract_go_depth(remainder, default_depth > 0 ? default_depth : 4);
      SearchParameters params{};
      params.depth = depth;
      params.alpha = -INF;
      params.beta = INF;
      const auto result = engine.search(board, params);
      const auto elapsed_ms = std::max<std::uint64_t>(
          1, result.elapsed_ms); // avoid division by zero in nps calculation
      const bool has_move = result.best_move.moving_pc != OccupancyType::empty;
      const std::string bestmove = has_move ? move_to_uci(result.best_move) : "0000";
      const std::uint64_t nodes = result.nodes;
      const std::uint64_t nps = (nodes * 1000ULL) / std::max<std::uint64_t>(1, elapsed_ms);

      std::ostringstream info_line;
      info_line << "info depth " << depth << " seldepth " << depth << " score ";
      const int mate_threshold = INF - 1000;
      if (result.score > mate_threshold) {
        const int mate_ply = INF - result.score;
        const int mate_moves = std::max(1, (mate_ply + 1) / 2);
        info_line << "mate " << mate_moves;
      } else if (result.score < -mate_threshold) {
        const int mate_ply = INF - std::abs(result.score);
        const int mate_moves = std::max(1, (mate_ply + 1) / 2);
        info_line << "mate -" << mate_moves;
      } else {
        info_line << "cp " << result.score;
      }
      info_line << " time " << elapsed_ms << " nodes " << nodes << " nps " << nps;
      if (has_move) {
        info_line << " pv " << bestmove;
      }
      const auto info_str = info_line.str();
      log_uci("out", info_str);
      std::cout << info_str << '\n';
      std::cout.flush();

      const std::string response = "bestmove " + bestmove;
      log_uci("out", response);
      std::cout << response << '\n';
      std::cout.flush();
    } else if (keyword == "quit") {
      break;
    } else if (keyword == "stop") {
      // synchronous search, nothing to stop
    } else if (keyword == "setoption") {
      // not supported yet
    }
  }
}

} // namespace chess
