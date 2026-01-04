#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/build_config.hpp"
#include "chess/cli.hpp"
#include "chess/defaults.hpp"
#include "chess/demo_debug.hpp"
#include "chess/engine.hpp"
#include "chess/engine_params.hpp"
#include "chess/moves.hpp"
#include "chess/params_loader.hpp"
#include "chess/perf.hpp"
#include "chess/polyglot.hpp"
#include "chess/pst_tables.hpp"
#include "chess/scoring_rules.hpp"
#include "chess/search.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"
#include "chess/uci.hpp"
#include "chess/version.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

constexpr std::array<const char*, static_cast<std::size_t>(chess::TermId::Count)>
    kTermNames = {"Material",      "PawnCenter",   "CenterControl",
                  "Attacking",     "KingSafety",   "KingMobility",
                  "Pins",          "PstMg",        "PstEg",
                  "PassedPawns",   "Initiative",   "Hanging",
                  "KingRing",      "BishopPair",   "RookFiles",
                  "MinorMobility", "PawnStructure"};

chess::SearchLimits to_search_limits(const chess::TimeControlOptions& options);

double game_outcome(chess::Board board) {
  const bool king_captured = board.king_captured != chess::PieceColor::None;
  const bool has_moves = chess::has_legal_moves(board, board.side_to_move);
  const bool side_in_check = chess::is_check(board, board.side_to_move);

  if (king_captured) {
    return board.king_captured == chess::PieceColor::White ? 0.0 : 1.0;
  }
  if (!has_moves && side_in_check) {
    return board.side_to_move == chess::SideToMove::White ? 0.0 : 1.0;
  }
  return 0.5;
}

struct ArenaSummary {
  int wins = 0;
  int losses = 0;
  int draws = 0;
  int games = 0;
  double score = 0.0;
};

ArenaSummary run_internal_arena(const chess::CliOptions& opts,
                                const chess::EngineParams& base_params,
                                const chess::EngineParams& cand_params,
                                bool show_progress, int thread_count) {
  chess::Engine engine;
  engine.set_thread_count(std::max(thread_count, 1));
  ArenaSummary summary{};

  const bool use_time = opts.time_control.enabled;
  const auto limits = to_search_limits(opts.time_control);

  for (int game_idx = 0; game_idx < opts.arena_games; ++game_idx) {
    chess::Board board{};
    try {
      board = chess::initial_board(opts.fen);
    } catch (const std::exception& ex) {
      throw std::runtime_error(std::string("Failed to load FEN: ") + ex.what());
    }
    engine.reset_history(board);

    const bool cand_white = (game_idx % 2 == 0);
    int plies_played = 0;
    const int max_plies = opts.max_full_moves * 2;

    while (plies_played < max_plies) {
      const bool cand_to_move = (board.side_to_move == chess::SideToMove::White)
                                    ? cand_white
                                    : !cand_white;
      if (cand_to_move) {
        chess::set_engine_params(cand_params);
      } else {
        chess::set_engine_params(base_params);
      }

      chess::SearchParameters params{};
      if (use_time) {
        params.depth = static_cast<int>(chess::MAX_PLY) - 1;
        params.limits = limits;
      } else {
        params.depth = opts.search_depth;
      }
      params.alpha = -chess::INF;
      params.beta = chess::INF;

      auto result = engine.search(board, params);
      const bool has_move =
          result.best_move.moving_pc != chess::OccupancyType::empty;
      if (!has_move) {
        break;
      }

      const bool irreversible = chess::move_is_irreversible(result.best_move);
      chess::make_move(board, result.best_move);
      engine.record_position(board.position_key, irreversible);
      ++plies_played;

      if (board.is_terminal()) {
        break;
      }
    }

    const double outcome = game_outcome(board);
    if (outcome > 0.5) {
      summary.wins += 1;
    } else if (outcome < 0.5) {
      summary.losses += 1;
    } else {
      summary.draws += 1;
    }
    summary.games += 1;

    if (show_progress) {
      std::cout << "\r[arena] game " << (game_idx + 1) << "/" << opts.arena_games
                << " W/L/D=" << summary.wins << "/" << summary.losses << "/"
                << summary.draws << std::flush;
    }
  }

  if (show_progress) {
    std::cout << "\n";
  }

  const int total = summary.wins + summary.losses + summary.draws;
  summary.score =
      (total > 0) ? (summary.wins + 0.5 * summary.draws) / total : 0.0;
  return summary;
}

chess::SearchLimits to_search_limits(const chess::TimeControlOptions& options) {
  chess::SearchLimits limits{};
  if (!options.enabled) {
    return limits;
  }
  limits.use_time = true;
  limits.per_move = options.per_move;
  limits.move_time_ms = options.move_time_ms;
  limits.white_time_ms = options.white_time_ms;
  limits.black_time_ms = options.black_time_ms;
  limits.white_increment_ms = options.white_increment_ms;
  limits.black_increment_ms = options.black_increment_ms;
  limits.moves_to_go = options.moves_to_go;
  return limits;
}

} // namespace

int main(int argc, char** argv) {
  const auto cli = chess::parse_cli(argc, argv);
  if (cli.parse_error) {
    std::cerr << "Error: " << cli.message << "\n";
    return EXIT_FAILURE;
  }
  if (cli.show_help) {
    std::cout << cli.message << "\n";
    return EXIT_SUCCESS;
  }

  if (cli.options.show_version) {
    std::cout << chess::kEngineName << " version " << chess::kEngineVersion
              << "\n";
    if (cli.options.show_extended_version) {
      std::cout << "Optimizations:\n";
      for (const auto feature : chess::kOptimizationFeatures) {
        std::cout << " - " << feature << "\n";
      }
      if (chess::kCompiledWithNeon) {
        std::cout << " - Compiled with NEON eval_linear path" << "\n";
      }
    } else {
      std::cout << "Use -vv for details.\n";
    }
    return EXIT_SUCCESS;
  }

  chess::reset_engine_params();

  chess::EngineParams baseline_params = chess::default_engine_params();
  chess::EngineParams candidate_params = baseline_params;
  if (cli.options.params_override) {
    std::string error;
    if (!chess::load_engine_params_from_file(cli.options.params_path,
                                             candidate_params, error)) {
      std::cerr << "Failed to load params: " << error << "\n";
      return EXIT_FAILURE;
    }
  }

  chess::set_engine_params(candidate_params);

  const auto resolve_thread_count = [](int requested) -> int {
    if (requested <= 0) {
      const unsigned hw = std::thread::hardware_concurrency();
      if (hw == 0) {
        return 1;
      }
      return static_cast<int>(hw);
    }
    return std::max(requested, 1);
  };

  const int thread_count = resolve_thread_count(cli.options.thread_count);

  chess::Engine engine;
  engine.set_thread_count(thread_count);

  if (cli.options.arena_mode) {
    try {
      const auto summary = run_internal_arena(
          cli.options, baseline_params, candidate_params, true, thread_count);
      std::cout << "{\"score\":" << summary.score << ","
                << "\"wins\":" << summary.wins << ","
                << "\"losses\":" << summary.losses << ","
                << "\"draws\":" << summary.draws << ","
                << "\"games\":" << summary.games << "}" << std::endl;
      return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
      std::cerr << "Arena failed: " << ex.what() << "\n";
      return EXIT_FAILURE;
    }
  }

  if (cli.options.static_eval || cli.options.eval_breakdown) {
    chess::Board board{};
    try {
      board = chess::initial_board(cli.options.fen);
    } catch (const std::exception& ex) {
      std::cerr << "Failed to load FEN: " << ex.what() << "\n";
      return EXIT_FAILURE;
    }

    engine.reset_history(board);

    const chess::EvalVector eval_vec = chess::compute_eval_vector(board);
    const int raw_linear = chess::eval_linear(eval_vec, chess::phase_weights());
    const int white_eval = engine.evaluate(board);
    const int stm_eval = (board.side_to_move == chess::SideToMove::White)
                             ? white_eval
                             : -white_eval;

    if (cli.options.eval_breakdown) {
      const double mg_ratio = static_cast<double>(eval_vec.mg_phase) /
                              static_cast<double>(chess::kPstPhaseMax);
      const double eg_ratio = 1.0 - mg_ratio;
      std::cout << "eval_terms {\"mg_phase\":" << eval_vec.mg_phase
                << ",\"eg_phase\":" << eval_vec.eg_phase
                << ",\"mg_ratio\":" << mg_ratio << ",\"eg_ratio\":" << eg_ratio
                << ",\"raw_linear\":" << raw_linear
                << ",\"static_eval_white\":" << white_eval
                << ",\"static_eval_stm\":" << stm_eval << ",\"term_names\":[";
      for (std::size_t i = 0; i < kTermNames.size(); ++i) {
        if (i > 0) {
          std::cout << ',';
        }
        std::cout << '\"' << kTermNames[i] << '\"';
      }
      std::cout << "],\"term_values\":[";
      for (std::size_t i = 0; i < static_cast<std::size_t>(chess::TermId::Count);
           ++i) {
        if (i > 0) {
          std::cout << ',';
        }
        std::cout << eval_vec.f[i];
      }
      std::cout << "]}" << "\n";
    }

    if (cli.options.static_eval) {
      std::cout << "static_eval_white " << white_eval << "\n";
      std::cout << "static_eval_stm " << stm_eval << "\n";
    }
    return EXIT_SUCCESS;
  }

  chess::polyglot::Book opening_book;
  bool opening_book_ready = false;
  std::filesystem::path opening_book_path;
  const bool use_weighted_book = true;
  if (cli.options.polyglot) {
    std::optional<std::filesystem::path> override_path;
    if (cli.options.polyglot_book_override) {
      override_path = cli.options.polyglot_book_path;
    }
    std::vector<std::filesystem::path> attempted_paths;
    auto book_path = chess::polyglot::resolve_book_path(
        cli.options.executable_path, override_path, attempted_paths);
    if (!book_path) {
      std::cerr << "Failed to locate Polyglot book.\n";
      if (!attempted_paths.empty()) {
        std::cerr << "Checked paths:\n";
        for (const auto& candidate : attempted_paths) {
          std::cerr << "  - " << candidate << "\n";
        }
      }
      return EXIT_FAILURE;
    }
    const bool loaded =
        chess::polyglot::load_book(book_path->string(), opening_book);
    if (!loaded) {
      std::cerr << "Failed to load Polyglot book from " << book_path->string()
                << "\n";
      return EXIT_FAILURE;
    }
    opening_book_ready = true;
    opening_book_path = *book_path;
    if (!cli.options.use_uci) {
      std::cout << "Polyglot book loaded from " << opening_book_path.string()
                << "\n";
    }
  }

  if (cli.options.perf_mode) {
    return run_perf_mode(engine, cli.options);
  }

  if (cli.options.best_move) {
    chess::Board board{};
    try {
      board = chess::initial_board(cli.options.fen);
    } catch (const std::exception& ex) {
      std::cerr << "Failed to load FEN: " << ex.what() << "\n";
      return EXIT_FAILURE;
    }
    engine.reset_history(board);

    auto move_to_uci = [](const chess::Move& move) -> std::string {
      if (move.moving_pc == chess::OccupancyType::empty) {
        return "0000";
      }
      std::string result = chess::square_to_string(move.from);
      result += chess::square_to_string(move.to);
      const auto promo = move.promo_pc;
      char promo_char = '\0';
      switch (promo) {
      case chess::OccupancyType::wQ:
      case chess::OccupancyType::bQ:
        promo_char = 'q';
        break;
      case chess::OccupancyType::wR:
      case chess::OccupancyType::bR:
        promo_char = 'r';
        break;
      case chess::OccupancyType::wB:
      case chess::OccupancyType::bB:
        promo_char = 'b';
        break;
      case chess::OccupancyType::wN:
      case chess::OccupancyType::bN:
        promo_char = 'n';
        break;
      default:
        break;
      }
      if (promo_char != '\0') {
        result.push_back(promo_char);
      }
      return result;
    };

    std::optional<chess::Move> best_move;
    bool move_from_book = false;
    if (opening_book_ready) {
      auto encoded_book_move = chess::polyglot::choose_move(
          opening_book, board, use_weighted_book,
          static_cast<uint64_t>(
              std::chrono::steady_clock::now().time_since_epoch().count()));
      if (encoded_book_move) {
        best_move = chess::decode_move(*encoded_book_move);
        move_from_book = true;
      }
    }

    chess::SearchResult search_result{};
    if (!best_move) {
      chess::SearchParameters params{};
      if (cli.options.time_control.enabled) {
        params.depth = static_cast<int>(chess::MAX_PLY) - 1;
        params.limits = to_search_limits(cli.options.time_control);
      } else {
        params.depth = cli.options.search_depth;
      }
      params.alpha = -chess::INF;
      params.beta = chess::INF;
      search_result = engine.search(board, params);
      if (search_result.best_move.moving_pc != chess::OccupancyType::empty) {
        best_move = search_result.best_move;
      }
    }

    uint16_t legal_count = 0;
    auto legal_moves =
        chess::generate_legal_moves(board, board.side_to_move, legal_count);

    const auto encode_move_if_present = [](const chess::Move& move) {
      if (move.moving_pc == chess::OccupancyType::empty) {
        return uint32_t{0};
      }
      return chess::encode_move(move.from, move.to, move.moving_pc,
                                move.captured_pc, move.promo_pc, move.flags);
    };

    if (best_move) {
      const uint32_t encoded_best = encode_move_if_present(*best_move);
      bool found = false;
      for (uint16_t idx = 0; idx < legal_count; ++idx) {
        if (legal_moves[idx] == encoded_best) {
          found = true;
          break;
        }
      }
      if (!found) {
        std::cerr << "Best move is illegal for provided FEN." << std::endl;
        return EXIT_FAILURE;
      }
    } else if (legal_count > 0) {
      std::cerr << "Engine did not find a move despite legal options."
                << std::endl;
      return EXIT_FAILURE;
    }

    const std::string output =
        best_move ? move_to_uci(*best_move) : std::string("0000");
    std::cout << output << "\n";
    if (move_from_book) {
      std::cout << "info string move sourced from polyglot book" << std::endl;
    }

    return EXIT_SUCCESS;
  }

  if (cli.options.use_uci) {
    std::optional<chess::UciPolyglotContext> ctx;
    if (opening_book_ready) {
      ctx = chess::UciPolyglotContext{&opening_book, opening_book_path,
                                      use_weighted_book};
    }
    chess::run_uci_loop(engine, cli.options.search_depth, ctx);
    return EXIT_SUCCESS;
  }

  chess::Board board{};
  try {
    board = chess::initial_board(cli.options.fen);
  } catch (const std::exception& ex) {
    std::cerr
        << "Failed to load FEN: " << ex.what()
        << "\nHint: pass custom positions with --fen \"<fen>\" or -f \"<fen>\"."
        << std::endl;
    return EXIT_FAILURE;
  }
  engine.reset_history(board);

  const bool profile =
      (std::getenv("SKAKS_PROFILE") != nullptr) || cli.options.enable_profile;
  std::chrono::steady_clock::time_point total_start{};
  if (profile) {
    total_start = std::chrono::steady_clock::now();
  }

  if (cli.options.only_fen) {
    std::cout << chess::board_to_fen(board) << "\n";
    std::cout.flush();
  } else {
    std::cout << "Board of color to move: " << board.side_to_move << "\n";
    chess::terminal_board_print(board);
  }
  int move_number = 1;
  const int max_full_moves = cli.options.max_full_moves;
  bool move_limit_reached = false;
  bool result_announced = false;

  while (true) {
    std::chrono::steady_clock::time_point move_start{};
    if (profile) {
      move_start = std::chrono::steady_clock::now();
    }
    const int current_full_move = (move_number / 2) + 1;
    if (!cli.options.only_fen) {
      std::cout << "Move: " << current_full_move << " Ply: " << move_number
                << ", " << board.side_to_move << " to move.\n";
    }
    chess::SearchParameters params{};
    if (cli.options.time_control.enabled) {
      params.depth = static_cast<int>(chess::MAX_PLY) - 1;
      params.limits = to_search_limits(cli.options.time_control);
    } else {
      params.depth = cli.options.search_depth;
    }
    params.alpha = -10000;
    params.beta = 10000;

    std::optional<chess::Move> move_to_play;
    std::optional<chess::SearchResult> search_result;
    bool move_from_book = false;

    if (opening_book_ready) {
      auto encoded_book_move = chess::polyglot::choose_move(
          opening_book, board, use_weighted_book,
          static_cast<uint64_t>(
              std::chrono::steady_clock::now().time_since_epoch().count()));
      if (encoded_book_move) {
        move_to_play = chess::decode_move(*encoded_book_move);
        move_from_book = true;
        if (!cli.options.only_fen) {
          std::cout << "Book move selected from " << opening_book_path.string()
                    << "\n";
        }
      }
    }

    if (!move_to_play) {
      auto result = engine.search(board, params);
      move_to_play = result.best_move;
      search_result = result;
    }

    if (profile && !cli.options.only_fen && search_result) {
      const auto move_end = std::chrono::steady_clock::now();
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          move_end - move_start);
      std::cout << "[timing] search_ms=" << elapsed.count() << "\n";
    }

    if (!cli.options.only_fen) {
      if (search_result) {
        std::cout << "Best move score: " << search_result->score << "\n";
      } else if (move_from_book) {
        std::cout << "Best move score: (book)\n";
      }
    }

    const bool has_move =
        move_to_play && move_to_play->moving_pc != chess::OccupancyType::empty;
    const auto outcome = search_result
                             ? search_result->outcome
                             : chess::SearchResult::Outcome::InProgress;
    if (!has_move) {
      if (!cli.options.only_fen) {
        const bool side_in_check = chess::is_check(board, board.side_to_move);
        if (side_in_check) {
          const auto winner = chess::flip_side(board.side_to_move);
          std::cout << "Checkmate! " << chess::to_string(winner) << " wins.\n";
        } else {
          std::cout << "Stalemate.\n";
        }
      }
      result_announced = true;
      break;
    }
    if (!cli.options.only_fen &&
        outcome != chess::SearchResult::Outcome::InProgress) {
      switch (outcome) {
      case chess::SearchResult::Outcome::Mate:
        std::cout << "(search) Mate sequence detected.\n";
        break;
      case chess::SearchResult::Outcome::DrawByRepetition:
        std::cout << "(search) Draw by repetition forecast.\n";
        break;
      case chess::SearchResult::Outcome::DrawByStalemate:
        std::cout << "(search) Stalemate forecast.\n";
        break;
      case chess::SearchResult::Outcome::InProgress:
        break;
      }
    }

    if (!cli.options.only_fen) {
      if (move_to_play) {
        std::cout << "Best move from "
                  << chess::square_to_string(move_to_play->from) << " to "
                  << chess::square_to_string(move_to_play->to);
        if (move_from_book) {
          std::cout << " (book)";
        }
        std::cout << "\n";
      }
    }
    const bool irreversible = chess::move_is_irreversible(*move_to_play);
    chess::make_move(board, *move_to_play);
    engine.record_position(board.position_key, irreversible);
    if (cli.options.only_fen) {
      std::cout << chess::board_to_fen(board) << "\n";
      std::cout.flush();
    } else {
      chess::terminal_board_print(board);
      std::cout << "FEN: " << chess::board_to_fen(board) << "\n\n";
    }
    move_number++;

    const bool limit_hit = ((move_number / 2) + 1 > max_full_moves);
    if (limit_hit || board.is_terminal()) {
      move_limit_reached = limit_hit;
      break;
    }
  }
  if (!cli.options.only_fen && !result_announced) {
    const bool king_was_captured =
        board.king_captured != chess::PieceColor::None;
    const bool has_moves = chess::has_legal_moves(board, board.side_to_move);
    const bool terminal_position =
        king_was_captured || !has_moves || board.is_terminal();

    if (terminal_position) {
      std::cout << "Game over detected.\n";
      if (king_was_captured) {
        if (board.king_captured == chess::PieceColor::White) {
          std::cout << "Black wins!\n";
        } else {
          std::cout << "White wins!\n";
        }
      } else if (!has_moves && chess::is_check(board, board.side_to_move)) {
        const auto winner = chess::flip_side(board.side_to_move);
        std::cout << "Checkmate! " << chess::to_string(winner) << " wins.\n";
      } else {
        std::cout << "Stalemate.\n";
      }
      result_announced = true;
    } else if (move_limit_reached) {
      std::cout << "Move limit reached. Draw!\n";
      result_announced = true;
    }
  }

  if (profile && !cli.options.only_fen) {
    const auto total_end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start);
    std::cout << "[timing] total_ms=" << elapsed.count() << "\n";
  }

  return 0;
}
