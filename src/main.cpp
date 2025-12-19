#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/cli.hpp"
#include "chess/defaults.hpp"
#include "chess/demo_debug.hpp"
#include "chess/engine.hpp"
#include "chess/moves.hpp"
#include "chess/perf.hpp"
#include "chess/polyglot.hpp"
#include "chess/search.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"
#include "chess/uci.hpp"
#include "chess/version.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>

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
    } else {
      std::cout << "Use -vv for details.\n";
    }
    return EXIT_SUCCESS;
  }

  chess::Engine engine;

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
    params.depth = cli.options.search_depth;
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
