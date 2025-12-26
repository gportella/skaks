#include "chess/board.hpp"
#include "chess/engine.hpp"
#include "chess/engine_params.hpp"
#include "chess/evaluation_params.hpp"
#include "chess/search.hpp"
#include "chess/search_params.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

template <typename T>
void assign_if_present(const py::dict& src, const char* key, T& target) {
  if (src.contains(key)) {
    target = py::cast<T>(src[key]);
  }
}

template <std::size_t N>
void assign_array_if_present(const py::dict& src, const char* key,
                             std::array<int, N>& target) {
  if (src.contains(key)) {
    auto seq = py::cast<py::sequence>(src[key]);
    if (seq.size() != static_cast<py::ssize_t>(N)) {
      throw std::invalid_argument(std::string(key) + " must have " +
                                  std::to_string(N) + " elements");
    }
    for (std::size_t i = 0; i < N; ++i) {
      target[i] = py::cast<int>(seq[i]);
    }
  }
}

struct PinPair {
  int base;
  int mobility;
};

PinPair parse_pin_pair(const py::handle& obj, const char* key) {
  // Accept either a sequence of length 2 or a dict with base/mobility.
  if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj)) {
    auto seq = py::cast<py::sequence>(obj);
    if (seq.size() != 2) {
      throw std::invalid_argument(std::string(key) + " needs 2 elements");
    }
    return {py::cast<int>(seq[0]), py::cast<int>(seq[1])};
  }
  if (py::isinstance<py::dict>(obj)) {
    auto d = py::cast<py::dict>(obj);
    if (!d.contains("base") || !d.contains("mobility")) {
      throw std::invalid_argument(std::string(key) +
                                  " dict needs base and mobility");
    }
    return {py::cast<int>(d["base"]), py::cast<int>(d["mobility"])};
  }
  throw std::invalid_argument(std::string(key) + " must be sequence[2] or dict");
}

chess::EngineParams params_from_dict(const py::dict& root) {
  chess::EngineParams params = chess::default_engine_params();

  if (root.contains("evaluation")) {
    auto ev = py::cast<py::dict>(root["evaluation"]);
    assign_if_present(ev, "check_penalty", params.evaluation.check_penalty);
    assign_if_present(ev, "pawn_shield_bonus",
                      params.evaluation.pawn_shield_bonus);
    assign_if_present(ev, "castling_bonus", params.evaluation.castling_bonus);
    assign_if_present(ev, "tempo_bonus", params.evaluation.tempo_bonus);
    assign_if_present(ev, "threat_weight", params.evaluation.threat_weight);
    assign_if_present(ev, "passed_pawn_base",
                      params.evaluation.passed_pawn_base);
    assign_if_present(ev, "passed_pawn_advance",
                      params.evaluation.passed_pawn_advance);
    assign_if_present(ev, "hanging_divisor", params.evaluation.hanging_divisor);
    assign_if_present(ev, "hanging_min_penalty",
                      params.evaluation.hanging_min_penalty);
    assign_if_present(ev, "king_ring_base", params.evaluation.king_ring_base);
    assign_if_present(ev, "king_ring_defended_scale",
                      params.evaluation.king_ring_defended_scale);
    assign_if_present(ev, "king_ring_enemy_occupier",
                      params.evaluation.king_ring_enemy_occupier);
    assign_if_present(ev, "king_ring_enemy_piece_material_scale",
                      params.evaluation.king_ring_enemy_piece_material_scale);
    assign_if_present(ev, "bishop_pair_bonus",
                      params.evaluation.bishop_pair_bonus);
    assign_if_present(ev, "rook_open_file_bonus",
                      params.evaluation.rook_open_file_bonus);
    assign_if_present(ev, "rook_semi_open_file_bonus",
                      params.evaluation.rook_semi_open_file_bonus);
    assign_if_present(ev, "mobility_scaling",
                      params.evaluation.mobility_scaling);
    assign_if_present(ev, "knight_dev_bonus",
                      params.evaluation.knight_dev_bonus);
    assign_if_present(ev, "bishop_dev_bonus",
                      params.evaluation.bishop_dev_bonus);
    assign_if_present(ev, "connect_rooks_bonus",
                      params.evaluation.connect_rooks_bonus);
    assign_if_present(ev, "central_pawn_bonus",
                      params.evaluation.central_pawn_bonus);
    assign_if_present(ev, "castle_urgency", params.evaluation.castle_urgency);
    assign_if_present(ev, "early_queen_penalty",
                      params.evaluation.early_queen_penalty);
    assign_if_present(ev, "flank_pawn_penalty",
                      params.evaluation.flank_pawn_penalty);
    assign_array_if_present(ev, "king_attack_weights",
                            params.evaluation.king_attack_weights);
    assign_array_if_present(ev, "threat_base", params.evaluation.threat_base);

    if (ev.contains("bishop_pin_penalty")) {
      auto pair = parse_pin_pair(ev["bishop_pin_penalty"], "bishop_pin_penalty");
      params.evaluation.bishop_pin_penalty.base = pair.base;
      params.evaluation.bishop_pin_penalty.mobility = pair.mobility;
    }
    if (ev.contains("rook_pin_penalty")) {
      auto pair = parse_pin_pair(ev["rook_pin_penalty"], "rook_pin_penalty");
      params.evaluation.rook_pin_penalty.base = pair.base;
      params.evaluation.rook_pin_penalty.mobility = pair.mobility;
    }
    if (ev.contains("knight_pin_penalty")) {
      auto pair = parse_pin_pair(ev["knight_pin_penalty"], "knight_pin_penalty");
      params.evaluation.knight_pin_penalty.base = pair.base;
      params.evaluation.knight_pin_penalty.mobility = pair.mobility;
    }
    if (ev.contains("pawn_pin_straight_penalty")) {
      auto pair = parse_pin_pair(ev["pawn_pin_straight_penalty"],
                                 "pawn_pin_straight_penalty");
      params.evaluation.pawn_pin_straight_penalty.base = pair.base;
      params.evaluation.pawn_pin_straight_penalty.mobility = pair.mobility;
    }
    if (ev.contains("pawn_pin_diagonal_penalty")) {
      auto pair = parse_pin_pair(ev["pawn_pin_diagonal_penalty"],
                                 "pawn_pin_diagonal_penalty");
      params.evaluation.pawn_pin_diagonal_penalty.base = pair.base;
      params.evaluation.pawn_pin_diagonal_penalty.mobility = pair.mobility;
    }
  }

  if (root.contains("search")) {
    auto s = py::cast<py::dict>(root["search"]);
    assign_if_present(s, "aspiration_window_initial",
                      params.search.aspiration_window_initial);
    assign_if_present(s, "aspiration_window_max",
                      params.search.aspiration_window_max);
    assign_if_present(s, "quiescence_delta_margin",
                      params.search.quiescence_delta_margin);
    assign_if_present(s, "quiescence_max_ply", params.search.quiescence_max_ply);
    assign_if_present(s, "quiescence_max_noisy_moves",
                      params.search.quiescence_max_noisy_moves);
    assign_if_present(s, "quiescence_zero_gain_skip_index",
                      params.search.quiescence_zero_gain_skip_index);
    assign_if_present(s, "null_move_reduction",
                      params.search.null_move_reduction);
    assign_if_present(s, "null_move_min_depth",
                      params.search.null_move_min_depth);
  }

  return params;
}

struct EvalOneResult {
  int cp = 0;
  std::string error;
};

EvalOneResult eval_one(const std::string& fen) {
  EvalOneResult res{};
  try {
    chess::Board b = chess::initial_board(fen);
    chess::Engine engine;
    res.cp = engine.evaluate(b);
  } catch (const std::exception& ex) {
    res.error = ex.what();
  }
  return res;
}

py::dict eval_fens(const std::vector<std::string>& fens,
                   const std::optional<py::dict>& params, int threads) {
  chess::EngineParams p =
      params ? params_from_dict(*params) : chess::default_engine_params();
  chess::set_engine_params(p);

  const std::size_t n = fens.size();
  std::vector<int> scores(n, 0);
  std::vector<std::string> errors(n);

  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  const unsigned t = threads > 0 ? static_cast<unsigned>(threads) : hw;
  const unsigned worker_count = std::max(1u, t);
  const std::size_t chunk = (n + worker_count - 1) / worker_count;

  {
    py::gil_scoped_release release;
    std::vector<std::thread> pool;
    pool.reserve(worker_count);
    for (unsigned w = 0; w < worker_count; ++w) {
      const std::size_t start = w * chunk;
      if (start >= n)
        break;
      const std::size_t end = std::min(n, start + chunk);
      pool.emplace_back([&, start, end]() {
        chess::Engine engine;
        for (std::size_t i = start; i < end; ++i) {
          try {
            chess::Board b = chess::initial_board(fens[i]);
            scores[i] = engine.evaluate(b);
            errors[i].clear();
          } catch (const std::exception& ex) {
            errors[i] = ex.what();
          }
        }
      });
    }
    for (auto& th : pool) {
      th.join();
    }
  }

  py::array_t<int> cp_array(static_cast<py::ssize_t>(n));
  auto r = cp_array.mutable_unchecked();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(n); ++i) {
    r(i) = scores[static_cast<std::size_t>(i)];
  }

  py::list err_list;
  for (const auto& e : errors) {
    if (e.empty()) {
      err_list.append(py::none());
    } else {
      err_list.append(e);
    }
  }

  return py::dict("cp"_a = cp_array, "errors"_a = err_list);
}

py::object eval_fen_single(const std::string& fen,
                           const std::optional<py::dict>& params) {
  chess::EngineParams p =
      params ? params_from_dict(*params) : chess::default_engine_params();
  chess::set_engine_params(p);
  auto res = eval_one(fen);
  if (!res.error.empty()) {
    return py::dict("ok"_a = false, "error"_a = res.error);
  }
  return py::dict("ok"_a = true, "cp"_a = res.cp);
}

struct SelfPlayOptions {
  int depth = 4;
  std::uint64_t movetime_ms = 0; // 0 means unused
  int max_plies = 160;
  int sample_stride = 4;
};

struct SelfPlayResult {
  std::vector<std::string> fens;
  std::vector<double> outcomes;
  std::vector<std::string> side_to_move;
  int games_played = 0;
};

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

chess::SearchParameters make_search_params(const SelfPlayOptions& opts) {
  chess::SearchParameters params{};
  if (opts.movetime_ms > 0) {
    params.depth = static_cast<int>(chess::MAX_PLY) - 1;
    chess::SearchLimits limits{};
    limits.use_time = true;
    limits.per_move = true;
    limits.move_time_ms = opts.movetime_ms;
    params.limits = limits;
  } else {
    params.depth = opts.depth;
  }
  params.alpha = -10000;
  params.beta = 10000;
  return params;
}

SelfPlayResult selfplay_many(const std::vector<std::string>& start_fens,
                             const std::optional<py::dict>& params,
                             const SelfPlayOptions& opts) {
  if (start_fens.empty()) {
    throw std::invalid_argument("start_fens must not be empty");
  }
  if ((opts.depth <= 0 && opts.movetime_ms == 0) || opts.max_plies <= 0 ||
      opts.sample_stride <= 0) {
    throw std::invalid_argument("invalid self-play options");
  }

  chess::EngineParams p =
      params ? params_from_dict(*params) : chess::default_engine_params();
  chess::set_engine_params(p);

  SelfPlayResult out{};
  const chess::SearchParameters search_params = make_search_params(opts);

  py::gil_scoped_release release;
  for (std::size_t game_idx = 0; game_idx < start_fens.size(); ++game_idx) {
    chess::Board board = chess::initial_board(start_fens[game_idx]);
    chess::Engine engine;
    engine.reset_history(board);

    int plies_played = 0;

    while (plies_played < opts.max_plies) {
      auto result = engine.search(board, search_params);
      const bool has_move =
          result.best_move.moving_pc != chess::OccupancyType::empty;
      if (!has_move) {
        break;
      }

      const bool irreversible = chess::move_is_irreversible(result.best_move);
      chess::make_move(board, result.best_move);
      engine.record_position(board.position_key, irreversible);
      ++plies_played;

      if ((plies_played - 1) % opts.sample_stride == 0) {
        out.fens.push_back(chess::board_to_fen(board));
        out.side_to_move.push_back(board.side_to_move == chess::SideToMove::White
                                       ? std::string("w")
                                       : std::string("b"));
      }

      if (board.is_terminal()) {
        break;
      }
    }

    const double outcome = game_outcome(board);
    for (std::size_t i = out.outcomes.size(); i < out.fens.size(); ++i) {
      out.outcomes.push_back(outcome);
    }
    out.games_played += 1;
  }
  return out;
}

} // namespace

PYBIND11_MODULE(skaks_eval, m) {
  m.doc() = "skaks evaluation bindings";

  m.def("eval_fens", &eval_fens, py::arg("fens"),
        py::arg("params") = std::nullopt, py::arg("threads") = 0,
        "Evaluate many FENs in parallel. params is an optional dict with"
        " 'evaluation'/'search' overrides.");

  m.def("eval_fen", &eval_fen_single, py::arg("fen"),
        py::arg("params") = std::nullopt,
        "Evaluate a single FEN. Returns {ok, cp or error}.");

  m.def(
      "selfplay",
      [](const std::vector<std::string>& start_fens,
         const std::optional<py::dict>& params, int depth, int movetime_ms,
         int max_plies, int sample_stride) {
        SelfPlayOptions opts{};
        opts.depth = depth;
        opts.movetime_ms =
            movetime_ms > 0 ? static_cast<std::uint64_t>(movetime_ms) : 0ULL;
        opts.max_plies = max_plies;
        opts.sample_stride = sample_stride;
        auto res = selfplay_many(start_fens, params, opts);
        return py::dict("fen"_a = res.fens, "outcome"_a = res.outcomes,
                        "side_to_move"_a = res.side_to_move,
                        "games_played"_a = res.games_played);
      },
      py::arg("start_fens"), py::arg("params") = std::nullopt,
      py::arg("depth") = 0, py::arg("movetime_ms") = 200,
      py::arg("max_plies") = 160, py::arg("sample_stride") = 4,
      "Run internal self-play over start_fens and return sampled FENs."
      " Exactly one of depth or movetime_ms must be positive.");
}
