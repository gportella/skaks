#include "nnue.h"

#include "chess/board.hpp"
#include "chess/types.hpp"

#include <algorithm>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {

struct NetworkPayload {
  std::string path;
  std::size_t size = 0;
};

std::mutex g_mutex;
std::shared_ptr<NetworkPayload> g_payload;

chess::OccupancyType code_to_occupancy(int code) {
  switch (code) {
  case 1:
    return chess::OccupancyType::wK;
  case 2:
    return chess::OccupancyType::wQ;
  case 3:
    return chess::OccupancyType::wR;
  case 4:
    return chess::OccupancyType::wB;
  case 5:
    return chess::OccupancyType::wN;
  case 6:
    return chess::OccupancyType::wP;
  case 7:
    return chess::OccupancyType::bK;
  case 8:
    return chess::OccupancyType::bQ;
  case 9:
    return chess::OccupancyType::bR;
  case 10:
    return chess::OccupancyType::bB;
  case 11:
    return chess::OccupancyType::bN;
  case 12:
    return chess::OccupancyType::bP;
  default:
    return chess::OccupancyType::empty;
  }
}

int piece_value(chess::OccupancyType occ) {
  switch (occ) {
  case chess::OccupancyType::wP:
    return 100;
  case chess::OccupancyType::wN:
    return 320;
  case chess::OccupancyType::wB:
    return 330;
  case chess::OccupancyType::wR:
    return 500;
  case chess::OccupancyType::wQ:
    return 900;
  case chess::OccupancyType::wK:
    return 0;
  case chess::OccupancyType::bP:
    return -100;
  case chess::OccupancyType::bN:
    return -320;
  case chess::OccupancyType::bB:
    return -330;
  case chess::OccupancyType::bR:
    return -500;
  case chess::OccupancyType::bQ:
    return -900;
  case chess::OccupancyType::bK:
    return 0;
  default:
    return 0;
  }
}

int evaluate_material(int player, const int* pieces, const int* squares) {
  (void)squares;
  int score = 0;
  for (int idx = 0; idx < 33; ++idx) {
    const int code = pieces[idx];
    if (code <= 0) {
      break;
    }
    const auto occ = code_to_occupancy(code);
    score += piece_value(occ);
  }
  return player == 0 ? score : -score;
}

} // namespace

extern "C" void nnue_init(const char* evalFile) {
  if (!evalFile || *evalFile == '\0') {
    throw std::runtime_error("NNUE init received empty path");
  }

  std::ifstream in(evalFile, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("NNUE network file not found: " +
                             std::string(evalFile));
  }

  auto payload = std::make_shared<NetworkPayload>();
  payload->path = evalFile;
  payload->size = static_cast<std::size_t>(in.tellg());

  std::lock_guard<std::mutex> lock(g_mutex);
  g_payload = std::move(payload);
}

static int eval_stub() {
  return 0;
}

extern "C" int nnue_evaluate(int player, int* pieces, int* squares) {
  if (!pieces || !squares) {
    return eval_stub();
  }
  const int clipped_player = (player < 0) ? 0 : (player > 1 ? 1 : player);
  return evaluate_material(clipped_player, pieces, squares);
}

extern "C" int nnue_evaluate_incremental(int player, int* pieces, int* squares,
                                         NNUEdata** nnue_data) {
  (void)nnue_data;
  return nnue_evaluate(player, pieces, squares);
}

extern "C" int nnue_evaluate_fen(const char* fen) {
  if (!fen) {
    return eval_stub();
  }
  try {
    chess::Board board = chess::initial_board(fen);
    int score = 0;
    for (const auto occ : board.pieces) {
      score += piece_value(occ);
    }
    return board.side_to_move == chess::SideToMove::White ? score : -score;
  } catch (...) {
    return eval_stub();
  }
}
