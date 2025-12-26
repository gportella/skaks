#include "chess/board.hpp"
#include "chess/nnue.hpp"

#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

py::array_t<std::int8_t> features_from_fen(const std::string& fen) {
  chess::Board b = chess::initial_board(fen);
  chess::NnueFeatures f = chess::make_nnue_features(b);

  py::array_t<std::int8_t> arr({static_cast<py::ssize_t>(chess::kNnueInputs)});
  auto buf = arr.mutable_unchecked<1>();
  for (std::size_t i = 0; i < chess::kNnueInputs; ++i) {
    buf(static_cast<py::ssize_t>(i)) = f.values[i];
  }
  return arr;
}

std::vector<float>
to_vector_flat(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
               std::size_t expected) {
  const auto buf = arr.request();
  if (buf.ndim == 2) {
    if (buf.shape[0] * buf.shape[1] != static_cast<py::ssize_t>(expected)) {
      throw std::invalid_argument("Unexpected size for weight matrix");
    }
  } else if (buf.ndim == 1) {
    if (buf.shape[0] != static_cast<py::ssize_t>(expected)) {
      throw std::invalid_argument("Unexpected size for weight vector");
    }
  } else {
    throw std::invalid_argument("Weights must be 1D or 2D");
  }
  const float* data = static_cast<const float*>(buf.ptr);
  return std::vector<float>(data, data + expected);
}

std::vector<std::int8_t> to_vector_int8(
    py::array_t<std::int8_t, py::array::c_style | py::array::forcecast> arr,
    std::size_t expected) {
  const auto buf = arr.request();
  if (buf.ndim != 1 || buf.shape[0] != static_cast<py::ssize_t>(expected)) {
    throw std::invalid_argument("features must be 1D of length kNnueInputs");
  }
  const std::int8_t* data = static_cast<const std::int8_t*>(buf.ptr);
  return std::vector<std::int8_t>(data, data + expected);
}

double forward_from_arrays(
    py::array_t<float, py::array::c_style | py::array::forcecast> w1,
    py::array_t<float, py::array::c_style | py::array::forcecast> b1,
    py::array_t<float, py::array::c_style | py::array::forcecast> w2,
    py::array_t<float, py::array::c_style | py::array::forcecast> b2,
    py::object features, py::object fen_opt) {
  const auto b1_vec = to_vector_flat(b1, static_cast<std::size_t>(b1.size()));
  const std::size_t hidden = b1_vec.size();
  const auto w1_vec = to_vector_flat(w1, hidden * chess::kNnueInputs);
  const auto w2_vec = to_vector_flat(w2, hidden);
  const auto b2_vec = to_vector_flat(b2, 1);

  chess::NnueNetwork net{};
  net.w1 = w1_vec;
  net.b1 = b1_vec;
  net.w2 = w2_vec;
  net.b2 = b2_vec.front();

  chess::NnueFeatures f{};
  if (!fen_opt.is_none()) {
    const auto fen = fen_opt.cast<std::string>();
    f = chess::make_nnue_features(chess::initial_board(fen));
  } else if (!features.is_none()) {
    auto feat_vec = to_vector_int8(
        features.cast<py::array_t<std::int8_t,
                                  py::array::c_style | py::array::forcecast>>(),
        chess::kNnueInputs);
    for (std::size_t i = 0; i < chess::kNnueInputs; ++i) {
      f.values[i] = feat_vec[i];
    }
  } else {
    throw std::invalid_argument("Provide either features or fen");
  }

  return static_cast<double>(net.forward(f));
}

py::dict load_yaml_weights(const std::string& path) {
  chess::NnueNetwork net{};
  std::string error;
  if (!chess::load_nnue_from_file(path, net, error)) {
    throw std::invalid_argument(error);
  }
  const auto hidden = static_cast<py::ssize_t>(net.hidden_size());
  py::dict d;
  d["hidden"] = hidden;

  py::array_t<float> w1_arr(
      {hidden, static_cast<py::ssize_t>(chess::kNnueInputs)});
  std::memcpy(w1_arr.mutable_data(), net.w1.data(),
              net.w1.size() * sizeof(float));
  d["w1"] = std::move(w1_arr);

  py::array_t<float> b1_arr({hidden});
  std::memcpy(b1_arr.mutable_data(), net.b1.data(),
              net.b1.size() * sizeof(float));
  d["b1"] = std::move(b1_arr);

  py::array_t<float> w2_arr({hidden});
  std::memcpy(w2_arr.mutable_data(), net.w2.data(),
              net.w2.size() * sizeof(float));
  d["w2"] = std::move(w2_arr);

  d["b2"] = net.b2;
  return d;
}

} // namespace

void bind_nnue(py::module_& m) {
  // Ensure NumPy is imported before constructing arrays to avoid lazy-import
  // recursion issues in some Python environments.
  py::module_::import("numpy");
  m.def("features_from_fen", &features_from_fen,
        "Return NNUE input features (int8 array) for a FEN");
  m.def("nnue_forward", &forward_from_arrays, py::arg("w1"), py::arg("b1"),
        py::arg("w2"), py::arg("b2"), py::arg("features") = py::none(),
        py::arg("fen") = py::none(),
        "Run NNUE forward given weights and either features or a FEN");
  m.def("load_nnue_yaml", &load_yaml_weights, py::arg("path"),
        "Load NNUE weights from YAML and return a dict of arrays");
}
