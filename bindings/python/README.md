# skaks-eval Python bindings

Build with `pip install .` from this directory (requires Python >=3.10, scikit-build-core, pybind11, CMake 3.24+).

Example:

```bash
cd bindings/python
UV_INDEX_URL=${UV_INDEX_URL:-https://pypi.org/simple} \
  pip install .
```

Usage:

```python
import skaks_eval
fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "8/8/8/8/8/8/8/8 w - - 0 1"]
res = skaks_eval.eval_fens(fens)
print(res["cp"])        # numpy array of ints
print(res["errors"])    # list of None or error strings
```

`params` can override evaluation/search fields via nested dicts matching the C++ structs; arrays must match expected lengths.
