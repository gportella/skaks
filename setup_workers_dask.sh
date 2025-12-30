cat setup_workers.sh
#!/usr/bin/env bash
set -euo pipefail

# Configurable paths
REPO_ROOT="${REPO_ROOT:-/mnt/skaks}"
BINDINGS_DIR="${BINDINGS_DIR:-/mnt/skaks/bindings/python}"
PY="/opt/conda/bin/python"

echo "[setup] Using repo: ${REPO_ROOT}"
echo "[setup] Using bindings: ${BINDINGS_DIR}"
echo "[setup] Using python: ${PY}"

# 1) System deps (libc++-17)
echo "[setup] Installing libc++-17 and libc++abi-17..."
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y libc++-17-dev libc++abi-17-dev
else
  apt-get update
  apt-get install -y libc++-17-dev libc++abi-17-dev
fi

# 2) Python build helpers
echo "[setup] Installing Python build tools..."
${PY} -m pip install --upgrade pip
${PY} -m pip install --no-cache-dir build wheel "scikit-build-core>=0.9" cmake ninja

# 3) Install from repo root (CMake with Clang-17/libc++)
echo "[setup] Building and installing tuning from repo root..."
cd "${REPO_ROOT}"
cmake --preset linux-ninja-libcxx-release
cmake --build --preset linux-ninja-libcxx-release

# 4) Install explicit bindings (if needed)
echo "[setup] Building and installing Python bindings subpackage..."
cd "${BINDINGS_DIR}"
SKBUILD_CMAKE_GENERATOR=Ninja CC=/usr/bin/clang-17 CXX=/usr/bin/clang++-17 \
  ${PY} -m pip install . -v \
  --config-settings=cmake.define.CMAKE_BUILD_TYPE=Release \
  --config-settings=cmake.define.CMAKE_CXX_STANDARD=23 \
  --config-settings=cmake.define.CMAKE_CXX_EXTENSIONS=OFF \
  --config-settings=cmake.define.CMAKE_POSITION_INDEPENDENT_CODE=ON \
  --config-settings=cmake.define.CMAKE_CXX_FLAGS="-stdlib=libc++" \
  --config-settings=cmake.define.CMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -Wl,-rpath,/usr/lib/llvm-17/lib -L/usr/lib/llvm-17/lib" \
  --config-settings=cmake.define.CMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -Wl,-rpath,/usr/lib/llvm-17/lib -L/usr/lib/llvm-17/lib" \
  --config-settings=cmake.define.CMAKE_INSTALL_RPATH="/usr/lib/llvm-17/lib"

# 5) Pin Dask/Distributed
DASK_VERSION="${DASK_VERSION:-2025.3.0}"
echo "[setup] Installing Dask/Distributed ${DASK_VERSION}..."
${PY} -m pip install --no-cache-dir "dask[complete]==${DASK_VERSION}" "distributed==${DASK_VERSION}"

cd ../../

# 6) Verify imports
echo "[setup] Verifying tuning imports..."
${PY} - <<'PY'
import importlib.util
assert importlib.util.find_spec('tuning'), 'Cannot import tuning'
assert importlib.util.find_spec('tuning.param_optimize'), 'Cannot import tuning.param_optimize'
print('tuning import OK')
PY

# 7) Patch ha.py to use /opt/conda/bin/python for subprocess
HA="${REPO_ROOT}/ha.py"
if [ -f "$HA" ]; then
  echo "[setup] Patching ha.py to use /opt/conda/bin/python..."
  python3 - <<'PY'
import pathlib, re
p = pathlib.Path("/mnt/skaks/ha.py")
s = p.read_text()
s = re.sub(r'(?m)^\s*cmd\s*=\s*\[\s*"python"\s*,', 'cmd = ["/opt/conda/bin/python",', s)
p.write_text(s)
print("ha.py patched")
PY
else
  echo "[setup] ha.py not found at ${HA}; skipping patch."
fi

echo "[setup] Completed successfully."
