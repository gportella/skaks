git pull
git stauts
git switch nnue_version
cmake --preset linux-ninja-libcxx-release
cmake --build --preset linux-ninja-libcxx-release
cmake --preset linux-ninja-libcxx-release
cmake --build --preset linux-ninja-libcxx-release
sudo apt-get update && apt-get install -y libc++-17-dev libc++abi-17-dev || sudo apt-get install -y libc++-dev libc++abi-dev
sudo apt-get install -y libc++-17-dev libc++abi-17-dev || sudo apt-get install -y libc++-dev libc++abi-dev
cmake --preset linux-ninja-libcxx-release
cmake --build --preset linux-ninja-libcxx-release
sudo apt-get install -y libc++-17-dev libc++abi-17-dev
SKBUILD_CMAKE_GENERATOR=Ninja CC=/usr/bin/clang-17 CXX=/usr/bin/clang++-17 pip install . -v --config-settings=cmake.define.CMAKE_BUILD_TYPE=Release --config-settings=cmake.define.CMAKE_CXX_STANDARD=23 --config-settings=cmake.define.CMAKE_CXX_EXTENSIONS=OFF --config-settings=cmake.define.CMAKE_POSITION_INDEPENDENT_CODE=ON --config-settings=cmake.define.CMAKE_CXX_FLAGS="-stdlib=libc++" --config-settings=cmake.define.CMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -Wl,-rpath,/usr/lib/llvm-17/lib -L/usr/lib/llvm-17/lib" --config-settings=cmake.define.CMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -Wl,-rpath,/usr/lib/llvm-17/lib -L/usr/lib/llvm-17/lib" --config-settings=cmake.define.CMAKE_INSTALL_RPATH="/usr/lib/llvm-17/lib"
cd bindings/python/
sudo apt-get install -y libc++-17-dev libc++abi-17-dev
SKBUILD_CMAKE_GENERATOR=Ninja CC=/usr/bin/clang-17 CXX=/usr/bin/clang++-17 pip install . -v --config-settings=cmake.define.CMAKE_BUILD_TYPE=Release --config-settings=cmake.define.CMAKE_CXX_STANDARD=23 --config-settings=cmake.define.CMAKE_CXX_EXTENSIONS=OFF --config-settings=cmake.define.CMAKE_POSITION_INDEPENDENT_CODE=ON --config-settings=cmake.define.CMAKE_CXX_FLAGS="-stdlib=libc++" --config-settings=cmake.define.CMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -Wl,-rpath,/usr/lib/llvm-17/lib -L/usr/lib/llvm-17/lib" --config-settings=cmake.define.CMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -Wl,-rpath,/usr/lib/llvm-17/lib -L/usr/lib/llvm-17/lib" --config-settings=cmake.define.CMAKE_INSTALL_RPATH="/usr/lib/llvm-17/lib"

pip install dask[complete]==2025.3.0
