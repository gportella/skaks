conda install ninja
sudo apt-get update && apt-get install -y libc++-17-dev libc++abi-17-dev || apt-get install -y libc++-dev libc++abi-dev
sudo apt-get install -y libc++-17-dev libc++abi-17-dev || apt-get install -y libc++-dev libc++abi-dev

cmake -S . -B build/linux-release-ninja -G Ninja -D CMAKE_C_COMPILER=/usr/bin/clang-17 -D CMAKE_CXX_COMPILER=/usr/bin/clang++-17 -D CMAKE_CXX_STANDARD=23 -D CMAKE_CXX_EXTENSIONS=OFF -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_CXX_FLAGS="-nostdinc++ -isystem /usr/include/c++/v1 -Wno-error=unused-variable -Wno-error=unused-command-line-argument" -D CMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -lc++abi -L/usr/lib/x86_64-linux-gnu" -D CMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -lc++abi -L/usr/lib/x86_64-linux-gnu" -D CMAKE_AR=/usr/bin/ar -D CMAKE_RANLIB=/usr/bin/ranlib

cmake -S . -B build/linux-release-ninja -G Ninja -D CMAKE_C_COMPILER=/usr/bin/clang-17 -D CMAKE_CXX_COMPILER=/usr/bin/clang++-17 -D CMAKE_CXX_STANDARD=23 -D CMAKE_CXX_EXTENSIONS=OFF -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_CXX_FLAGS="-nostdinc++ -isystem /usr/include/c++/v1 -Wno-error=unused-variable -Wno-error=unused-command-line-argument" -D CMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -lc++abi -L/usr/lib/x86_64-linux-gnu" -D CMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -lc++abi -L/usr/lib/x86_64-linux-gnu" -D CMAKE_AR=/usr/bin/ar -D CMAKE_RANLIB=/usr/bin/ranlib

cd bindings/python/

rm -rf build dist *.egg-info _skbuild && SKBUILD_CMAKE_GENERATOR="Ninja" CC=/usr/bin/clang-17 CXX=/usr/bin/clang++-17 pip install . -v --config-settings=cmake.define.CMAKE_BUILD_TYPE=Release --config-settings=cmake.define.CMAKE_C_COMPILER=/usr/bin/clang-17 --config-settings=cmake.define.CMAKE_CXX_COMPILER=/usr/bin/clang++-17 --config-settings=cmake.define.CMAKE_CXX_STANDARD=23 --config-settings=cmake.define.CMAKE_CXX_EXTENSIONS=OFF --config-settings=cmake.define.CMAKE_POSITION_INDEPENDENT_CODE=ON --config-settings=cmake.define.CMAKE_CXX_FLAGS="-nostdinc++ -isystem /usr/include/c++/v1 -Wno-error=unused-variable -Wno-error=unused-command-line-argument" --config-settings=cmake.define.CMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -lc++abi -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu" --config-settings=cmake.define.CMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -lc++abi -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu"

cd ..
