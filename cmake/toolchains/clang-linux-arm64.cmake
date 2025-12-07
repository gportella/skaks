# Sample clang-based toolchain for cross-compiling to arm64 Linux from macOS.
# Update sysroot/toolchain paths to match your environment.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER clang CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER clang++ CACHE FILEPATH "" FORCE)

set(CMAKE_C_COMPILER_TARGET "aarch64-linux-gnu" CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER_TARGET "aarch64-linux-gnu" CACHE STRING "" FORCE)

set(CHESS_LINUX_SYSROOT "/opt/sysroots/aarch64-linux-gnu" CACHE PATH "Linux sysroot path")

set(CMAKE_SYSROOT "${CHESS_LINUX_SYSROOT}" CACHE PATH "" FORCE)

set(CMAKE_EXE_LINKER_FLAGS_INIT "--target=aarch64-linux-gnu" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_INIT "--target=aarch64-linux-gnu" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_INIT "--target=aarch64-linux-gnu" CACHE STRING "" FORCE)
