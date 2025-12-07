# Sample clang-based toolchain for cross-compiling to x86_64 Linux from macOS.
# Update the sysroot and paths to match your SDK or cross environment.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER clang CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER clang++ CACHE FILEPATH "" FORCE)

set(CMAKE_C_COMPILER_TARGET "x86_64-linux-gnu" CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER_TARGET "x86_64-linux-gnu" CACHE STRING "" FORCE)

set(CHESS_LINUX_SYSROOT "/opt/sysroots/x86_64-linux-gnu" CACHE PATH "Linux sysroot path")

set(CMAKE_SYSROOT "${CHESS_LINUX_SYSROOT}" CACHE PATH "" FORCE)

set(CMAKE_EXE_LINKER_FLAGS_INIT "--target=x86_64-linux-gnu" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_INIT "--target=x86_64-linux-gnu" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_INIT "--target=x86_64-linux-gnu" CACHE STRING "" FORCE)
