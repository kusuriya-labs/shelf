cmake_minimum_required(VERSION 3.24)

# Resolves the ROCm root and pins the default C++ compiler to the ROCm
# toolchain clang++ before project() selects a compiler.
include(cmake/rocm_root.cmake)

project(
    PhaseShift
    VERSION 0.1.0
    LANGUAGES CXX HIP
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options and platform detection.
include(cmake/options.cmake)

# Production / library targets.
include(cmake/targets.cmake)

# User-facing application entrypoints.
include(cmake/apps.cmake)

# Resident model host (internal runtime component).
include(cmake/model_host.cmake)

# Test suite.
if(PHASESHIFT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
