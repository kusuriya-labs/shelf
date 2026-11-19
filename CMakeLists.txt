cmake_minimum_required(VERSION 3.24)

# Resolves the ROCm root and pins the default C++ compiler to the ROCm
# toolchain clang++ before project() selects a compiler.
include(cmake/rocm_root.cmake)

option(SHELF_HY4_ONLY "Hy4-flash 専用の shelf-compute を構築する" OFF)
if(SHELF_HY4_ONLY AND NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx1201 CACHE STRING "Hy4-flash のホスト側 HIP ターゲット")
endif()

project(
    Shelf
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

# Test suite.
if(SHELF_BUILD_TESTS)
    enable_testing()
    if(SHELF_HY4_ONLY)
        add_executable(test_hy4_flash_layer EXCLUDE_FROM_ALL tests/kernels/test_hy4_flash_layer.cpp)
        target_link_libraries(test_hy4_flash_layer PRIVATE shelf_hy4_model)
        target_include_directories(test_hy4_flash_layer SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
        shelf_set_rocm_rpath(test_hy4_flash_layer)
        set_property(TARGET test_hy4_flash_layer APPEND PROPERTY BUILD_RPATH "${SHELF_LIBCXX_LLVM_ROOT}/lib")
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        add_test(NAME hy4_flash_cpu COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/unit/test_hy4_flash.py)
    else()
        add_subdirectory(tests)
    endif()
endif()
