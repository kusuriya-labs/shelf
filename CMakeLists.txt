cmake_minimum_required(VERSION 3.24)

# Resolves the ROCm root and pins the default C++ compiler to the ROCm
# toolchain clang++ before project() selects a compiler.
include(cmake/rocm_root.cmake)

option(SHELF_HY4_RDMA "Hy4 Engram の RDMA 読み出しを有効にする" OFF)
option(SHELF_HY4_ONLY "Hy4-flash 専用の shelf-compute を構築する" ON)
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
        add_executable(bench_hy4_v620_expert EXCLUDE_FROM_ALL tests/kernels/bench_hy4_v620_expert.cpp)
        target_link_libraries(bench_hy4_v620_expert PRIVATE shelf_core hip::host shelf_hy4_flash)
        shelf_set_rocm_rpath(bench_hy4_v620_expert)
        set_property(TARGET bench_hy4_v620_expert APPEND PROPERTY BUILD_RPATH "${SHELF_LIBCXX_LLVM_ROOT}/lib")
        add_executable(verify_hy4_swiglu EXCLUDE_FROM_ALL tests/kernels/verify_hy4_swiglu.hip)
        target_link_libraries(verify_hy4_swiglu PRIVATE shelf_core hip::host)
        shelf_set_rocm_rpath(verify_hy4_swiglu)
        set_property(TARGET verify_hy4_swiglu APPEND PROPERTY BUILD_RPATH "${SHELF_LIBCXX_LLVM_ROOT}/lib")
        add_executable(test_safetensors_population tests/unit/test_safetensors_population.cpp)
        target_link_libraries(test_safetensors_population PRIVATE shelf_io)
        set_property(TARGET test_safetensors_population APPEND PROPERTY BUILD_RPATH "${SHELF_LIBCXX_LLVM_ROOT}/lib")
        add_test(NAME safetensors_population COMMAND test_safetensors_population)
        if(SHELF_HY4_RDMA)
            add_executable(test_hy4_snapshot_rdma EXCLUDE_FROM_ALL tests/unit/test_hy4_snapshot_rdma.cpp)
            target_link_libraries(test_hy4_snapshot_rdma PRIVATE shelf_hy4_model)
            shelf_set_rocm_rpath(test_hy4_snapshot_rdma)
            set_property(TARGET test_hy4_snapshot_rdma APPEND PROPERTY BUILD_RPATH "${SHELF_LIBCXX_LLVM_ROOT}/lib")
            add_executable(test_hy4_engram_rdma EXCLUDE_FROM_ALL tests/unit/test_hy4_engram_rdma.cpp)
            target_include_directories(test_hy4_engram_rdma SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
            target_link_libraries(test_hy4_engram_rdma PRIVATE shelf_hy4_model)
            add_executable(test_hy4_engram_rdma_timeout EXCLUDE_FROM_ALL tests/unit/test_hy4_engram_rdma_timeout.cpp)
            target_link_libraries(test_hy4_engram_rdma_timeout PRIVATE shelf_hy4_model)
            foreach(rdma_test test_hy4_engram_rdma test_hy4_engram_rdma_timeout)
                shelf_set_rocm_rpath(${rdma_test})
                set_property(TARGET ${rdma_test} APPEND PROPERTY BUILD_RPATH "${SHELF_LIBCXX_LLVM_ROOT}/lib")
            endforeach()
            add_library(hy4_rdma_fault_clock SHARED EXCLUDE_FROM_ALL tests/unit/hy4_rdma_fault_clock.cpp)
            target_link_libraries(hy4_rdma_fault_clock PRIVATE dl)
        endif()
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        add_test(NAME hy4_flash_cpu COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/unit/test_hy4_flash.py)
    else()
        add_subdirectory(tests)
    endif()
endif()
