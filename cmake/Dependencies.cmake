# Third-party dependencies for shelf.
#
# Phase 1 core must build CPU-only with the standard library alone.
# Optional backends pull their toolchains here.

if(SHELF_ENABLE_HIP)
    # ROCm root (and therefore the hip::host/hip::device config) was resolved
    # in cmake/RocmRoot.cmake via -DSHELF_ROCM_ROOT / ROCM_PATH /
    # `rocm-sdk path --root` — machine-specific paths are never hardcoded.
    find_package(hip CONFIG REQUIRED)

    # HIP language TUs (kernels). Compiler was pinned to the same ROCm
    # clang++ in RocmRoot.cmake; device arch is a single explicit target.
    if(NOT DEFINED CMAKE_HIP_COMPILER)
        set(CMAKE_HIP_COMPILER "${SHELF_ROCM_ROOT}/lib/llvm/bin/clang++")
    endif()
    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx1201")
    endif()
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)

    # ROCm/CMake leaves CMAKE_HIP_FLAGS_<CONFIG> empty; without -O3 the
    # kernels compile at -O0 and invalidate any performance measurement
    # (lesson carried from the PhaseShift build system).
    string(FIND "${CMAKE_HIP_FLAGS}" "-O" _shelf_hip_has_opt)
    if(_shelf_hip_has_opt EQUAL -1)
        string(APPEND CMAKE_HIP_FLAGS " -O3")
    endif()
endif()

# SYCL runtime: AdaptiveCpp (hipSYCL). Not required by CPU/HIP-only builds.
# Discovery order (machine-specific paths are never hardcoded):
#   -DAdaptiveCpp_DIR=<dir with AdaptiveCppConfig.cmake>
#   -DSHELF_SYCL_ROOT=<install prefix>   (or ACPP_PATH env var)
#   `acpp` on PATH (installation root parsed from `acpp --help`)
if(SHELF_ENABLE_SYCL)
    if(NOT DEFINED AdaptiveCpp_DIR)
        set(_shelf_sycl_root "${SHELF_SYCL_ROOT}")
        if(_shelf_sycl_root STREQUAL "" AND DEFINED ENV{ACPP_PATH})
            set(_shelf_sycl_root "$ENV{ACPP_PATH}")
        endif()
        if(_shelf_sycl_root STREQUAL "")
            find_program(SHELF_ACPP_EXECUTABLE NAMES acpp)
            if(SHELF_ACPP_EXECUTABLE)
                execute_process(
                    COMMAND "${SHELF_ACPP_EXECUTABLE}" --help
                    OUTPUT_VARIABLE _shelf_acpp_help ERROR_QUIET)
                string(REGEX MATCH "Installation root: ([^ \n]+)"
                       _shelf_acpp_match "${_shelf_acpp_help}")
                if(CMAKE_MATCH_1)
                    set(_shelf_sycl_root "${CMAKE_MATCH_1}")
                endif()
            endif()
        endif()
        if(NOT _shelf_sycl_root STREQUAL "")
            set(AdaptiveCpp_DIR "${_shelf_sycl_root}/lib/cmake/AdaptiveCpp")
        endif()
        unset(_shelf_sycl_root)
    endif()
    find_package(AdaptiveCpp CONFIG REQUIRED)
endif()

# Vendored header-only deps (see third_party/README.md for licenses/provenance).
add_library(shelf_nlohmann INTERFACE)
add_library(shelf::nlohmann ALIAS shelf_nlohmann)
target_include_directories(shelf_nlohmann SYSTEM INTERFACE
    "${PROJECT_SOURCE_DIR}/third_party/nlohmann/include")
