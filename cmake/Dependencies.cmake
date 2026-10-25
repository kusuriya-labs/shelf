# Third-party dependencies for shelf.
#
# Phase 1 core must build CPU-only with the standard library alone.
# Optional backends pull their toolchains here.

if(SHELF_ENABLE_HIP)
    # ROCm SDK location is provided via CMAKE_PREFIX_PATH (see the `hip`
    # preset). On this machine: pip package _rocm_sdk_devel (ROCm 7.14),
    # whose lib/cmake/hip exports hip::host / hip::device.
    find_package(hip CONFIG REQUIRED)

    # HIP language TUs (kernels). The compiler must come from the same ROCm
    # root as hip-config, and the whole project is then built with that
    # clang++ (preset pins CMAKE_CXX_COMPILER) so host and device TUs share
    # one C++ ABI. Device arch is a single explicit target (gfx1201).
    if(NOT DEFINED CMAKE_HIP_COMPILER)
        get_filename_component(_shelf_rocm_root "${hip_DIR}" DIRECTORY)
        get_filename_component(_shelf_rocm_root "${_shelf_rocm_root}" DIRECTORY)
        get_filename_component(_shelf_rocm_root "${_shelf_rocm_root}" DIRECTORY)
        set(CMAKE_HIP_COMPILER "${_shelf_rocm_root}/lib/llvm/bin/clang++")
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

# SYCL runtime is NOT installed on this machine (planned). backend/sycl stays
# empty until a SYCL toolchain lands; do not silently substitute HIP for it.

# Vendored header-only deps (see third_party/README.md for licenses/provenance).
add_library(shelf_nlohmann INTERFACE)
add_library(shelf::nlohmann ALIAS shelf_nlohmann)
target_include_directories(shelf_nlohmann SYSTEM INTERFACE
    "${PROJECT_SOURCE_DIR}/third_party/nlohmann/include")
