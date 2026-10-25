# Third-party dependencies for shelf.
#
# Phase 1 core must build CPU-only with the standard library alone.
# Optional backends pull their toolchains here.

if(SHELF_ENABLE_HIP)
    # ROCm SDK location is provided via CMAKE_PREFIX_PATH (see the `hip`
    # preset). On this machine: pip package _rocm_sdk_devel (ROCm 7.14),
    # whose lib/cmake/hip exports hip::host / hip::device.
    find_package(hip CONFIG REQUIRED)
endif()

# SYCL runtime is NOT installed on this machine (planned). backend/sycl stays
# empty until a SYCL toolchain lands; do not silently substitute HIP for it.

# Vendored header-only deps (see third_party/README.md for licenses/provenance).
add_library(shelf_nlohmann INTERFACE)
add_library(shelf::nlohmann ALIAS shelf_nlohmann)
target_include_directories(shelf_nlohmann SYSTEM INTERFACE
    "${PROJECT_SOURCE_DIR}/third_party/nlohmann/include")
