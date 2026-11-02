# PhaseShift build options and platform detection.
#
# Included from root CMakeLists.txt.

# ---------------------------------------------------------------------------
# Toolchain enforcement: GCC and libstdc++ are banned.
# All C++ / HIP translation units are compiled by the ROCm toolchain clang
# with the libc++ standard library (see overview.md "Toolchain").
# ---------------------------------------------------------------------------

if(NOT PHASESHIFT_ROCM_ROOT)
    message(
        FATAL_ERROR
        "PHASESHIFT_ROCM_ROOT is empty. Set -DPHASESHIFT_ROCM_ROOT=<path>, "
        "the ROCM_PATH environment variable, or provide the rocm-sdk CLI."
    )
endif()

string(REGEX REPLACE "/+$" "" _ps_llvm_root "${PHASESHIFT_ROCM_ROOT}")
set(_ps_llvm_prefix "${_ps_llvm_root}/lib/llvm/")

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(
        FATAL_ERROR
        "GCC compiler is banned. The C++ compiler must be the ROCm toolchain "
        "clang++ (got: ${CMAKE_CXX_COMPILER}). Remove -DCMAKE_CXX_COMPILER=g++."
    )
endif()

if(DEFINED CMAKE_HIP_COMPILER_ID AND NOT CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
    message(
        FATAL_ERROR
        "The HIP compiler must be the ROCm toolchain clang++ (got: ${CMAKE_HIP_COMPILER})."
    )
endif()

string(FIND "${CMAKE_CXX_COMPILER}" "${_ps_llvm_prefix}" _ps_cxx_prefix_pos)
if(NOT _ps_cxx_prefix_pos EQUAL 0)
    message(
        FATAL_ERROR
        "The C++ compiler must come from the ROCm toolchain ${_ps_llvm_prefix} "
        "(got: ${CMAKE_CXX_COMPILER})."
    )
endif()

string(FIND "${CMAKE_HIP_COMPILER}" "${_ps_llvm_prefix}" _ps_hip_prefix_pos)
if(NOT _ps_hip_prefix_pos EQUAL 0)
    message(
        FATAL_ERROR
        "The HIP compiler must come from the ROCm toolchain ${_ps_llvm_prefix} "
        "(got: ${CMAKE_HIP_COMPILER})."
    )
endif()

set(
    PHASESHIFT_LIBCXX_LLVM_ROOT
    "/usr/lib/llvm-23"
    CACHE PATH
    "System LLVM root providing the host libc++ (libc++-23-dev / libc++abi-23-dev)"
)

if(NOT EXISTS "${PHASESHIFT_LIBCXX_LLVM_ROOT}/include/c++/v1")
    message(
        FATAL_ERROR
        "host libc++ headers were not found under ${PHASESHIFT_LIBCXX_LLVM_ROOT}. "
        "Install libc++-23-dev and libc++abi-23-dev (e.g. ./llvm.sh 23 all), "
        "or point PHASESHIFT_LIBCXX_LLVM_ROOT at a valid LLVM root."
    )
endif()

if(NOT EXISTS "${PHASESHIFT_LIBCXX_LLVM_ROOT}/lib/libc++.so")
    message(
        FATAL_ERROR
        "host libc++ library was not found under ${PHASESHIFT_LIBCXX_LLVM_ROOT}. "
        "Install libc++-23-dev and libc++abi-23-dev."
    )
endif()

# The space-separated form of -stdlib++-isystem is required: the AMD clang
# 23 driver ignores the = form. Applies to CXX and HIP (the only languages).
# The HIP pipeline forwards -stdlib=libc++ to a device-side compilation that
# does not consume it; silence that warning only.
add_compile_options(
    -stdlib=libc++
    -stdlib++-isystem
    "${PHASESHIFT_LIBCXX_LLVM_ROOT}/include/c++/v1"
    -Wno-unused-command-line-argument
)

# The host libc++ / libc++abi are linked by absolute path from the pinned
# LLVM root. Do NOT pass -stdlib=libc++ at link time: the driver's
# auto-link emits `-lc++`, which lld resolves by searching the toolchain
# -L directories in order. On this host /usr/lib/x86_64-linux-gnu/libc++.so
# is a 28-byte libtool linker script (not an ELF), and lld skips the
# non-ELF file (then dedups it against the absolute-path library), leaving
# every std::__* symbol undefined. Linking the pinned libraries directly,
# with no -lc++ on the line, makes resolution independent of -l search.
add_link_options(
    "${PHASESHIFT_LIBCXX_LLVM_ROOT}/lib/libc++abi.so"
    "${PHASESHIFT_LIBCXX_LLVM_ROOT}/lib/libc++.so"
)

# ---------------------------------------------------------------------------
# GPU architecture: gfx1201 only (AMD Radeon AI PRO R9700).
# CMAKE_HIP_ARCHITECTURES may be auto-detected (the local GPU codename) or set
# explicitly. Any other architecture is rejected at configure time.
# ---------------------------------------------------------------------------

if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1201")
    message(
        FATAL_ERROR
        "PhaseShift targets gfx1201 (AMD Radeon AI PRO R9700) only. "
        "Got CMAKE_HIP_ARCHITECTURES='${CMAKE_HIP_ARCHITECTURES}'."
    )
endif()

message(STATUS "PhaseShift GPU architecture: ${CMAKE_HIP_ARCHITECTURES}")

list(PREPEND CMAKE_PREFIX_PATH "${PHASESHIFT_ROCM_ROOT}")

function(phaseshift_set_rocm_rpath target)
    if(TARGET "${target}" AND NOT "${PHASESHIFT_ROCM_ROOT}" STREQUAL "")
        set_target_properties(
            "${target}"
            PROPERTIES
                BUILD_RPATH "${PHASESHIFT_ROCM_ROOT}/lib"
                INSTALL_RPATH "${PHASESHIFT_ROCM_ROOT}/lib"
        )
    endif()
endfunction()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# HIP optimization policy
#
# CMAKE_HIP_FLAGS_<CONFIG> is empty in the current ROCm/CMake environment,
# so explicitly guarantee optimization for HIP translation units.
# Without this, HIP kernels compile with the compiler default (-O0) even for
# Release builds, invalidating every performance measurement.
# ---------------------------------------------------------------------------

add_compile_options(
    "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:Release>>:-O3>"

    "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:RelWithDebInfo>>:-O2>"
    "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:RelWithDebInfo>>:-g>"

    "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:Debug>>:-O0>"
    "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:Debug>>:-g>"

    "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:MinSizeRel>>:-Os>"
)

add_compile_definitions(
    "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:Release>>:NDEBUG>"
    "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:RelWithDebInfo>>:NDEBUG>"
    "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:MinSizeRel>>:NDEBUG>"
)

find_package(hip REQUIRED CONFIG)

find_package(hipfile QUIET CONFIG)
if(hipfile_FOUND)
    message(STATUS "hipFile ${hipfile_VERSION} found")
else()
    message(STATUS "hipFile not found: SSD row gather falls back to POSIX")
endif()

# Applies the single selected GPU architecture to a HIP target. CMake
# already adds CMAKE_HIP_ARCHITECTURES to every HIP translation unit;
# this appends the explicit --offload-arch flag for determinism.
#
# Must use target_compile_options (append), never
# set_target_properties(... COMPILE_OPTIONS ...): replacing the target
# property makes the Ninja generator drop the directory-level
# add_compile_options flags (-stdlib=libc++), which would compile HIP
# translation units with the default libstdc++ and break the link
# against the libc++-compiled C++ objects.
function(phaseshift_set_hip_archs target)
    target_compile_options("${target}" PRIVATE "--offload-arch=${CMAKE_HIP_ARCHITECTURES}")
endfunction()

# ROCTx detection (optional). Used by benches only.
find_library(
    ROCTX_LIBRARY
    NAMES rocprofiler-sdk-roctx
    PATHS "${PHASESHIFT_ROCM_ROOT}/lib"
    NO_DEFAULT_PATH
)
find_path(
    ROCTX_INCLUDE_DIR
    NAMES rocprofiler-sdk-roctx/roctx.h
    PATHS "${PHASESHIFT_ROCM_ROOT}/include"
    NO_DEFAULT_PATH
)
if(ROCTX_LIBRARY AND ROCTX_INCLUDE_DIR)
    set(ROCTX_FOUND TRUE)
else()
    set(ROCTX_FOUND FALSE)
endif()
message(STATUS "ROCTx detection: ${ROCTX_FOUND}")

# Test suite toggle. Preserves existing CLI compatibility:
#   -DPHASESHIFT_BUILD_TESTS=ON
option(PHASESHIFT_BUILD_TESTS "Build PhaseShift tests" ON)

# Optional / external / heavy tests. OFF by default: optional test
# executables are not even created at configure time.
option(PHASESHIFT_BUILD_OPTIONAL_TESTS "Build optional / external / heavy tests" OFF)

option(PHASESHIFT_BUILD_BENCHMARKS "Build PhaseShift benchmark executables" ON)

# HIP graph execution mode for the Qwen3.5 runtime (capture/replay of the
# per-submission kernel sequence). Experimental.
option(PHASESHIFT_HIP_GRAPH "Enable HIP graph execution mode" OFF)

