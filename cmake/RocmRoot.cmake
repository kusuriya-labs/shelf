# shelf ROCm root discovery + default compiler pinning.
#
# Included from the root CMakeLists.txt BEFORE project(), so the default
# C++/HIP compilers come from the ROCm toolchain clang++ (host and device
# TUs must share one ABI when kernel TUs exist).
#
# The SDK location is PERSONAL to the machine — never hardcode it.
# Discovery order:
#   -DSHELF_ROCM_ROOT=<path>
#   ROCM_PATH env var
#   `rocm-sdk path --root` (pip rocm-sdk tool)
#
# (same resolution ladder the PhaseShift build system uses)

if(SHELF_ENABLE_HIP AND NOT DEFINED SHELF_ROCM_ROOT)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(SHELF_ROCM_ROOT "$ENV{ROCM_PATH}" CACHE PATH "ROCm installation root")
    else()
        find_program(ROCM_SDK_EXECUTABLE rocm-sdk)
        if(ROCM_SDK_EXECUTABLE)
            execute_process(
                COMMAND "${ROCM_SDK_EXECUTABLE}" path --root
                OUTPUT_VARIABLE _shelf_rocm_sdk_root
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(NOT _shelf_rocm_sdk_root STREQUAL "")
                set(SHELF_ROCM_ROOT "${_shelf_rocm_sdk_root}"
                    CACHE PATH "ROCm installation root")
            endif()
        endif()
    endif()
endif()

if(SHELF_ENABLE_HIP)
    if(NOT DEFINED SHELF_ROCM_ROOT OR SHELF_ROCM_ROOT STREQUAL "")
        message(FATAL_ERROR
            "SHELF_ENABLE_HIP=ON but no ROCm root found. "
            "Pass -DSHELF_ROCM_ROOT=<path>, set ROCM_PATH, or install rocm-sdk "
            "so that `rocm-sdk path --root` works.")
    endif()
    if(NOT EXISTS "${SHELF_ROCM_ROOT}/lib/llvm/bin/clang++")
        message(FATAL_ERROR
            "ROCm toolchain clang++ not found under ${SHELF_ROCM_ROOT}/lib/llvm/bin")
    endif()

    # Single clang++ for CXX and HIP (one C++ ABI across host/device TUs).
    if(NOT DEFINED CACHE{CMAKE_CXX_COMPILER} AND NOT DEFINED ENV{CXX})
        set(CMAKE_CXX_COMPILER "${SHELF_ROCM_ROOT}/lib/llvm/bin/clang++")
    endif()
    if(NOT DEFINED CACHE{CMAKE_HIP_COMPILER})
        set(CMAKE_HIP_COMPILER "${SHELF_ROCM_ROOT}/lib/llvm/bin/clang++")
    endif()

    # find_package(hip) in cmake/Dependencies.cmake resolves from here.
    list(APPEND CMAKE_PREFIX_PATH "${SHELF_ROCM_ROOT}/lib/cmake")
endif()
