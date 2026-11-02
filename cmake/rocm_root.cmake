# Shelf ROCm root discovery + default compiler pinning.
#
# Included from the root CMakeLists.txt BEFORE project(), so that the
# default C++ compiler is the ROCm toolchain clang++ (GCC banned).
#
# Discovery order (CLI compatibility preserved):
#   -DSHELF_ROCM_ROOT=<path>
#   ROCM_PATH env var
#   rocm-sdk tool

if(NOT DEFINED SHELF_ROCM_ROOT)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(
            SHELF_ROCM_ROOT
            "$ENV{ROCM_PATH}"
            CACHE PATH
            "ROCm installation root"
        )
    else()
        find_program(ROCM_SDK_EXECUTABLE rocm-sdk)
        if(ROCM_SDK_EXECUTABLE)
            execute_process(
                COMMAND
                    "${ROCM_SDK_EXECUTABLE}"
                    path
                    --root
                OUTPUT_VARIABLE
                    ROCM_SDK_ROOT_OUTPUT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(NOT ROCM_SDK_ROOT_OUTPUT STREQUAL "")
                set(
                    SHELF_ROCM_ROOT
                    "${ROCM_SDK_ROOT_OUTPUT}"
                    CACHE PATH
                    "ROCm installation root"
                )
            endif()
        endif()
    endif()
endif()

set(
    CMAKE_HIP_COMPILER_ROCM_ROOT
    "${SHELF_ROCM_ROOT}"
    CACHE PATH
    "ROCm root for HIP compiler"
)

if(NOT DEFINED CACHE{CMAKE_CXX_COMPILER} AND NOT DEFINED ENV{CXX})
    if(SHELF_ROCM_ROOT)
        set(CMAKE_CXX_COMPILER "${SHELF_ROCM_ROOT}/lib/llvm/bin/clang++")
    endif()
endif()

if(SHELF_ROCM_ROOT AND NOT EXISTS "${SHELF_ROCM_ROOT}/lib/llvm/bin/clang++")
    message(
        FATAL_ERROR
        "ROCm toolchain clang++ was not found under ${SHELF_ROCM_ROOT}/lib/llvm/bin. "
        "Fix -DSHELF_ROCM_ROOT / ROCM_PATH, or pass -DCMAKE_CXX_COMPILER explicitly."
    )
endif()
