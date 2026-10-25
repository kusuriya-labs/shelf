# shelf — inference runtime (skeleton: Phase 1)
cmake_minimum_required(VERSION 3.25)

# ROCm root must be discovered before project() pins the compilers
# (never hardcode a machine-specific SDK path).
include(cmake/RocmRoot.cmake)

project(shelf
    VERSION 0.1.0
    DESCRIPTION "Model-agnostic LLM inference runtime (skeleton)"
    LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(Options)
include(Dependencies)
include(Targets)

add_subdirectory(apps)

if(SHELF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(SHELF_ENABLE_HIP)
    add_subdirectory(tools)
endif()
