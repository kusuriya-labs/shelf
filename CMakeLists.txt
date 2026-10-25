# shelf — inference runtime (skeleton: Phase 1)
cmake_minimum_required(VERSION 3.25)

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
