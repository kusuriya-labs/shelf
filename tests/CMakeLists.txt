cmake_minimum_required(VERSION 3.24)

# Thin wrapper: all test target definitions live in cmake/tests.cmake
# so that the build/test taxonomy is co-located with the other CMake modules.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/tests.cmake)
