# shelf tests (Phase 1: unit + architecture). No external framework yet.

add_executable(test_foundation unit/test_foundation.cpp)
target_link_libraries(test_foundation PRIVATE shelf::foundation)
add_test(NAME unit.foundation COMMAND test_foundation)

add_executable(test_architecture_boundaries architecture/test_architecture_boundaries.cpp)
target_compile_definitions(test_architecture_boundaries PRIVATE
    SHELF_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")
target_compile_features(test_architecture_boundaries PRIVATE cxx_std_20)
add_test(NAME architecture.boundaries COMMAND test_architecture_boundaries)
