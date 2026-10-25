# shelf tests (Phase 1: unit + architecture). No external framework yet.

add_executable(test_foundation unit/test_foundation.cpp)
target_link_libraries(test_foundation PRIVATE shelf::foundation)
add_test(NAME unit.foundation COMMAND test_foundation)

add_executable(test_artifact_safetensors unit/test_artifact_safetensors.cpp)
target_link_libraries(test_artifact_safetensors PRIVATE shelf::artifact)
add_test(NAME unit.artifact_safetensors COMMAND test_artifact_safetensors)

add_executable(test_weights_catalog unit/test_weights_catalog.cpp)
target_link_libraries(test_weights_catalog PRIVATE shelf::weights)
add_test(NAME unit.weights_catalog COMMAND test_weights_catalog)

add_executable(test_architecture_boundaries architecture/test_architecture_boundaries.cpp)
target_compile_definitions(test_architecture_boundaries PRIVATE
    SHELF_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")
target_compile_features(test_architecture_boundaries PRIVATE cxx_std_20)
add_test(NAME architecture.boundaries COMMAND test_architecture_boundaries)

if(SHELF_ENABLE_HIP)
    add_executable(test_hip_device unit/test_hip_device.cpp)
    target_link_libraries(test_hip_device PRIVATE shelf::backend_hip)
    add_test(NAME unit.hip_device COMMAND test_hip_device)
endif()
