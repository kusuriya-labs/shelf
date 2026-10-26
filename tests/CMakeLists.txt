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

add_executable(test_ir_graph_validation unit/test_ir_graph_validation.cpp)
target_link_libraries(test_ir_graph_validation PRIVATE shelf::ir)
add_test(NAME unit.ir_graph_validation COMMAND test_ir_graph_validation)

add_executable(test_reference_executor unit/test_reference_executor.cpp)
target_link_libraries(test_reference_executor PRIVATE shelf::backend_cpu)
add_test(NAME unit.reference_executor COMMAND test_reference_executor)

add_executable(test_paged_kv_abi unit/test_paged_kv_abi.cpp)
target_link_libraries(test_paged_kv_abi PRIVATE shelf::backend_common shelf::runtime_state)
add_test(NAME unit.paged_kv_abi COMMAND test_paged_kv_abi)

if(SHELF_ENABLE_HIP)
    add_executable(test_hip_gemm_bf16 unit/test_hip_gemm_bf16.cpp)
    target_link_libraries(test_hip_gemm_bf16 PRIVATE shelf::backend_hip)
    add_test(NAME unit.hip_gemm_bf16 COMMAND test_hip_gemm_bf16)

    add_executable(test_hip_attention unit/test_hip_attention.cpp)
    target_link_libraries(test_hip_attention PRIVATE shelf::backend_hip)
    add_test(NAME unit.hip_attention COMMAND test_hip_attention)

    add_executable(test_hip_gdn unit/test_hip_gdn.cpp)
    target_link_libraries(test_hip_gdn PRIVATE shelf::backend_hip)
    add_test(NAME unit.hip_gdn COMMAND test_hip_gdn)

    add_executable(test_hip_elementwise_kernels unit/test_hip_elementwise_kernels.cpp)
    target_link_libraries(test_hip_elementwise_kernels PRIVATE shelf::backend_hip)
    add_test(NAME unit.hip_elementwise_kernels COMMAND test_hip_elementwise_kernels)

    add_executable(test_hip_step_executor unit/test_hip_step_executor.cpp)
    target_link_libraries(test_hip_step_executor PRIVATE shelf::backend_hip
        shelf::model_qwen35)
    add_test(NAME unit.hip_step_executor COMMAND test_hip_step_executor)
endif()

if(SHELF_ENABLE_SYCL)
    add_executable(test_sycl_device unit/test_sycl_device.cpp)
    target_link_libraries(test_sycl_device PRIVATE shelf::backend_sycl)
    add_sycl_to_target(TARGET test_sycl_device SOURCES unit/test_sycl_device.cpp)
    add_test(NAME unit.sycl_device COMMAND test_sycl_device)

    add_executable(test_sycl_kernels unit/test_sycl_kernels.cpp)
    target_link_libraries(test_sycl_kernels PRIVATE shelf::backend_sycl)
    add_sycl_to_target(TARGET test_sycl_kernels SOURCES unit/test_sycl_kernels.cpp)
    add_test(NAME unit.sycl_kernels COMMAND test_sycl_kernels)

    add_executable(test_sycl_step_executor unit/test_sycl_step_executor.cpp)
    target_link_libraries(test_sycl_step_executor PRIVATE shelf::backend_sycl
        shelf::model_qwen35)
    add_sycl_to_target(TARGET test_sycl_step_executor
        SOURCES unit/test_sycl_step_executor.cpp)
    add_test(NAME unit.sycl_step_executor COMMAND test_sycl_step_executor)
endif()

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
