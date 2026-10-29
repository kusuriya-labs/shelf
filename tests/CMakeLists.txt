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

add_executable(test_quant_fp8 unit/test_quant_fp8.cpp)
target_link_libraries(test_quant_fp8 PRIVATE shelf::quant)
add_test(NAME unit.quant_fp8 COMMAND test_quant_fp8)

# quantization_config の解析（apps/common の実装を使う）
add_executable(test_fp8_format unit/test_fp8_format.cpp)
target_link_libraries(test_fp8_format PRIVATE shelf_app_common)
add_test(NAME unit.fp8_format COMMAND test_fp8_format)

add_executable(test_ir_graph_validation unit/test_ir_graph_validation.cpp)
target_link_libraries(test_ir_graph_validation PRIVATE shelf::ir)
add_test(NAME unit.ir_graph_validation COMMAND test_ir_graph_validation)

add_executable(test_step_graph unit/test_step_graph.cpp)
target_link_libraries(test_step_graph PRIVATE shelf::model_qwen35)
add_test(NAME unit.step_graph COMMAND test_step_graph)

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

    add_executable(test_hip_gemm_fp8 unit/test_hip_gemm_fp8.cpp)
    target_link_libraries(test_hip_gemm_fp8 PRIVATE shelf::backend_hip shelf::quant)
    add_test(NAME unit.hip_gemm_fp8 COMMAND test_hip_gemm_fp8)

    # pipeline-parallel（レイヤ分割）が数値を変えないこと（全ステージ同一 GPU で検証）
    add_executable(test_hip_pipeline unit/test_hip_pipeline.cpp)
    target_link_libraries(test_hip_pipeline PRIVATE
        shelf::backend_hip
        shelf::model_qwen35
        shelf_app_common)
    add_test(NAME unit.hip_pipeline COMMAND test_hip_pipeline)

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

# shelf-bench の CLI 契約（phaseshift-bench と同じ規約）:
# --help は装置初期化・モデルロードの前に必ず exit 0、用法の誤りは非 0。
add_test(NAME bench.help COMMAND shelf-bench --help)
add_test(NAME bench.kernel.help COMMAND shelf-bench kernel --help)
add_test(NAME bench.unknown_command COMMAND shelf-bench bogus)
set_tests_properties(bench.unknown_command PROPERTIES WILL_FAIL TRUE)
add_test(NAME bench.kernel.unknown_name COMMAND shelf-bench kernel no-such-kernel)
set_tests_properties(bench.kernel.unknown_name PROPERTIES WILL_FAIL TRUE)

# 未知の --key は exit 2（hip では実装が無いとカーネル計測まで進み exit 0 になる）。
add_test(NAME bench.unknown_option
         COMMAND shelf-bench kernel rmsnorm --rows 8 --features 8
                 --warmup 0 --samples 1 --launches 1 --bogus 1)
set_tests_properties(bench.unknown_option PROPERTIES WILL_FAIL TRUE)
