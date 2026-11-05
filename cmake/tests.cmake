# Shelf test targets.
#
# Test policy:
#   - required: self-contained contract / correctness tests.
#     Repository + ROCm + 1 GPU only. No external files.
#   - optional: external-model / heavy tests. Built only when
#     SHELF_BUILD_OPTIONAL_TESTS=ON.
#
# File layout follows the same taxonomy, in this order:
#   1. test helper (shelf-gpu-test-runner, shared fixtures)
#   2. required — self-contained contract / correctness
#   3. benchmarks-only smoke (SHELF_BUILD_BENCHMARKS)
#   4. optional — current regression / infrastructure acceptance
#
#
# Included via tests/CMakeLists.txt (thin wrapper).

set(_SHELF_TESTS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(shelf-gpu-test-runner
    "${_SHELF_TESTS_ROOT}/support/gpu_test_runner/main.cpp"
)
target_compile_features(shelf-gpu-test-runner PRIVATE cxx_std_20)

# shelf_add_test(
#   NAME <test>
#   SOURCE <relpath from tests/>
#   [LABELS <label...>]
#   [LIBRARIES <libs...>]
#   [TIMEOUT <sec>]
#   [GPU_COUNT <n>]
#   [GPU_COST_GB <gb>]
# )
#
# Links exactly the listed libraries (never the shelf aggregate).
# GPU_COUNT > 0 wraps the test in shelf-gpu-test-runner:
# integrated GPUs are dropped from the candidate list, VRAM budget reservation
# (per-GPU budget derived from the device's own VRAM, forced by --budget-gb /
# env SHELF_TEST_GPU_BUDGET_GB), GPU assignment via HIP_VISIBLE_DEVICES,
# hard deadline (SIGKILL on timeout), exit code normalization. Exit 77 is
# reported as CTest SKIP, but required acceptance still counts it as a failure.
function(shelf_add_test)
    cmake_parse_arguments(
        SHELF_TEST
        ""
        "NAME;SOURCE;TIMEOUT;GPU_COUNT;GPU_COST_GB"
        "LIBRARIES;LABELS;DEFS;DEPENDS"
        ${ARGN}
    )
    add_executable(${SHELF_TEST_NAME} "${_SHELF_TESTS_ROOT}/${SHELF_TEST_SOURCE}")
    shelf_set_rocm_rpath(${SHELF_TEST_NAME})
    shelf_set_hip_archs(${SHELF_TEST_NAME})
    target_compile_features(${SHELF_TEST_NAME} PRIVATE cxx_std_20)
    target_include_directories(${SHELF_TEST_NAME} PRIVATE "${_SHELF_TESTS_ROOT}")
    target_link_libraries(${SHELF_TEST_NAME} PRIVATE ${SHELF_TEST_LIBRARIES})
    if(SHELF_TEST_DEFS)
        target_compile_definitions(${SHELF_TEST_NAME} PRIVATE ${SHELF_TEST_DEFS})
    endif()
    if(SHELF_TEST_DEPENDS)
        add_dependencies(${SHELF_TEST_NAME} ${SHELF_TEST_DEPENDS})
    endif()
    if(SHELF_TEST_GPU_COUNT GREATER 0)
        if(NOT SHELF_TEST_TIMEOUT)
            set(SHELF_TEST_TIMEOUT 120)
        endif()
        if(NOT SHELF_TEST_GPU_COST_GB)
            set(SHELF_TEST_GPU_COST_GB 1)
        endif()
        math(EXPR SHELF_TEST_BACKSTOP "${SHELF_TEST_TIMEOUT} + 960")
        add_test(NAME ${SHELF_TEST_NAME} COMMAND shelf-gpu-test-runner
            --gpu-count ${SHELF_TEST_GPU_COUNT}
            --cost-gb ${SHELF_TEST_GPU_COST_GB}
            --timeout ${SHELF_TEST_TIMEOUT}
            --state-dir "${CMAKE_BINARY_DIR}/gpu-test-state"
            -- $<TARGET_FILE:${SHELF_TEST_NAME}>)
        add_dependencies(${SHELF_TEST_NAME} shelf-gpu-test-runner)
        set_tests_properties(${SHELF_TEST_NAME} PROPERTIES TIMEOUT ${SHELF_TEST_BACKSTOP})
    else()
        add_test(NAME ${SHELF_TEST_NAME} COMMAND ${SHELF_TEST_NAME})
        if(SHELF_TEST_TIMEOUT)
            set_tests_properties(${SHELF_TEST_NAME} PROPERTIES TIMEOUT ${SHELF_TEST_TIMEOUT})
        endif()
    endif()
    set_tests_properties(${SHELF_TEST_NAME} PROPERTIES
        LABELS "${SHELF_TEST_LABELS}"
        SKIP_RETURN_CODE 77)
endfunction()

# ---------------------------------------------------------------------------
# required — self-contained contract / correctness tests
# ---------------------------------------------------------------------------

shelf_add_test(NAME test_safetensors_writer SOURCE unit/test_safetensors_writer.cpp LABELS "cpu;required" LIBRARIES shelf_io)
shelf_add_test(NAME test_expert_slice_fixture SOURCE unit/test_expert_slice_fixture.cpp LABELS "cpu;required" TIMEOUT 120 LIBRARIES shelf_weights shelf_io)
shelf_add_test(NAME test_expert_slice_load SOURCE unit/test_expert_slice_load.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_weights shelf_kernels_correctness shelf_qwen35_kernels_optimized)
shelf_add_test(NAME test_safetensors_partial_read SOURCE unit/test_safetensors_partial_read.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES shelf_io shelf_weights)
shelf_add_test(NAME test_quant_format_layout SOURCE unit/test_quant_format_layout.cpp LABELS "cpu;required" LIBRARIES shelf_quant_format)
shelf_add_test(NAME test_quant_spec SOURCE unit/test_quant_spec.cpp LABELS "cpu;required" LIBRARIES shelf_quant_format)
shelf_add_test(NAME test_manifest_v4 SOURCE unit/test_manifest_v4.cpp LABELS "cpu;required" LIBRARIES shelf_quant_format)
shelf_add_test(NAME test_int8_compute_contract SOURCE unit/test_int8_compute_contract.cpp LABELS "cpu;required" LIBRARIES shelf_quant_reference)
shelf_add_test(NAME test_physical_alignment_contract SOURCE unit/test_physical_alignment_contract.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES shelf_core shelf_inference_state)
shelf_add_test(NAME test_ocp_e4m3_codec SOURCE unit/test_ocp_e4m3_codec.cpp LABELS "cpu;required" LIBRARIES shelf_quant_reference)
shelf_add_test(NAME test_activation_quantize_sensitivity SOURCE unit/test_activation_quantize_sensitivity.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES shelf_quant_reference)
shelf_add_test(NAME test_fp8_payload SOURCE unit/test_fp8_payload.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_quant_reference shelf_quant_format)
shelf_add_test(NAME test_mxfp4_codec SOURCE unit/test_mxfp4_codec.cpp LABELS "cpu;required" TIMEOUT 120 LIBRARIES shelf_quant_reference)
shelf_add_test(NAME test_exl3_codec SOURCE unit/test_exl3_codec.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES shelf_quant_reference)
shelf_add_test(NAME test_exl3_linear SOURCE unit/test_exl3_linear.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_kernels_optimized shelf_quant_reference shelf_gpu)
shelf_add_test(NAME test_exl3_dense SOURCE unit/test_exl3_dense.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_kernels_optimized shelf_quant_reference shelf_gpu)
shelf_add_test(NAME test_exl3_grouped SOURCE unit/test_exl3_grouped.cpp LABELS "gpu1;optional;external_files" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_optimized shelf_quant_reference shelf_io)
target_include_directories(test_exl3_grouped SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
find_library(SHELF_TEST_HIPBLASLT_LIBRARY NAMES hipblaslt PATHS "${SHELF_ROCM_ROOT}/lib" NO_DEFAULT_PATH)
if(SHELF_TEST_HIPBLASLT_LIBRARY)
    target_link_libraries(test_exl3_dense PRIVATE "${SHELF_TEST_HIPBLASLT_LIBRARY}")
endif()
shelf_add_test(NAME test_mxfp4_canonical_payload SOURCE unit/test_mxfp4_canonical_payload.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES shelf_quant_reference shelf_quant_bundle shelf_weights)
shelf_add_test(NAME test_mxfp4_linear SOURCE unit/test_mxfp4_linear.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_kernels_optimized shelf_quant_reference shelf_runtime shelf_gpu)
shelf_add_test(NAME test_fp8_wmma_subnormal SOURCE unit/test_fp8_wmma_subnormal.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_gpu)
shelf_add_test(NAME test_activation_rowref_fold SOURCE unit/test_activation_rowref_fold.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_kernels_optimized shelf_quant_reference shelf_gpu)
shelf_add_test(NAME test_hf_manifest SOURCE unit/test_hf_manifest.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES shelf_weights shelf_quant_bundle shelf_io)
shelf_add_test(NAME test_tensor_partition SOURCE unit/test_tensor_partition.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_weights)
shelf_add_test(NAME test_canonical_partition SOURCE unit/test_canonical_partition.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES shelf_weights shelf_quant_reference)
shelf_add_test(NAME test_fp8_frag_layout SOURCE unit/test_fp8_frag_layout.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_quant_format)
shelf_add_test(NAME test_qwen35_tensor_parallel_plan SOURCE unit/test_qwen35_tensor_parallel_plan.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35)
shelf_add_test(NAME test_qwen35_tp_context SOURCE unit/test_qwen35_tp_context.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35)
shelf_add_test(NAME test_qwen35_pipeline_stage SOURCE unit/test_qwen35_pipeline_stage.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35)
shelf_add_test(NAME test_tp_reduction SOURCE unit/test_tp_reduction.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_runtime)
target_include_directories(test_tp_reduction PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_tp_barrier_group SOURCE unit/test_tp_barrier_group.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_runtime)
target_include_directories(test_tp_barrier_group PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_tp_transport SOURCE unit/test_tp_transport.cpp LABELS "gpu2;optional" TIMEOUT 120 GPU_COUNT 2 GPU_COST_GB 1 LIBRARIES shelf_runtime)
shelf_add_test(NAME test_qwen35_tp_execution SOURCE unit/test_qwen35_tp_execution.cpp LABELS "gpu2;optional" TIMEOUT 600 GPU_COUNT 2 GPU_COST_GB 2 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_tp_execution PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35moe_tp_layer_trace SOURCE unit/test_qwen35moe_tp_layer_trace.cpp LABELS "gpu2;optional;external_files" TIMEOUT 3600 GPU_COUNT 2 GPU_COST_GB 26 LIBRARIES shelf_models shelf_qwen35moe shelf_qwen35 shelf_qwen35_runtime shelf_qwen_hybrid shelf_inference_execution shelf_inference_state shelf_runtime shelf_kernels_optimized shelf_weights shelf_io shelf_quant_bundle shelf_gpu)
shelf_add_test(NAME test_qwen35moe_tp_execution SOURCE unit/test_qwen35moe_tp_execution.cpp LABELS "gpu2;optional" TIMEOUT 900 GPU_COUNT 2 GPU_COST_GB 3 LIBRARIES shelf_qwen35moe shelf_qwen35 shelf_qwen35_runtime shelf_qwen_hybrid shelf_inference_execution shelf_inference_state shelf_runtime shelf_kernels_optimized shelf_weights shelf_io shelf_quant_bundle shelf_gpu)
target_include_directories(test_qwen35moe_tp_execution PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/tests/unit")
shelf_add_test(NAME test_qwen35_tp_e2e SOURCE unit/test_qwen35_tp_e2e.cpp LABELS "gpu2;optional;external_files" TIMEOUT 2400 GPU_COUNT 2 GPU_COST_GB 20 LIBRARIES shelf_qwen35_runtime)

# Qwen3.8-Flash-Next (qwen4_exp) EXL3 load: TP=2 expert-axis split, body + PLE table.
shelf_add_test(NAME test_qwen4exp_load SOURCE unit/test_qwen4exp_load.cpp LABELS "gpu2;optional;external_files" TIMEOUT 3600 GPU_COUNT 2 GPU_COST_GB 30 LIBRARIES shelf_qwen4exp shelf_qwen_hybrid shelf_inference_execution shelf_gpu shelf_io shelf_quant_bundle)
target_include_directories(test_qwen4exp_load SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
shelf_add_test(NAME test_qwen35_tp_weight_load SOURCE unit/test_qwen35_tp_weight_load.cpp LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_qwen35)
target_include_directories(test_qwen35_tp_e2e PRIVATE "${CMAKE_SOURCE_DIR}/src")

# DFlash2 + TP: drafter は非分割、verify のみ TP2 で回す経路の E2E。
shelf_add_test(NAME test_dflash2_tp_e2e SOURCE unit/test_dflash2_tp_e2e.cpp LABELS "gpu2;optional;external_files" TIMEOUT 3600 GPU_COUNT 2 GPU_COST_GB 20 LIBRARIES shelf_weights shelf_qwen35 shelf_qwen35_runtime)
target_include_directories(test_dflash2_tp_e2e PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test_qwen35_tp_execution PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_weight_load SOURCE unit/test_weight_load.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_weights shelf_qwen35)
shelf_add_test(NAME test_qwen35_mtp_weight_load SOURCE unit/test_qwen35_mtp_weight_load.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_weights shelf_qwen35)
shelf_add_test(NAME test_dflash2_config SOURCE unit/test_dflash2_config.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35)
shelf_add_test(NAME test_thinking_budget SOURCE unit/test_thinking_budget.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
shelf_add_test(NAME test_dflash2_footprint SOURCE unit/test_dflash2_footprint.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
shelf_add_test(NAME test_executor_footprint SOURCE unit/test_executor_footprint.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
shelf_add_test(NAME test_quality_metrics SOURCE unit/test_quality_metrics.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_core)
target_include_directories(test_quality_metrics PRIVATE "${CMAKE_SOURCE_DIR}/src/apps/bench")
shelf_add_test(NAME test_qwen35_config_tie SOURCE unit/test_qwen35_config_tie.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35)
shelf_add_test(NAME test_moe_config SOURCE unit/test_moe_config.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES shelf_qwen35moe shelf_qwen35)
shelf_add_test(NAME test_model_identity SOURCE unit/test_model_identity.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35moe shelf_qwen35)
shelf_add_test(NAME test_moe_weight_plan SOURCE unit/test_moe_weight_plan.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES shelf_qwen35moe)
shelf_add_test(NAME test_qwen35moe_tensor_parallel_plan SOURCE unit/test_qwen35moe_tensor_parallel_plan.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES shelf_qwen35moe shelf_qwen35)
shelf_add_test(NAME test_moe_expert_slice_fixture SOURCE unit/test_moe_expert_slice_fixture.cpp LABELS "cpu;required" TIMEOUT 120 LIBRARIES shelf_qwen35moe)
shelf_add_test(NAME test_weight_slot_validation SOURCE unit/test_weight_slot_validation.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_inference_execution shelf_qwen35moe shelf_qwen_hybrid shelf_core)
shelf_add_test(NAME test_decode_perf_stats SOURCE unit/test_decode_perf_stats.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_inference_execution shelf_runtime)
shelf_add_test(NAME test_moe_expert_load SOURCE unit/test_moe_expert_load.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35moe shelf_gpu)
shelf_add_test(NAME test_moe_router SOURCE unit/test_moe_router.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_moe_grouped_linear SOURCE unit/test_moe_grouped_linear.hip LABELS "gpu1;required" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_optimized shelf_qwen35moe shelf_quant_bundle shelf_gpu)
shelf_add_test(NAME test_moe_combine SOURCE unit/test_moe_combine.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_qwen35moe_model_load SOURCE unit/test_qwen35moe_model_load.hip LABELS "gpu1;required" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_models shelf_qwen35moe shelf_qwen35 shelf_quant_bundle shelf_gpu)
shelf_add_test(NAME test_qwen35moe_lowering SOURCE unit/test_qwen35moe_lowering.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_qwen35moe shelf_qwen35 shelf_qwen35_runtime shelf_qwen_hybrid shelf_runtime shelf_weights shelf_io shelf_quant_bundle shelf_gpu)
shelf_add_test(NAME test_qwen35moe_executor SOURCE unit/test_qwen35moe_executor.hip LABELS "gpu1;required" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_models shelf_qwen35moe shelf_qwen35 shelf_qwen35_runtime shelf_qwen_hybrid shelf_runtime shelf_inference_state shelf_inference_execution shelf_inference_scheduling shelf_kernels_correctness shelf_kernels_optimized shelf_qwen35_kernels_optimized shelf_weights shelf_io shelf_quant_bundle shelf_gpu)
shelf_add_test(NAME test_qwen35moe_mtp_executor SOURCE unit/test_qwen35moe_mtp_executor.hip LABELS "gpu1;required" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_models shelf_qwen35moe shelf_qwen35 shelf_qwen35_runtime shelf_qwen_hybrid shelf_runtime shelf_inference_state shelf_inference_execution shelf_inference_scheduling shelf_kernels_correctness shelf_kernels_optimized shelf_qwen35_kernels_optimized shelf_weights shelf_io shelf_quant_bundle shelf_gpu)
shelf_add_test(NAME test_qwen35moe_rows_parity SOURCE unit/test_qwen35moe_rows_parity.hip LABELS "gpu1;required" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_models shelf_qwen35moe shelf_qwen35 shelf_qwen35_runtime shelf_qwen_hybrid shelf_runtime shelf_inference_state shelf_inference_execution shelf_inference_scheduling shelf_kernels_correctness shelf_kernels_optimized shelf_qwen35_kernels_optimized shelf_weights shelf_io shelf_quant_bundle shelf_gpu)
shelf_add_test(NAME test_moe_layer_e2e SOURCE unit/test_moe_layer_e2e.hip LABELS "gpu1;required" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 2 DEFS SHELF_MOE_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures/qwen35moe" LIBRARIES shelf_kernels_optimized shelf_qwen35moe shelf_quant_bundle shelf_io shelf_gpu)
shelf_add_test(NAME test_dflash2_weight_contract SOURCE unit/test_dflash2_weight_contract.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_weights shelf_qwen35)
shelf_add_test(NAME test_dflash2_feature_concat SOURCE unit/test_dflash2_feature_concat.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35_kernels_optimized)
shelf_add_test(NAME test_dflash2_grouped_dynamic_conv SOURCE unit/test_dflash2_grouped_dynamic_conv.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35_kernels_optimized)
shelf_add_test(NAME test_dflash2_bf16_geometry SOURCE unit/test_dflash2_bf16_geometry.hip LABELS "gpu1;required" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 3 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_dflash2_bf16_geometry PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_dflash2_rope SOURCE unit/test_dflash2_rope.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35_kernels_optimized)
shelf_add_test(NAME test_dflash2_attention SOURCE unit/test_dflash2_attention.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35_kernels_optimized)
shelf_add_test(NAME test_dflash2_topk SOURCE unit/test_dflash2_topk.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35_kernels_optimized)
shelf_add_test(NAME test_gdn_spec_history SOURCE unit/test_gdn_spec_history.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_qwen35_kernels_optimized shelf_qwen35_runtime)
shelf_add_test(NAME test_prefix_cache SOURCE unit/test_prefix_cache.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35_runtime)
shelf_add_test(NAME test_prefix_cache_batcher SOURCE unit/test_prefix_cache_batcher.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_qwen35_runtime shelf_qwen35)
shelf_add_test(NAME test_executor_move_assignment SOURCE unit/test_executor_move_assignment.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35_runtime shelf_gpu)
shelf_add_test(NAME test_dflash2_candidate_selector SOURCE unit/test_dflash2_candidate_selector.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35_kernels_optimized)
shelf_add_test(NAME test_dflash2_kv_ring SOURCE unit/test_dflash2_kv_ring.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35_kernels_optimized)
shelf_add_test(NAME test_dflash2_attention_ring SOURCE unit/test_dflash2_attention_ring.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35_kernels_optimized)
shelf_add_test(NAME test_dflash2_radix_topn SOURCE unit/test_dflash2_radix_topn.hip LABELS "gpu1;required" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 3 LIBRARIES shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_dflash2_radix_topn_perf SOURCE unit/test_dflash2_radix_topn_perf.hip LABELS "gpu1;perf" TIMEOUT 1800 GPU_COUNT 1 GPU_COST_GB 6 LIBRARIES shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_qwen35_mtp_lowering SOURCE unit/test_qwen35_mtp_lowering.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35 shelf_qwen35_runtime)
shelf_add_test(NAME test_qwen35_lowering_contract SOURCE unit/test_qwen35_lowering_contract.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35)
shelf_add_test(NAME test_qwen35_pipeline_lowering SOURCE unit/test_qwen35_pipeline_lowering.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35)
shelf_add_test(NAME test_qwen35_spec_verify SOURCE unit/test_qwen35_spec_verify.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)

# 合成 Qwen3.5 checkpoint（output_gate_type 非宣言）の層 0 = GDN を f64 参照で固定する。
add_test(NAME test_qwen35_gdn_gate_reference
         COMMAND $<TARGET_FILE:shelf-gpu-test-runner>
             --gpu-count 1 --cost-gb 2 --timeout 300
             --state-dir "${CMAKE_BINARY_DIR}/gpu-test-state"
             -- ${CMAKE_COMMAND} -E env
                "SHELF_BUILD_DIR=${CMAKE_BINARY_DIR}"
                python3 "${CMAKE_SOURCE_DIR}/tests/correctness/test_qwen35_gdn_gate_reference.py")
set_tests_properties(test_qwen35_gdn_gate_reference PROPERTIES
                     LABELS "gpu1;required"
                     TIMEOUT 420)
shelf_add_test(NAME test_paged_types SOURCE unit/test_paged_types.cpp LABELS "cpu;required" LIBRARIES shelf_inference_state)
shelf_add_test(NAME test_tensor SOURCE unit/test_tensor.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 LIBRARIES shelf_gpu)
shelf_add_test(NAME test_correctness_primitives SOURCE kernels/common/test_correctness_primitives.hip LABELS "gpu1;required" TIMEOUT 30 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_kernels_correctness)
shelf_add_test(NAME test_architecture_boundaries SOURCE unit/test_architecture_boundaries.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES shelf_core DEFS SHELF_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")
shelf_add_test(NAME test_program_workspace_lifetime SOURCE unit/test_program_workspace_lifetime.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_runtime)
shelf_add_test(NAME test_grouped_row_shape SOURCE unit/test_grouped_row_shape.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_runtime)
shelf_add_test(NAME test_program_moe_plan SOURCE unit/test_program_moe_plan.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_runtime)
shelf_add_test(NAME test_program_hyper_connection SOURCE unit/test_program_hyper_connection.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_runtime)
shelf_add_test(NAME test_qwen4exp_hyper_site SOURCE unit/test_qwen4exp_hyper_site.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen4exp shelf_runtime)
shelf_add_test(NAME test_qwen4exp_gdn_gate SOURCE unit/test_qwen4exp_gdn_gate.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen4exp shelf_runtime)
shelf_add_test(NAME test_ple_gate SOURCE unit/test_ple_gate.cpp LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_ple_conv SOURCE unit/test_ple_conv.cpp LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_moe_program_execute SOURCE unit/test_moe_program_execute.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_inference_execution shelf_kernels_optimized shelf_runtime shelf_gpu shelf_core)
shelf_add_test(NAME test_program_block_scaled_lowering SOURCE unit/test_program_block_scaled_lowering.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_runtime)
shelf_add_test(NAME test_linear_quant_contract SOURCE unit/test_linear_quant_contract.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_runtime)
shelf_add_test(NAME test_activation_workspace_layout SOURCE unit/test_activation_workspace_layout.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_runtime)
shelf_add_test(NAME test_program_quant_contract SOURCE unit/test_program_quant_contract.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_runtime)
shelf_add_test(NAME test_dispatch_staging_size SOURCE unit/test_dispatch_staging_size.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_runtime)
shelf_add_test(NAME test_gpu_arena_vmm SOURCE unit/test_gpu_arena_vmm.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_gpu)

shelf_add_test(NAME test_gemm_bf16_correctness SOURCE kernels/linear/test_gemm_bf16_correctness.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_kernels_correctness)
shelf_add_test(NAME test_gemm_bf16_exact_rows SOURCE kernels/linear/test_gemm_bf16_exact_rows.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 4 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_activation_quantize_fp8_finegrained SOURCE kernels/optimized/test_activation_quantize_fp8_finegrained.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_activation_quantize_tp_split SOURCE kernels/optimized/test_activation_quantize_tp_split.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_gemm_fp8_optimized SOURCE kernels/linear/test_gemm_fp8_optimized.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_quant_reference shelf_gpu)
shelf_add_test(NAME test_gemm_bf16_wmma SOURCE kernels/optimized/test_gemm_bf16_wmma.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_gemm_bf16_verify_exact SOURCE kernels/optimized/test_gemm_bf16_verify_exact.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_argmax_f32_top2 SOURCE kernels/optimized/test_argmax_f32_top2.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_rmsnorm SOURCE kernels/optimized/test_rmsnorm.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_l2_normalize SOURCE kernels/optimized/test_l2_normalize.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_output_gather SOURCE kernels/optimized/test_output_gather.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_kv_append SOURCE kernels/optimized/test_kv_append.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_paged_attention SOURCE kernels/optimized/test_paged_attention.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_rope SOURCE kernels/optimized/test_rope.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_rope_yarn SOURCE kernels/optimized/test_rope_yarn.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_gdn_recurrence SOURCE kernels/optimized/test_gdn_recurrence.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_gdn_recurrence_exact_history SOURCE kernels/optimized/test_gdn_recurrence_exact_history.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 4 LIBRARIES shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_gdn_compact_log_commit SOURCE kernels/optimized/test_gdn_compact_log_commit.hip LABELS "gpu1;optional" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 4 LIBRARIES shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_gdn_compact_commit_poc SOURCE unit/test_gdn_compact_commit_poc.hip LABELS "gpu1;optional" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_gpu)
shelf_add_test(NAME test_gdn_conv1d SOURCE kernels/optimized/test_gdn_conv1d.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_elementwise SOURCE kernels/optimized/test_elementwise.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_hyper_connection SOURCE unit/test_hyper_connection.cpp LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_embedding SOURCE kernels/optimized/test_embedding.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_argmax SOURCE kernels/optimized/test_argmax.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_qwen35_kernel_mode SOURCE unit/test_qwen35_kernel_mode.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_kernel_mode PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35_linear_selector SOURCE unit/test_qwen35_linear_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_linear_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35_rmsnorm_selector SOURCE unit/test_qwen35_rmsnorm_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_rmsnorm_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35_l2_normalize_selector SOURCE unit/test_qwen35_l2_normalize_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_l2_normalize_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35_kv_append_selector SOURCE unit/test_qwen35_kv_append_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_kv_append_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35_rope_selector SOURCE unit/test_qwen35_rope_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_rope_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35_rope_profile SOURCE unit/test_qwen35_rope_profile.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35)
shelf_add_test(NAME test_qwen35_paged_attention_selector SOURCE unit/test_qwen35_paged_attention_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
shelf_add_test(NAME test_paged_attention_plan_splits SOURCE unit/test_paged_attention_plan_splits.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_inference_execution shelf_runtime shelf_core)
target_include_directories(test_qwen35_paged_attention_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_paged_attention_split_reduce_exact SOURCE unit/test_paged_attention_split_reduce_exact.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35_kernels_optimized)
shelf_add_test(NAME test_qwen35_gdn_recurrence_selector SOURCE unit/test_qwen35_gdn_recurrence_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_gdn_recurrence_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35_gdn_conv_selector SOURCE unit/test_qwen35_gdn_conv_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_gdn_conv_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35_elementwise_selector SOURCE unit/test_qwen35_elementwise_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_elementwise_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35_embedding_selector SOURCE unit/test_qwen35_embedding_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_embedding_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35_sampling_selector SOURCE unit/test_qwen35_sampling_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_sampling_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35_sampling_params SOURCE unit/test_qwen35_sampling_params.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES shelf_qwen35_runtime)
target_include_directories(test_qwen35_sampling_params PRIVATE "${CMAKE_SOURCE_DIR}/src")
shelf_add_test(NAME test_qwen35_sampling_rng SOURCE unit/test_qwen35_sampling_rng.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_stochastic_sampling SOURCE kernels/optimized/test_stochastic_sampling.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)
shelf_add_test(NAME test_stochastic_topk SOURCE kernels/optimized/test_stochastic_topk.hip LABELS "gpu1;required" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 3 LIBRARIES shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_gpu)

# Default required build unit. Required acceptance builds only this target.
add_custom_target(
    shelf-required-tests
    DEPENDS
        test_safetensors_writer
        test_expert_slice_fixture
        test_expert_slice_load
        test_safetensors_partial_read
        test_quant_format_layout
        test_quant_spec
        test_manifest_v4
        test_int8_compute_contract
        test_physical_alignment_contract
        test_ocp_e4m3_codec
        test_activation_quantize_sensitivity
        test_fp8_payload
        test_mxfp4_codec
        test_exl3_codec
        test_exl3_linear
        test_exl3_dense
        test_mxfp4_canonical_payload
        test_mxfp4_linear
        test_fp8_wmma_subnormal
        test_activation_rowref_fold
        test_hf_manifest
        test_tensor_partition
        test_canonical_partition
        test_qwen35_tensor_parallel_plan
        test_qwen35_tp_context
        test_tp_reduction
        test_weight_load
        test_qwen35_tp_weight_load
        test_qwen35_mtp_weight_load
        test_dflash2_config
        test_thinking_budget
        test_dflash2_footprint
        test_executor_footprint
        test_quality_metrics
        test_qwen35_config_tie
        test_moe_config
        test_model_identity
        test_moe_weight_plan
        test_qwen35moe_tensor_parallel_plan
        test_moe_expert_slice_fixture
        test_weight_slot_validation
        test_decode_perf_stats
        test_moe_expert_load
        test_moe_router
        test_moe_grouped_linear
        test_moe_combine
        test_moe_layer_e2e
        test_qwen35moe_model_load
        test_qwen35moe_lowering
        test_qwen35moe_executor
        test_qwen35moe_mtp_executor
        test_qwen35moe_rows_parity
        test_dflash2_weight_contract
        test_dflash2_feature_concat
        test_dflash2_grouped_dynamic_conv
        test_dflash2_bf16_geometry
        test_dflash2_rope
        test_dflash2_attention
        test_dflash2_topk
        test_gdn_spec_history
        test_prefix_cache
        test_executor_move_assignment
        test_dflash2_candidate_selector
        test_dflash2_kv_ring
        test_dflash2_attention_ring
        test_dflash2_radix_topn
        test_paged_types
        test_paged_attention_split_reduce_exact
        test_tensor
        test_correctness_primitives
        test_gemm_bf16_correctness
        test_gemm_bf16_exact_rows
        test_activation_quantize_fp8_finegrained
        test_activation_quantize_tp_split
        test_gemm_fp8_optimized
        test_gemm_bf16_wmma
        test_rmsnorm
        test_l2_normalize
        test_output_gather
        test_kv_append
        test_paged_attention
        test_rope
        test_rope_yarn
        test_gdn_recurrence
        test_gdn_recurrence_exact_history
        test_gdn_conv1d
        test_elementwise
        test_embedding
        test_argmax
        test_qwen35_kernel_mode
        test_qwen35_linear_selector
        test_qwen35_rmsnorm_selector
        test_qwen35_l2_normalize_selector
        test_qwen35_kv_append_selector
        test_qwen35_rope_selector
        test_qwen35_rope_profile
        test_qwen35_paged_attention_selector
        test_paged_attention_plan_splits
        test_qwen35_gdn_recurrence_selector
        test_qwen35_gdn_conv_selector
        test_qwen35_elementwise_selector
        test_qwen35_embedding_selector
        test_qwen35_sampling_selector
        test_qwen35_sampling_params
        test_qwen35_sampling_rng
        test_stochastic_sampling
        test_stochastic_topk
        test_prefix_cache_batcher
        test_qwen35_mtp_lowering
        test_qwen35_lowering_contract
        test_qwen35_pipeline_stage
        test_qwen35_pipeline_lowering
        test_qwen35_spec_verify
        test_architecture_boundaries
        test_gemm_bf16_verify_exact
        test_argmax_f32_top2
        test_program_workspace_lifetime
        test_grouped_row_shape
        test_program_moe_plan
        test_moe_program_execute
        test_program_block_scaled_lowering
        test_linear_quant_contract
        test_activation_workspace_layout
        test_program_quant_contract
        test_dispatch_staging_size
        test_gpu_arena_vmm
)

# ---------------------------------------------------------------------------
# shelf-compute decode backend contract (CPU-only: no GPU, no model).
# --decode-backend must be decided before any model load: `hip` alone is
# accepted and every other value is rejected with an explicit error.
# ---------------------------------------------------------------------------
add_test(NAME test_compute_decode_backend_hip
         COMMAND $<TARGET_FILE:shelf-compute> --decode-backend hip)
set_tests_properties(test_compute_decode_backend_hip PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "model-dir required"
                     FAIL_REGULAR_EXPRESSION "bad --decode-backend")

add_test(NAME test_compute_decode_backend_reject_non_hip
         COMMAND $<TARGET_FILE:shelf-compute> --decode-backend gpu-mcu)
set_tests_properties(test_compute_decode_backend_reject_non_hip PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "bad --decode-backend"
                     FAIL_REGULAR_EXPRESSION "model-dir required")

# ---------------------------------------------------------------------------
# --arena-gib accepts `auto` and rejects 0. Both decisions happen before any
# GPU query / model load.
# ---------------------------------------------------------------------------
add_test(NAME test_compute_arena_gib_auto
         COMMAND $<TARGET_FILE:shelf-compute> --arena-gib auto)
set_tests_properties(test_compute_arena_gib_auto PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "model-dir required"
                     FAIL_REGULAR_EXPRESSION "bad --arena-gib")

add_test(NAME test_compute_arena_gib_reject_zero
         COMMAND $<TARGET_FILE:shelf-compute> --arena-gib 0)
set_tests_properties(test_compute_arena_gib_reject_zero PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "bad --arena-gib")

# ---------------------------------------------------------------------------
# --rope-factor accepts auto | 1 | 2 | 4 and rejects anything else before any
# GPU query / model load.
# ---------------------------------------------------------------------------
add_test(NAME test_compute_rope_factor_accepts_modes
         COMMAND $<TARGET_FILE:shelf-compute> --rope-factor 2)
set_tests_properties(test_compute_rope_factor_accepts_modes PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "model-dir required"
                     FAIL_REGULAR_EXPRESSION "bad --rope-factor")

add_test(NAME test_compute_rope_factor_reject_three
         COMMAND $<TARGET_FILE:shelf-compute> --rope-factor 3)
set_tests_properties(test_compute_rope_factor_reject_three PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "bad --rope-factor")

add_test(NAME test_compute_rope_factor_reject_yarn_word
         COMMAND $<TARGET_FILE:shelf-compute> --rope-factor yarn)
set_tests_properties(test_compute_rope_factor_reject_yarn_word PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "bad --rope-factor")

# ---------------------------------------------------------------------------
# --rope-factor caps --max-seq-len at min(original_len * factor, 1000000).
# Rejection happens before any GPU query / model load.
# ---------------------------------------------------------------------------
set(_shelf_rope_fixture "${CMAKE_SOURCE_DIR}/tests/fixtures/rope_cap")
# ---------------------------------------------------------------------------
# MTP drafter option contract. Every rejection happens before any model load.
# ---------------------------------------------------------------------------
add_test(NAME test_compute_mtp_rejects_dflash2
         COMMAND $<TARGET_FILE:shelf-compute>
             --model-dir placeholder --mtp --dflash2-model-dir placeholder)
set_tests_properties(test_compute_mtp_rejects_dflash2 PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "mutually exclusive")

add_test(NAME test_compute_rejects_tp_abbreviation
         COMMAND $<TARGET_FILE:shelf-compute>
             --model-dir placeholder --tp 2 --device 0,1)
set_tests_properties(test_compute_rejects_tp_abbreviation PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "unknown argument: --tp")

add_test(NAME test_compute_mtp_rejects_tp2
         COMMAND $<TARGET_FILE:shelf-compute>
             --model-dir placeholder --mtp --tensor-parallel 2 --device 0,1)
set_tests_properties(test_compute_mtp_rejects_tp2 PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "requires --tensor-parallel 1")

add_test(NAME test_compute_mtp_drafts_reject_over_ceiling
         COMMAND $<TARGET_FILE:shelf-compute>
             --model-dir placeholder --mtp --mtp-drafts 32)
set_tests_properties(test_compute_mtp_drafts_reject_over_ceiling PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "bad --mtp-drafts")

add_test(NAME test_compute_rope_cap_native_over
         COMMAND $<TARGET_FILE:shelf-compute>
             --model-dir "${_shelf_rope_fixture}" --rope-factor 1 --max-seq-len 262145)
set_tests_properties(test_compute_rope_cap_native_over PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "exceeds the rope factor limit 262144")

add_test(NAME test_compute_rope_cap_2x_over
         COMMAND $<TARGET_FILE:shelf-compute>
             --model-dir "${_shelf_rope_fixture}" --rope-factor 2 --max-seq-len 524289)
set_tests_properties(test_compute_rope_cap_2x_over PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "exceeds the rope factor limit 524288")

add_test(NAME test_compute_rope_cap_4x_over
         COMMAND $<TARGET_FILE:shelf-compute>
             --model-dir "${_shelf_rope_fixture}" --rope-factor 4 --max-seq-len 1000001)
set_tests_properties(test_compute_rope_cap_4x_over PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "exceeds the rope factor limit 1000000")


add_test(NAME test_quality_gate_selftest
         COMMAND python3 "${CMAKE_SOURCE_DIR}/tools/quality_gate.py" selftest)
set_tests_properties(test_quality_gate_selftest PROPERTIES LABELS "cpu;required")

# ---------------------------------------------------------------------------
# shelf-bench CLI help smoke (CPU-only: no GPU, no model).
# --help must exit 0 before any hipSetDevice / model load / file load.
# ---------------------------------------------------------------------------
if(SHELF_BUILD_BENCHMARKS)
    add_test(NAME test_bench_help COMMAND $<TARGET_FILE:shelf-bench> --help)
    set_tests_properties(test_bench_help PROPERTIES LABELS "cpu;required")

    foreach(_cmd
            pp
            tg
            quality
            batch
            activation-quantize
            gemm
            rmsnorm
            l2-normalize
            kv-append
            paged-attention
            rope
            gdn-recurrence
            gdn-conv1d
            elementwise
            embedding
            sampling
            tp-reduce
            gpu-memory)
        string(REPLACE "-" "_" _bench_name ${_cmd})
        add_test(NAME test_bench_help_${_bench_name}
                 COMMAND $<TARGET_FILE:shelf-bench> ${_cmd} --help)
        set_tests_properties(test_bench_help_${_bench_name} PROPERTIES LABELS "cpu;required")
    endforeach()

    foreach(_bad wmma direct all)
        add_test(NAME test_bench_gemm_reject_${_bad}
                 COMMAND $<TARGET_FILE:shelf-bench> gemm --variant ${_bad})
        set_tests_properties(test_bench_gemm_reject_${_bad} PROPERTIES
                             LABELS "cpu;required" WILL_FAIL TRUE)
    endforeach()

    add_test(NAME test_cli_docs_sync
             COMMAND python3 "${CMAKE_SOURCE_DIR}/tools/gen_cli_docs.py"
                 --check --sources cpp --build-dir "${CMAKE_BINARY_DIR}")
    set_tests_properties(test_cli_docs_sync PROPERTIES LABELS "cpu;required")

    # FP8 kernel ISA 資源契約: VGPR / SGPR spill と local memory 使用を 0 に保つ。
    add_test(NAME test_fp8_kernel_spill_contract
             COMMAND python3 "${CMAKE_SOURCE_DIR}/tools/check_kernel_spills.py"
                 --build-dir "${CMAKE_BINARY_DIR}")
    set_tests_properties(test_fp8_kernel_spill_contract
                         PROPERTIES LABELS "cpu;required" TIMEOUT 900)

    add_test(NAME test_cli_docs_sync_python
             COMMAND python3 "${CMAKE_SOURCE_DIR}/tools/gen_cli_docs.py"
                 --check --sources all --build-dir "${CMAKE_BINARY_DIR}")
    set_tests_properties(test_cli_docs_sync_python PROPERTIES LABELS "cpu;optional")
endif()

if(SHELF_BUILD_OPTIONAL_TESTS)
    # -----------------------------------------------------------------------
    # optional: current regression / infrastructure acceptance（external model）
    # -----------------------------------------------------------------------

    # MTP weight contract against a real model directory.
    # Model dir via SHELF_MODEL_DIR_MTP (preferred) or SHELF_MODEL_DIR.
    # Skips (exit 77) when none is set.
    shelf_add_test(NAME test_qwen35_mtp_weight_real SOURCE unit/test_qwen35_mtp_weight_real.hip LABELS "gpu1;optional;external_files" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES shelf_weights shelf_qwen35)

    add_test(NAME test_mtp_parity
             COMMAND $<TARGET_FILE:shelf-gpu-test-runner>
                 --gpu-count 1 --cost-gb 30 --timeout 1800
                 --state-dir "${CMAKE_BINARY_DIR}/gpu-test-state"
                 -- ${CMAKE_COMMAND} -E env
                    "SHELF_BUILD_DIR=${CMAKE_BINARY_DIR}"
                    python3 "${CMAKE_SOURCE_DIR}/tests/correctness/test_mtp_parity.py")
    set_tests_properties(test_mtp_parity PROPERTIES
                         LABELS "gpu1;optional;external_files"
                         TIMEOUT 1860
                         SKIP_RETURN_CODE 77)

    # DFlash2 drafter weight contract against a real model directory.
    # Model dir via SHELF_MODEL_DIR_DFLASH2. Skips (exit 77) when unset.
    shelf_add_test(NAME test_dflash2_weight_real SOURCE unit/test_dflash2_weight_real.hip LABELS "gpu1;optional;external_files" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 8 LIBRARIES shelf_weights shelf_qwen35)

    # FP8 KV quality regression: BF16 KV vs FP8 KV on a fixed corpus
    # (perplexity / KL / top-1 / generation agreement / fixed end-task).
    # Model dir via SHELF_MODEL_DIR. Skips (exit 77) when unset.
    add_test(NAME test_quality_fp8_kv
             COMMAND $<TARGET_FILE:shelf-gpu-test-runner>
                 --gpu-count 1 --cost-gb 30 --timeout 3600
                 --state-dir "${CMAKE_BINARY_DIR}/gpu-test-state"
                 -- python3 "${CMAKE_SOURCE_DIR}/tests/quality/run_quality_gate.py")
    set_tests_properties(test_quality_fp8_kv PROPERTIES
                         LABELS "gpu1;quality;optional;external_files"
                         TIMEOUT 4560
                         SKIP_RETURN_CODE 77)

    # CPU f64 fixture と correctness path の layer hidden / greedy token を比較する。
    # Model dir via SHELF_MODEL_DIR. Skips (exit 77) when unset.
    shelf_add_test(NAME test_qwen35moe_model_load_real SOURCE unit/test_qwen35moe_model_load_real.hip LABELS "gpu1;optional;external_files" TIMEOUT 2400 GPU_COUNT 1 GPU_COST_GB 24 LIBRARIES shelf_models shelf_qwen35moe shelf_qwen35 shelf_quant_bundle shelf_gpu)

    add_test(NAME test_qwen35_correctness_fixture
             COMMAND $<TARGET_FILE:shelf-gpu-test-runner>
                 --gpu-count 1 --cost-gb 30 --timeout 1800
                 --state-dir "${CMAKE_BINARY_DIR}/gpu-test-state"
                 -- ${CMAKE_COMMAND} -E env
                    "SHELF_BUILD_DIR=${CMAKE_BINARY_DIR}"
                    python3 "${CMAKE_SOURCE_DIR}/tests/correctness/test_qwen35_correctness_fixture.py")
    set_tests_properties(test_qwen35_correctness_fixture PROPERTIES
                         LABELS "gpu1;optional;external_files"
                         TIMEOUT 2760
                         SKIP_RETURN_CODE 77)

    # DFlash2 batched target verify: 2 requests in one verify batch must match the
    # per-request single path exactly.
    shelf_add_test(NAME test_dflash2_batch_e2e SOURCE unit/test_dflash2_batch_e2e.hip LABELS "gpu1;optional;external_files" TIMEOUT 2400 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES shelf_weights shelf_qwen35 shelf_qwen35_runtime)

    # DFlash2 batched target verify throughput scaling (perf, not correctness).
    shelf_add_test(NAME test_dflash2_batch_perf SOURCE unit/test_dflash2_batch_perf.hip LABELS "gpu1;perf;optional;external_files" TIMEOUT 2400 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES shelf_weights shelf_qwen35 shelf_qwen35_runtime)

    # DFlash2 mixed batch: decode region + verify region in one ScheduledBatch must
    # match the per-region single-class batches exactly.
    shelf_add_test(NAME test_dflash2_mixed_batch SOURCE unit/test_dflash2_mixed_batch.hip LABELS "gpu1;optional;external_files" TIMEOUT 2400 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES shelf_weights shelf_qwen35 shelf_qwen35_runtime)

    # DFlash2 target hidden taps: external tap == internal layer output.
    # Target model dir via SHELF_MODEL_DIR_DFLASH2_TARGET (skips when unset);
    # tap layer ids come from SHELF_MODEL_DIR_DFLASH2 when available.
    shelf_add_test(NAME test_dflash2_target_taps SOURCE unit/test_dflash2_target_taps.hip LABELS "gpu1;optional;external_files" TIMEOUT 1800 GPU_COUNT 1 GPU_COST_GB 24 LIBRARIES shelf_weights shelf_qwen35 shelf_qwen35_runtime)
    target_include_directories(test_dflash2_target_taps PRIVATE "${CMAKE_SOURCE_DIR}/src")

    # Target lm_head certified proxy: real hidden probe (num_output_rows large).

    # Target lm_head certified proxy: real hidden shadow (pool / margin sweep).

    # Target lm_head certified proxy: isolated real-head performance (Gate 6/7).



    # Model dir via SHELF_MODEL_DIR_MTP or SHELF_MODEL_DIR.


endif()
