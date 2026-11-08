# Shelf production / library targets.
#
# Included from root CMakeLists.txt. Defines the library targets and their
# dependency graph.

# Base INTERFACE target: public include root.
add_library(shelf_core INTERFACE)
target_compile_features(shelf_core INTERFACE cxx_std_20)
target_include_directories(
    shelf_core
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# GPU abstraction.
add_library(shelf_gpu STATIC
    src/shelf/core/memory/tensor.cpp
    src/shelf/core/gpu/scoped_device.cpp
    src/shelf/core/memory/arena.hip
)
target_compile_features(shelf_gpu PRIVATE cxx_std_20)
target_include_directories(shelf_gpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(shelf_gpu PUBLIC shelf_core hip::host)
target_compile_options(shelf_gpu PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
shelf_set_hip_archs(shelf_gpu)

# File I/O.
add_library(shelf_io STATIC
    src/shelf/io/safetensors_reader.cpp
    src/shelf/io/safetensors_writer.cpp
)

target_compile_features(shelf_io PRIVATE cxx_std_20)
target_include_directories(shelf_io PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_io SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries(shelf_io PUBLIC shelf_core hip::host)
target_compile_options(shelf_io PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Generic weight loading (model-independent): safetensors / quantized
# safetensors -> GPU MatrixWeight / Tensor. Depends on format + quantization
# layers, never on models/.
add_library(shelf_weights STATIC
    src/shelf/weights/weight_loader.cpp
    src/shelf/weights/tensor_partition.cpp
    src/shelf/weights/canonical_partition.cpp
)
target_compile_features(shelf_weights PRIVATE cxx_std_20)
target_include_directories(shelf_weights PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_weights SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries(shelf_weights PUBLIC
    shelf_core
    shelf_gpu
    shelf_io
    shelf_quant_format
    shelf_quant_bundle
    shelf_quant_reference
)
target_compile_options(shelf_weights PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Model-independent correctness reference kernels. One model reference executor = one translation
# unit. model_dispatch_correctness.hip is the single __global__ dispatch and
# includes detail/*.inc (elementwise / normalization / linear / attention / gdn
# / embedding / output) as source fragments. standalone/ keeps the per-op
# reference kernels used by kernel-level tests.
add_library(shelf_kernels_correctness STATIC
    src/shelf/kernels/correctness/model_dispatch_correctness.hip
    src/shelf/kernels/correctness/standalone/rmsnorm_correctness.hip
    src/shelf/kernels/correctness/standalone/silu_correctness.hip
    src/shelf/kernels/correctness/standalone/swiglu_correctness.hip
    src/shelf/kernels/correctness/standalone/gemm_bf16_correctness.hip
    src/shelf/kernels/correctness/standalone/gemm_block_scaled_correctness.hip
)
target_compile_features(shelf_kernels_correctness PRIVATE cxx_std_20)
target_include_directories(shelf_kernels_correctness PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_kernels_correctness PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(shelf_kernels_correctness PUBLIC shelf_gpu shelf_runtime shelf_inference_state)
target_compile_options(shelf_kernels_correctness PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
shelf_set_hip_archs(shelf_kernels_correctness)

# Model-independent optimized kernels (linear / attention / gdn / elementwise /
# normalization / embedding / quantize / sampling). One source per implementation
# family (the same optimization problem), not per KernelId or dtype. Public
# launchers live in the matching kernels/optimized/*.h and select
# dtype/algorithm variants internally. Shared by Qwen35 and future Qwen-family
# models; never depends on models/.
set(SHELF_OPTIMIZED_KERNEL_SOURCES
    src/shelf/kernels/optimized/linear/bf16.hip
    src/shelf/kernels/optimized/linear/fp8.hip
    src/shelf/kernels/optimized/linear/fp8_ptpc.hip
    src/shelf/kernels/optimized/linear/mxfp4.hip
    src/shelf/kernels/optimized/linear/exl3.hip
    src/shelf/kernels/optimized/linear/exl3_mul24_grouped.hip
    src/shelf/kernels/optimized/linear/exl3_int8.hip
    src/shelf/kernels/optimized/linear/exl3_int8_prefill.hip
    src/shelf/kernels/optimized/moe/router.hip
    src/shelf/kernels/optimized/moe/combine.hip
    src/shelf/kernels/optimized/attention/paged_decode.hip
    src/shelf/kernels/optimized/attention/paged_prefill.hip
    src/shelf/kernels/optimized/attention/rope.hip
    src/shelf/kernels/optimized/embedding.hip
    src/shelf/kernels/optimized/elementwise.hip
    src/shelf/kernels/optimized/rmsnorm.hip
    src/shelf/kernels/optimized/hyper_connection.hip
    src/shelf/kernels/optimized/mean_streams.hip
    src/shelf/kernels/optimized/ple.hip
    src/shelf/kernels/optimized/ple_conv.hip
    src/shelf/kernels/optimized/repeat_streams.hip
    src/shelf/kernels/optimized/qsa.hip
    src/shelf/kernels/optimized/l2_normalize.hip
    src/shelf/kernels/optimized/output_gather.hip
    src/shelf/kernels/optimized/activation_quantize.hip
    src/shelf/kernels/optimized/kv_append.hip
    src/shelf/kernels/optimized/gdn/recurrence.hip
    src/shelf/kernels/optimized/gdn/conv1d.hip
    src/shelf/kernels/optimized/sampling.hip
    src/shelf/kernels/optimized/stochastic_sampling.hip
    src/shelf/kernels/optimized/radix_topn.hip
)

# Fused kernels: more than one logical operation in one launch, separate from the
# single-operation optimized family.
set(SHELF_FUSED_KERNEL_SOURCES
    src/shelf/kernels/fused/hyper_connection.hip
    src/shelf/kernels/fused/silu_scale.hip
    src/shelf/kernels/fused/tp_combine.hip
)

add_library(shelf_kernels_fused STATIC ${SHELF_FUSED_KERNEL_SOURCES})
target_compile_features(shelf_kernels_fused PRIVATE cxx_std_20)
target_include_directories(shelf_kernels_fused PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(shelf_kernels_fused PUBLIC shelf_core shelf_gpu)
target_compile_options(shelf_kernels_fused PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
shelf_set_hip_archs(shelf_kernels_fused)

# Qwen35 speculative-decode kernels (DFlash2). Model-specific drafter kernels
# stay under models/qwen35 until a second model needs them.
set(SHELF_QWEN35_SPEC_KERNEL_SOURCES
    src/shelf/models/qwen35/kernels/dflash2/candidate_selector.hip
    src/shelf/models/qwen35/kernels/dflash2/feature_concat.hip
    src/shelf/models/qwen35/kernels/dflash2/grouped_dynamic_conv.hip
    src/shelf/models/qwen35/kernels/dflash2/rope.hip
    src/shelf/models/qwen35/kernels/dflash2/rmsnorm.hip
    src/shelf/models/qwen35/kernels/dflash2/topk.hip
    src/shelf/models/qwen35/kernels/dflash2/topk_optimized.hip
    src/shelf/models/qwen35/kernels/dflash2/swiglu.hip
    src/shelf/models/qwen35/kernels/dflash2/attention.hip
    src/shelf/models/qwen35/kernels/dflash2/kv_ring.hip
    src/shelf/models/qwen35/kernels/dflash2/noise_input.hip
)

add_library(shelf_kernels_optimized STATIC ${SHELF_OPTIMIZED_KERNEL_SOURCES})
target_compile_features(shelf_kernels_optimized PRIVATE cxx_std_20)
target_include_directories(shelf_kernels_optimized PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_kernels_optimized PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(shelf_kernels_optimized PUBLIC shelf_gpu shelf_runtime shelf_inference_state shelf_kernels_fused)
target_compile_options(shelf_kernels_optimized PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
shelf_set_hip_archs(shelf_kernels_optimized)

add_library(shelf_qwen35_kernels_optimized STATIC ${SHELF_QWEN35_SPEC_KERNEL_SOURCES})
target_compile_features(shelf_qwen35_kernels_optimized PRIVATE cxx_std_20)
target_include_directories(shelf_qwen35_kernels_optimized PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_qwen35_kernels_optimized PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(shelf_qwen35_kernels_optimized PUBLIC shelf_gpu shelf_runtime shelf_kernels_optimized)
target_compile_options(shelf_qwen35_kernels_optimized PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
shelf_set_hip_archs(shelf_qwen35_kernels_optimized)

# Qwen3.5 paged sequence state management.
add_library(shelf_runtime STATIC
    src/shelf/runtime/batch/device_batch_context.hip
    src/shelf/runtime/graph/primitive_graph.cpp
    src/shelf/runtime/stream_bridge.cpp
    src/shelf/runtime/program/program.cpp
    src/shelf/runtime/tp/tp_execution.cpp
    src/shelf/runtime/tp/tp_barrier_group.cpp
    src/shelf/runtime/tp/tp_transports.cpp
    src/shelf/runtime/tp/hip_peer_tp_transport.hip
)
target_compile_features(shelf_runtime PRIVATE cxx_std_20)
target_include_directories(shelf_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(shelf_runtime PUBLIC shelf_core shelf_gpu shelf_kernels_fused)
target_compile_options(shelf_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
shelf_set_hip_archs(shelf_runtime)

add_library(shelf_inference_state STATIC
    src/shelf/inference/state/paged_sequence_state.cpp
    src/shelf/inference/state/sequence_slot_pool.cpp
    src/shelf/inference/state/gdn_state_pool.cpp
    src/shelf/inference/state/qsa_state_pool.cpp
    src/shelf/inference/state/ple_conv_state_pool.cpp
    src/shelf/inference/state/paged_kv_pool.cpp
)
target_compile_features(shelf_inference_state PRIVATE cxx_std_20)
target_include_directories(shelf_inference_state PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(shelf_inference_state PUBLIC shelf_core shelf_gpu)
target_compile_options(shelf_inference_state PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Model-independent scheduling (sampling params / scheduled batch / schedule
# plan / KV capacity / KV banker). Consumed by the Qwen35 runtime today and by
# future Qwen-family runtimes without copying.
add_library(shelf_inference_scheduling STATIC
    src/shelf/inference/scheduling/sampling_params.cpp
    src/shelf/inference/scheduling/scheduled_batch.cpp
    src/shelf/inference/scheduling/kv_capacity_manager.cpp
    src/shelf/inference/scheduling/kv_banker.cpp
)
target_compile_features(shelf_inference_scheduling PRIVATE cxx_std_20)
target_include_directories(shelf_inference_scheduling PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(shelf_inference_scheduling PUBLIC shelf_core shelf_gpu shelf_inference_state)
target_compile_options(shelf_inference_scheduling PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Model-independent paged-KV prefix cache. Owns the radix tree, refcounted page
# sharing and eviction; model-specific recurrent state is supplied by a
# per-family adapter.
add_library(shelf_inference_prefix STATIC
    src/shelf/inference/prefix/prefix_kv_cache.cpp
)
target_compile_features(shelf_inference_prefix PRIVATE cxx_std_20)
target_include_directories(shelf_inference_prefix PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(shelf_inference_prefix PUBLIC shelf_core shelf_gpu shelf_inference_state)
target_compile_options(shelf_inference_prefix PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Model-independent op selectors (choose optimized kernel variants from
# Program / DispatchBinding shape facts). Shared by Qwen35 and future
# Qwen-family runtimes; never touches model weights or model config.
add_library(shelf_inference_execution STATIC
    src/shelf/inference/execution/selectors/linear_selector.cpp
    src/shelf/inference/execution/selectors/rmsnorm_selector.cpp
    src/shelf/inference/execution/selectors/l2_normalize_selector.cpp
    src/shelf/inference/execution/selectors/kv_append_selector.cpp
    src/shelf/inference/execution/selectors/paged_attention_selector.cpp
    src/shelf/inference/execution/selectors/gdn_recurrence_selector.cpp
    src/shelf/inference/execution/selectors/gdn_conv_selector.cpp
    src/shelf/inference/execution/selectors/rope_selector.cpp
    src/shelf/inference/execution/selectors/elementwise_selector.cpp
    src/shelf/inference/execution/selectors/embedding_selector.cpp
    src/shelf/inference/execution/selectors/sampling_selector.cpp
    src/shelf/inference/execution/decode_perf_stats.cpp
    src/shelf/inference/execution/elementwise_dispatch.hip
    src/shelf/inference/execution/exl3_dense_gemm.hip
    src/shelf/inference/execution/gemm_lt.hip
    src/shelf/inference/execution/fp8_ptpc_dispatch.hip
    src/shelf/inference/execution/embedding_dispatch.hip
    src/shelf/inference/execution/gdn_conv_dispatch.hip
    src/shelf/inference/execution/gdn_recurrence_dispatch.hip
    src/shelf/inference/execution/kv_append_dispatch.hip
    src/shelf/inference/execution/kv_calib_dump.cpp
    src/shelf/inference/execution/linear_input_dump.cpp
    src/shelf/inference/execution/moe_dispatch.hip
    src/shelf/inference/execution/hyper_connection_dispatch.hip
    src/shelf/inference/execution/ple_dispatch.hip
    src/shelf/inference/execution/qsa_dispatch.hip
    src/shelf/inference/execution/l2_normalize_dispatch.hip
    src/shelf/inference/execution/optimized_dispatch.hip
    src/shelf/inference/execution/paged_attention_dispatch.hip
    src/shelf/inference/execution/physical_launch.hip
    src/shelf/inference/execution/program_executor.hip
    src/shelf/inference/execution/weight_slot_validation.cpp
    src/shelf/inference/execution/rope_dispatch.hip
    src/shelf/inference/execution/sampling_dispatch.hip
)
target_compile_features(shelf_inference_execution PRIVATE cxx_std_20)
target_include_directories(shelf_inference_execution PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
find_library(SHELF_HIPBLASLT_LIBRARY NAMES hipblaslt PATHS "${SHELF_ROCM_ROOT}/lib" NO_DEFAULT_PATH)
target_link_libraries(shelf_inference_execution PUBLIC shelf_core shelf_gpu shelf_runtime shelf_inference_state shelf_inference_scheduling shelf_kernels_optimized shelf_kernels_correctness)
if(SHELF_HIPBLASLT_LIBRARY)
    target_link_libraries(shelf_inference_execution PUBLIC "${SHELF_HIPBLASLT_LIBRARY}")
else()
    message(FATAL_ERROR "hipblaslt was not found under ${SHELF_ROCM_ROOT}/lib")
endif()
target_compile_options(shelf_inference_execution PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Qwen3.5 model sources.
set(
    SHELF_QWEN35_SOURCES
    src/shelf/models/qwen35/weights/model_weights.cpp
    src/shelf/models/qwen35/weights/tensor_parallel_plan.cpp
    src/shelf/models/qwen35/model/qwen35_config.cpp
    src/shelf/models/qwen35/model/rope_profile.cpp
    src/shelf/models/qwen35/model/tensor_parallel_context.cpp
    src/shelf/models/qwen35/model/qwen35_model.cpp
    src/shelf/models/qwen35/model/lower_to_primitives.cpp
    src/shelf/models/qwen35/dflash2/config.cpp
    src/shelf/models/qwen35/dflash2/weights.cpp
    src/shelf/models/qwen35/dflash2/context_state.hip
)

# Qwen hybrid backbone: the model geometry and weight loading that a dense model and an
# MoE model share. It knows nothing about either FFN.
set(
    SHELF_QWEN_HYBRID_SOURCES
    src/shelf/models/common/qwen_hybrid/config.cpp
    src/shelf/models/common/qwen_hybrid/weight_loader.cpp
    src/shelf/models/common/qwen_hybrid/weight_validation.cpp
    src/shelf/models/common/qwen_hybrid/weight_slot.cpp
    src/shelf/models/common/qwen_hybrid/tensor_partition.cpp
    src/shelf/models/common/qwen_hybrid/tensor_parallel_context.cpp
)

add_library(shelf_qwen_hybrid STATIC ${SHELF_QWEN_HYBRID_SOURCES})
target_compile_features(shelf_qwen_hybrid PRIVATE cxx_std_20)
target_include_directories(shelf_qwen_hybrid PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_qwen_hybrid PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(shelf_qwen_hybrid SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
target_link_libraries(shelf_qwen_hybrid PUBLIC shelf_weights shelf_io shelf_quant_bundle)
target_compile_options(shelf_qwen_hybrid PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

add_library(shelf_qwen35 STATIC ${SHELF_QWEN35_SOURCES})
target_compile_features(shelf_qwen35 PRIVATE cxx_std_20)
target_include_directories(shelf_qwen35 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_qwen35 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(shelf_qwen35 SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
target_link_libraries(shelf_qwen35 PUBLIC shelf_qwen_hybrid shelf_weights shelf_io shelf_inference_state shelf_quant_reference shelf_quant_format shelf_quant_bundle shelf_runtime)
target_compile_options(shelf_qwen35 PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Qwen3.5 MoE (model_type qwen3_5_moe): routed expert weights and config.
set(
    SHELF_QWEN35MOE_SOURCES
    src/shelf/models/qwen35moe/model/qwen35moe_config.cpp
    src/shelf/models/qwen35moe/model/tensor_parallel_context.cpp
    src/shelf/models/qwen35moe/model/lower_to_primitives.cpp
    src/shelf/models/qwen35moe/runtime/executor_footprint.cpp
    src/shelf/models/qwen35moe/model/qwen35moe_model.cpp
    src/shelf/models/qwen35moe/weights/moe_weight_plan.cpp
    src/shelf/models/qwen35moe/weights/moe_expert_load.cpp
    src/shelf/models/qwen35moe/weights/moe_tensor_parallel_plan.cpp
    src/shelf/models/qwen35moe/weights/qwen35moe_model_load.cpp
)

add_library(shelf_qwen35moe STATIC ${SHELF_QWEN35MOE_SOURCES})
target_compile_features(shelf_qwen35moe PRIVATE cxx_std_20)
target_include_directories(shelf_qwen35moe PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_qwen35moe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(shelf_qwen35moe SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
target_link_libraries(shelf_qwen35moe PUBLIC shelf_qwen_hybrid shelf_weights shelf_io shelf_quant_bundle shelf_gpu)
target_compile_options(shelf_qwen35moe PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Qwen3.8-Flash-Next (model_type qwen4_exp): EXL3 body, PLE n-gram table and the
# expert-axis tensor-parallel load path. This target owns the load contract only; the
# forward/executor path is a later family port.
set(
    SHELF_QWEN4EXP_SOURCES
    src/shelf/models/qwen4exp/model/qwen4exp_config.cpp
    src/shelf/models/qwen4exp/model/qwen4exp_model.cpp
    src/shelf/models/qwen4exp/model/lower_to_primitives.cpp
    src/shelf/models/qwen4exp/model/hyper_connection_lowering.cpp
    src/shelf/models/qwen4exp/model/moe_lowering.cpp
    src/shelf/models/qwen4exp/model/layer_lowering.cpp
    src/shelf/models/qwen4exp/model/ple_lowering.cpp
    src/shelf/models/qwen4exp/model/qsa_lowering.cpp
    src/shelf/models/qwen4exp/weights/ple_table.cpp
    src/shelf/models/qwen4exp/runtime/ple_input.cpp
    src/shelf/models/qwen4exp/runtime/executor_footprint.cpp
    src/shelf/models/qwen4exp/runtime/ple_input.cpp
    src/shelf/models/qwen4exp/weights/qwen4exp_load.cpp
)

add_library(shelf_qwen4exp STATIC ${SHELF_QWEN4EXP_SOURCES})
target_compile_features(shelf_qwen4exp PRIVATE cxx_std_20)
target_include_directories(shelf_qwen4exp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_qwen4exp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(shelf_qwen4exp SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
target_link_libraries(shelf_qwen4exp PUBLIC shelf_qwen_hybrid shelf_weights shelf_io shelf_quant_bundle shelf_gpu)
target_compile_options(shelf_qwen4exp PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Model weight loading entry for the qwen35 family: the one place that links both the
# dense and the MoE loaders and picks between them.
add_library(shelf_models STATIC
    src/shelf/models/model_weights_factory.cpp
    src/shelf/models/loaded_model.cpp
    src/shelf/models/pipeline_stage_model.cpp
    src/shelf/models/tp_rank_model.cpp)
target_compile_features(shelf_models PRIVATE cxx_std_20)
target_include_directories(shelf_models PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_models PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(shelf_models PUBLIC shelf_qwen35 shelf_qwen35_runtime shelf_qwen35moe shelf_qwen4exp shelf_gpu)
target_compile_options(shelf_models PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

add_library(shelf_qwen35_runtime STATIC
    src/shelf/models/qwen35/runtime/runtime_request.cpp
    src/shelf/models/qwen35/runtime/token_budget_scheduler.cpp
    src/shelf/models/qwen35/runtime/continuous_batcher.cpp
    src/shelf/models/qwen35/runtime/prefix_cache.cpp
    src/shelf/models/qwen35/runtime/tensor_parallel.cpp
    src/shelf/models/qwen35/runtime/pipeline_parallel.cpp
    src/shelf/models/qwen35/runtime/executor.hip
    src/shelf/models/qwen35/runtime/spec_decode.cpp
    src/shelf/models/qwen35/runtime/mtp_drafter.cpp
    src/shelf/models/qwen35/runtime/dflash2_spec_decoder.cpp
    src/shelf/models/qwen35/runtime/dflash2_spec_batch.cpp
    src/shelf/models/qwen35/runtime/dflash2_session_pool.cpp
    src/shelf/models/qwen35/runtime/dflash2_footprint.cpp
    src/shelf/models/qwen35/runtime/executor_footprint.cpp
    src/shelf/models/qwen35/runtime/gdn_spec_history.hip
    src/shelf/models/qwen35/runtime/decode_backend.cpp
    src/shelf/models/qwen35/dflash2/executor.hip
    src/shelf/models/common/drafter/linear.hip
)
target_compile_features(shelf_qwen35_runtime PRIVATE cxx_std_20)
target_include_directories(shelf_qwen35_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_qwen35_runtime PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(shelf_qwen35_runtime PUBLIC shelf_qwen35 shelf_inference_state shelf_inference_scheduling shelf_inference_execution shelf_inference_prefix shelf_kernels_correctness shelf_qwen35_kernels_optimized shelf_runtime shelf_gpu)
target_compile_options(shelf_qwen35_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
shelf_set_hip_archs(shelf_qwen35_runtime)
if(SHELF_HIP_GRAPH)
    set_source_files_properties(
        src/shelf/inference/execution/program_executor.hip
        src/shelf/models/qwen35/runtime/executor.hip
        PROPERTIES COMPILE_DEFINITIONS "SHELF_HIP_GRAPH")
endif()

# Aggregate INTERFACE target. Tests and benchmarks link against this.
add_library(shelf INTERFACE)
target_link_libraries(
    shelf
    INTERFACE
        shelf_core
        shelf_gpu
        shelf_io
        shelf_weights
        shelf_kernels_correctness
        shelf_qwen35_kernels_optimized
        shelf_runtime
        shelf_inference_state
        shelf_inference_scheduling
        shelf_inference_execution
        shelf_qwen35
        shelf_qwen35_runtime
        shelf_qwen35moe
        shelf_models
        shelf_quant_format
        shelf_quant_bundle
)

# Quantized safetensors manifest contract (schema / CRC32). Header-only
# runtime resolution lives in include/shelf/quantization/manifest/.
add_library(shelf_quant_format STATIC
    src/shelf/quantization/manifest/crc32.cpp
    src/shelf/quantization/manifest/manifest_v4.cpp
    src/shelf/quantization/manifest/quantized_manifest.cpp
)
target_compile_features(shelf_quant_format PRIVATE cxx_std_20)
target_include_directories(shelf_quant_format PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_quant_format SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries(shelf_quant_format PUBLIC shelf_core)
target_compile_options(shelf_quant_format PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Quantized bundle I/O (bundle reader / HF adapter) over the manifest contract.
add_library(shelf_quant_bundle STATIC
    src/shelf/quantization/bundle/hf_manifest.cpp
    src/shelf/quantization/bundle/quantized_model_reader.cpp
)
target_compile_features(shelf_quant_bundle PRIVATE cxx_std_20)
target_include_directories(shelf_quant_bundle PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(shelf_quant_bundle PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(shelf_quant_bundle SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries(shelf_quant_bundle PUBLIC shelf_quant_format shelf_io)
target_compile_options(shelf_quant_bundle PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Reference quantizers (pure CPU math; independent of model_format).
add_library(shelf_quant_reference STATIC
    src/shelf/quantization/codec/e4m3.cpp
    src/shelf/quantization/codec/ue4m3.cpp
    src/shelf/quantization/fp8/fp8.cpp
    src/shelf/quantization/fp8/fp8_native.cpp
    src/shelf/quantization/mxfp4/mxfp4.cpp
    src/shelf/quantization/mxfp4/mxfp4_native.cpp
    src/shelf/quantization/exl3/exl3.cpp
    src/shelf/quantization/exl3/exl3_int8.cpp
)
set_source_files_properties(
    src/shelf/quantization/fp8/fp8.cpp
    src/shelf/quantization/fp8/fp8_native.cpp
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_compile_features(shelf_quant_reference PRIVATE cxx_std_20)
target_include_directories(shelf_quant_reference PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(shelf_quant_reference PUBLIC shelf_core)
target_compile_options(shelf_quant_reference PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

