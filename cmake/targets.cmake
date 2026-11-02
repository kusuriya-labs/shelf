# PhaseShift production / library targets.
#
# Included from root CMakeLists.txt. Defines the library targets and their
# dependency graph.

# Base INTERFACE target: public include root.
add_library(phaseshift_core INTERFACE)
target_compile_features(phaseshift_core INTERFACE cxx_std_20)
target_include_directories(
    phaseshift_core
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# GPU abstraction.
add_library(phaseshift_gpu STATIC
    src/phaseshift/core/memory/tensor.cpp
    src/phaseshift/core/gpu/scoped_device.cpp
    src/phaseshift/core/memory/arena.hip
)
target_compile_features(phaseshift_gpu PRIVATE cxx_std_20)
target_include_directories(phaseshift_gpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_gpu PUBLIC phaseshift_core hip::host)
target_compile_options(phaseshift_gpu PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_gpu)

# File I/O.
add_library(phaseshift_io STATIC
    src/phaseshift/io/safetensors_reader.cpp
    src/phaseshift/io/safetensors_writer.cpp
    src/phaseshift/io/row_reader.cpp
    src/phaseshift/io/hipfile_reader.cpp
)

target_compile_features(phaseshift_io PRIVATE cxx_std_20)
target_include_directories(phaseshift_io PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_io SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries(phaseshift_io PUBLIC phaseshift_core)
target_compile_options(phaseshift_io PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
if(hipfile_FOUND)
    target_compile_definitions(phaseshift_io PRIVATE PHASESHIFT_HAS_HIPFILE=1)
    target_link_libraries(phaseshift_io PUBLIC hip::host hip::hipfile)
else()
    target_link_libraries(phaseshift_io PUBLIC hip::host)
endif()

# Generic weight loading (model-independent): safetensors / quantized
# safetensors -> GPU MatrixWeight / Tensor. Depends on format + quantization
# layers, never on models/.
add_library(phaseshift_weights STATIC
    src/phaseshift/weights/weight_loader.cpp
    src/phaseshift/weights/tensor_partition.cpp
    src/phaseshift/weights/canonical_partition.cpp
)
target_compile_features(phaseshift_weights PRIVATE cxx_std_20)
target_include_directories(phaseshift_weights PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_weights SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries(phaseshift_weights PUBLIC
    phaseshift_core
    phaseshift_gpu
    phaseshift_io
    phaseshift_quant_format
    phaseshift_quant_bundle
    phaseshift_quant_reference
)
target_compile_options(phaseshift_weights PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Model-independent correctness reference kernels. One model reference executor = one translation
# unit. model_dispatch_correctness.hip is the single __global__ dispatch and
# includes detail/*.inc (elementwise / normalization / linear / attention / gdn
# / embedding / output) as source fragments. standalone/ keeps the per-op
# reference kernels used by kernel-level tests.
add_library(phaseshift_kernels_correctness STATIC
    src/phaseshift/kernels/correctness/model_dispatch_correctness.hip
    src/phaseshift/kernels/correctness/standalone/rmsnorm_correctness.hip
    src/phaseshift/kernels/correctness/standalone/silu_correctness.hip
    src/phaseshift/kernels/correctness/standalone/swiglu_correctness.hip
    src/phaseshift/kernels/correctness/standalone/gemm_bf16_correctness.hip
    src/phaseshift/kernels/correctness/standalone/gemm_block_scaled_correctness.hip
)
target_compile_features(phaseshift_kernels_correctness PRIVATE cxx_std_20)
target_include_directories(phaseshift_kernels_correctness PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_kernels_correctness PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(phaseshift_kernels_correctness PUBLIC phaseshift_gpu phaseshift_runtime phaseshift_inference_state)
target_compile_options(phaseshift_kernels_correctness PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_kernels_correctness)

# Model-independent optimized kernels (linear / attention / gdn / elementwise /
# normalization / embedding / quantize / sampling). One source per implementation
# family (the same optimization problem), not per KernelId or dtype. Public
# launchers live in the matching kernels/optimized/*.h and select
# dtype/algorithm variants internally. Shared by Qwen35 and future Qwen-family
# models; never depends on models/.
set(PHASESHIFT_OPTIMIZED_KERNEL_SOURCES
    src/phaseshift/kernels/optimized/linear/bf16.hip
    src/phaseshift/kernels/optimized/linear/fp8.hip
    src/phaseshift/kernels/optimized/linear/psq5.hip
    src/phaseshift/kernels/optimized/attention/paged_decode.hip
    src/phaseshift/kernels/optimized/attention/paged_prefill.hip
    src/phaseshift/kernels/optimized/attention/rope.hip
    src/phaseshift/kernels/optimized/embedding.hip
    src/phaseshift/kernels/optimized/elementwise.hip
    src/phaseshift/kernels/optimized/rmsnorm.hip
    src/phaseshift/kernels/optimized/l2_normalize.hip
    src/phaseshift/kernels/optimized/output_gather.hip
    src/phaseshift/kernels/optimized/row_gather.hip
    src/phaseshift/kernels/optimized/activation_quantize.hip
    src/phaseshift/kernels/optimized/swiglu_quantize.hip
    src/phaseshift/kernels/optimized/rmsnorm_quantize.hip
    src/phaseshift/kernels/optimized/resid_rmsnorm_quantize.hip
    src/phaseshift/kernels/optimized/attn_post_quantize.hip
    src/phaseshift/kernels/optimized/kv_append.hip
    src/phaseshift/kernels/optimized/kv_relocate.hip
    src/phaseshift/kernels/optimized/gdn/recurrence.hip
    src/phaseshift/kernels/optimized/gdn/conv1d.hip
    src/phaseshift/kernels/optimized/gdn/fused_gdn_post.hip
    src/phaseshift/kernels/optimized/gdn/fused_gdn_conv_silu.hip
    src/phaseshift/kernels/optimized/gdn/fused_gdn_qk_prepare.hip
    src/phaseshift/kernels/optimized/sampling.hip
    src/phaseshift/kernels/optimized/stochastic_sampling.hip
    src/phaseshift/kernels/optimized/radix_topn.hip
    src/phaseshift/kernels/optimized/group_ops.hip
    src/phaseshift/kernels/optimized/moe_ops.hip
    src/phaseshift/kernels/optimized/attention/qsa_indexer.hip
    src/phaseshift/kernels/optimized/attention/qsa_sparse_attention.hip
)

# Qwen35 speculative-decode kernels (DFlash2 / EAGLE4). Model-specific drafter
# kernels stay under models/qwen35 until a second model needs them.
set(PHASESHIFT_QWEN35_SPEC_KERNEL_SOURCES
    src/phaseshift/models/qwen35/kernels/dflash2/candidate_selector.hip
    src/phaseshift/models/qwen35/kernels/dflash2/feature_concat.hip
    src/phaseshift/models/qwen35/kernels/dflash2/grouped_dynamic_conv.hip
    src/phaseshift/models/qwen35/kernels/dflash2/rope.hip
    src/phaseshift/models/qwen35/kernels/dflash2/rmsnorm.hip
    src/phaseshift/models/qwen35/kernels/dflash2/topk.hip
    src/phaseshift/models/qwen35/kernels/dflash2/topk_optimized.hip
    src/phaseshift/models/qwen35/kernels/dflash2/swiglu.hip
    src/phaseshift/models/qwen35/kernels/dflash2/attention.hip
    src/phaseshift/models/qwen35/kernels/dflash2/kv_ring.hip
    src/phaseshift/models/qwen35/kernels/dflash2/noise_input.hip
    src/phaseshift/models/qwen35/kernels/eagle4/attention.hip
    src/phaseshift/models/qwen35/kernels/eagle4/candidate_ops.hip
    src/phaseshift/models/qwen35/kernels/eagle4/ops.hip
)

add_library(phaseshift_kernels_optimized STATIC ${PHASESHIFT_OPTIMIZED_KERNEL_SOURCES})
target_compile_features(phaseshift_kernels_optimized PRIVATE cxx_std_20)
target_include_directories(phaseshift_kernels_optimized PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_kernels_optimized PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(phaseshift_kernels_optimized PUBLIC phaseshift_gpu phaseshift_runtime phaseshift_inference_state)
target_compile_options(phaseshift_kernels_optimized PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_kernels_optimized)

add_library(phaseshift_qwen35_kernels_optimized STATIC ${PHASESHIFT_QWEN35_SPEC_KERNEL_SOURCES})
target_compile_features(phaseshift_qwen35_kernels_optimized PRIVATE cxx_std_20)
target_include_directories(phaseshift_qwen35_kernels_optimized PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_qwen35_kernels_optimized PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(phaseshift_qwen35_kernels_optimized PUBLIC phaseshift_gpu phaseshift_runtime phaseshift_kernels_optimized)
target_compile_options(phaseshift_qwen35_kernels_optimized PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_qwen35_kernels_optimized)

# Qwen3.5 paged sequence state management.
add_library(phaseshift_runtime STATIC
    src/phaseshift/runtime/batch/device_batch_context.hip
    src/phaseshift/runtime/graph/primitive_graph.cpp
    src/phaseshift/runtime/stream_bridge.cpp
    src/phaseshift/runtime/program/program.cpp
    src/phaseshift/runtime/tp/tp_execution.cpp
    src/phaseshift/runtime/tp/tp_barrier_group.cpp
    src/phaseshift/runtime/tp/tp_transports.cpp
    src/phaseshift/runtime/tp/hip_peer_tp_transport.hip
)
target_compile_features(phaseshift_runtime PRIVATE cxx_std_20)
target_include_directories(phaseshift_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_runtime PUBLIC phaseshift_core phaseshift_gpu)
target_compile_options(phaseshift_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_runtime)

add_library(phaseshift_inference_state STATIC
    src/phaseshift/inference/state/paged_sequence_state.cpp
    src/phaseshift/inference/state/sequence_slot_pool.cpp
    src/phaseshift/inference/state/gdn_state_pool.cpp
    src/phaseshift/inference/state/paged_kv_pool.cpp
)
target_compile_features(phaseshift_inference_state PRIVATE cxx_std_20)
target_include_directories(phaseshift_inference_state PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_inference_state PUBLIC phaseshift_core phaseshift_gpu)
target_compile_options(phaseshift_inference_state PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Model-independent scheduling (sampling params / scheduled batch / schedule
# plan / KV capacity / KV banker). Consumed by the Qwen35 runtime today and by
# future Qwen-family runtimes without copying.
add_library(phaseshift_inference_scheduling STATIC
    src/phaseshift/inference/scheduling/sampling_params.cpp
    src/phaseshift/inference/scheduling/scheduled_batch.cpp
    src/phaseshift/inference/scheduling/kv_capacity_manager.cpp
    src/phaseshift/inference/scheduling/kv_banker.cpp
)
target_compile_features(phaseshift_inference_scheduling PRIVATE cxx_std_20)
target_include_directories(phaseshift_inference_scheduling PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_inference_scheduling PUBLIC phaseshift_core phaseshift_gpu phaseshift_inference_state)
target_compile_options(phaseshift_inference_scheduling PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Model-independent op selectors (choose optimized kernel variants from
# Program / DispatchBinding shape facts). Shared by Qwen35 and future
# Qwen-family runtimes; never touches model weights or model config.
add_library(phaseshift_inference_execution STATIC
    src/phaseshift/inference/execution/selectors/linear_selector.cpp
    src/phaseshift/inference/execution/selectors/rmsnorm_selector.cpp
    src/phaseshift/inference/execution/selectors/l2_normalize_selector.cpp
    src/phaseshift/inference/execution/selectors/kv_append_selector.cpp
    src/phaseshift/inference/execution/selectors/paged_attention_selector.cpp
    src/phaseshift/inference/execution/selectors/gdn_recurrence_selector.cpp
    src/phaseshift/inference/execution/selectors/gdn_conv_selector.cpp
    src/phaseshift/inference/execution/selectors/rope_selector.cpp
    src/phaseshift/inference/execution/selectors/elementwise_selector.cpp
    src/phaseshift/inference/execution/selectors/embedding_selector.cpp
    src/phaseshift/inference/execution/selectors/sampling_selector.cpp
    src/phaseshift/inference/execution/attn_post_quantize_fusion.hip
    src/phaseshift/inference/execution/decode_perf_stats.cpp
    src/phaseshift/inference/execution/elementwise_dispatch.hip
    src/phaseshift/inference/execution/embedding_dispatch.hip
    src/phaseshift/inference/execution/gdn_conv_dispatch.hip
    src/phaseshift/inference/execution/gdn_conv_silu_fusion.hip
    src/phaseshift/inference/execution/gdn_post_fusion.hip
    src/phaseshift/inference/execution/gdn_qk_prepare_fusion.hip
    src/phaseshift/inference/execution/gdn_recurrence_dispatch.hip
    src/phaseshift/inference/execution/kv_append_dispatch.hip
    src/phaseshift/inference/execution/kv_calib_dump.cpp
    src/phaseshift/inference/execution/l2_normalize_dispatch.hip
    src/phaseshift/inference/execution/optimized_dispatch.hip
    src/phaseshift/inference/execution/paged_attention_dispatch.hip
    src/phaseshift/inference/execution/physical_launch.hip
    src/phaseshift/inference/execution/program_executor.hip
    src/phaseshift/inference/execution/resid_rmsnorm_quantize_fusion.hip
    src/phaseshift/inference/execution/rmsnorm_quantize_fusion.hip
    src/phaseshift/inference/execution/rope_dispatch.hip
    src/phaseshift/inference/execution/sampling_dispatch.hip
    src/phaseshift/inference/execution/swiglu_quantize_fusion.hip
)
target_compile_features(phaseshift_inference_execution PRIVATE cxx_std_20)
target_include_directories(phaseshift_inference_execution PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_inference_execution PUBLIC phaseshift_core phaseshift_gpu phaseshift_runtime phaseshift_inference_state phaseshift_inference_scheduling phaseshift_kernels_optimized phaseshift_kernels_correctness)
target_compile_options(phaseshift_inference_execution PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Qwen3.5 model sources.
set(
    PHASESHIFT_QWEN35_SOURCES
    src/phaseshift/models/qwen35/weights/model_weights.cpp
    src/phaseshift/models/qwen35/weights/tensor_parallel_plan.cpp
    src/phaseshift/models/qwen35/model/qwen35_config.cpp
    src/phaseshift/models/qwen35/model/rope_profile.cpp
    src/phaseshift/models/qwen35/model/tensor_parallel_context.cpp
    src/phaseshift/models/qwen35/model/qwen35_model.cpp
    src/phaseshift/models/qwen35/model/lower_to_primitives.cpp
    src/phaseshift/models/qwen35/dflash2/config.cpp
    src/phaseshift/models/qwen35/dflash2/weights.cpp
    src/phaseshift/models/qwen35/dflash2/context_state.hip
    src/phaseshift/models/qwen35/eagle4/config.cpp
    src/phaseshift/models/qwen35/eagle4/weights.cpp
    src/phaseshift/models/qwen35/eagle4/tensor_parallel_plan.cpp
    src/phaseshift/models/qwen35/eagle4/context_state.hip
)

add_library(phaseshift_qwen35 STATIC ${PHASESHIFT_QWEN35_SOURCES})
target_compile_features(phaseshift_qwen35 PRIVATE cxx_std_20)
target_include_directories(phaseshift_qwen35 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_qwen35 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(phaseshift_qwen35 SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
target_link_libraries(phaseshift_qwen35 PUBLIC phaseshift_weights phaseshift_io phaseshift_inference_state phaseshift_quant_reference phaseshift_quant_format phaseshift_quant_bundle phaseshift_runtime)
target_compile_options(phaseshift_qwen35 PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Qwen4Exp (Qwen3.8-Flash-Next). Model-local PLE / Expert / kernel staging.
add_library(phaseshift_qwen4exp STATIC
    src/phaseshift/models/qwen4exp/model/qwen4exp_config.cpp
    src/phaseshift/models/qwen4exp/model/qwen4exp_weights.cpp
    src/phaseshift/models/qwen4exp/ple/ple_row_gather.cpp
)
target_compile_features(phaseshift_qwen4exp PRIVATE cxx_std_20)
target_include_directories(phaseshift_qwen4exp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_qwen4exp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(phaseshift_qwen4exp SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
target_link_libraries(phaseshift_qwen4exp PUBLIC phaseshift_weights phaseshift_io phaseshift_gpu phaseshift_kernels_optimized)
target_compile_options(phaseshift_qwen4exp PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_qwen4exp)

target_sources(phaseshift_qwen4exp PRIVATE
    src/phaseshift/models/qwen4exp/ple/ngram_hash.cpp
    src/phaseshift/models/qwen4exp/ple/ple_conv.cpp
    src/phaseshift/models/qwen4exp/hyper/gated_residual.cpp
    src/phaseshift/models/qwen4exp/runtime/qwen4exp_state.cpp
    src/phaseshift/models/qwen4exp/model/lower_to_primitives.cpp
    src/phaseshift/models/qwen4exp/ple/ple_forward.cpp
)

# Qwen3.5 continuous batching runtime.
add_library(phaseshift_qwen35_runtime STATIC
    src/phaseshift/models/qwen35/runtime/runtime_request.cpp
    src/phaseshift/models/qwen35/runtime/token_budget_scheduler.cpp
    src/phaseshift/models/qwen35/runtime/continuous_batcher.cpp
    src/phaseshift/models/qwen35/runtime/prefix_cache.cpp
    src/phaseshift/models/qwen35/runtime/tensor_parallel.cpp
    src/phaseshift/models/qwen35/runtime/executor.hip
    src/phaseshift/models/qwen35/runtime/spec_decode.cpp
    src/phaseshift/models/qwen35/runtime/eagle4_tree.cpp
    src/phaseshift/models/qwen35/runtime/eagle4_cost.cpp
    src/phaseshift/models/qwen35/runtime/dflash2_spec_decoder.cpp
    src/phaseshift/models/qwen35/runtime/dflash2_spec_batch.cpp
    src/phaseshift/models/qwen35/runtime/dflash2_session_pool.cpp
    src/phaseshift/models/qwen35/runtime/gdn_spec_history.hip
    src/phaseshift/models/qwen35/runtime/decode_backend.cpp
    src/phaseshift/models/qwen35/dflash2/executor.hip
    src/phaseshift/models/qwen35/eagle4/executor.hip
)
target_compile_features(phaseshift_qwen35_runtime PRIVATE cxx_std_20)
target_include_directories(phaseshift_qwen35_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_qwen35_runtime PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(phaseshift_qwen35_runtime PUBLIC phaseshift_qwen35 phaseshift_inference_state phaseshift_inference_scheduling phaseshift_inference_execution phaseshift_kernels_correctness phaseshift_qwen35_kernels_optimized phaseshift_runtime phaseshift_gpu phaseshift_resident phaseshift_model_source)
target_compile_options(phaseshift_qwen35_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_qwen35_runtime)
if(PHASESHIFT_HIP_GRAPH)
    set_source_files_properties(
        src/phaseshift/inference/execution/program_executor.hip
        src/phaseshift/models/qwen35/runtime/executor.hip
        PROPERTIES COMPILE_DEFINITIONS "PHASESHIFT_HIP_GRAPH")
endif()

# Resident model client: attaches GPU model weights owned by the internal
# phaseshift-model-host component (see cmake/model_host.cmake). Used by
# phaseshift-compute and by test workers. Enabled per process by
# --model-host / PHASESHIFT_MODEL_HOST_SOCKET; not a user-facing entrypoint.
# Contract: docs/developer/resident_model.md
add_library(phaseshift_resident STATIC
    src/phaseshift/resident/resident_model_key.cpp
    src/phaseshift/resident/resident_format.cpp
    src/phaseshift/resident/resident_protocol.cpp
    src/phaseshift/resident/resident_client.cpp
    src/phaseshift/resident/resident_host.cpp
)
target_compile_features(phaseshift_resident PRIVATE cxx_std_20)
target_include_directories(phaseshift_resident PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_resident PUBLIC
    phaseshift_core
    phaseshift_gpu
    phaseshift_weights
    phaseshift_qwen35
)
target_compile_options(phaseshift_resident PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Model source selection: local safetensors load or resident model host attach.
# Startup path only; the choice is resolved once per process.
add_library(phaseshift_model_source STATIC
    src/phaseshift/models/model_source.cpp
)
target_compile_features(phaseshift_model_source PRIVATE cxx_std_20)
target_include_directories(phaseshift_model_source PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_model_source PUBLIC
    phaseshift_qwen35
    phaseshift_resident
)
target_compile_options(phaseshift_model_source PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Aggregate INTERFACE target. Tests and benchmarks link against this.
add_library(phaseshift INTERFACE)
target_link_libraries(
    phaseshift
    INTERFACE
        phaseshift_core
        phaseshift_gpu
        phaseshift_io
        phaseshift_weights
        phaseshift_kernels_correctness
        phaseshift_qwen35_kernels_optimized
        phaseshift_runtime
        phaseshift_inference_state
        phaseshift_inference_scheduling
        phaseshift_inference_execution
        phaseshift_qwen35
        phaseshift_qwen35_runtime
        phaseshift_qwen4exp
        phaseshift_resident
        phaseshift_quant_format
        phaseshift_quant_bundle
)

# Quantized safetensors manifest contract (schema / CRC32). Header-only
# runtime resolution lives in include/phaseshift/quantization/manifest/.
add_library(phaseshift_quant_format STATIC
    src/phaseshift/quantization/manifest/crc32.cpp
    src/phaseshift/quantization/manifest/manifest_v4.cpp
    src/phaseshift/quantization/manifest/quantized_manifest.cpp
)
target_compile_features(phaseshift_quant_format PRIVATE cxx_std_20)
target_include_directories(phaseshift_quant_format PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_quant_format SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries(phaseshift_quant_format PUBLIC phaseshift_core)
target_compile_options(phaseshift_quant_format PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Quantized bundle I/O (bundle reader / HF adapter) over the manifest contract.
add_library(phaseshift_quant_bundle STATIC
    src/phaseshift/quantization/bundle/hf_manifest.cpp
    src/phaseshift/quantization/bundle/psq5_manifest.cpp
    src/phaseshift/quantization/bundle/quantized_model_reader.cpp
)
target_compile_features(phaseshift_quant_bundle PRIVATE cxx_std_20)
target_include_directories(phaseshift_quant_bundle PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_quant_bundle PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(phaseshift_quant_bundle SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries(phaseshift_quant_bundle PUBLIC phaseshift_quant_format phaseshift_io)
target_compile_options(phaseshift_quant_bundle PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Reference quantizers (pure CPU math; independent of model_format).
add_library(phaseshift_quant_reference STATIC
    src/phaseshift/quantization/codec/e4m3.cpp
    src/phaseshift/quantization/codec/ue4m3.cpp
    src/phaseshift/quantization/fp8/fp8.cpp
    src/phaseshift/quantization/fp8/fp8_native.cpp
    src/phaseshift/quantization/psq5/psq5.cpp
)
set_source_files_properties(
    src/phaseshift/quantization/fp8/fp8.cpp
    src/phaseshift/quantization/fp8/fp8_native.cpp
    src/phaseshift/quantization/psq5/psq5.cpp
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_compile_features(phaseshift_quant_reference PRIVATE cxx_std_20)
target_include_directories(phaseshift_quant_reference PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_quant_reference PUBLIC phaseshift_core)
target_compile_options(phaseshift_quant_reference PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

find_package(OpenSSL REQUIRED)

