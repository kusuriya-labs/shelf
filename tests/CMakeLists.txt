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

# MXFP4（E2M1/E8M0 コーデックと断片順 relayout）
add_executable(test_quant_mxfp4 unit/test_quant_mxfp4.cpp)
target_link_libraries(test_quant_mxfp4 PRIVATE shelf::quant)
add_test(NAME unit.quant_mxfp4 COMMAND test_quant_mxfp4)

# quantization_config の解析（apps/common の実装を使う）
# 性能の回帰ゲート（コストモデル。GPU 不要・決定論的）
add_executable(test_cost_model architecture/test_cost_model.cpp)
target_link_libraries(test_cost_model PRIVATE shelf::model_qwen35 shelf::ir)
add_test(NAME architecture.cost_model COMMAND test_cost_model)

add_executable(test_fp8_format unit/test_fp8_format.cpp)
target_link_libraries(test_fp8_format PRIVATE shelf_app_common)
add_test(NAME unit.fp8_format COMMAND test_fp8_format)

add_executable(test_ir_graph_validation unit/test_ir_graph_validation.cpp)
target_link_libraries(test_ir_graph_validation PRIVATE shelf::ir)
add_test(NAME unit.ir_graph_validation COMMAND test_ir_graph_validation)

# 起動の受理判定（D1）— GPU 不要の純粋関数
# 読み取りと実体化（D3）— 合成の供給元で GPU 不要
add_executable(test_runtime_load unit/test_runtime_load.cpp)
target_link_libraries(test_runtime_load PRIVATE shelf::runtime_load)
add_test(NAME unit.runtime_load COMMAND test_runtime_load)

# declare 相（D4）: アリーナ offsets の純粋計算
add_executable(test_arena_plan unit/test_arena_plan.cpp)
target_link_libraries(test_arena_plan PRIVATE shelf::compiler shelf::model_qwen35)
add_test(NAME unit.arena_plan COMMAND test_arena_plan)

add_executable(test_runtime_boot unit/test_runtime_boot.cpp)
target_link_libraries(test_runtime_boot PRIVATE shelf::runtime_boot)
add_test(NAME unit.runtime_boot COMMAND test_runtime_boot)

# placement（D2）: 予算と所在の純粋算術（GPU 不要）
add_executable(test_placement unit/test_placement.cpp)
target_link_libraries(test_placement PRIVATE shelf::placement)
add_test(NAME unit.placement COMMAND test_placement)

# D8 の C++ 側: sampling の契約（順序・greedy・seed の純粋性）と JSONL の型
add_executable(test_api_sampling unit/test_api_sampling.cpp)
target_link_libraries(test_api_sampling PRIVATE shelf::api)
add_test(NAME unit.api_sampling COMMAND test_api_sampling)

add_executable(test_api_protocol unit/test_api_protocol.cpp)
target_link_libraries(test_api_protocol PRIVATE shelf::api)
add_test(NAME unit.api_protocol COMMAND test_api_protocol)
# T01 / T03: 停止トークンの解決（generation_config + モデル設定）と
# request_id を照合する cancel。GPU もモデルも使わない純粋関数。
add_executable(test_runtime_generate_stop unit/test_runtime_generate_stop.cpp)
target_link_libraries(test_runtime_generate_stop PRIVATE shelf::runtime_generate)
add_test(NAME unit.runtime_generate_stop COMMAND test_runtime_generate_stop)
# T15: 連続バッチングのスケジューラ（受付・Prefill チャンク・decode バッチ・再利用）
add_executable(test_runtime_batch_scheduler unit/test_runtime_batch_scheduler.cpp)
target_link_libraries(test_runtime_batch_scheduler PRIVATE shelf::runtime_generate)
add_test(NAME unit.runtime_batch_scheduler COMMAND test_runtime_batch_scheduler)

# フェイクエンジンとのセッション（パイプ越し・ポートもモデルも使わない）
add_executable(test_api_session unit/test_api_session.cpp)
target_link_libraries(test_api_session PRIVATE shelf::api)
add_test(NAME unit.api_session COMMAND test_api_session)

add_executable(test_step_graph unit/test_step_graph.cpp)
target_link_libraries(test_step_graph PRIVATE shelf::model_qwen35)
add_test(NAME unit.step_graph COMMAND test_step_graph)

add_executable(test_reference_executor unit/test_reference_executor.cpp)
target_link_libraries(test_reference_executor PRIVATE shelf::backend_cpu)
add_test(NAME unit.reference_executor COMMAND test_reference_executor)

# T17: MoE ブロックの契約（config の MoE 欄・moe_mlp の形状検証・CPU 参照の
# top-k ルーティングと共有エキスパート・step_graph の lowering）
add_executable(test_ir_moe unit/test_ir_moe.cpp)
target_link_libraries(test_ir_moe PRIVATE shelf::backend_cpu shelf::model_qwen35)
add_test(NAME unit.ir_moe COMMAND test_ir_moe)

add_executable(test_paged_kv_abi unit/test_paged_kv_abi.cpp)
target_link_libraries(test_paged_kv_abi PRIVATE shelf::backend_common shelf::runtime_state)
add_test(NAME unit.paged_kv_abi COMMAND test_paged_kv_abi)
# T16: Prefix Cache（最長共通 prefix・参照カウント・LRU・COW・世代）
add_executable(test_runtime_prefix_cache unit/test_runtime_prefix_cache.cpp)
target_link_libraries(test_runtime_prefix_cache PRIVATE shelf::runtime_state)
add_test(NAME unit.runtime_prefix_cache COMMAND test_runtime_prefix_cache)

if(SHELF_ENABLE_HIP)
    add_executable(test_hip_gemm_bf16 unit/test_hip_gemm_bf16.cpp)
    target_link_libraries(test_hip_gemm_bf16 PRIVATE shelf::backend_hip)
    add_test(NAME unit.hip_gemm_bf16 COMMAND test_hip_gemm_bf16)

    add_executable(test_hip_gemm_fp8 unit/test_hip_gemm_fp8.cpp)
    target_link_libraries(test_hip_gemm_fp8 PRIVATE shelf::backend_hip shelf::quant)
    add_test(NAME unit.hip_gemm_fp8 COMMAND test_hip_gemm_fp8)

    # MXFP4 W4A8（E2M1 重み × E4M3 活性化の GEMV）
    add_executable(test_hip_gemm_mxfp4 unit/test_hip_gemm_mxfp4.cpp)
    target_link_libraries(test_hip_gemm_mxfp4 PRIVATE shelf::backend_hip shelf::quant)
    add_test(NAME unit.hip_gemm_mxfp4 COMMAND test_hip_gemm_mxfp4)

    # decode 1 ステップの実行間決定性（実モデルに近い形で 2 回実行してビット比較）
    add_executable(test_hip_decode_determinism unit/test_hip_decode_determinism.cpp)
    target_link_libraries(test_hip_decode_determinism PRIVATE
        shelf::backend_hip
        shelf::model_qwen35
        shelf::compiler)
    add_test(NAME unit.hip_decode_determinism COMMAND test_hip_decode_determinism)

    # N01: decode1（GEMV）とバッチ GEMM の数値差（加算順の違いだけに収まること）
    add_executable(test_hip_gemm_row_parity unit/test_hip_gemm_row_parity.cpp)
    target_link_libraries(test_hip_gemm_row_parity PRIVATE shelf::backend_hip shelf::quant)
    add_test(NAME unit.hip_gemm_row_parity COMMAND test_hip_gemm_row_parity)

    # pipeline-parallel（レイヤ分割）が数値を変えないこと（全ステージ同一 GPU で検証）
    add_executable(test_hip_pipeline unit/test_hip_pipeline.cpp)
    target_link_libraries(test_hip_pipeline PRIVATE
        shelf::backend_hip
        shelf::model_qwen35
        shelf_app_common)
    add_test(NAME unit.hip_pipeline COMMAND test_hip_pipeline)

    # 状態の初期化: reset を挟んで同じ列を流すと一致するか（serve の欠陥の再現）
    add_executable(test_hip_reset unit/test_hip_reset.cpp)
    target_link_libraries(test_hip_reset PRIVATE shelf::backend_hip shelf::model_qwen35)
    add_test(NAME unit.hip_reset COMMAND test_hip_reset)

    add_executable(test_hip_attention unit/test_hip_attention.cpp)
    target_link_libraries(test_hip_attention PRIVATE shelf::backend_hip)
    add_test(NAME unit.hip_attention COMMAND test_hip_attention)

    add_executable(test_hip_gdn unit/test_hip_gdn.cpp)
    target_link_libraries(test_hip_gdn PRIVATE shelf::backend_hip)
    add_test(NAME unit.hip_gdn COMMAND test_hip_gdn)

    # GDN のチャンク版（FP16 WMMA）が逐次版と丸めの範囲で一致すること
    add_executable(test_hip_gdn_chunk unit/test_hip_gdn_chunk.cpp)
    target_link_libraries(test_hip_gdn_chunk PRIVATE shelf::backend_hip)
    add_test(NAME unit.hip_gdn_chunk COMMAND test_hip_gdn_chunk)

    add_executable(test_hip_elementwise_kernels unit/test_hip_elementwise_kernels.cpp)
    target_link_libraries(test_hip_elementwise_kernels PRIVATE shelf::backend_hip)
    add_test(NAME unit.hip_elementwise_kernels COMMAND test_hip_elementwise_kernels)

    add_executable(test_hip_step_executor unit/test_hip_step_executor.cpp)
    target_link_libraries(test_hip_step_executor PRIVATE shelf::backend_hip
        shelf::model_qwen35)
    add_test(NAME unit.hip_step_executor COMMAND test_hip_step_executor)

    # T17: MoE ブロック（router + top-k + エキスパート + 共有）の HIP 実行が
    # CPU リファレンスと一致すること。
    add_executable(test_hip_moe unit/test_hip_moe.cpp)
    target_link_libraries(test_hip_moe PRIVATE shelf::backend_hip shelf::backend_cpu)
    add_test(NAME unit.hip_moe COMMAND test_hip_moe)

    # T17: MXFP4（W4A4）エキスパートの GPU / CPU 分担が同じ値になること。
    # N08: 複数行のルーター logits が 1 行版とビット一致すること
    add_executable(test_hip_moe_router unit/test_hip_moe_router.cpp)
    target_link_libraries(test_hip_moe_router PRIVATE shelf::backend_hip)
    add_test(NAME unit.hip_moe_router COMMAND test_hip_moe_router)

    add_executable(test_hip_moe_mxfp4 unit/test_hip_moe_mxfp4.cpp)
    target_link_libraries(test_hip_moe_mxfp4 PRIVATE shelf::backend_hip shelf::quant)
    add_test(NAME unit.hip_moe_mxfp4 COMMAND test_hip_moe_mxfp4)
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

# 性能の回帰ゲート（壁時計）。実行ファイルは常に作り、ctest への登録だけ opt-in。
if(SHELF_ENABLE_HIP)
    add_executable(test_perf_gate perf/test_perf_gate.cpp)
    target_link_libraries(test_perf_gate PRIVATE shelf::foundation)
    if(SHELF_ENABLE_PERF_GATE)
        add_test(NAME gate.perf.hip
                 COMMAND test_perf_gate $<TARGET_FILE:shelf-bench>
                         ${CMAKE_CURRENT_SOURCE_DIR}/perf/baseline_hip.txt)
        set_tests_properties(gate.perf.hip PROPERTIES
            LABELS perf
            TIMEOUT 900)
    endif()
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

if(SHELF_ENABLE_HIP)
    # shelf-compute のプロトコル（モデル・GPU 不要。--help と ping/generate の枠）
    add_test(NAME serve.protocol
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/serve_protocol.sh
                     $<TARGET_FILE:shelf-compute>)
endif()

# Python 面（apps/server）の pytest。aiohttp / pytest が無ければ skip（77）。
# `unit.*` と同じ ctest に載せる（CI は `ctest` 1 本で C++ と Python の両方を見る）。
find_program(SHELF_PYTHON NAMES python3 python)
if(SHELF_PYTHON)
    add_test(NAME serve.python
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_python_tests.sh
                     ${SHELF_PYTHON} ${CMAKE_SOURCE_DIR}/apps/server/tests)
    set_tests_properties(serve.python PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 300)
endif()

