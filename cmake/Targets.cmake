# shelf library targets.
#
# Plan §3.1: one CMake target per module. Targets are NOT necessarily shared
# libraries; Phase 1 uses static/object libraries only.
#
# Dependency direction (enforced by tests/architecture + CMake links):
#   foundation <- model/ir/artifact/quant/weights/... <- api
#
# Only `foundation` has code in the skeleton. Additional targets appear with
# their implementing PRs (PR-003..), never as empty placeholders.

# ---------------------------------------------------------------- foundation
add_library(shelf_foundation STATIC
    src/foundation/foundation.cpp)
add_library(shelf::foundation ALIAS shelf_foundation)

target_include_directories(shelf_foundation PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"              # public facade
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/foundation/include>") # module-internal headers

target_compile_features(shelf_foundation PUBLIC cxx_std_20)

# ---------------------------------------------------------------- backend/common
add_library(shelf_backend_common STATIC
    src/backend/common/backend_common.cpp
    src/backend/common/paged_kv.cpp)
add_library(shelf::backend_common ALIAS shelf_backend_common)
target_include_directories(shelf_backend_common PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/backend/common/include>")
target_link_libraries(shelf_backend_common PUBLIC shelf::foundation)

# ---------------------------------------------------------------- runtime/state
# Prefix-sharing identity contract (pool/eviction arrive with batching).
add_library(shelf_runtime_state STATIC
    src/runtime/state/prefix_key.cpp)
add_library(shelf::runtime_state ALIAS shelf_runtime_state)
target_include_directories(shelf_runtime_state PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/runtime/state/include>")
target_link_libraries(shelf_runtime_state PUBLIC shelf::foundation)

# ---------------------------------------------------------------- compiler
# declare 相（D4）: グラフから offsets を決める純粋計算。ヘッダのみ。
add_library(shelf_compiler INTERFACE)
add_library(shelf::compiler ALIAS shelf_compiler)
target_include_directories(shelf_compiler INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/compiler/include>")
target_link_libraries(shelf_compiler INTERFACE shelf::ir shelf::foundation)

# ---------------------------------------------------------------- runtime/boot
# 起動の分離（D1）: 構造化された probe と、要求機能による受理判定（純粋）。
add_library(shelf_runtime_boot STATIC
    src/runtime/boot/probe.cpp)
add_library(shelf::runtime_boot ALIAS shelf_runtime_boot)
target_include_directories(shelf_runtime_boot PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/runtime/boot/include>")
target_link_libraries(shelf_runtime_boot PUBLIC shelf::backend_common shelf::foundation)

# ---------------------------------------------------------------- runtime/pipeline
# pipeline-parallel の編成（D3/エンジン側）。ステージの生成は apps が行い、
# ここは非所有のポインタで受け取って順に回すだけ（HIP を知らない）。
add_library(shelf_runtime_pipeline STATIC
    src/runtime/pipeline/pipeline.cpp)
add_library(shelf::runtime_pipeline ALIAS shelf_runtime_pipeline)
target_include_directories(shelf_runtime_pipeline PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/runtime/pipeline/include>")
target_link_libraries(shelf_runtime_pipeline PUBLIC
    shelf::backend_common
    shelf::model_qwen35
    shelf::foundation)

# ---------------------------------------------------------------- runtime/load
# 読み取りの分離（D3）: 供給元（TensorSource）から検証・実体化まで。
# artifact には依存しない（供給元はアプリが実装して差し込む）。
add_library(shelf_runtime_load STATIC
    src/runtime/load/materialize.cpp)
add_library(shelf::runtime_load ALIAS shelf_runtime_load)
target_include_directories(shelf_runtime_load PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/runtime/load/include>")
target_link_libraries(shelf_runtime_load PUBLIC
    shelf::foundation
    shelf::quant
    shelf::nlohmann)

# ---------------------------------------------------------------- backend/hip
# Host-side runtime API + device kernels. HIP headers/types stay confined here.
if(SHELF_ENABLE_HIP)
    file(GLOB_RECURSE SHELF_HIP_KERNEL_SRCS CONFIGURE_DEPENDS
         "${PROJECT_SOURCE_DIR}/src/backend/hip/kernels/*.hip")
    add_library(shelf_backend_hip STATIC
        src/backend/hip/hip_api.cpp
        src/backend/hip/step_executor.cpp
        ${SHELF_HIP_KERNEL_SRCS})
    if(SHELF_EXECUTOR_O0)
        set_source_files_properties(src/backend/hip/step_executor.cpp
            PROPERTIES COMPILE_OPTIONS "-O0")
    endif()
    if(SHELF_EXECUTOR_SANITIZE)
        set_source_files_properties(src/backend/hip/step_executor.cpp
            PROPERTIES COMPILE_OPTIONS "${SHELF_EXECUTOR_SANITIZE_FLAGS}")
    endif()
    add_library(shelf::backend_hip ALIAS shelf_backend_hip)
    target_include_directories(shelf_backend_hip PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/backend/hip/include>")
    target_link_libraries(shelf_backend_hip
        PUBLIC shelf::backend_common shelf::compiler shelf::ir shelf::quant hip::host
        PRIVATE hip::device)
    # CMake passes CMAKE_HIP_ARCHITECTURES; pin it explicitly for determinism.
    target_compile_options(shelf_backend_hip PRIVATE
        "--offload-arch=${CMAKE_HIP_ARCHITECTURES}")
endif()

# ---------------------------------------------------------------- ir
add_library(shelf_ir INTERFACE)
add_library(shelf::ir ALIAS shelf_ir)
target_include_directories(shelf_ir INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/ir/include>")
target_link_libraries(shelf_ir INTERFACE shelf::foundation)

# ---------------------------------------------------------------- backend/cpu
# CPU reference executor (correctness golden; SIMD kernels arrive later).
add_library(shelf_backend_cpu STATIC
    src/backend/cpu/reference_executor.cpp)
add_library(shelf::backend_cpu ALIAS shelf_backend_cpu)
target_include_directories(shelf_backend_cpu PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/backend/cpu/include>")
target_link_libraries(shelf_backend_cpu PUBLIC shelf::ir shelf::foundation)

# ---------------------------------------------------------------- model/qwen35
add_library(shelf_model_qwen35 STATIC
    src/model/qwen35/text_config.cpp
    src/model/qwen35/step_graph.cpp)
add_library(shelf::model_qwen35 ALIAS shelf_model_qwen35)
target_include_directories(shelf_model_qwen35 PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/model/qwen35/include>")
target_link_libraries(shelf_model_qwen35 PUBLIC shelf::ir)
target_link_libraries(shelf_model_qwen35 PRIVATE shelf::nlohmann)

# ---------------------------------------------------------------- artifact
add_library(shelf_artifact STATIC
    src/artifact/safetensors/reader.cpp)
add_library(shelf::artifact ALIAS shelf_artifact)
target_include_directories(shelf_artifact PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/artifact/common/include>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/artifact/safetensors/include>")
target_link_libraries(shelf_artifact PUBLIC shelf::foundation PRIVATE shelf::nlohmann)

# ---------------------------------------------------------------- quant
# FP8 W8A8（compressed-tensors / float-quantized）の契約: E4M3 コーデックと
# 断片順 relayout（ホスト側のみ）。デバイス側の e4m3 変換命令は backend/hip の
# e4m3_codec に残り、意味論（スケールの粒度）はここが定義する。
add_library(shelf_quant STATIC
    src/quant/fp8_w8a8.cpp)
add_library(shelf::quant ALIAS shelf_quant)
target_include_directories(shelf_quant PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/quant/include>")
target_link_libraries(shelf_quant PUBLIC shelf::foundation)

# ---------------------------------------------------------------- weights
# Header-only Canonical Catalog for now (binding/materialization later).
add_library(shelf_weights INTERFACE)
add_library(shelf::weights ALIAS shelf_weights)
target_include_directories(shelf_weights INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/weights/catalog/include>")
target_link_libraries(shelf_weights INTERFACE shelf::artifact shelf::foundation)

# ---------------------------------------------------------------- D7 整合確認
# 1 ディレクトリ ≒ 1 ターゲット。定義の最後に一度だけ検査する（configure で止まる）。
include("${CMAKE_CURRENT_LIST_DIR}/CheckModuleTargets.cmake")
