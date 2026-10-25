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
    src/backend/common/backend_common.cpp)
add_library(shelf::backend_common ALIAS shelf_backend_common)
target_include_directories(shelf_backend_common PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/backend/common/include>")
target_link_libraries(shelf_backend_common PUBLIC shelf::foundation)

# ---------------------------------------------------------------- backend/hip
# Host-side runtime API from ordinary C++ TUs (no kernels yet; Phase 5 will
# add HIP-language TUs for device code). HIP headers/types stay confined here.
if(SHELF_ENABLE_HIP)
    add_library(shelf_backend_hip STATIC
        src/backend/hip/hip_api.cpp)
    add_library(shelf::backend_hip ALIAS shelf_backend_hip)
    target_include_directories(shelf_backend_hip PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/backend/hip/include>")
    target_link_libraries(shelf_backend_hip PUBLIC shelf::backend_common hip::host)
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

# ---------------------------------------------------------------- weights
# Header-only Canonical Catalog for now (binding/materialization later).
add_library(shelf_weights INTERFACE)
add_library(shelf::weights ALIAS shelf_weights)
target_include_directories(shelf_weights INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/weights/catalog/include>")
target_link_libraries(shelf_weights INTERFACE shelf::artifact shelf::foundation)

# ---------------------------------------------------------------- api (PR-008+)
# shelf_api will be added when a facade exists. include/shelf/ stays minimal.
