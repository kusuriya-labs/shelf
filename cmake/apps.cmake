# PhaseShift application entrypoints.
#
# The four user-facing executables:
#   phaseshift-compute   raw token ID in/out inference engine
#   phaseshift-launcher  resident model host orchestrator (Python)
#   phaseshift-server    OpenAI-compatible serving product (aiohttp adapter)
#   phaseshift-bench     E2E benchmark (pp / tg) + GEMM kernel micro-benchmark

# Inference engine (raw token IDs in, GENERATED_IDS out).
add_executable(phaseshift-compute
    src/apps/compute/main.hip
    src/apps/compute/compute_cli.cpp
    src/apps/compute/single_request.hip
    src/apps/compute/stdio_server.hip
    src/apps/compute/compute_runtime.hip
)
phaseshift_set_rocm_rpath(phaseshift-compute)
target_compile_features(phaseshift-compute PRIVATE cxx_std_20)
target_include_directories(phaseshift-compute SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
target_include_directories(phaseshift-compute PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(phaseshift-compute PRIVATE phaseshift)
phaseshift_set_hip_archs(phaseshift-compute)

# Launcher: Python script copied into the build directory.
set(PHASESHIFT_LAUNCHER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/src/apps/launcher/phaseshift_launcher.py")
set(PHASESHIFT_LAUNCHER_TARGET "${CMAKE_CURRENT_BINARY_DIR}/phaseshift-launcher")
add_custom_command(
    OUTPUT "${PHASESHIFT_LAUNCHER_TARGET}"
    COMMAND ${CMAKE_COMMAND} -E copy "${PHASESHIFT_LAUNCHER_SCRIPT}" "${PHASESHIFT_LAUNCHER_TARGET}"
    COMMAND /bin/sh -c "chmod +x '${PHASESHIFT_LAUNCHER_TARGET}'"
    DEPENDS "${PHASESHIFT_LAUNCHER_SCRIPT}"
    COMMENT "Staging phaseshift-launcher"
)
add_custom_target(phaseshift-launcher-stage ALL DEPENDS "${PHASESHIFT_LAUNCHER_TARGET}")

# Server product: Python launcher plus the OpenAI protocol adapter, the async
# compute client and the shared chat codec. Everything the server needs lives
# under build/phaseshift-server-lib/; nothing is imported from the source tree
# at runtime.
set(PHASESHIFT_SERVER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/src/apps/server/phaseshift_server.py")
set(PHASESHIFT_SERVER_TARGET "${CMAKE_CURRENT_BINARY_DIR}/phaseshift-server")
set(PHASESHIFT_SERVER_LIB "${CMAKE_CURRENT_BINARY_DIR}/phaseshift-server-lib")
set(PHASESHIFT_SERVERLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/apps/server/serverlib")
set(PHASESHIFT_CHAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/apps/common/phaseshift_chat")
set(PHASESHIFT_CHAT_STAGE_DIR "${PHASESHIFT_SERVER_LIB}/common/phaseshift_chat")
add_custom_command(
    OUTPUT "${PHASESHIFT_SERVER_TARGET}"
    COMMAND ${CMAKE_COMMAND} -E copy "${PHASESHIFT_SERVER_SCRIPT}" "${PHASESHIFT_SERVER_TARGET}"
    COMMAND /bin/sh -c "chmod +x '${PHASESHIFT_SERVER_TARGET}'"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${PHASESHIFT_SERVER_LIB}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${PHASESHIFT_SERVER_LIB}/serverlib"
        "${PHASESHIFT_CHAT_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${PHASESHIFT_SERVERLIB_DIR}" "${PHASESHIFT_SERVER_LIB}/serverlib"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${PHASESHIFT_CHAT_DIR}" "${PHASESHIFT_CHAT_STAGE_DIR}"
    DEPENDS
        "${PHASESHIFT_SERVER_SCRIPT}"
        "${PHASESHIFT_SERVERLIB_DIR}/__init__.py"
        "${PHASESHIFT_SERVERLIB_DIR}/openai_protocol.py"
        "${PHASESHIFT_SERVERLIB_DIR}/compute_client.py"
        "${PHASESHIFT_SERVERLIB_DIR}/chat_service.py"
        "${PHASESHIFT_CHAT_DIR}/__init__.py"
        "${PHASESHIFT_CHAT_DIR}/codec.py"
        "${PHASESHIFT_CHAT_DIR}/reasoning.py"
    COMMENT "Staging phaseshift-server"
)
add_custom_target(phaseshift-server-stage ALL DEPENDS "${PHASESHIFT_SERVER_TARGET}")

# E2E benchmark (pp / tg)
# + Qwen35 micro-benchmarks
if(PHASESHIFT_BUILD_BENCHMARKS)
    add_executable(phaseshift-bench
        src/apps/bench/main.cpp
        src/apps/bench/pp.hip
        src/apps/bench/qwen4exp.hip
        src/apps/bench/tg.hip
        src/apps/bench/quality.hip
        src/apps/bench/batch.hip
        src/apps/bench/gemm.hip
        src/apps/bench/activation_quantize.hip
        src/apps/bench/rmsnorm.hip
        src/apps/bench/l2_normalize.hip
        src/apps/bench/kv_append.hip
        src/apps/bench/paged_attention.hip
        src/apps/bench/rope.hip
        src/apps/bench/gdn_recurrence.hip
        src/apps/bench/gdn_conv1d.hip
        src/apps/bench/elementwise.hip
        src/apps/bench/embedding.hip
        src/apps/bench/sampling.hip
        src/apps/bench/gpu_memory.hip
        src/apps/bench/tp_reduce.cpp
    )
    phaseshift_set_rocm_rpath(phaseshift-bench)
    target_compile_features(phaseshift-bench PRIVATE cxx_std_20)
    target_link_libraries(phaseshift-bench PRIVATE phaseshift)
    phaseshift_set_hip_archs(phaseshift-bench)
    target_include_directories(phaseshift-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    execute_process(
        COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --short HEAD
        OUTPUT_VARIABLE PHASESHIFT_GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PHASESHIFT_GIT_SHA)
        target_compile_definitions(
            phaseshift-bench PRIVATE PHASESHIFT_GIT_SHA="${PHASESHIFT_GIT_SHA}")
    endif()
    if(ROCTX_FOUND)
        target_compile_definitions(phaseshift-bench PRIVATE PHASESHIFT_HAS_ROCTX=1)
        target_include_directories(phaseshift-bench PRIVATE "${ROCTX_INCLUDE_DIR}")
        target_link_libraries(phaseshift-bench PRIVATE "${ROCTX_LIBRARY}")
    endif()
endif()
