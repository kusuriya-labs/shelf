# Shelf application entrypoints.
#
# The three user-facing executables:
#   shelf-compute   raw token ID in/out inference engine
#   shelf-server    OpenAI-compatible serving product (aiohttp adapter)
#   shelf-bench     E2E benchmark (pp / tg) + GEMM kernel micro-benchmark

# Inference engine (raw token IDs in, GENERATED_IDS out).
add_executable(shelf-compute
    src/apps/compute/main.hip
    src/apps/compute/compute_cli.cpp
    src/apps/compute/single_request.hip
    src/apps/compute/stdio_server.hip
    src/apps/compute/compute_runtime.hip
)
shelf_set_rocm_rpath(shelf-compute)
target_compile_features(shelf-compute PRIVATE cxx_std_20)
target_include_directories(shelf-compute SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
target_include_directories(shelf-compute PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(shelf-compute PRIVATE shelf)
shelf_set_hip_archs(shelf-compute)

# Server product: Python entrypoint plus the OpenAI protocol adapter, the async
# compute client and the shared chat codec. Everything the server needs lives
# under build/shelf-server-lib/; nothing is imported from the source tree
# at runtime.
set(SHELF_SERVER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/src/apps/server/shelf_server.py")
set(SHELF_SERVER_TARGET "${CMAKE_CURRENT_BINARY_DIR}/shelf-server")
set(SHELF_SERVER_LIB "${CMAKE_CURRENT_BINARY_DIR}/shelf-server-lib")
set(SHELF_SERVERLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/apps/server/serverlib")
set(SHELF_CHAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/apps/common/shelf_chat")
set(SHELF_CHAT_STAGE_DIR "${SHELF_SERVER_LIB}/common/shelf_chat")
add_custom_command(
    OUTPUT "${SHELF_SERVER_TARGET}"
    COMMAND ${CMAKE_COMMAND} -E copy "${SHELF_SERVER_SCRIPT}" "${SHELF_SERVER_TARGET}"
    COMMAND /bin/sh -c "chmod +x '${SHELF_SERVER_TARGET}'"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${SHELF_SERVER_LIB}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${SHELF_SERVER_LIB}/serverlib"
        "${SHELF_CHAT_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${SHELF_SERVERLIB_DIR}" "${SHELF_SERVER_LIB}/serverlib"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${SHELF_CHAT_DIR}" "${SHELF_CHAT_STAGE_DIR}"
    DEPENDS
        "${SHELF_SERVER_SCRIPT}"
        "${SHELF_SERVERLIB_DIR}/__init__.py"
        "${SHELF_SERVERLIB_DIR}/openai_protocol.py"
        "${SHELF_SERVERLIB_DIR}/compute_client.py"
        "${SHELF_SERVERLIB_DIR}/chat_service.py"
        "${SHELF_CHAT_DIR}/__init__.py"
        "${SHELF_CHAT_DIR}/codec.py"
        "${SHELF_CHAT_DIR}/reasoning.py"
    COMMENT "Staging shelf-server"
)
add_custom_target(shelf-server-stage ALL DEPENDS "${SHELF_SERVER_TARGET}")

# E2E benchmark (pp / tg)
# + Qwen35 micro-benchmarks
if(SHELF_BUILD_BENCHMARKS)
    add_executable(shelf-bench
        src/apps/bench/main.cpp
        src/apps/bench/pp.hip
        src/apps/bench/tg.hip
        src/apps/bench/quality.hip
        src/apps/bench/batch.hip
        src/apps/bench/gemm.hip
        src/apps/bench/exl3_int8.hip
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
    shelf_set_rocm_rpath(shelf-bench)
    target_compile_features(shelf-bench PRIVATE cxx_std_20)
    target_link_libraries(shelf-bench PRIVATE shelf)
    shelf_set_hip_archs(shelf-bench)
    target_include_directories(shelf-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    execute_process(
        COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --short HEAD
        OUTPUT_VARIABLE SHELF_GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(SHELF_GIT_SHA)
        target_compile_definitions(
            shelf-bench PRIVATE SHELF_GIT_SHA="${SHELF_GIT_SHA}")
    endif()
    if(ROCTX_FOUND)
        target_compile_definitions(shelf-bench PRIVATE SHELF_HAS_ROCTX=1)
        target_include_directories(shelf-bench PRIVATE "${ROCTX_INCLUDE_DIR}")
        target_link_libraries(shelf-bench PRIVATE "${ROCTX_LIBRARY}")
    endif()
endif()
