# shelf apps — thin end-to-end drivers over the library modules.

# apps/common — shelf-run / shelf-bench が共有するモデル読取・重みアップロード。
# （D3: runtime/load/ への分離の対象）
add_library(shelf_app_common STATIC common/app_support.cpp)
target_include_directories(shelf_app_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(shelf_app_common PUBLIC
    shelf::model_qwen35
    shelf::artifact
    shelf::nlohmann)

add_executable(shelf-run shelf_run.cpp)
target_link_libraries(shelf-run PRIVATE
    shelf_app_common
    shelf::backend_cpu)

# shelf-validate — 無損失経路と量子化経路の比較検証ドライバ
# (draft-docs/shelf-validate.md, V0: アプリ骨格)
add_executable(shelf-validate shelf_validate.cpp)
target_link_libraries(shelf-validate PRIVATE
    shelf::model_qwen35
    shelf::backend_cpu
    shelf::artifact
    shelf::nlohmann)

# shelf-bench — pp / tg / kernel <name> の計測エントリ
# (docs/spec/bench.md / docs/guide/bench.md)。HIP 無しのビルドでも --help の契約
# （GPU 非依存で exit 0）を守れるようにする。
add_executable(shelf-bench
    bench/main.cpp
    bench/bench_common.cpp
    bench/pp.cpp
    bench/tg.cpp
    bench/kernel.cpp)
target_link_libraries(shelf-bench PRIVATE shelf_app_common)

if(SHELF_ENABLE_HIP)
    target_link_libraries(shelf-run PRIVATE shelf::backend_hip)
    target_compile_definitions(shelf-run PRIVATE SHELF_HAVE_HIP=1)
    target_link_libraries(shelf-validate PRIVATE shelf::backend_hip)
    target_compile_definitions(shelf-validate PRIVATE SHELF_HAVE_HIP=1)
    target_link_libraries(shelf-bench PRIVATE shelf::backend_hip)
    target_compile_definitions(shelf-bench PRIVATE SHELF_HAVE_HIP=1)
endif()
