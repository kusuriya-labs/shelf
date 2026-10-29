# shelf build options.
#
# Option names follow docs plan (shelf_rebuild_plan §Phase 1).

option(SHELF_ENABLE_HIP "Enable HIP (AMD ROCm) backend targets" OFF)
option(SHELF_BUILD_TESTS "Build unit / architecture tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 性能の回帰ゲート（壁時計）。既定 OFF:
#   * 時間は装置・負荷でぶれるため、通常の ctest には入れない
#   * 有効化: `cmake -DSHELF_ENABLE_PERF_GATE=ON ...` で configure し、
#     `ctest -R gate.perf` を空いている GPU で実行する（基準の取り直しは
#     `test_perf_gate --dump`）
option(SHELF_ENABLE_PERF_GATE "Add the wall-clock performance regression gate" OFF)
