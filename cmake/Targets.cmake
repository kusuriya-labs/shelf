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

# ---------------------------------------------------------------- api (PR-008+)
# shelf_api will be added when a facade exists. include/shelf/ stays minimal.
