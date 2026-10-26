# shelf tools — developer-facing GPU benches (not part of the library).
add_executable(bench_gemv_hip bench_gemv_hip.cpp)
target_link_libraries(bench_gemv_hip PRIVATE shelf::backend_hip)

# tile-refactor bitwise snapshots (procedure §2.3): record/verify against a dir.
add_executable(hip_tile_snapshots hip_tile_snapshots.cpp)
target_link_libraries(hip_tile_snapshots PRIVATE shelf::backend_hip)
add_test(NAME tools.hip_tile_snapshots_selfcheck
         COMMAND hip_tile_snapshots roundtrip)
