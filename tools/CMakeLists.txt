# shelf tools — developer-facing GPU benches (not part of the library).
add_executable(bench_gemv_hip bench_gemv_hip.cpp)
target_link_libraries(bench_gemv_hip PRIVATE shelf::backend_hip)
