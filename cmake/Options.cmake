# shelf build options.
#
# Option names follow docs plan (shelf_rebuild_plan §Phase 1).

option(SHELF_ENABLE_SYCL "Enable SYCL backend targets" OFF)
option(SHELF_ENABLE_HIP "Enable HIP (AMD ROCm) backend targets" OFF)
option(SHELF_BUILD_TESTS "Build unit / architecture tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(SHELF_ENABLE_SYCL)
    # backend/sycl is implemented (SYCL port of backend/hip). It compiles
    # through AdaptiveCpp (hipSYCL): cmake/Dependencies.cmake resolves
    # AdaptiveCppConfig, and the backend's TUs are routed through the acpp
    # driver by add_sycl_to_target. Choose the SYCL compilation targets with
    # the SHELF_SYCL_TARGETS cache variable (default: generic SSCP;
    # e.g. "hip:gfx1201" for AOT on AMD, "omp" for CPU-only correctness
    # testing — wave32 kernels report unsupported on non-32-lane devices).
    set(SHELF_SYCL_TARGETS "generic" CACHE STRING
        "AdaptiveCpp --acpp-targets value for the SYCL backend")
endif()

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()
