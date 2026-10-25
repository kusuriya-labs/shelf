# shelf build options.
#
# Option names follow docs plan (shelf_rebuild_plan §Phase 1).

option(SHELF_ENABLE_SYCL "Enable SYCL backend targets" OFF)
option(SHELF_BUILD_TESTS "Build unit / architecture tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(SHELF_ENABLE_SYCL)
    # Phase 5 will introduce backend/sycl. Until then the flag is inert so
    # CI presets stay valid.
    message(WARNING
        "SHELF_ENABLE_SYCL=OFF is currently the only supported setting; "
        "the SYCL backend is not implemented yet (skeleton). Ignoring.")
endif()

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()
