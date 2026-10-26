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
