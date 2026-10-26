# shelf apps — thin end-to-end drivers over the library modules.
add_executable(shelf-run shelf_run.cpp)
target_link_libraries(shelf-run PRIVATE
    shelf::model_qwen35
    shelf::backend_cpu
    shelf::artifact
    shelf::nlohmann)
if(SHELF_ENABLE_HIP)
    target_link_libraries(shelf-run PRIVATE shelf::backend_hip)
    target_compile_definitions(shelf-run PRIVATE SHELF_HAVE_HIP=1)
endif()
if(SHELF_ENABLE_SYCL)
    target_link_libraries(shelf-run PRIVATE shelf::backend_sycl)
    target_compile_definitions(shelf-run PRIVATE SHELF_HAVE_SYCL=1)
    # the TU references SYCL types via the backend headers; route it through
    # the acpp launcher so glue headers compile with the right flags
    add_sycl_to_target(TARGET shelf-run SOURCES shelf_run.cpp)
endif()
