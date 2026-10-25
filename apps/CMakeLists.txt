# shelf apps — thin end-to-end drivers over the library modules.
add_executable(shelf-run shelf_run.cpp)
target_link_libraries(shelf-run PRIVATE
    shelf::model_qwen35
    shelf::backend_cpu
    shelf::artifact
    shelf::nlohmann)
