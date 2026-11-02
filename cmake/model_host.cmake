# Resident model host: internal runtime component that owns GPU model weights
# and hands HIP IPC handles to compute / test worker processes.
#
# Not a user-facing entrypoint (see cmake/apps.cmake); it is started by
# shelf-launcher, an operator, or a session runner, never by
# shelf-server or shelf-compute.

add_executable(shelf-model-host
    src/apps/model_host/main.cpp
)
shelf_set_rocm_rpath(shelf-model-host)
target_compile_features(shelf-model-host PRIVATE cxx_std_20)
target_link_libraries(shelf-model-host PRIVATE shelf_model_source shelf_resident)
target_compile_options(shelf-model-host PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
shelf_set_hip_archs(shelf-model-host)
