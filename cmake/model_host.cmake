# Resident model host: internal runtime component that owns GPU model weights
# and hands HIP IPC handles to compute / test worker processes.
#
# Not a user-facing entrypoint (see cmake/apps.cmake); it is started by
# phaseshift-launcher, an operator, or a session runner, never by
# phaseshift-server or phaseshift-compute.

add_executable(phaseshift-model-host
    src/apps/model_host/main.cpp
)
phaseshift_set_rocm_rpath(phaseshift-model-host)
target_compile_features(phaseshift-model-host PRIVATE cxx_std_20)
target_link_libraries(phaseshift-model-host PRIVATE phaseshift_model_source phaseshift_resident)
target_compile_options(phaseshift-model-host PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift-model-host)
