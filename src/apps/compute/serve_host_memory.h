#pragma once

#include <cstdlib>

namespace shelf::app {

inline bool configure_serve_host_memory() {
    if (std::getenv("HSA_USERPTR_FOR_PAGED_MEM") != nullptr ||
        std::getenv("HSA_SVM_FOR_PAGED_MEM") != nullptr ||
        std::getenv("HSA_USE_SVM") != nullptr) return true;

    return setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) == 0 &&
           setenv("HSA_USE_SVM", "0", 0) == 0;
}

}
