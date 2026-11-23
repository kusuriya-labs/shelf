#include "serve_host_memory.h"

#include <cstdio>
#include <cstring>

namespace {

bool check(const char* name, const char* expected) {
    const char* actual = std::getenv(name);
    if (actual == nullptr && expected == nullptr) return true;
    if (actual != nullptr && expected != nullptr && std::strcmp(actual, expected) == 0) return true;
    std::fprintf(stderr, "%s: expected %s, got %s\n", name, expected == nullptr ? "(unset)" : expected,
                 actual == nullptr ? "(unset)" : actual);
    return false;
}

}

int main() {
    constexpr const char* userptr = "HSA_USERPTR_FOR_PAGED_MEM";
    constexpr const char* paged_svm = "HSA_SVM_FOR_PAGED_MEM";
    constexpr const char* svm = "HSA_USE_SVM";
    if (unsetenv(userptr) != 0 || unsetenv(paged_svm) != 0 || unsetenv(svm) != 0 ||
        !shelf::app::configure_serve_host_memory() ||
        !check(userptr, "0") || !check(svm, "0") || !check(paged_svm, nullptr)) return 1;

    if (setenv(userptr, "1", 1) != 0 || unsetenv(svm) != 0 ||
        !shelf::app::configure_serve_host_memory() ||
        !check(userptr, "1") || !check(svm, nullptr)) return 1;

    if (unsetenv(userptr) != 0 || setenv(svm, "1", 1) != 0 ||
        !shelf::app::configure_serve_host_memory() ||
        !check(userptr, nullptr) || !check(svm, "1")) return 1;

    if (unsetenv(svm) != 0 || setenv(paged_svm, "0", 1) != 0 ||
        !shelf::app::configure_serve_host_memory() ||
        !check(userptr, nullptr) || !check(svm, nullptr) || !check(paged_svm, "0")) return 1;
    return 0;
}
