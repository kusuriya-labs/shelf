#include <atomic>
#include <cstdlib>
#include <dlfcn.h>
#include <time.h>

extern "C" int clock_gettime(clockid_t id, struct timespec *value) {
    static auto actual = reinterpret_cast<int (*)(clockid_t, struct timespec *)>(dlsym(RTLD_NEXT, "clock_gettime"));
    static std::atomic<int> calls{};
    int result = actual(id, value);
    if (!result && id == CLOCK_MONOTONIC && std::getenv("SHELF_TEST_TIMEOUT") && calls.fetch_add(1) > 0)
        value->tv_sec += 11;
    return result;
}
