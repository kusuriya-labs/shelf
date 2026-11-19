#include <shelf/inference/execution/hipblaslt_abi.h>

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <typeinfo>

struct Base { virtual ~Base() = default; };
struct Derived : Base {};
struct ShelfException : std::exception {};

void throw_shelf() { throw ShelfException{}; }
void throw_std() { throw std::runtime_error("test"); }

int main(int argc, char** argv) {
    void* lt = dlopen(SHELF_TEST_HIPBLASLT, RTLD_NOW | RTLD_GLOBAL);
    if (lt == nullptr) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }
    const bool expected = !(argc > 1 && std::strcmp(argv[1], "reject") == 0);
    if (shelf::execution::hipblaslt_abi_compatible() != expected) return 2;
    if (!expected) return 0;
    Derived derived;
    Base* base = &derived;
    if (dynamic_cast<Derived*>(base) != &derived || typeid(*base) != typeid(Derived)) return 3;
    try { throw ShelfException{}; } catch (const ShelfException&) {} catch (...) { return 4; }
    try { throw std::runtime_error("test"); } catch (const std::exception&) {} catch (...) { return 5; }
    try { throw std::bad_alloc{}; } catch (const std::bad_alloc&) {} catch (...) { return 6; }
    void* fixture = dlopen(SHELF_TEST_ABI_FIXTURE, RTLD_NOW | RTLD_LOCAL);
    if (fixture == nullptr) return 7;
    using Invoke = void (*)(void (*)());
    using Raise = void (*)();
    auto invoke = reinterpret_cast<Invoke>(dlsym(fixture, "shelf_abi_invoke"));
    auto raise = reinterpret_cast<Raise>(dlsym(fixture, "shelf_abi_bad_alloc"));
    if (invoke == nullptr || raise == nullptr) return 8;
    try { invoke(throw_shelf); } catch (const ShelfException&) {} catch (...) { return 9; }
    try { invoke(throw_std); } catch (const std::exception&) {} catch (...) { return 10; }
    try { raise(); } catch (const std::bad_alloc&) {} catch (...) { return 11; }
    dlclose(fixture);
    dlclose(lt);
    return 0;
}
