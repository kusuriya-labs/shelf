#include <new>

extern "C" void shelf_abi_invoke(void (*callback)()) { callback(); }
extern "C" void shelf_abi_bad_alloc() { throw std::bad_alloc{}; }
