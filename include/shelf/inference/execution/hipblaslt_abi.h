#pragma once

#include <cstdio>
#include <dlfcn.h>

namespace shelf::execution {

inline bool hipblaslt_abi_compatible() {
    static const bool compatible = [] {
        void* host = dlopen("libtensilelite-host.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (host == nullptr) return true;
        dlclose(host);
        void* libstdcxx = dlopen("libstdc++.so.6", RTLD_LAZY | RTLD_NOLOAD);
        if (libstdcxx == nullptr) {
            std::fprintf(stderr, "shelf: hipBLASLt C++ ABI 検査に失敗: libstdc++ が読み込まれていません\n");
            return false;
        }
        const char* symbols[] = {"_ZTVN10__cxxabiv117__class_type_infoE",
                                 "_ZTVN10__cxxabiv120__si_class_type_infoE",
                                 "_ZTVN10__cxxabiv121__vmi_class_type_infoE",
                                 "__dynamic_cast"};
        bool ok = true;
        for (const char* symbol : symbols) {
            void* global = dlsym(RTLD_DEFAULT, symbol);
            void* expected = dlsym(libstdcxx, symbol);
            if (global == nullptr || expected == nullptr || global != expected) {
                Dl_info info{};
                const char* owner = global != nullptr && dladdr(global, &info) != 0
                    ? info.dli_fname : "不明";
                std::fprintf(stderr, "shelf: hipBLASLt C++ ABI 検査に失敗: %s の束縛先は %s です。libstdc++ を libc++abi より先に読み込んでください\n", symbol, owner);
                ok = false;
            }
        }
        dlclose(libstdcxx);
        return ok;
    }();
    return compatible;
}

}
