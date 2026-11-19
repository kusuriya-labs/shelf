#include <shelf/models/hy4_flash/snapshot/transport.h>
#include <cstdio>
#include <chrono>
#include <stdexcept>
#include <vector>

int main() {
    try {
        using namespace shelf::hy4::snapshot;
        std::vector<uint8_t> first(17 * 1024 * 1024 + 11);
        for (size_t i = 0; i < first.size(); ++i)
            first[i] = (i * 157 + i / 37) % 256;
        const auto start = std::chrono::steady_clock::now();
        save(first);
        const auto saved = std::chrono::steady_clock::now();
        auto loaded = load(first.size());
        const auto restored = std::chrono::steady_clock::now();
        if (loaded != first) throw std::runtime_error("最初の世代が一致しません");
        double save_s = std::chrono::duration<double>(saved - start).count();
        double load_s = std::chrono::duration<double>(restored - saved).count();
        std::printf("bytes=%zu save_s=%.6f save_GBps=%.3f load_s=%.6f load_GBps=%.3f\n",
                    first.size(), save_s, first.size() / save_s / 1e9, load_s, first.size() / load_s / 1e9);
        bool rejected = false;
        try { (void)load(first.size() - 1); } catch (const std::exception &) { rejected = true; }
        if (!rejected) throw std::runtime_error("受信容量の超過が拒否されませんでした");
        std::vector<uint8_t> oversized(33 * 1024 * 1024, 42);
        rejected = false;
        try { save(oversized); } catch (const std::exception &) { rejected = true; }
        if (!rejected) throw std::runtime_error("送信容量の超過が拒否されませんでした");
        if (load(first.size()) != first) throw std::runtime_error("拒否後に保存世代が変化しました");
        std::vector<uint8_t> second(1024 * 1024 + 3, 123);
        save(second);
        if (load(second.size()) != second) throw std::runtime_error("次の世代が一致しません");
        std::puts("状態 RDMA の往復・容量拒否・世代切り替え成功");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
