#include <shelf/models/hy4_flash/residency.h>
#include <shelf/models/hy4_flash/absorbed.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <atomic>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#ifdef SHELF_HY4_RDMA
#include <openssl/evp.h>
#endif

namespace shelf::hy4 {
namespace {
size_t align(size_t n) { return (n + 255) & ~size_t(255); }
void check(hipError_t e) {
    if (e != hipSuccess)
        throw std::runtime_error(hipGetErrorString(e));
}
}
void *Region::next(size_t length) {
    size_t start = align(cursor);
    if (start > size || length > size - start)
        throw std::runtime_error("作業域の容量が不足しています");
    cursor = start + length;
    return base + start;
}
Residency::Residency(Checkpoint &model, size_t tokens) : model(model), tokens(tokens) {
#ifdef SHELF_HY4_RDMA
    if (model.rdma) {
        engram_client = std::make_unique<rdma::Client>();
        std::array<uint64_t, 4> rows{};
        for (int g = 0; g < 4; ++g)
            for (int size : model.moduli[g])
                rows[g] += size;
        engram_client->validate_rows(rows);
    }
#endif
    for (const auto &[name, dims] : model.shape)
        if (name.ends_with(".self_attn.kv_b_proj"))
            folded[name] = nullptr;
    if (!tokens || tokens > 2097152)
        throw std::runtime_error("KV の予約長が範囲外です");
}
Residency::~Residency() noexcept {
    try {
        shutdown();
    } catch (const std::exception &e) {
        fprintf(stderr, "Hy4 の解放エラー: %s\n", e.what());
    }
}
void Residency::reserve() {
    int count = 0;
    check(hipGetDeviceCount(&count));
    if (count != 3)
        throw std::runtime_error("R9700 と V620 2 台を指定してください");
    int primary = 0, worker = 1;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t props{};
        check(hipGetDeviceProperties(&props, device));
        std::string_view isa = props.gcnArchName;
        int rank;
        if (isa.starts_with("gfx1201") && primary++ == 0)
            rank = 0;
        else if (isa.starts_with("gfx1030") && worker < 3)
            rank = worker++;
        else
            throw std::runtime_error("R9700 と V620 2 台の構成が必要です");
        physical[rank] = device;
        char bus[32]{};
        check(hipDeviceGetPCIBusId(bus, sizeof(bus), device));
        fprintf(stderr, "rank=%d device=%d isa=%s pci=%s\n", rank, device, props.gcnArchName, bus);
    }
    std::array<size_t, 3> available{}, overhead{};
    std::array<int, 3> used{};
    std::array<int64_t, 3> remaining{};
    for (const auto &row : model.owners)
        for (int r : row)
            ++used[r];
    for (int r = 0; r < 3; ++r) {
        check(hipSetDevice(physical[r]));
        size_t total = 0;
        check(hipMemGetInfo(&available[r], &total));
        // 空きを使い切ると arena の hipMalloc が丸めや他プロセスの揺れで std::bad_alloc になるため、余裕を残す。
        // 実測では起動後の追加確保は無く、必要なのは数 MiB（余裕 0.29 MiB で 21 回中 1 回失敗、1.7 MiB 以上は成功）。
        // 失敗が起きるたびに、この値を 2 倍にすること（8 → 16 → 32 → ... MiB）。
        available[r] -= std::min(available[r], size_t(8) << 20);
        overhead[r] = (r ? size_t(512) << 20 : size_t(2) << 30) + matrix_bytes;
        if (!r)
            for (int i = 0; i < 42; ++i) {
                bool sparse = i == 0 || i % 4 == 1;
                overhead[r] += align(294 * (sparse ? tokens : std::min(tokens, size_t(518))));
                if (sparse)
                    overhead[r] += align((tokens / 4 + 12) * 256);
            }
        if (!r)
            for (const auto &[name, ignored] : folded)
                overhead[0] += size_t(model.shape.at(name)[1]) * 512 * 2;
        remaining[r] = int64_t(available[r]) - int64_t(align(model.bytes[r]) + overhead[r]);
    }
    if (remaining[0] + remaining[1] + remaining[2] < 0)
        throw std::bad_alloc();
    auto plan = model.owners;
    int moved = 0;
    for (int expert = 511; expert >= 0; --expert)
        for (int layer = 0; layer < 41; ++layer) {
            int from = plan[layer][expert];
            if (remaining[from] >= 0)
                continue;
            int64_t bytes = 3 * (1536ll * 1536 * model.expert_bits[layer][expert] / 8 + 1536 * 4);
            int dest = std::max_element(remaining.begin(), remaining.end()) - remaining.begin();
            if (remaining[dest] < bytes)
                continue;
            plan[layer][expert] = dest;
            remaining[from] += bytes;
            remaining[dest] -= bytes;
            --used[from];
            ++used[dest];
            ++moved;
        }
    if (*std::min_element(remaining.begin(), remaining.end()) < 0)
        throw std::bad_alloc();
    if (moved)
        model.relocate(plan);
    fprintf(stderr, "relocated_experts=%d rank_counts=%d,%d,%d\n", moved, used[0], used[1], used[2]);
    for (int r = 0; r < 3; ++r) {
        check(hipSetDevice(physical[r]));
        size_t free, total;
        check(hipMemGetInfo(&free, &total));
        size_t work = r ? size_t(512) << 20 : size_t(2) << 30;
        size_t bytes = align(model.bytes[r]) + overhead[r];
        fprintf(stderr, "rank=%d weight=%zu workspace=%zu request=%zu free=%zu total=%zu\n", r, model.bytes[r],
                work, bytes, free, total);
        auto result =
            gpu::GpuArena::create(physical[r], bytes, r ? gpu::DeviceArch::gfx1030 : gpu::DeviceArch::gfx1201);
        if (!result.ok()) {
            if (result.status().code() == Status::Code::insufficient_memory)
                throw std::bad_alloc();
            throw std::runtime_error(result.status().message());
        }
        arenas.push_back(result.release());
        auto allocation = arenas.back().allocate_aligned(bytes, 256);
        if (!allocation.ok())
            throw std::runtime_error(allocation.status().message());
        Region buffer{static_cast<uint8_t *>(allocation.value().data()), bytes};
        weight_base[r] = static_cast<uint8_t *>(buffer.next(model.bytes[r]));
        if (!r)
            for (int i = 0; i < 42; ++i) {
                bool sparse = i == 0 || i % 4 == 1;
                cache[i] = static_cast<uint8_t *>(
                    buffer.next(294 * (sparse ? tokens : std::min(tokens, size_t(518)))));
                if (sparse)
                    index_keys[i] = static_cast<uint16_t *>(buffer.next((tokens / 4 + 12) * 256));
            }
        if (!r)
            for (auto &[name, ptr] : folded)
                ptr = buffer.next(size_t(model.shape.at(name)[1]) * 512 * 2);
        matrices[r] = static_cast<Hy4ExpertProjection *>(buffer.next(matrix_bytes));
        workspace[r] = {static_cast<uint8_t *>(buffer.next(work)), work};
        check(hipMemGetInfo(&free, &total));
        fprintf(stderr, "rank=%d committed=%zu free_after=%zu\n", r, arenas.back().committed(), free);
    }
    check(hipSetDevice(physical[0]));
}
void Residency::upload_experts(int r, int layer) {
    std::vector<Hy4ExpertProjection> table(512 * 3);
    for (int e = 0; e < 512; ++e)
        if (model.owners[layer - 1][e] == r) {
            std::string key =
                "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + ".";
            int kind = 0;
            for (auto suffix : {"gate_proj", "up_proj", "down_proj"}) {
                auto name = key + suffix;
                table[e * 3 + kind++] = {static_cast<const uint16_t *>(device(model.at(name + ".suh"))),
                                         static_cast<const uint16_t *>(device(model.at(name + ".svh"))),
                                         static_cast<const uint32_t *>(device(model.at(name + ".trellis"))),
                                         model.at(name + ".trellis").spec.shape[2] / 16};
            }
        }
    check(hipMemcpy(matrices[r] + (layer - 1) * 1536, table.data(), table.size() * sizeof(table[0]),
                    hipMemcpyHostToDevice));
}
void Residency::upload() {
    auto profile = std::getenv("SHELF_HY4_PROFILE_EP");
    for (int dst = 0; dst < 3; ++dst)
        for (int src = 0; src < 3; ++src) {
            if (src == dst || (dst && src))
                continue;
            int accessible = 0;
            check(hipSetDevice(physical[dst]));
            check(hipDeviceCanAccessPeer(&accessible, physical[dst], physical[src]));
            if (accessible) {
                auto status = hipDeviceEnablePeerAccess(physical[src], 0);
                peer[dst][src] = status == hipSuccess || status == hipErrorPeerAccessAlreadyEnabled;
                owns_peer[dst][src] = status == hipSuccess;
                if (status != hipSuccess)
                    (void)hipGetLastError();
            }
            if (peer[dst][src]) {
                check(hipSetDevice(physical[src]));
                check(
                    hipEventCreateWithFlags(&ready[src][dst], hipEventDisableTiming | hipEventReleaseToSystem));
            }
            if (!peer[dst][src] && !staging)
                check(hipHostMalloc(&staging, max_prefill_chunk * 1536u * 2u, hipHostMallocPortable));
            fprintf(stderr, "transfer=%d->%d method=%s\n", src, dst, peer[dst][src] ? "peer" : "host");
        }
    const auto begin = std::chrono::steady_clock::now();
#ifdef SHELF_HY4_RDMA
    auto snapshot_env = std::getenv("SHELF_HY4_SNAPSHOT");
    bool snapshot_enabled = snapshot_env && std::string_view(snapshot_env) == "1";
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> digest_ctx(snapshot_enabled ? EVP_MD_CTX_new() : nullptr, EVP_MD_CTX_free);
    if (snapshot_enabled && (!digest_ctx || EVP_DigestInit_ex(digest_ctx.get(), EVP_sha256(), nullptr) != 1))
        throw std::runtime_error("重みのフィンガープリントを初期化できません");
    auto hash = [&](const void *data, size_t length) {
        if (snapshot_enabled && EVP_DigestUpdate(digest_ctx.get(), data, length) != 1)
            throw std::runtime_error("重みのフィンガープリントを計算できません");
    };
#endif
    constexpr size_t chunk = 16ull << 20;
    void *buffer = nullptr;
    check(hipHostMalloc(&buffer, 3 * chunk, hipHostMallocPortable));
    try {
        std::memset(buffer, 0, 3 * chunk);
        std::array<size_t, 3> base{}, used{};
        auto flush = [&](int r) {
            if (!used[r])
                return;
            check(hipSetDevice(physical[r]));
            check(hipMemcpy(weight_base[r] + base[r], static_cast<uint8_t *>(buffer) + r * chunk, used[r],
                            hipMemcpyHostToDevice));
            used[r] = 0;
        };
#ifdef SHELF_HY4_RDMA
        // 重み全体の SHA-256（約 44 秒）をやめ、shard の大きさと配置で同一性を見る。
        // shortcut: 同じ大きさのまま中身だけ書き換えられた場合は検出できない、重みを差し替えるときは snapshot を消すこと。
        for (const auto &file : model.files) {
            auto size = file.file_size();
            hash(&size, sizeof(size));
        }
#endif
        // 先読み: ファイルの mmap をページフォルトで同期的に読むと遅いので、読み出し位置の少し先に MADV_WILLNEED を出す。
        std::atomic<size_t> consumed{0};
        std::atomic<bool> stop_prefetch{false};
        std::thread prefetch([&] {
            constexpr size_t ahead = size_t(8) << 30;
            const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
            size_t advised = 0;
            for (const auto *w : model.upload_order) {
                while (!stop_prefetch && advised > consumed + ahead)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (stop_prefetch)
                    return;
                auto begin_address = reinterpret_cast<uintptr_t>(w->host) & ~(page - 1);
                auto end_address = reinterpret_cast<uintptr_t>(w->host) + w->bytes;
                (void)madvise(reinterpret_cast<void *>(begin_address), end_address - begin_address, MADV_WILLNEED);
                advised += w->bytes;
            }
        });
        struct Join {
            std::atomic<bool> &stop;
            std::thread &thread;
            ~Join() {
                stop = true;
                if (thread.joinable())
                    thread.join();
            }
        } join{stop_prefetch, prefetch};
        for (const auto *w : model.upload_order) {
            const int r = w->rank;
#ifdef SHELF_HY4_RDMA
            hash(&w->rank, sizeof(w->rank));
            hash(&w->offset, sizeof(w->offset));
            hash(&w->bytes, sizeof(w->bytes));
#endif
            for (size_t n = 0; n < w->bytes;) {
                size_t pos = w->offset + n, first = pos / chunk * chunk;
                if (first != base[r]) {
                    flush(r);
                    base[r] = first;
                }
                size_t local = pos - first, bytes = std::min(chunk - local, w->bytes - n);
                std::memcpy(static_cast<uint8_t *>(buffer) + r * chunk + local,
                            static_cast<const uint8_t *>(w->host) + n, bytes);
                used[r] = local + bytes;
                n += bytes;
            }
            consumed += w->bytes;
        }
        for (int r = 0; r < 3; ++r)
            flush(r);
    } catch (...) {
        check(hipHostFree(buffer));
        throw;
    }
    check(hipHostFree(buffer));
#ifdef SHELF_HY4_RDMA
    if (snapshot_enabled) {
        unsigned length = 0;
        if (EVP_DigestFinal_ex(digest_ctx.get(), snapshot_upload_digest.data(), &length) != 1 ||
            length != snapshot_upload_digest.size())
            throw std::runtime_error("重みのフィンガープリントを確定できません");
        snapshot_upload_digest_ready = true;
    }
#endif
    fprintf(stderr, "weight_upload_s=%.3f\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
    for (int r = 0; r < 3; ++r) {
        check(hipSetDevice(physical[r]));
        if (profile && std::string_view(profile) == "1") {
            check(hipEventCreate(&expert_begin[r]));
            check(hipEventCreate(&expert_end[r]));
        }
        for (int layer = 1; layer < 42; ++layer)
            upload_experts(r, layer);
    }
    check(hipSetDevice(physical[0]));
    if (!folded.empty())
        fold_kv_weights(*this);
}
void Residency::transfer(void *dst, int destination, const void *src, int source, size_t bytes) {
    transfer_bytes[source][destination] += bytes;
    std::chrono::steady_clock::time_point begin;
    if (expert_begin[0])
        begin = std::chrono::steady_clock::now();
    if (peer[destination][source]) {
        check(hipSetDevice(physical[source]));
        check(hipEventRecord(ready[source][destination], nullptr));
        check(hipSetDevice(physical[destination]));
        check(hipStreamWaitEvent(nullptr, ready[source][destination], 0));
        check(hipMemcpyPeerAsync(dst, physical[destination], src, physical[source], bytes, nullptr));
    } else {
        if (!staging || bytes > max_prefill_chunk * 1536u * 2u)
            throw std::runtime_error("転送作業域の範囲外です");
        check(hipSetDevice(physical[source]));
        check(hipMemcpy(staging, src, bytes, hipMemcpyDeviceToHost));
        check(hipSetDevice(physical[destination]));
        check(hipMemcpy(dst, staging, bytes, hipMemcpyHostToDevice));
    }
    if (expert_begin[0])
        transfer_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}
void *Residency::route_buffer(int rank, size_t bytes) {
    constexpr size_t capacity =
        max_prefill_chunk * (16 * (sizeof(Hy4Route) + sizeof(Hy4ExpertBatch) + sizeof(int)) + sizeof(int)) +
        sizeof(int);
    if (bytes > capacity)
        throw std::runtime_error("routing 情報が作業域の上限を超えました");
    if (!route_host[rank])
        check(hipHostMalloc(&route_host[rank], capacity, hipHostMallocPortable));
    return route_host[rank];
}
void Residency::shutdown() {
    std::string errors;
    for (auto it = arenas.rbegin(); it != arenas.rend(); ++it) {
        auto status = it->shutdown();
        if (!status.ok())
            errors += status.message() + "\n";
    }
    arenas.clear();
    auto collect = [&](hipError_t e) {
        if (e != hipSuccess)
            errors += std::string(hipGetErrorString(e)) + "\n";
    };
    for (int r = 0; r < 3; ++r)
        if (expert_begin[r] || expert_end[r]) {
            auto status = hipSetDevice(physical[r]);
            collect(status);
            if (status == hipSuccess) {
                if (expert_begin[r])
                    collect(hipEventDestroy(expert_begin[r]));
                if (expert_end[r])
                    collect(hipEventDestroy(expert_end[r]));
            }
            expert_begin[r] = expert_end[r] = nullptr;
        }
    for (int r = 0; r < 3; ++r) {
        if (route_host[r]) {
            collect(hipHostFree(route_host[r]));
            route_host[r] = nullptr;
        }
        for (auto &event : ready[r])
            if (event) {
                auto status = hipSetDevice(physical[r]);
                collect(status);
                if (status == hipSuccess)
                    collect(hipEventDestroy(event));
                event = nullptr;
            }
    }
    if (staging) {
        collect(hipHostFree(staging));
        staging = nullptr;
    }
    for (int dst = 0; dst < 3; ++dst)
        for (int src = 0; src < 3; ++src)
            if (owns_peer[dst][src]) {
                auto e = hipSetDevice(physical[dst]);
                collect(e);
                if (e == hipSuccess)
                    collect(hipDeviceDisablePeerAccess(physical[src]));
                owns_peer[dst][src] = false;
            }
#ifdef SHELF_HY4_RDMA
    if (engram_pending.valid())
        engram_pending.wait();
    if (engram_client) {
        try {
            engram_client->shutdown();
        } catch (const std::exception &e) {
            errors += std::string(e.what()) + "\n";
        }
        engram_client.reset();
    }
#endif
    if (!errors.empty())
        throw std::runtime_error(errors);
}
const void *Residency::device(const Weight &w) const {
    if (w.rank < 0)
        throw std::runtime_error("ホスト常駐 tensor です");
    return weight_base.at(w.rank) + w.offset;
}
}
