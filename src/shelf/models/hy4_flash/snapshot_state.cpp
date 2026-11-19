#include <shelf/models/hy4_flash/snapshot/state.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace shelf::hy4::snapshot {
namespace {
constexpr uint64_t magic = 0x3150414e53345948ULL;
struct Header {
    uint64_t signature;
    uint64_t capacity;
    uint64_t length;
};
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
size_t bytes(size_t capacity, size_t length) {
    if (!length || length > capacity || capacity > 1048576) throw std::runtime_error("保存する履歴長が不正です");
    const size_t sliding = std::min(capacity, size_t(518));
    return sizeof(Header) + length * sizeof(int) + 12 * (294 * length + 256 * (length / 4) + 3072)
           + 30 * 294 * sliding;
}
void copy_state(Residency &gpu, uint8_t *buffer, size_t length, bool output) {
    check(hipSetDevice(gpu.physical[0]));
    auto copy = [&](void *device, size_t count) {
        if (output) check(hipMemcpy(buffer, device, count, hipMemcpyDeviceToHost));
        else check(hipMemcpy(device, buffer, count, hipMemcpyHostToDevice));
        buffer += count;
    };
    for (int layer = 0; layer < 42; ++layer) {
        bool sparse = layer == 0 || layer % 4 == 1;
        copy(gpu.cache[layer], 294 * (sparse ? length : std::min(gpu.tokens, size_t(518))));
        if (sparse) {
            copy(gpu.index_keys[layer], 256 * (length / 4));
            copy(gpu.index_keys[layer] + (gpu.tokens / 4) * 128, 3072);
        }
    }
}
}
std::vector<uint8_t> capture(Residency &gpu, const std::vector<int> &history) {
    std::vector<uint8_t> result(bytes(gpu.tokens, history.size()));
    Header h{magic, gpu.tokens, history.size()};
    std::memcpy(result.data(), &h, sizeof(h));
    std::memcpy(result.data() + sizeof(h), history.data(), history.size() * sizeof(int));
    copy_state(gpu, result.data() + sizeof(h) + history.size() * sizeof(int), history.size(), true);
    return result;
}
std::vector<int> inspect(Residency &gpu, std::span<const uint8_t> state) {
    if (state.size() < sizeof(Header)) throw std::runtime_error("状態ヘッダーが不足しています");
    Header h{};
    std::memcpy(&h, state.data(), sizeof(h));
    if (h.signature != magic || h.capacity != gpu.tokens || h.length > gpu.tokens ||
        state.size() != bytes(h.capacity, h.length)) throw std::runtime_error("状態の形式または予約長が異なります");
    std::vector<int> history(h.length);
    std::memcpy(history.data(), state.data() + sizeof(h), history.size() * sizeof(int));
    if (std::any_of(history.begin(), history.end(), [](int id) { return id < 0 || id >= 120832; }))
        throw std::runtime_error("保存された token ID が不正です");
    return history;
}
std::vector<int> restore(Residency &gpu, std::span<const uint8_t> state) {
    auto history = inspect(gpu, state);
    copy_state(gpu, const_cast<uint8_t *>(state.data()) + sizeof(Header) + history.size() * sizeof(int), history.size(), false);
    return history;
}
void clear_for_test(Residency &gpu) {
    check(hipSetDevice(gpu.physical[0]));
    for (int layer = 0; layer < 42; ++layer) {
        bool sparse = layer == 0 || layer % 4 == 1;
        check(hipMemset(gpu.cache[layer], 0, 294 * (sparse ? gpu.tokens : std::min(gpu.tokens, size_t(518)))));
        if (sparse) check(hipMemset(gpu.index_keys[layer], 0, (gpu.tokens / 4 + 12) * 256));
    }
    check(hipDeviceSynchronize());
}
}
