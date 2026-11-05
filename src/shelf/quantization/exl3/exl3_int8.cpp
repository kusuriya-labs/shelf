#include <shelf/quantization/exl3/exl3_int8.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <thread>
#include <vector>

namespace shelf::quantization::exl3 {

namespace {

struct TileMap {
    std::array<uint8_t, 256> row;
    std::array<uint8_t, 256> col;
};

TileMap make_tile_map() {
    TileMap m{};
    for (uint32_t t = 0; t < 256; ++t) {
        const uint32_t lane = t / 8u;
        const uint32_t j = t % 8u;
        m.row[t] = static_cast<uint8_t>(8u * (lane / 16u) + j);
        m.col[t] = static_cast<uint8_t>(lane % 16u);
    }
    return m;
}

inline int8_t q_from_state(uint32_t state) {
    const uint32_t x = state * kExl3Mul24Multiplier;
    const int b = static_cast<int>((x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24));
    return static_cast<int8_t>((b - 509) >> 2);
}

void decode_rows(
    const uint16_t* trellis, uint64_t kt_begin, uint64_t kt_end, uint64_t kt, uint64_t nt,
    uint32_t bits, int8_t* out, const TileMap& map) {
    const uint32_t nbits = 256u * bits;
    const uint32_t nwords = nbits / 32u;
    const uint64_t k_total = kt * kExl3TileDim;
    std::vector<uint32_t> words(nwords);
    for (uint64_t a = kt_begin; a < kt_end; ++a) {
        for (uint64_t c = 0; c < nt; ++c) {
            const uint16_t* tile = trellis + (a * nt + c) * (16ull * bits);
            std::memcpy(words.data(), tile, nwords * sizeof(uint32_t));
            for (uint32_t t = 0; t < 256; ++t) {
                const uint32_t p0 = (t * bits + bits - 16u + nbits) % nbits;
                const uint32_t wi = p0 / 32u;
                const uint32_t off = p0 % 32u;
                const uint64_t merged =
                    (static_cast<uint64_t>(words[wi]) << 32) | words[(wi + 1u) % nwords];
                const uint32_t state = static_cast<uint32_t>((merged >> (64u - off - 16u)) & 0xFFFFu);
                const uint64_t n = c * kExl3TileDim + map.col[t];
                const uint64_t k = a * kExl3TileDim + map.row[t];
                out[n * k_total + k] = q_from_state(state);
            }
        }
    }
}

}

bool exl3_decode_int8(
    const uint16_t* trellis, uint64_t kt, uint64_t nt, uint32_t bits, int8_t* out, unsigned threads) {
    if (trellis == nullptr || out == nullptr || kt == 0 || nt == 0 || bits < 1 || bits > 8) return false;
    static const TileMap map = make_tile_map();
    const unsigned workers = static_cast<unsigned>(
        std::max<uint64_t>(1, std::min<uint64_t>(threads == 0 ? 1u : threads, kt)));
    if (workers == 1) {
        decode_rows(trellis, 0, kt, kt, nt, bits, out, map);
        return true;
    }
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (unsigned w = 0; w < workers; ++w) {
        const uint64_t begin = kt * w / workers;
        const uint64_t end = kt * (w + 1) / workers;
        pool.emplace_back(decode_rows, trellis, begin, end, kt, nt, bits, out, std::cref(map));
    }
    for (std::thread& t : pool) t.join();
    return true;
}

bool exl3_tiles_to_nmajor(
    const uint16_t* src, uint64_t kt, uint64_t nt, uint32_t bits, uint16_t* dst, unsigned threads) {
    if (src == nullptr || dst == nullptr || kt == 0 || nt == 0 || bits < 1 || bits > 8) return false;
    const size_t tile_bytes = 32u * bits;
    auto work = [&](uint64_t a_begin, uint64_t a_end) {
        for (uint64_t a = a_begin; a < a_end; ++a)
            for (uint64_t c = 0; c < nt; ++c)
                std::memcpy(dst + (c * kt + a) * 16u * bits, src + (a * nt + c) * 16u * bits, tile_bytes);
    };
    const unsigned workers = static_cast<unsigned>(
        std::max<uint64_t>(1, std::min<uint64_t>(threads == 0 ? 1u : threads, kt)));
    if (workers == 1) {
        work(0, kt);
        return true;
    }
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (unsigned w = 0; w < workers; ++w) pool.emplace_back(work, kt * w / workers, kt * (w + 1) / workers);
    for (std::thread& t : pool) t.join();
    return true;
}

}
