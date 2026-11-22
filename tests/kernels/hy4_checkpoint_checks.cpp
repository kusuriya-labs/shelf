#include <shelf/models/hy4_flash/target.h>
#include <shelf/models/hy4_flash/generate.h>
#include <shelf/io/token_ids.h>
#include <shelf/models/hy4_flash/routes.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <stdexcept>

namespace shelf::hy4 {
namespace {
size_t check_weights(Residency &gpu, int rank) {
    std::vector<uint8_t> buffer(16u << 20);
    auto &model = gpu.model;
    size_t base = SIZE_MAX, total = 0;
    for (const auto *w : model.upload_order[rank]) {
        for (size_t i = 0; i < w->bytes;) {
            size_t offset = w->offset + i;
            size_t start = offset / buffer.size() * buffer.size();
            if (base != start) {
                base = start;
                size_t length = std::min(buffer.size(), model.bytes[rank] - base);
                checked(hipMemcpy(buffer.data(), gpu.weight_base[rank] + base, length, hipMemcpyDeviceToHost));
            }
            size_t local = offset - base, length = std::min(buffer.size() - local, w->bytes - i);
            if (std::memcmp(buffer.data() + local, static_cast<const uint8_t *>(w->host) + i, length)) {
                for (const auto &[key, weight] : model.local[rank])
                    if (&weight == w)
                        throw std::runtime_error("転送した重みが checkpoint と一致しません: " + key +
                                                 " offset=" + std::to_string(i));
                throw std::runtime_error("転送した重みの所属を確認できません");
            }
            i += length;
        }
        total += w->bytes;
    }
    return total;
}

void check_expert_tables(Residency &gpu, int rank) {
    auto &model = gpu.model;
    if (gpu.moe_owners[rank]) {
        Owners owners{};
        checked(hipMemcpy(owners.data(), gpu.moe_owners[rank], sizeof(owners), hipMemcpyDeviceToHost));
        if (owners != model.owners)
            throw std::runtime_error("GPU 上の expert 配置表が指定と一致しません");
    }
    int group = rank / gpu.tp_size;
    std::vector<Hy4ExpertProjection> table(512 * 3);
    for (int layer = 1; layer < 42; ++layer) {
        checked(hipMemcpy(table.data(), gpu.matrices[rank] + (layer - 1) * table.size(),
                          table.size() * sizeof(table[0]), hipMemcpyDeviceToHost));
        validate_expert_table(table, model.owners[layer - 1], group);
        for (int e = 0; e < 512; ++e) {
            if (model.owners[layer - 1][e] != group)
                continue;
            std::string prefix = "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + ".";
            int kind = 0;
            for (auto suffix : {"gate_proj", "up_proj", "down_proj"}) {
                std::string key = prefix + suffix;
                const auto &entry = table[3 * e + kind++];
                const auto &w = model.at(key + ".trellis", rank);
                const auto *shape = model.shape_of(key, rank);
                if (entry.trellis != gpu.device(w) || entry.bits != w.spec.shape[2] / 16 ||
                    entry.suh != gpu.device(model.at(key + ".suh", rank)) ||
                    entry.svh != gpu.device(model.at(key + ".svh", rank)) ||
                    !shape || entry.k != shape->at(0) || entry.n != shape->at(1))
                    throw std::runtime_error("GPU 上の expert 射影表に不一致があります: " + key);
            }
        }
    }
}

void check_fold(Residency &gpu, int rank, const std::string &key, const void *device) {
    using namespace quantization::exl3;
    const auto &trellis = gpu.model.at(key + ".trellis", rank);
    const auto &scale = gpu.model.at(key + ".svh", rank);
    size_t n = scale.spec.shape.at(0);
    std::vector<int8_t> decoded(n * 512);
    if (!exl3_decode_int8(static_cast<const uint16_t *>(trellis.host), 32, n / 16, 4, decoded.data(), 4))
        throw std::runtime_error("KV 射影を復号できません: " + key);
    std::vector<_Float16> folded(n * 512);
    checked(hipMemcpy(folded.data(), device, folded.size() * sizeof(_Float16), hipMemcpyDeviceToHost));
    auto svh = static_cast<const _Float16 *>(scale.host);
    for (size_t first = 0; first < n; first += 128) {
        for (int k = 0; k < 512; ++k) {
            std::array<int, 128> values;
            for (int j = 0; j < 128; ++j)
                values[j] = decoded[(first + j) * 512 + k];
            for (int step = 1; step < 128; step *= 2)
                for (int base = 0; base < 128; base += 2 * step)
                    for (int j = base; j < base + step; ++j) {
                        int a = values[j], b = values[j + step];
                        values[j] = a + b;
                        values[j + step] = a - b;
                    }
            for (int j = 0; j < 128; ++j) {
                float rotated = values[j] * kExl3Int8GridStep * 0.08838834764831845f;
                double value = double(rotated) * double(svh[first + j]);
                if (folded[(first + j) * 512 + k] != static_cast<_Float16>(value))
                    throw std::runtime_error("fold した KV 射影が CPU 基準と一致しません: " + key +
                                             " out=" + std::to_string(first + j) + " k=" + std::to_string(k) +
                                             " expected=" + std::to_string(float(static_cast<_Float16>(value))) +
                                             " actual=" + std::to_string(float(folded[(first + j) * 512 + k])));
            }
        }
    }
}

void load_layer_experts(Ops &op, int layer, std::vector<uint8_t> &buffer) {
    const auto &weights = op.gpu.model.local[0];
    const std::string prefix = "model.layers." + std::to_string(layer) + ".mlp.experts.";
    auto first = weights.lower_bound(prefix), last = first;
    size_t begin = SIZE_MAX, end = 0;
    while (last != weights.end() && last->first.starts_with(prefix)) {
        const auto &w = last->second;
        begin = std::min(begin, w.offset);
        end = std::max(end, w.offset + w.bytes);
        ++last;
    }
    if (end <= begin || end > op.gpu.model.bytes[0])
        throw std::runtime_error("層の expert 領域が不正です");
    buffer.resize(end - begin);
    for (auto it = first; it != last; ++it) {
        const auto &w = it->second;
        std::memcpy(buffer.data() + (w.offset - begin), w.host, w.bytes);
    }
    op.copy(op.gpu.weight_base[0] + begin, buffer.data(), buffer.size(), hipMemcpyHostToDevice);
}

}

void verify_hy4_layers(Residency &gpu, const Path &input, const Path &dump) {
    auto ids = io::read_input_ids(input);
    if (ids.empty() || ids.size() > 32)
        throw std::invalid_argument("層別照合の入力は 1〜32 token で指定してください");
    std::vector<int> chunks;
    for (int n : {1, 8, 15, 16, 20})
        if (size_t(n) < ids.size())
            chunks.push_back(n);
    chunks.push_back(int(ids.size()));
    Ops op(gpu);
    auto embedded = op.embed(ids, true);
    std::vector<uint16_t> state(embedded.bytes() / 2), expected(state.size()), output(state.size());
    op.copy(state.data(), embedded.p, embedded.bytes(), hipMemcpyDeviceToHost);
    std::vector<uint8_t> buffer;
    for (int layer = 0; layer < 42; ++layer) {
        if (layer > 1)
            load_layer_experts(op, layer, buffer);
        for (int chunk : chunks) {
            for (size_t first = 0; first < ids.size(); first += chunk) {
                size_t end = std::min(first + chunk, ids.size());
                op.arena.cursor = 0;
                Mat x = op.mat(end - first, 12288);
                op.copy(x.p, state.data() + first * 12288, x.bytes(), hipMemcpyHostToDevice);
                Selection selection;
                decoder_layer(op, layer, first, x, {ids.begin(), ids.begin() + first},
                              {ids.begin() + first, ids.begin() + end}, selection);
                op.copy(output.data() + first * 12288, x.p, x.bytes(), hipMemcpyDeviceToHost);
            }
            if (chunk == chunks.front())
                expected = output;
            else if (output != expected)
                throw std::runtime_error("層出力が異なります: layer=" + std::to_string(layer) +
                                         " chunk=" + std::to_string(chunk));
        }
        state.swap(expected);
    }
    Target target(gpu, 0.f, 0, 0, 1.f);
    Mat x = op.mat(ids.size(), 12288);
    op.copy(x.p, state.data(), x.bytes(), hipMemcpyHostToDevice);
    float *post;
    auto device = target.head_logits(op.collapse_norm(x, "model.hc_head.hc_head_", post, "model.norm.weight"));
    std::vector<float> logits(ids.size() * 120832);
    op.copy(logits.data(), device, logits.size() * sizeof(float), hipMemcpyDeviceToHost);
    std::ofstream file(dump, std::ios::binary);
    if (!file.write(reinterpret_cast<const char *>(logits.data()), logits.size() * sizeof(float)))
        throw std::runtime_error("層別照合の logits を保存できません");
    auto row = logits.end() - 120832;
    printf("ep=%d rows=%zu chunks=%zu greedy=%td\n", gpu.ranks, ids.size(), chunks.size(),
           std::max_element(row, logits.end()) - row);
}

bool trace_hy4_prefill(Residency &gpu, const Path &input, int chunk, const Path &dump) {
    auto ids = io::read_input_ids(input);
    if (ids.empty() || ids.size() >= gpu.tokens || chunk < 1 || chunk > max_prefill_chunk)
        throw std::invalid_argument("照合用の入力長またはチャンク長が不正です");
    generate(gpu, ids, 1, chunk, 0.f, 0, dump);
    std::ifstream file(dump, std::ios::binary);
    std::vector<float> expected(120832);
    if (!file.read(reinterpret_cast<char *>(expected.data()), expected.size() * sizeof(float)))
        throw std::runtime_error("比較対象の logits を読み込めません");
    Target target(gpu, 0.f, 0, 0, 1.f);
    auto &op = target.op;
    std::vector<int> prior;
    std::array<uint64_t, 42> hashes;
    hashes.fill(14695981039346656037ull);
    std::vector<uint16_t> state;
    Mat last{};
    for (size_t i = 0; i < ids.size(); i += chunk) {
        size_t end = std::min(i + chunk, ids.size());
        std::vector<int> batch(ids.begin() + i, ids.begin() + end);
        op.arena.cursor = target.base;
        Mat x = op.embed(batch, true);
        state.resize(x.bytes() / sizeof(uint16_t));
        for (int layer = 0; layer < 42; ++layer) {
            Selection selection;
            decoder_layer(op, layer, i, x, prior, batch, selection);
            op.copy(state.data(), x.p, x.bytes(), hipMemcpyDeviceToHost);
            for (uint16_t word : state)
                hashes[layer] = (hashes[layer] ^ word) * 1099511628211ull;
        }
        last = x.row(x.rows - 1);
        prior.insert(prior.end(), batch.begin(), batch.end());
    }
    for (int r = 0; r < gpu.ranks; ++r) {
        char bus[32]{};
        checked(hipDeviceGetPCIBusId(bus, sizeof(bus), gpu.physical[r]));
        printf("rank=%d pci=%s weights=%p workspace=%p\n", r, bus, static_cast<void *>(gpu.weight_base[r]),
               static_cast<void *>(gpu.workspace[r].base));
    }
    for (int layer = 0; layer < 42; ++layer)
        printf("layer=%d hidden_hash=%016llx\n", layer, static_cast<unsigned long long>(hashes[layer]));
    float *post;
    auto logits = target.head_logits(op.collapse_norm(last, "model.hc_head.hc_head_", post, "model.norm.weight"));
    std::vector<float> traced(expected.size());
    op.copy(traced.data(), logits, traced.size() * sizeof(float), hipMemcpyDeviceToHost);
    bool match = std::memcmp(traced.data(), expected.data(), traced.size() * sizeof(float)) == 0;
    printf("trace_matches_first_prefill=%d\n", match);
    return match;
}

void verify_hy4_upload(Residency &gpu) {
    for (int r = 0; r < gpu.ranks; ++r) {
        checked(hipSetDevice(gpu.physical[r]));
        size_t bytes = check_weights(gpu, r);
        check_expert_tables(gpu, r);
        size_t projections = 0;
        for (const auto &[key, ptr] : gpu.folded[r])
            if (ptr) {
                check_fold(gpu, r, key, ptr);
                ++projections;
            }
        printf("rank=%d verified_weight_bytes=%zu verified_kv_projections=%zu\n", r, bytes, projections);
    }
    checked(hipSetDevice(gpu.physical[0]));
}
}
