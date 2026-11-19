#include <shelf/models/hy4_flash/snapshot/state.h>
#include <shelf/models/hy4_flash/snapshot/transport.h>
#include <openssl/evp.h>
#include <fstream>
#include <cstring>
#include <memory>

namespace shelf::hy4 {
std::vector<uint64_t> engram_indices(const Checkpoint &, int, const std::vector<int> &, const std::vector<int> &);
namespace snapshot {
Digest fingerprint(Residency &gpu, const std::vector<int> &history) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("フィンガープリントを初期化できません");
    auto update = [&](const void *data, size_t size) {
        if (EVP_DigestUpdate(ctx.get(), data, size) != 1) throw std::runtime_error("フィンガープリントの計算に失敗しました");
    };
    if (!gpu.snapshot_model_digest_ready) {
        const char version[] = "hy4-snapshot-v1-octave294";
        update(version, sizeof(version));
        update(&gpu.tokens, sizeof(gpu.tokens));
        int runtime = 0, driver = 0;
        if (hipRuntimeGetVersion(&runtime) != hipSuccess || hipDriverGetVersion(&driver) != hipSuccess)
            throw std::runtime_error("GPU runtime の版を取得できません");
        update(&runtime, sizeof(runtime));
        update(&driver, sizeof(driver));
        for (int device : gpu.physical) {
            hipDeviceProp_t props{};
            char bus[32]{};
            if (hipGetDeviceProperties(&props, device) != hipSuccess || hipDeviceGetPCIBusId(bus, sizeof(bus), device) != hipSuccess)
                throw std::runtime_error("GPU 配置を取得できません");
            update(props.gcnArchName, std::strlen(props.gcnArchName));
            update(bus, std::strlen(bus));
        }
        update(gpu.model.owners.data(), sizeof(gpu.model.owners));
        update(gpu.model.moduli.data(), sizeof(gpu.model.moduli));
        update(gpu.model.expert_bits.data(), sizeof(gpu.model.expert_bits));
        update(&gpu.model.mixed, sizeof(gpu.model.mixed));
        update(&gpu.model.has_draft, sizeof(gpu.model.has_draft));
        for (const auto &[name, dimensions] : gpu.model.shape) {
            size_t name_length = name.size();
            update(&name_length, sizeof(name_length));
            update(name.data(), name.size());
            update(dimensions.data(), sizeof(dimensions));
        }
        if (!gpu.snapshot_upload_digest_ready) throw std::runtime_error("重みのフィンガープリントがありません");
        update(gpu.snapshot_upload_digest.data(), gpu.snapshot_upload_digest.size());
        for (const auto &[key, weight] : gpu.model.weights) {
            if (weight.rank < 0) continue;
            size_t name_length = key.size();
            update(&name_length, sizeof(name_length));
            update(key.data(), key.size());
            update(&weight.spec.dtype, sizeof(weight.spec.dtype));
            size_t dims = weight.spec.shape.size();
            update(&dims, sizeof(dims));
            update(weight.spec.shape.data(), dims * sizeof(size_t));
            update(&weight.rank, sizeof(weight.rank));
            update(&weight.offset, sizeof(weight.offset));
            update(&weight.bytes, sizeof(weight.bytes));
        }
        std::ifstream executable("/proc/self/exe", std::ios::binary);
        if (!executable) throw std::runtime_error("実行ファイルを確認できません");
        std::vector<char> block(1024 * 1024);
        while (executable) {
            executable.read(block.data(), block.size());
            update(block.data(), executable.gcount());
        }
        if (!executable.eof()) throw std::runtime_error("実行ファイルの読み取りに失敗しました");
        unsigned model_length = 0;
        if (EVP_DigestFinal_ex(ctx.get(), gpu.snapshot_model_digest.data(), &model_length) != 1 ||
            model_length != gpu.snapshot_model_digest.size())
            throw std::runtime_error("モデルのフィンガープリントを確定できません");
        gpu.snapshot_model_digest_ready = true;
    }
    if (EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("フィンガープリントを初期化できません");
    update(gpu.snapshot_model_digest.data(), gpu.snapshot_model_digest.size());
    for (const auto &[key, weight] : gpu.model.weights) {
        if (weight.rank >= 0) continue;
        size_t name_length = key.size();
        update(&name_length, sizeof(name_length));
        update(key.data(), key.size());
        update(&weight.bytes, sizeof(weight.bytes));
        update(weight.host, weight.bytes);
    }
    update(history.data(), history.size() * sizeof(int));
    if (gpu.model.rdma) {
        if (!gpu.engram_client) throw std::runtime_error("Engram 接続がありません");
        for (size_t first = 0; first < history.size(); first += 1024) {
            size_t end = std::min(first + 1024, history.size());
            std::vector<int> ids(history.begin() + first, history.begin() + end);
            std::vector<int> prior(history.begin() + (first > 2 ? first - 2 : 0), history.begin() + first);
            rdma::Indices indices;
            for (int g = 0; g < 4; ++g) indices[g] = engram_indices(gpu.model, g, prior, ids);
            auto rows = gpu.engram_client->read(indices);
            for (auto data : rows) update(data.data(), data.size());
        }
    }
    Digest result{};
    unsigned length = 0;
    if (EVP_DigestFinal_ex(ctx.get(), result.data(), &length) != 1 || length != result.size())
        throw std::runtime_error("フィンガープリントの確定に失敗しました");
    return result;
}
}
}
