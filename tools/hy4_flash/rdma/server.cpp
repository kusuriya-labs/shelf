#include <arpa/inet.h>
#include <cstring>
#include <csignal>
#include <cerrno>
#include <fcntl.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <shelf/models/hy4_flash/rdma/protocol.h>
#include <sys/mman.h>
#include <sys/stat.h>

using namespace shelf::hy4::rdma;
using Json = nlohmann::json;

struct File {
    int fd = -1;
    size_t size = 0;
    uint8_t *data = nullptr;
    explicit File(const std::string &path) {
        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        struct stat st{};
        if (fd < 0 || fstat(fd, &st) || st.st_size < 8)
            throw std::runtime_error("Engram ファイルを開けません: " + path);
        size = st.st_size;
        data = static_cast<uint8_t *>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data == MAP_FAILED)
            throw std::runtime_error("Engram ファイルを mmap できません: " + path);
    }
    ~File() {
        if (data && data != MAP_FAILED && munmap(data, size))
            std::cerr << "Engram ファイルの mmap 解除に失敗しました\n";
        if (fd >= 0 && close(fd))
            std::cerr << "Engram ファイルの close に失敗しました\n";
    }
};

volatile std::sig_atomic_t stopping = 0;
void stop_handler(int) { stopping = 1; }

struct Resources {
    ibv_context *ctx = nullptr;
    ibv_pd *pd = nullptr;
    std::array<ibv_mr *, 4> regions{};
    std::array<uint8_t *, 4> memory{};
    std::array<size_t, 4> lengths{};
    std::array<bool, 4> locked{};
    int listener = -1;
    std::string shutdown() {
        std::string errors;
        if (listener >= 0 && close(listener))
            errors += "制御ポートの close に失敗しました\n";
        listener = -1;
        for (int i = 3; i >= 0; --i) {
            if (regions[i] && ibv_dereg_mr(regions[i]))
                errors += "MR の登録解除に失敗しました\n";
            regions[i] = nullptr;
            if (locked[i] && munlock(memory[i], lengths[i]))
                errors += "Engram メモリの固定解除に失敗しました\n";
            locked[i] = false;
            if (memory[i] && munmap(memory[i], lengths[i]))
                errors += "Engram メモリの解放に失敗しました\n";
            memory[i] = nullptr;
        }
        if (pd && ibv_dealloc_pd(pd))
            errors += "PD の解放に失敗しました\n";
        pd = nullptr;
        if (ctx && ibv_close_device(ctx))
            errors += "RDMA デバイスの解放に失敗しました\n";
        ctx = nullptr;
        return errors;
    }
    ~Resources() {
        auto errors = shutdown();
        if (!errors.empty())
            std::cerr << "Engram RDMA 解放エラー: " << errors;
    }
};

int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("使用方法: engram-rdma-server 表のディレクトリ");
        struct sigaction action{};
        action.sa_handler = stop_handler;
        sigemptyset(&action.sa_mask);
        if (sigaction(SIGTERM, &action, nullptr) || sigaction(SIGINT, &action, nullptr))
            throw std::runtime_error("終了シグナルを設定できません");
        Resources resources;
        auto devices = ibv_get_device_list(nullptr);
        if (!devices)
            throw std::runtime_error("RDMA デバイスを列挙できません");
        ibv_context *&ctx = resources.ctx;
        for (int i = 0; devices[i]; ++i)
            if (std::string(ibv_get_device_name(devices[i])) == "hfi1_0")
                ctx = ibv_open_device(devices[i]);
        ibv_free_device_list(devices);
        if (!ctx)
            throw std::runtime_error("hfi1_0 を開けません");
        ibv_port_attr port{};
        if (ibv_query_port(ctx, 1, &port) || port.state != IBV_PORT_ACTIVE || port.active_mtu < IBV_MTU_4096)
            throw std::runtime_error("RDMA ポートが使用できません");
        auto pd = ibv_alloc_pd(ctx);
        resources.pd = pd;
        if (!pd)
            throw std::runtime_error("RDMA PD を作成できません");
        Endpoint local{};
        local.version_id = version;
        local.lid = port.lid;
        for (int g = 0; g < 4; ++g) {
            auto key = "model.layers." + std::to_string(1 + 10 * g) +
                       ".engram_layer.multi_head_embedding.embedding.weight";
            File file(std::string(argv[1]) + "/" + key + ".fp8.safetensors");
            uint64_t header_len;
            std::memcpy(&header_len, file.data, 8);
            if (header_len > file.size - 8 || header_len > 16 * 1024 * 1024)
                throw std::runtime_error("safetensors ヘッダーが不正です");
            auto header = Json::parse(file.data + 8, file.data + 8 + header_len);
            auto rows = header.at(key).at("shape").at(0).get<uint64_t>();
            if (!rows || rows > 200000000)
                throw std::runtime_error("Engram の行数が不正です");
            auto tensor = [&](const std::string &name, const char *dtype, size_t width) {
                auto &entry = header.at(name);
                auto shape = entry.at("shape").get<std::vector<size_t>>();
                auto offsets = entry.at("data_offsets").get<std::array<uint64_t, 2>>();
                uint64_t base = 8 + header_len;
                if (entry.at("dtype") != dtype || shape != std::vector<size_t>{rows, width} ||
                    offsets[0] > offsets[1] || offsets[1] > file.size - base ||
                    offsets[1] - offsets[0] != rows * (width == 1 ? 2 : width))
                    throw std::runtime_error("Engram tensor の型または長さが不正です: " + name);
                return file.data + base + offsets[0];
            };
            const auto bits = tensor(key, "F8_E4M3", 96);
            const auto scale = tensor(key + ".scale", "BF16", 1);
            size_t length = rows * 98;
            auto packed = static_cast<uint8_t *>(
                mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            if (packed == MAP_FAILED)
                throw std::runtime_error("Engram のメモリを確保できません");
            resources.memory[g] = packed;
            resources.lengths[g] = length;
            for (uint64_t row = 0; row < rows; ++row) {
                if (!(row & 0xfffff) && stopping)
                    throw std::runtime_error("Engram RDMA サーバーを停止しました");
                std::memcpy(packed + row * 98, bits + row * 96, 96);
                std::memcpy(packed + row * 98 + 96, scale + row * 2, 2);
            }
            if (mlock(packed, length))
                throw std::runtime_error("Engram の固定に失敗しました。memlock 上限を確認してください");
            resources.locked[g] = true;
            auto mr = ibv_reg_mr(pd, packed, length, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
            if (!mr)
                throw std::runtime_error("Engram MR の登録に失敗しました。memlock 上限を確認してください");
            resources.regions[g] = mr;
            local.address[g] = reinterpret_cast<uint64_t>(packed);
            local.rows[g] = rows;
            local.rkey[g] = mr->rkey;
            std::cerr << "Engram 表 " << g << " 登録完了\n";
        }
        int listener = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        resources.listener = listener;
        int reuse = 1;
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(19390);
        inet_pton(AF_INET, "10.255.254.1", &addr.sin_addr);
        if (listener < 0 || bind(listener, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) ||
            listen(listener, 1))
            throw std::runtime_error("RDMA 制御ポートを開けません");
        std::cerr << "Engram RDMA 接続待機中: 10.255.254.1:19390\n";
        while (!stopping) {
            int fd = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
            if (fd < 0) {
                if (stopping)
                    break;
                if (errno == EINTR)
                    continue;
                throw std::runtime_error("RDMA 制御接続の受け付けに失敗しました");
            }
            auto cq = ibv_create_cq(ctx, 1024, nullptr, nullptr, 0);
            std::array<ibv_qp *, qp_count> qps{};
            try {
                if (!cq)
                    throw std::runtime_error("RDMA CQ を作成できません");
                for (int i = 0; i < qp_count; ++i) {
                    qps[i] = create_qp(pd, cq);
                    local.qpn[i] = qps[i]->qp_num;
                }
                Endpoint peer{};
                transfer(fd, &peer, sizeof(peer), false);
                if (peer.version_id != version)
                    throw std::runtime_error("RDMA プロトコルの版が異なります");
                for (int i = 0; i < qp_count; ++i)
                    connect_qp(qps[i], peer, i);
                transfer(fd, &local, sizeof(local), true);
                char byte;
                while (!stopping && recv(fd, &byte, 1, 0) > 0) {
                }
            } catch (const std::exception &e) {
                std::cerr << "接続エラー: " << e.what() << '\n';
            }
            std::string errors;
            if (close(fd))
                errors += "制御接続の close に失敗しました\n";
            for (auto qp : qps)
                if (qp && ibv_destroy_qp(qp))
                    errors += "QP の解放に失敗しました\n";
            if (cq && ibv_destroy_cq(cq))
                errors += "CQ の解放に失敗しました\n";
            if (!errors.empty())
                throw std::runtime_error(errors);
        }
        auto errors = resources.shutdown();
        if (!errors.empty())
            throw std::runtime_error("Engram RDMA 解放エラー: " + errors);
    } catch (const std::exception &e) {
        std::cerr << "Engram RDMA サーバー: " << e.what() << '\n';
        return 1;
    }
}
