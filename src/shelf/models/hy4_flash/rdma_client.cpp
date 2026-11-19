#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <shelf/models/hy4_flash/rdma/client.h>
#include <shelf/models/hy4_flash/rdma/protocol.h>

namespace shelf::hy4::rdma {
struct Client::Impl {
    int fd = -1;
    ibv_context *ctx = nullptr;
    ibv_pd *pd = nullptr;
    ibv_cq *cq = nullptr;
    std::array<ibv_qp *, qp_count> qps{};
    ibv_mr *mr = nullptr;
    uint8_t *buffer = nullptr;
    size_t capacity = 0;
    Endpoint remote{};
    bool broken = false;
    std::string shutdown() {
        std::string errors;
        if (fd >= 0 && close(fd))
            errors += "制御接続の close に失敗しました\n";
        fd = -1;
        for (auto &qp : qps) {
            if (qp && ibv_destroy_qp(qp))
                errors += "QP の解放に失敗しました\n";
            qp = nullptr;
        }
        if (mr && ibv_dereg_mr(mr))
            errors += "MR の登録解除に失敗しました\n";
        mr = nullptr;
        free(buffer);
        buffer = nullptr;
        capacity = 0;
        if (cq && ibv_destroy_cq(cq))
            errors += "CQ の解放に失敗しました\n";
        cq = nullptr;
        if (pd && ibv_dealloc_pd(pd))
            errors += "PD の解放に失敗しました\n";
        pd = nullptr;
        if (ctx && ibv_close_device(ctx))
            errors += "RDMA デバイスの解放に失敗しました\n";
        ctx = nullptr;
        broken = true;
        return errors;
    }
    ~Impl() {
        auto errors = shutdown();
        if (!errors.empty())
            std::fprintf(stderr, "Engram RDMA 解放エラー: %s", errors.c_str());
    }
    void reserve(size_t bytes) {
        if (bytes <= capacity)
            return;
        if (mr && ibv_dereg_mr(mr))
            throw std::runtime_error("RDMA 作業域の登録解除に失敗しました");
        mr = nullptr;
        free(buffer);
        buffer = nullptr;
        capacity = 0;
        if (posix_memalign(reinterpret_cast<void **>(&buffer), 4096, bytes))
            throw std::runtime_error("RDMA 作業域を確保できません");
        mr = ibv_reg_mr(pd, buffer, bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!mr)
            throw std::runtime_error("RDMA 作業域の登録に失敗しました");
        capacity = bytes;
    }
};
Client::Client() : impl(std::make_unique<Impl>()) {
    auto &s = *impl;
    auto devices = ibv_get_device_list(nullptr);
    if (!devices)
        throw std::runtime_error("RDMA デバイスを列挙できません");
    for (int i = 0; devices[i]; ++i)
        if (std::string(ibv_get_device_name(devices[i])) == "hfi1_0")
            s.ctx = ibv_open_device(devices[i]);
    ibv_free_device_list(devices);
    if (!s.ctx)
        throw std::runtime_error("hfi1_0 を開けません");
    ibv_port_attr port{};
    if (ibv_query_port(s.ctx, 1, &port) || port.state != IBV_PORT_ACTIVE || port.active_mtu < IBV_MTU_4096)
        throw std::runtime_error("RDMA ポートが使用できません");
    s.pd = ibv_alloc_pd(s.ctx);
    if (!s.pd)
        throw std::runtime_error("RDMA PD を作成できません");
    s.cq = ibv_create_cq(s.ctx, 1024, nullptr, nullptr, 0);
    if (!s.cq)
        throw std::runtime_error("RDMA CQ を作成できません");
    Endpoint local{};
    local.version_id = version;
    local.lid = port.lid;
    for (int i = 0; i < qp_count; ++i) {
        s.qps[i] = create_qp(s.pd, s.cq);
        local.qpn[i] = s.qps[i]->qp_num;
    }
    s.fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    inet_pton(AF_INET, "10.255.254.2", &address.sin_addr);
    if (s.fd < 0 || bind(s.fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)))
        throw std::runtime_error("RDMA 制御元アドレスを bind できません");
    timeval timeout{10, 0};
    setsockopt(s.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(s.fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    address.sin_port = htons(19390);
    inet_pton(AF_INET, "10.255.254.1", &address.sin_addr);
    if (connect(s.fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)))
        throw std::runtime_error("RDMA サーバーに接続できません");
    transfer(s.fd, &local, sizeof(local), true);
    transfer(s.fd, &s.remote, sizeof(s.remote), false);
    if (s.remote.version_id != version)
        throw std::runtime_error("RDMA プロトコルの版が異なります");
    for (int i = 0; i < qp_count; ++i)
        connect_qp(s.qps[i], s.remote, i);
}
void Client::validate_rows(const std::array<uint64_t, 4> &rows) const {
    if (impl->remote.rows != rows)
        throw std::runtime_error("RDMA Engram 表の行数が checkpoint と一致しません");
}
Tables Client::read(const Indices &indices) {
    auto &s = *impl;
    if (s.broken)
        throw std::runtime_error("RDMA 接続が失敗済みです。再起動してください");
    try {
        char byte;
        auto state = recv(s.fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
        if (state == 0 || (state < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR))
            throw std::runtime_error("RDMA 制御接続が切断されました");
        Tables output{};
        size_t total_rows = 0, largest = 0;
        for (const auto &input : indices) {
            if (input.size() > 1024 * 16)
                throw std::runtime_error("RDMA Engram の要求が大きすぎます");
            total_rows += input.size();
            largest = std::max(largest, input.size());
        }
        if (!total_rows)
            return output;
        s.reserve((total_rows + largest) * 98);
        size_t output_offset = largest * 98;
        for (int g = 0; g < 4; ++g) {
            const auto &input = indices[g];
            if (input.empty())
                continue;
            auto destination = s.buffer + output_offset;
            output[g] = std::span<const uint8_t>(destination, input.size() * 98);
            output_offset += input.size() * 98;
            std::vector<uint64_t> sorted = input;
            std::sort(sorted.begin(), sorted.end());
            sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
            if (sorted.back() >= s.remote.rows[g])
                throw std::runtime_error("RDMA Engram の行番号が範囲外です");
            struct Read {
                uint64_t row;
                size_t offset;
                size_t count;
            };
            std::vector<Read> reads;
            for (size_t i = 0; i < sorted.size();) {
                size_t end = i + 1;
                while (end < sorted.size() && sorted[end] == sorted[end - 1] + 1 && end - i < 4096)
                    ++end;
                reads.push_back({sorted[i], i, end - i});
                i = end;
            }
            for (size_t base = 0; base < reads.size();) {
                std::array<ibv_send_wr, qp_count * 16> wr{};
                std::array<ibv_sge, qp_count * 16> sge{};
                size_t count = std::min(reads.size() - base, wr.size());
                for (size_t i = 0; i < count; ++i) {
                    auto &r = reads[base + i];
                    sge[i].addr = reinterpret_cast<uint64_t>(s.buffer + r.offset * 98);
                    sge[i].length = r.count * 98;
                    sge[i].lkey = s.mr->lkey;
                    wr[i].wr_id = i;
                    wr[i].sg_list = &sge[i];
                    wr[i].num_sge = 1;
                    wr[i].opcode = IBV_WR_RDMA_READ;
                    wr[i].wr.rdma.remote_addr = s.remote.address[g] + r.row * 98;
                    wr[i].wr.rdma.rkey = s.remote.rkey[g];
                }
                for (int q = 0; q < qp_count; ++q) {
                    size_t begin = size_t(q) * 16;
                    if (begin >= count)
                        break;
                    size_t end = std::min(begin + 16, count);
                    for (size_t i = begin; i + 1 < end; ++i)
                        wr[i].next = &wr[i + 1];
                    wr[end - 1].send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
                    ibv_send_wr *bad = nullptr;
                    if (ibv_post_send(s.qps[q], &wr[begin], &bad))
                        throw std::runtime_error("RDMA READ の発行に失敗しました");
                }
                auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                size_t done = 0;
                size_t expected = (count + 15) / 16;
                while (done < expected) {
                    ibv_wc completions[128];
                    int n = ibv_poll_cq(s.cq, 128, completions);
                    if (n < 0)
                        throw std::runtime_error("RDMA 完了の取得に失敗しました");
                    for (int i = 0; i < n; ++i)
                        if (completions[i].status != IBV_WC_SUCCESS)
                            throw std::runtime_error(std::string("RDMA READ に失敗しました: ") +
                                                     ibv_wc_status_str(completions[i].status));
                    done += n;
                    if (std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error("RDMA READ がタイムアウトしました");
                }
                base += count;
            }
            for (size_t i = 0; i < input.size(); ++i) {
                size_t pos = std::lower_bound(sorted.begin(), sorted.end(), input[i]) - sorted.begin();
                std::memcpy(destination + i * 98, s.buffer + pos * 98, 98);
            }
        }
        return output;
    } catch (...) {
        s.broken = true;
        throw;
    }
}
void Client::shutdown() {
    auto errors = impl->shutdown();
    if (!errors.empty())
        throw std::runtime_error("Engram RDMA 解放エラー: " + errors);
}
Client::~Client() = default;
}
