#include <shelf/models/hy4_flash/snapshot/wire.h>
#include <arpa/inet.h>
#include <openssl/sha.h>
#include <chrono>
#include <cerrno>
#include <fcntl.h>
#include <cstdlib>
#include <cstring>
#include <poll.h>

namespace shelf::hy4::snapshot {
Digest digest(std::span<const uint8_t> data) {
    Digest result{};
    SHA256(data.data(), data.size(), result.data());
    return result;
}
namespace {
struct Connection {
    int fd = -1;
    ibv_context *ctx = nullptr;
    ibv_pd *pd = nullptr;
    ibv_cq *cq = nullptr;
    ibv_qp *qp = nullptr;
    ibv_mr *mr = nullptr;
    uint8_t *buffer = nullptr;
    static constexpr size_t chunk = 4 * 1024 * 1024;
    ~Connection() {
        if (fd >= 0) close(fd);
        if (qp) ibv_destroy_qp(qp);
        if (mr) ibv_dereg_mr(mr);
        free(buffer);
        if (cq) ibv_destroy_cq(cq);
        if (pd) ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }
    void open(uint32_t action, uint64_t length, Message &answer) {
        auto list = ibv_get_device_list(nullptr);
        if (!list) throw std::runtime_error("RDMA デバイスを列挙できません");
        for (int i = 0; list[i]; ++i)
            if (std::string(ibv_get_device_name(list[i])) == "hfi1_0") ctx = ibv_open_device(list[i]);
        ibv_free_device_list(list);
        if (!ctx) throw std::runtime_error("hfi1_0 を開けません");
        ibv_port_attr port{};
        if (ibv_query_port(ctx, 1, &port) || port.state != IBV_PORT_ACTIVE) throw std::runtime_error("RDMA ポートが使用できません");
        pd = ibv_alloc_pd(ctx);
        if (!pd) throw std::runtime_error("PD を作成できません");
        cq = ibv_create_cq(ctx, 8, nullptr, nullptr, 0);
        if (!cq) throw std::runtime_error("CQ を作成できません");
        qp = make_qp(pd, cq);
        if (posix_memalign(reinterpret_cast<void **>(&buffer), 4096, chunk)) throw std::bad_alloc();
        mr = ibv_reg_mr(pd, buffer, chunk, IBV_ACCESS_LOCAL_WRITE);
        if (!mr) throw std::runtime_error("状態転送バッファを登録できません");
        fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0) throw std::runtime_error("状態サーバーへの socket を作成できません");
        timeval timeout{10, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        inet_pton(AF_INET, "10.255.254.2", &addr.sin_addr);
        if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr))) throw std::runtime_error("状態転送元を bind できません");
        addr.sin_port = htons(19391);
        inet_pton(AF_INET, "10.255.254.1", &addr.sin_addr);
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK)) throw std::runtime_error("状態接続を設定できません");
        if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr))) {
            if (errno != EINPROGRESS) throw std::runtime_error("状態サーバーに接続できません");
            pollfd pending{fd, POLLOUT, 0};
            int status = poll(&pending, 1, 10000);
            int error = 0;
            socklen_t length = sizeof(error);
            if (status <= 0 || getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length) || error)
                throw std::runtime_error("状態サーバーに接続できません");
        }
        if (fcntl(fd, F_SETFL, flags)) throw std::runtime_error("状態接続を設定できません");
        Message request{};
        request.action = action;
        request.length = length;
        request.endpoint.lid = port.lid;
        request.endpoint.qpn[0] = qp->qp_num;
        send_message(fd, request);
        answer = receive_message(fd);
        if (answer.status != ok) throw std::runtime_error("状態サーバーが要求を拒否しました");
        rdma::connect_qp(qp, answer.endpoint, 0);
    }
    void transfer_bytes(uint8_t *data, size_t size, const Message &remote, bool write) {
        for (size_t offset = 0; offset < size;) {
            size_t count = std::min(chunk, size - offset);
            if (write) std::memcpy(buffer, data + offset, count);
            ibv_sge sge{};
            sge.addr = reinterpret_cast<uint64_t>(buffer);
            sge.length = count;
            sge.lkey = mr->lkey;
            ibv_send_wr wr{};
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = write ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
            wr.send_flags = IBV_SEND_SIGNALED;
            wr.wr.rdma.remote_addr = remote.address + offset;
            wr.wr.rdma.rkey = remote.rkey;
            ibv_send_wr *bad = nullptr;
            if (ibv_post_send(qp, &wr, &bad)) throw std::runtime_error("状態 RDMA を発行できません");
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            ibv_wc wc{};
            int n;
            while (!(n = ibv_poll_cq(cq, 1, &wc)))
                if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("状態 RDMA がタイムアウトしました");
            if (n < 0 || wc.status != IBV_WC_SUCCESS) throw std::runtime_error("状態 RDMA が失敗しました");
            if (!write) std::memcpy(data + offset, buffer, count);
            offset += count;
        }
    }
};
}
void save(std::span<const uint8_t> data) {
    if (data.empty()) throw std::runtime_error("空の状態は保存できません");
    Connection conn;
    Message remote{};
    conn.open(put, data.size(), remote);
    conn.transfer_bytes(const_cast<uint8_t *>(data.data()), data.size(), remote, true);
    Message request{};
    request.action = commit;
    request.length = data.size();
    request.checksum = digest(data);
    send_message(conn.fd, request);
    auto reply = receive_message(conn.fd);
    if (reply.status != ok) throw std::runtime_error("状態の確定に失敗しました");
}
std::vector<uint8_t> load(size_t limit) {
    Connection conn;
    Message remote{};
    conn.open(get, 0, remote);
    if (!remote.length || remote.length > limit) throw std::runtime_error("保存状態のサイズが範囲外です");
    std::vector<uint8_t> result(remote.length);
    conn.transfer_bytes(result.data(), result.size(), remote, false);
    if (digest(result) != remote.checksum) throw std::runtime_error("保存状態のチェックサムが一致しません");
    Message done{};
    done.action = get;
    send_message(conn.fd, done);
    return result;
}
}
