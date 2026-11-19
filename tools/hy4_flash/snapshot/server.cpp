#include <shelf/models/hy4_flash/snapshot/wire.h>
#include <arpa/inet.h>
#include <openssl/sha.h>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <sys/mman.h>

using namespace shelf::hy4::snapshot;
volatile std::sig_atomic_t stopping = 0;
void stop_handler(int) { stopping = 1; }
struct Resources {
    ibv_context *ctx = nullptr;
    ibv_pd *pd = nullptr;
    std::array<uint8_t *, 2> buffers{};
    std::array<ibv_mr *, 2> regions{};
    size_t capacity = 0;
    int listener = -1;
    ~Resources() {
        if (listener >= 0) close(listener);
        for (int i = 0; i < 2; ++i) {
            if (regions[i]) ibv_dereg_mr(regions[i]);
            if (buffers[i]) munmap(buffers[i], capacity);
        }
        if (pd) ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }
};
struct Client {
    int fd = -1;
    ibv_cq *cq = nullptr;
    ibv_qp *qp = nullptr;
    ~Client() {
        if (fd >= 0) close(fd);
        if (qp) ibv_destroy_qp(qp);
        if (cq) ibv_destroy_cq(cq);
    }
};
int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("使用方法: 状態サーバー 1世代あたりの容量(byte)");
        char *end = nullptr;
        auto capacity = std::strtoull(argv[1], &end, 10);
        if (!end || *end || !capacity || capacity > (uint64_t(8) << 30)) throw std::runtime_error("状態領域の容量が不正です");
        struct sigaction action{};
        action.sa_handler = stop_handler;
        sigemptyset(&action.sa_mask);
        sigaction(SIGTERM, &action, nullptr);
        sigaction(SIGINT, &action, nullptr);
        Resources r;
        r.capacity = capacity;
        auto list = ibv_get_device_list(nullptr);
        if (!list) throw std::runtime_error("RDMA デバイスを列挙できません");
        for (int i = 0; list[i]; ++i)
            if (std::string(ibv_get_device_name(list[i])) == "hfi1_0") r.ctx = ibv_open_device(list[i]);
        ibv_free_device_list(list);
        if (!r.ctx) throw std::runtime_error("hfi1_0 を開けません");
        ibv_port_attr port{};
        if (ibv_query_port(r.ctx, 1, &port) || port.state != IBV_PORT_ACTIVE) throw std::runtime_error("RDMA ポートが使用できません");
        r.pd = ibv_alloc_pd(r.ctx);
        if (!r.pd) throw std::runtime_error("PD を作成できません");
        for (int i = 0; i < 2; ++i) {
            auto p = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) throw std::runtime_error("状態領域を確保できません");
            r.buffers[i] = static_cast<uint8_t *>(p);
            r.regions[i] = ibv_reg_mr(r.pd, p, capacity, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
            if (!r.regions[i]) throw std::runtime_error("状態領域を RDMA 登録できません。memlock 上限を確認してください");
        }
        r.listener = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        int reuse = 1;
        setsockopt(r.listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(19391);
        inet_pton(AF_INET, "10.255.254.1", &addr.sin_addr);
        if (r.listener < 0 || bind(r.listener, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) || listen(r.listener, 2))
            throw std::runtime_error("状態サーバーのポートを開けません");
        std::cerr << "状態 RDMA 接続待機中: 10.255.254.1:19391 容量=" << capacity << '\n';
        int active = -1;
        uint64_t length = 0, generation = 0;
        Digest checksum{};
        while (!stopping) {
            Client client;
            client.fd = accept4(r.listener, nullptr, nullptr, SOCK_CLOEXEC);
            if (client.fd < 0) {
                if (stopping) break;
                if (errno == EINTR) continue;
                throw std::runtime_error("状態接続を受け付けできません");
            }
            timeval timeout{10, 0};
            setsockopt(client.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            setsockopt(client.fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            try {
                auto request = receive_message(client.fd);
                Message reply{};
                if ((request.action != put && request.action != get) ||
                    (request.action == put && (!request.length || request.length > capacity)) ||
                    (request.action == get && active < 0)) {
                    reply.status = failed;
                    send_message(client.fd, reply);
                    continue;
                }
                int slot = request.action == put ? (active == 0 ? 1 : 0) : active;
                client.cq = ibv_create_cq(r.ctx, 8, nullptr, nullptr, 0);
                if (!client.cq) throw std::runtime_error("CQ を作成できません");
                client.qp = make_qp(r.pd, client.cq);
                shelf::hy4::rdma::connect_qp(client.qp, request.endpoint, 0);
                reply.endpoint.lid = port.lid;
                reply.endpoint.qpn[0] = client.qp->qp_num;
                reply.address = reinterpret_cast<uint64_t>(r.buffers[slot]);
                reply.rkey = r.regions[slot]->rkey;
                reply.length = request.action == get ? length : capacity;
                reply.checksum = checksum;
                reply.generation = generation;
                send_message(client.fd, reply);
                if (request.action == put) {
                    auto done = receive_message(client.fd);
                    Digest actual{};
                    if (done.action != commit || done.length != request.length) throw std::runtime_error("状態確定要求が不正です");
                    SHA256(r.buffers[slot], done.length, actual.data());
                    Message result{};
                    if (actual != done.checksum) result.status = failed;
                    else {
                        active = slot;
                        length = done.length;
                        checksum = actual;
                        ++generation;
                        result.generation = generation;
                    }
                    send_message(client.fd, result);
                } else {
                    auto done = receive_message(client.fd);
                    if (done.action != get) throw std::runtime_error("状態読み出し完了要求が不正です");
                }
            } catch (const std::exception &e) {
                std::cerr << "状態接続エラー: " << e.what() << '\n';
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "状態 RDMA サーバー: " << e.what() << '\n';
        return 1;
    }
}
