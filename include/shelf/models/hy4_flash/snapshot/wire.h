#pragma once
#include <shelf/models/hy4_flash/rdma/protocol.h>
#include <shelf/models/hy4_flash/snapshot/transport.h>

namespace shelf::hy4::snapshot {
constexpr uint64_t magic = 0x485934534e415001ULL;
constexpr uint32_t put = 1, get = 2, commit = 3, ok = 0, failed = 1;
struct Message {
    uint64_t signature = magic;
    uint32_t action = 0;
    uint32_t status = 0;
    uint64_t length = 0;
    uint64_t address = 0;
    uint32_t rkey = 0;
    uint32_t reserved = 0;
    uint64_t generation = 0;
    Digest checksum{};
    rdma::Endpoint endpoint{};
};
inline void send_message(int fd, Message &message) { rdma::transfer(fd, &message, sizeof(message), true); }
inline Message receive_message(int fd) {
    Message message{};
    rdma::transfer(fd, &message, sizeof(message), false);
    if (message.signature != magic) throw std::runtime_error("状態保存プロトコルの版が異なります");
    return message;
}
inline ibv_qp *make_qp(ibv_pd *pd, ibv_cq *cq) {
    auto qp = rdma::create_qp(pd, cq);
    ibv_qp_attr attr{};
    attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(qp, &attr, IBV_QP_ACCESS_FLAGS)) {
        ibv_destroy_qp(qp);
        throw std::runtime_error("状態保存 QP のアクセス設定に失敗しました");
    }
    return qp;
}
}
