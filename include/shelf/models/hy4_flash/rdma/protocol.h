#pragma once

#include <array>
#include <cstdint>
#include <infiniband/verbs.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace shelf::hy4::rdma {
constexpr uint32_t version = 0x48593401;
constexpr int qp_count = 8;
struct Endpoint {
    uint32_t version_id;
    uint16_t lid;
    uint16_t reserved;
    std::array<uint32_t, qp_count> qpn;
    std::array<uint32_t, qp_count> psn;
    std::array<uint64_t, 4> address;
    std::array<uint64_t, 4> rows;
    std::array<uint32_t, 4> rkey;
};
inline void transfer(int fd, void *data, size_t bytes, bool send_data) {
    auto p = static_cast<uint8_t *>(data);
    while (bytes) {
        auto n = send_data ? ::send(fd, p, bytes, MSG_NOSIGNAL) : ::recv(fd, p, bytes, 0);
        if (n <= 0)
            throw std::runtime_error("RDMA 制御接続が切断されました");
        p += n;
        bytes -= n;
    }
}
inline void connect_qp(ibv_qp *qp, const Endpoint &peer, int index) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = peer.qpn[index];
    attr.rq_psn = peer.psn[index];
    attr.max_dest_rd_atomic = 16;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = peer.lid;
    attr.ah_attr.port_num = 1;
    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
        throw std::runtime_error("RDMA QP の RTR 移行に失敗しました");
    attr = {};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 3;
    attr.rnr_retry = 3;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 16;
    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC))
        throw std::runtime_error("RDMA QP の RTS 移行に失敗しました");
}
inline ibv_qp *create_qp(ibv_pd *pd, ibv_cq *cq) {
    ibv_qp_init_attr init{};
    init.send_cq = cq;
    init.recv_cq = cq;
    init.cap.max_send_wr = 128;
    init.cap.max_recv_wr = 1;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    init.qp_type = IBV_QPT_RC;
    auto qp = ibv_create_qp(pd, &init);
    if (!qp)
        throw std::runtime_error("RDMA QP の作成に失敗しました");
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        ibv_destroy_qp(qp);
        throw std::runtime_error("RDMA QP の初期化に失敗しました");
    }
    return qp;
}
}
