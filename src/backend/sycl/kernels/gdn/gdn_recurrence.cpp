// vendored from PhaseShift (own work); namespace/type adapted only.
//
// Ported from phaseshift recurrence.hip `gdn_recurrence_f32_impl` (portable
// decode recurrence; one warp of 32 threads owns 32 head_v columns of one
// (request, v_head) and walks the head_k rows of the state matrix).
// Algebra per value head (ground truth; matches shelf IR gdn_step CPU ref):
//   beta  = sigmoid(b)
//   g     = -exp(A_log) * softplus(a + dt_bias)   [softplus = log(1+exp(x))]
//   d_t   = exp(g)
//   kv_mem[j] = sum_i S_old[i][j] * d_t * k[i]            (decayed read)
//   delta[j]  = (v[j] - kv_mem[j]) * beta
//   S[i][j]   = S_old[i][j] * d_t + k[i] * delta[j]
//   out[j]    = sum_i S_new[i][j] * q[i]
// shelf adaptations: recurrent_state fp32 (PhaseShift fp16) — direct float
// access; GdnRequestDescriptor instead of ps::runtime::DeviceRequestDescriptor;
// the gfx12 WMMA chunk/decode1/decode_rows/commit variants (fp16 state, fixed
// head shapes) are not ported.
// (SYCL port of backend/hip kernels/gdn/gdn_recurrence.hip.)
#include <shelf/backend/sycl/kernels/gdn/gdn_recurrence.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

#include <cstdint>
#include <cstddef>
#include <cmath>

namespace shelf::syclk {

namespace {

inline float bf16f(const shelf::bf16_t* p) { return ::shelf::bf16_to_f32(*p); }

inline float gdn_softplus_exact(float x) {
    return ::sycl::log(1.0f + ::sycl::exp(x));
}

} // namespace

::shelf::Status launch_gdn_recurrence_f32(
    const GdnRecurrenceArgs& args, const ::shelf::sycl::Stream& stream) {
    if (args.head_k == 0u || args.head_k > 128u)
        return ::shelf::Status::invalid_argument("bad head_k", __FILE__, __LINE__);
    if (args.head_v == 0u)
        return ::shelf::Status::invalid_argument("head_v == 0", __FILE__, __LINE__);
    if (args.key_heads == 0u || args.num_v_heads == 0u)
        return ::shelf::Status::invalid_argument("bad head counts", __FILE__, __LINE__);
    if (args.num_v_heads % args.key_heads != 0u)
        return ::shelf::Status::invalid_argument("v heads % key heads != 0", __FILE__, __LINE__);
    if (args.num_requests == 0u || args.actual_rows == 0u)
        return ::shelf::Status::invalid_argument("empty launch rejected", __FILE__, __LINE__);
    if (args.q == nullptr || args.k == nullptr || args.v == nullptr || args.a == nullptr || args.b == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (args.dt_bias == nullptr || args.a_log == nullptr)
        return ::shelf::Status::invalid_argument("null weight", __FILE__, __LINE__);
    if (args.recurrent_state == nullptr || args.output == nullptr)
        return ::shelf::Status::invalid_argument("null state/output", __FILE__, __LINE__);
    if (args.requests == nullptr || args.max_sequences == 0u)
        return ::shelf::Status::invalid_argument("null requests", __FILE__, __LINE__);
    const uint32_t v_tiles = (args.head_v + 31u) >> 5;
    const uint32_t grid_x = args.num_requests * args.num_v_heads * v_tiles;
    if (grid_x == 0u)
        return ::shelf::Status::invalid_argument("empty grid", __FILE__, __LINE__);

    const GdnRecurrenceArgs a = args;
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{grid_x * 32u}, ::sycl::range<1>{32u}},
            [=](::sycl::nd_item<1> it) {
                const uint32_t head_k = a.head_k;
                const uint32_t head_v = a.head_v;
                const uint32_t vt = (head_v + 31u) >> 5;

                const uint32_t task = static_cast<uint32_t>(it.get_group(0));
                const uint32_t tile = task % vt;
                const uint32_t tmp = task / vt;
                const uint32_t v_head = tmp % a.num_v_heads;
                const uint32_t request_index = tmp / a.num_v_heads;

                const uint32_t lane = static_cast<uint32_t>(it.get_local_linear_id());
                const uint32_t vcol = tile * 32u + lane;
                const bool vcol_valid = vcol < head_v;

                if (request_index >= a.num_requests) return;
                const GdnRequestDescriptor req = a.requests[request_index];
                const uint32_t slot = req.slot;
                if (slot >= a.max_sequences) return;
                const uint32_t row_begin = req.row_begin;
                const uint32_t row_count = req.row_count;

                const uint32_t repeat = a.num_v_heads / a.key_heads;
                const uint32_t qk_head = v_head / repeat;

                float* state_head = a.recurrent_state
                    + static_cast<uint64_t>(slot) * a.recurrent_slot_stride
                    + static_cast<uint64_t>(a.state_index) * a.recurrent_layer_stride
                    + static_cast<uint64_t>(v_head) * head_k * head_v;
                float* state_col = state_head + vcol;

                const float dt_bias = bf16f(a.dt_bias + v_head);
                const float neg_exp_a_log = -::sycl::exp(bf16f(a.a_log + v_head));

                const float* k_head = a.k + static_cast<size_t>(qk_head) * head_k;
                const float* q_head = a.q + static_cast<size_t>(qk_head) * head_k;

                for (uint32_t t = 0u; t < row_count; ++t) {
                    const uint32_t row = row_begin + t;
                    if (row >= a.actual_rows) continue;

                    const float b_val = bf16f(a.b + static_cast<size_t>(row) * a.b_row_stride + v_head);
                    const float a_val = bf16f(a.a + static_cast<size_t>(row) * a.a_row_stride + v_head);
                    const float beta = 1.0f / (1.0f + ::sycl::exp(-b_val));
                    const float dt = a_val + dt_bias;
                    const float g = neg_exp_a_log * gdn_softplus_exact(dt);
                    const float d_t = ::sycl::exp(g);

                    const float* k_row = k_head + static_cast<size_t>(row) * a.k_row_stride;
                    const float* q_row = q_head + static_cast<size_t>(row) * a.q_row_stride;

                    const float v_val = vcol_valid
                        ? bf16f(a.v + static_cast<size_t>(row) * a.v_row_stride + static_cast<size_t>(v_head) * head_v + vcol)
                        : 0.0f;

                    if (!vcol_valid) continue;

                    float kv_mem = 0.0f;
                    for (uint32_t kk = 0u; kk < head_k; ++kk) {
                        const float s = state_col[static_cast<uint64_t>(kk) * head_v];
                        kv_mem += s * d_t * k_row[kk];
                    }

                    const float delta = (v_val - kv_mem) * beta;

                    for (uint32_t kk = 0u; kk < head_k; ++kk) {
                        float* sc = state_col + static_cast<uint64_t>(kk) * head_v;
                        *sc = *sc * d_t + k_row[kk] * delta;
                    }

                    float acc = 0.0f;
                    for (uint32_t kk = 0u; kk < head_k; ++kk) {
                        acc += state_col[static_cast<uint64_t>(kk) * head_v] * q_row[kk];
                    }

                    a.output[static_cast<size_t>(row) * a.output_row_stride + static_cast<size_t>(v_head) * head_v + vcol] = acc;
                }
            });
    });
}

} // namespace shelf::syclk
