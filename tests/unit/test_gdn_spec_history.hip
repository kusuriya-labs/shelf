#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/gdn/conv1d.h>
#include <shelf/kernels/optimized/gdn/recurrence.h>
#include <shelf/models/qwen35/runtime/gdn_spec_history.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using ps::bf16_t;
using ps::fp16_t;
using ps::state::GdnStatePool;
using ps::state::GdnStatePoolDeviceView;
using ps::state::GdnStatePoolLayout;
using ps::qwen35::runtime::GdnSpecHistory;
using ps::qwen35::runtime::create_gdn_spec_history;
using ps::qwen35::runtime::gdn_spec_history_view;
using ps::qwen35::runtime::restore_gdn_spec_history;
using ps::qwen35::runtime::shutdown_gdn_spec_history;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static bool alloc_bf16(
    ps::gpu::GpuArena& arena, const std::vector<std::size_t>& shape, ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(bf16_t), 256);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = ps::gpu::Tensor::create<bf16_t>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static bool alloc_f32(
    ps::gpu::GpuArena& arena, const std::vector<std::size_t>& shape, ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(float), 256);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = ps::gpu::Tensor::create<float>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static bool alloc_u8(
    ps::gpu::GpuArena& arena, std::size_t bytes, ps::gpu::Tensor& out) {
    auto view = arena.allocate_aligned(bytes, 256);
    if (!view.ok()) return false;
    auto backing = view.release();
    auto t = ps::gpu::Tensor::create<uint8_t>(backing, {bytes}, {1u});
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static bool same_bytes(const void* a, const void* b, std::size_t n) {
    return std::memcmp(a, b, n) == 0;
}

static float randf(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
}

static ps::bf16_t randb(uint32_t& s) {
    return ps::bf16_t{ps::f32_to_bf16_rne(randf(s) * 0.5f)};
}

struct RequestBuf {
    ps::gpu::Tensor t;
    ::ps::runtime::DeviceRequestDescriptor* ptr = nullptr;
};

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    auto arena_result = ps::gpu::GpuArena::create(0, 2ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        return 1;
    }

    const uint32_t kRows = 4u;
    const uint32_t kCapture = 3u;
    const uint32_t kKeyHeads = 16u;
    const uint32_t kNumVHeads = 48u;
    const uint32_t kHeadK = 128u;
    const uint32_t kHeadV = 128u;
    const uint32_t q_feat = kKeyHeads * kHeadK;
    const uint32_t v_feat = kNumVHeads * kHeadV;
    const uint32_t conv_dim = 2u * q_feat + v_feat;

    const GdnStatePoolLayout layout =
        GdnStatePoolLayout::make(1u, conv_dim, 3u, kNumVHeads, kHeadK, kHeadV);
    auto pool_result = GdnStatePool::create(arena, 1u, layout);
    check(pool_result.ok(), "pool create");
    if (!pool_result.ok()) return 1;
    GdnStatePool pool = pool_result.release();
    check(pool.initialize(0u, stream).ok(), "pool initialize");
    check(pool.reset(0u, stream).ok(), "pool reset");
    (void)hipStreamSynchronize(stream);

    auto history_result = create_gdn_spec_history(arena, pool, kRows);
    check(history_result.ok(), "history create");
    if (!history_result.ok()) return 1;
    GdnSpecHistory history = history_result.release();
    const GdnStatePoolDeviceView view = pool.device_view();
    check(history.conv_state_bytes ==
              static_cast<std::size_t>(view.conv_slot_stride) * sizeof(bf16_t),
          "history conv bytes");
    check(history.recurrent_state_bytes ==
              static_cast<std::size_t>(view.recurrent_slot_stride) * sizeof(fp16_t),
          "history recurrent bytes");
    std::printf("GATE11C_HISTORY conv_state_bytes=%zu recurrent_state_bytes=%zu rows=%u\n",
                history.conv_state_bytes, history.recurrent_state_bytes, history.rows);

    {
        auto bad = gdn_spec_history_view(history, kRows + 1u);
        check(bad.recurrent_history == nullptr && bad.capture_rows == 0u,
              "history view rejects overflow rows");
    }

    ps::gpu::Tensor q;
    ps::gpu::Tensor kk;
    ps::gpu::Tensor v;
    ps::gpu::Tensor a;
    ps::gpu::Tensor b;
    ps::gpu::Tensor dt;
    ps::gpu::Tensor al;
    ps::gpu::Tensor out;
    ps::gpu::Tensor conv_in;
    ps::gpu::Tensor conv_w;
    ps::gpu::Tensor conv_out;
    ps::gpu::Tensor req_mem;
    const bool alloc_ok =
        alloc_f32(arena, {kRows, q_feat}, q) && alloc_f32(arena, {kRows, q_feat}, kk) &&
        alloc_bf16(arena, {kRows, v_feat}, v) && alloc_bf16(arena, {kRows, kNumVHeads}, a) &&
        alloc_bf16(arena, {kRows, kNumVHeads}, b) && alloc_bf16(arena, {kNumVHeads}, dt) &&
        alloc_bf16(arena, {kNumVHeads}, al) && alloc_f32(arena, {kRows, v_feat}, out) &&
        alloc_bf16(arena, {kRows, conv_dim}, conv_in) &&
        alloc_bf16(arena, {conv_dim * 4u}, conv_w) &&
        alloc_f32(arena, {kRows, conv_dim}, conv_out) &&
        alloc_u8(arena, sizeof(::ps::runtime::DeviceRequestDescriptor), req_mem);
    check(alloc_ok, "buffer alloc");
    if (!alloc_ok) return 1;

    {
        std::vector<float> hq(static_cast<std::size_t>(kRows) * q_feat);
        std::vector<float> hk(static_cast<std::size_t>(kRows) * q_feat);
        std::vector<bf16_t> hv(static_cast<std::size_t>(kRows) * v_feat);
        std::vector<bf16_t> ha(static_cast<std::size_t>(kRows) * kNumVHeads);
        std::vector<bf16_t> hb(static_cast<std::size_t>(kRows) * kNumVHeads);
        std::vector<bf16_t> hdt(kNumVHeads);
        std::vector<bf16_t> hal(kNumVHeads);
        std::vector<bf16_t> hci(static_cast<std::size_t>(kRows) * conv_dim);
        std::vector<bf16_t> hcw(static_cast<std::size_t>(conv_dim) * 4u);
        uint32_t seed = 0x12345678u;
        for (float& x : hq) x = randf(seed);
        for (float& x : hk) x = randf(seed);
        for (bf16_t& x : hv) x = randb(seed);
        for (bf16_t& x : ha) x = randb(seed);
        for (bf16_t& x : hb) x = randb(seed);
        for (bf16_t& x : hdt) x = randb(seed);
        for (bf16_t& x : hal) x = randb(seed);
        for (bf16_t& x : hci) x = randb(seed);
        for (bf16_t& x : hcw) x = randb(seed);
        check(hipMemcpy(q.data<float>(), hq.data(), hq.size() * sizeof(float),
                        hipMemcpyHostToDevice) == hipSuccess,
              "q upload");
        check(hipMemcpy(kk.data<float>(), hk.data(), hk.size() * sizeof(float),
                        hipMemcpyHostToDevice) == hipSuccess,
              "k upload");
        check(hipMemcpy(v.data<bf16_t>(), hv.data(), hv.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "v upload");
        check(hipMemcpy(a.data<bf16_t>(), ha.data(), ha.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "a upload");
        check(hipMemcpy(b.data<bf16_t>(), hb.data(), hb.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "b upload");
        check(hipMemcpy(dt.data<bf16_t>(), hdt.data(), hdt.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "dt upload");
        check(hipMemcpy(al.data<bf16_t>(), hal.data(), hal.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "a_log upload");
        check(hipMemcpy(conv_in.data<bf16_t>(), hci.data(), hci.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "conv input upload");
        check(hipMemcpy(conv_w.data<bf16_t>(), hcw.data(), hcw.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "conv weight upload");
    }

    auto* req = reinterpret_cast<::ps::runtime::DeviceRequestDescriptor*>(
        req_mem.data<uint8_t>());
    {
        ::ps::runtime::DeviceRequestDescriptor hreq{};
        hreq.request_handle.slot = 0u;
        hreq.request_handle.generation = 1u;
        hreq.row_begin = 0u;
        hreq.row_count = kRows;
        check(hipMemcpy(req, &hreq, sizeof(hreq), hipMemcpyHostToDevice) == hipSuccess,
              "request upload");
    }

    const std::size_t rec_slot_bytes = history.recurrent_state_bytes;
    const std::size_t conv_slot_bytes = history.conv_state_bytes;
    std::vector<std::vector<fp16_t>> clean_rec(kRows + 1u);
    std::vector<std::vector<bf16_t>> clean_conv(kRows + 1u);

    auto make_rec_args = [&](uint32_t actual_rows) {
        ::ps::kernel::GdnRecurrenceArgs r{};
        r.q = q.data<float>();
        r.k = kk.data<float>();
        r.v = v.data<bf16_t>();
        r.a = a.data<bf16_t>();
        r.b = b.data<bf16_t>();
        r.dt_bias = dt.data<bf16_t>();
        r.a_log = al.data<bf16_t>();
        r.recurrent_state = view.recurrent_base;
        r.output = out.data<float>();
        r.requests = req;
        r.num_requests = 1u;
        r.actual_rows = actual_rows;
        r.q_row_stride = q_feat;
        r.k_row_stride = q_feat;
        r.v_row_stride = v_feat;
        r.a_row_stride = kNumVHeads;
        r.b_row_stride = kNumVHeads;
        r.output_row_stride = v_feat;
        r.state_index = 0u;
        r.max_sequences = pool.max_sequences();
        r.num_gdn_states = pool.num_gdn_states();
        r.key_heads = kKeyHeads;
        r.num_v_heads = kNumVHeads;
        r.head_k = kHeadK;
        r.head_v = kHeadV;
        r.recurrent_slot_stride = view.recurrent_slot_stride;
        r.recurrent_layer_stride = view.recurrent_layer_stride;
        return r;
    };

    auto make_conv_args = [&](uint32_t actual_rows) {
        ::ps::kernel::GdnConv1dArgs c{};
        c.input = conv_in.data<bf16_t>();
        c.output = conv_out.data<float>();
        c.conv_weight = conv_w.data<bf16_t>();
        c.conv_state = view.conv_base;
        c.requests = req;
        c.num_requests = 1u;
        c.actual_rows = actual_rows;
        c.max_request_rows = actual_rows;
        c.input_row_stride = conv_dim;
        c.output_row_stride = conv_dim;
        c.conv_dim = conv_dim;
        c.conv_history = 3u;
        c.state_index = 0u;
        c.max_sequences = pool.max_sequences();
        c.conv_slot_stride = view.conv_slot_stride;
        c.conv_layer_stride = view.conv_layer_stride;
        c.conv_history_stride = view.conv_history_stride;
        return c;
    };

    for (uint32_t n = 1u; n <= kCapture; ++n) {
        check(pool.reset(0u, stream).ok(), "clean reset");
        auto ra = make_rec_args(n);
        check(::ps::kernel::launch_gdn_recurrence_f32_wmma_decode1_serial(ra, stream) ==
                  hipSuccess,
              "clean recurrence launch");
        auto ca = make_conv_args(n);
        check(::ps::kernel::launch_gdn_conv1d_bf16_f32(ca, stream) == hipSuccess,
              "clean conv launch");
        check(hipStreamSynchronize(stream) == hipSuccess, "clean sync");
        clean_rec[n].resize(rec_slot_bytes / sizeof(fp16_t));
        clean_conv[n].resize(conv_slot_bytes / sizeof(bf16_t));
        check(hipMemcpy(clean_rec[n].data(), view.recurrent_base, rec_slot_bytes,
                        hipMemcpyDeviceToHost) == hipSuccess,
              "clean rec read");
        check(hipMemcpy(clean_conv[n].data(), view.conv_base, conv_slot_bytes,
                        hipMemcpyDeviceToHost) == hipSuccess,
              "clean conv read");
    }

    check(pool.reset(0u, stream).ok(), "capture reset");
    {
        auto ra = make_rec_args(kCapture);
        ra.recurrent_history = history.recurrent.data<fp16_t>();
        ra.recurrent_history_row_stride = static_cast<uint64_t>(history.recurrent.dim(1));
        ra.capture_rows = kCapture;
        check(::ps::kernel::launch_gdn_recurrence_f32_wmma_decode1_serial(ra, stream) ==
                  hipSuccess,
              "capture recurrence launch");
        auto ca = make_conv_args(kCapture);
        ca.conv_history_store = history.conv.data<bf16_t>();
        ca.conv_history_row_stride = static_cast<uint64_t>(history.conv.dim(1));
        ca.capture_rows = kCapture;
        check(::ps::kernel::launch_gdn_conv1d_bf16_f32(ca, stream) == hipSuccess,
              "capture conv launch");
        check(hipStreamSynchronize(stream) == hipSuccess, "capture sync");
    }

    for (uint32_t n = 1u; n <= kCapture; ++n) {
        std::vector<fp16_t> hist_rec(rec_slot_bytes / sizeof(fp16_t));
        std::vector<bf16_t> hist_conv(conv_slot_bytes / sizeof(bf16_t));
        const fp16_t* rec_row =
            history.recurrent.data<fp16_t>() +
            static_cast<std::size_t>(n - 1u) * history.recurrent.dim(1);
        const bf16_t* conv_row =
            history.conv.data<bf16_t>() +
            static_cast<std::size_t>(n - 1u) * history.conv.dim(1);
        check(hipMemcpy(hist_rec.data(), rec_row, rec_slot_bytes, hipMemcpyDeviceToHost) ==
                  hipSuccess,
              "history rec read");
        check(hipMemcpy(hist_conv.data(), conv_row, conv_slot_bytes,
                        hipMemcpyDeviceToHost) == hipSuccess,
              "history conv read");
        check(same_bytes(hist_rec.data(), clean_rec[n].data(), rec_slot_bytes),
              "recurrent history row " + std::to_string(n));
        check(same_bytes(hist_conv.data(), clean_conv[n].data(), conv_slot_bytes),
              "conv history row " + std::to_string(n));
    }

    check(pool.reset(0u, stream).ok(), "restore reset");
    check(restore_gdn_spec_history(pool, 0u, history, 1u, stream).ok(), "history restore");
    check(hipStreamSynchronize(stream) == hipSuccess, "restore sync");
    {
        std::vector<fp16_t> live_rec(rec_slot_bytes / sizeof(fp16_t));
        std::vector<bf16_t> live_conv(conv_slot_bytes / sizeof(bf16_t));
        check(hipMemcpy(live_rec.data(), view.recurrent_base, rec_slot_bytes,
                        hipMemcpyDeviceToHost) == hipSuccess,
              "restore rec read");
        check(hipMemcpy(live_conv.data(), view.conv_base, conv_slot_bytes,
                        hipMemcpyDeviceToHost) == hipSuccess,
              "restore conv read");
        const fp16_t* rec_row =
            history.recurrent.data<fp16_t>() + 1u * history.recurrent.dim(1);
        const bf16_t* conv_row = history.conv.data<bf16_t>() + 1u * history.conv.dim(1);
        std::vector<fp16_t> want_rec(rec_slot_bytes / sizeof(fp16_t));
        std::vector<bf16_t> want_conv(conv_slot_bytes / sizeof(bf16_t));
        (void)hipMemcpy(want_rec.data(), rec_row, rec_slot_bytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(want_conv.data(), conv_row, conv_slot_bytes, hipMemcpyDeviceToHost);
        check(same_bytes(live_rec.data(), want_rec.data(), rec_slot_bytes),
              "restore recurrent exact");
        check(same_bytes(live_conv.data(), want_conv.data(), conv_slot_bytes),
              "restore conv exact");
    }

    auto set_req_rows = [&](uint32_t rows) {
        ::ps::runtime::DeviceRequestDescriptor hreq{};
        hreq.request_handle.slot = 0u;
        hreq.request_handle.generation = 1u;
        hreq.row_begin = 0u;
        hreq.row_count = rows;
        (void)hipMemcpy(req, &hreq, sizeof(hreq), hipMemcpyHostToDevice);
    };

    ps::gpu::Tensor parents_dev;
    check(alloc_u8(arena, 4u * sizeof(int32_t), parents_dev), "parents alloc");
    {
        std::vector<int32_t> parents(kCapture);
        for (uint32_t i = 0; i < kCapture; ++i)
            parents[i] = static_cast<int32_t>(i) - 1;
        check(hipMemcpy(parents_dev.data<uint8_t>(), parents.data(),
                        parents.size() * sizeof(int32_t), hipMemcpyHostToDevice) ==
                  hipSuccess,
              "parents upload");
        const int32_t* parents_dev_ptr =
            reinterpret_cast<const int32_t*>(parents_dev.data<uint8_t>());
        set_req_rows(kCapture);
        check(pool.reset(0u, stream).ok(), "tree chain reset");
        auto ra = make_rec_args(kCapture);
        ra.recurrent_history = history.recurrent.data<fp16_t>();
        ra.recurrent_history_row_stride =
            static_cast<uint64_t>(history.recurrent.dim(1));
        ra.capture_rows = kCapture;
        ra.row_parent = parents_dev_ptr;
        check(::ps::kernel::launch_gdn_recurrence_f32_wmma_decode1_serial(ra, stream) ==
                  hipSuccess,
              "tree chain recurrence launch");
        auto ca = make_conv_args(kCapture);
        ca.conv_history_store = history.conv.data<bf16_t>();
        ca.conv_history_row_stride = static_cast<uint64_t>(history.conv.dim(1));
        ca.capture_rows = kCapture;
        ca.row_parent = parents_dev_ptr;
        check(::ps::kernel::launch_gdn_conv1d_bf16_f32(ca, stream) == hipSuccess,
              "tree chain conv launch");
        check(hipStreamSynchronize(stream) == hipSuccess, "tree chain sync");
        std::vector<fp16_t> trec(rec_slot_bytes / sizeof(fp16_t));
        std::vector<bf16_t> tconv(conv_slot_bytes / sizeof(bf16_t));
        check(hipMemcpy(trec.data(), view.recurrent_base, rec_slot_bytes,
                        hipMemcpyDeviceToHost) == hipSuccess,
              "tree chain rec read");
        check(hipMemcpy(tconv.data(), view.conv_base, conv_slot_bytes,
                        hipMemcpyDeviceToHost) == hipSuccess,
              "tree chain conv read");
        check(same_bytes(trec.data(), clean_rec[kCapture].data(), rec_slot_bytes),
              "tree chain parents recurrence equals chain");
        check(same_bytes(tconv.data(), clean_conv[kCapture].data(), conv_slot_bytes),
              "tree chain parents conv equals chain");
    }

    {
        const int32_t parents_a[4] = {-1, 0, 1, 1};
        check(hipMemcpy(parents_dev.data<uint8_t>(), parents_a, sizeof(parents_a),
                        hipMemcpyHostToDevice) == hipSuccess,
              "parents a upload");
        const int32_t* parents_a_dev =
            reinterpret_cast<const int32_t*>(parents_dev.data<uint8_t>());
        set_req_rows(4u);
        check(pool.reset(0u, stream).ok(), "tree branch reset A");
        auto ra = make_rec_args(4u);
        ra.recurrent_history = history.recurrent.data<fp16_t>();
        ra.recurrent_history_row_stride =
            static_cast<uint64_t>(history.recurrent.dim(1));
        ra.capture_rows = kCapture;
        ra.row_parent = parents_a_dev;
        check(::ps::kernel::launch_gdn_recurrence_f32_wmma_decode1_serial(ra, stream) ==
                  hipSuccess,
              "tree branch recurrence launch A");
        auto ca = make_conv_args(4u);
        ca.conv_history_store = history.conv.data<bf16_t>();
        ca.conv_history_row_stride = static_cast<uint64_t>(history.conv.dim(1));
        ca.capture_rows = kCapture;
        ca.row_parent = parents_a_dev;
        check(::ps::kernel::launch_gdn_conv1d_bf16_f32(ca, stream) == hipSuccess,
              "tree branch conv launch A");
        check(hipStreamSynchronize(stream) == hipSuccess, "tree branch sync A");
        std::vector<float> rec_a(v_feat);
        std::vector<float> conv_a(conv_dim);
        check(hipMemcpy(rec_a.data(), out.data<float>() + 3u * v_feat,
                        v_feat * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess,
              "tree branch rec row3 read");
        check(hipMemcpy(conv_a.data(), conv_out.data<float>() + 3u * conv_dim,
                        conv_dim * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess,
              "tree branch conv row3 read");

        auto copy_row3_to_row2 = [&](auto* base, uint32_t stride, std::size_t bytes) {
            return hipMemcpy(base + static_cast<std::size_t>(2u) * stride,
                             base + static_cast<std::size_t>(3u) * stride,
                             static_cast<std::size_t>(stride) * bytes,
                             hipMemcpyDeviceToDevice) == hipSuccess;
        };
        check(copy_row3_to_row2(q.data<float>(), q_feat, sizeof(float)),
              "copy q row");
        check(copy_row3_to_row2(kk.data<float>(), q_feat, sizeof(float)),
              "copy k row");
        check(copy_row3_to_row2(v.data<bf16_t>(), v_feat, sizeof(bf16_t)),
              "copy v row");
        check(copy_row3_to_row2(a.data<bf16_t>(), kNumVHeads, sizeof(bf16_t)),
              "copy a row");
        check(copy_row3_to_row2(b.data<bf16_t>(), kNumVHeads, sizeof(bf16_t)),
              "copy b row");
        check(copy_row3_to_row2(conv_in.data<bf16_t>(), conv_dim, sizeof(bf16_t)),
              "copy conv row");

        set_req_rows(3u);
        check(pool.reset(0u, stream).ok(), "tree branch reset B");
        auto rb = make_rec_args(3u);
        check(::ps::kernel::launch_gdn_recurrence_f32_wmma_decode1_serial(rb, stream) ==
                  hipSuccess,
              "tree branch recurrence launch B");
        auto cb = make_conv_args(3u);
        check(::ps::kernel::launch_gdn_conv1d_bf16_f32(cb, stream) == hipSuccess,
              "tree branch conv launch B");
        check(hipStreamSynchronize(stream) == hipSuccess, "tree branch sync B");
        std::vector<float> rec_b(v_feat);
        std::vector<float> conv_b(conv_dim);
        check(hipMemcpy(rec_b.data(), out.data<float>() + 2u * v_feat,
                        v_feat * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess,
              "tree branch rec row2 read");
        check(hipMemcpy(conv_b.data(), conv_out.data<float>() + 2u * conv_dim,
                        conv_dim * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess,
              "tree branch conv row2 read");
        check(same_bytes(rec_a.data(), rec_b.data(), v_feat * sizeof(float)),
              "tree branch parent recurrence equals chain control");
        check(same_bytes(conv_a.data(), conv_b.data(), conv_dim * sizeof(float)),
              "tree branch parent conv equals chain control");
    }

    check(shutdown_gdn_spec_history(history).ok(), "history shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    check(arena.shutdown().ok(), "arena shutdown");
    std::printf("test_gdn_spec_history: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
