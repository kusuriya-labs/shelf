#include <shelf/inference/execution/linear_input_dump.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/runtime/graph/primitive_graph.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace shelf::execution {
namespace {

struct LinearInputDumpState {
    bool initialized = false;
    bool enabled = false;
    uint32_t max_rows_per_k = 4096u;
    uint32_t max_tensors = 40u;
    uint32_t tensors_written = 0u;
    std::unordered_map<uint32_t, std::FILE*> files;
    std::unordered_map<uint32_t, uint32_t> rows_written;
    std::FILE* meta_file = nullptr;
    std::string dir;
};

LinearInputDumpState& state() {
    static LinearInputDumpState s;
    return s;
}

void initialize(LinearInputDumpState& s) {
    s.initialized = true;
    const char* dir = std::getenv("SHELF_LINEAR_INPUT_DUMP");
    if (dir == nullptr || dir[0] == '\0') return;
    const char* rows_env = std::getenv("SHELF_LINEAR_INPUT_DUMP_MAX_ROWS");
    if (rows_env != nullptr && rows_env[0] != '\0') {
        const long parsed = std::strtol(rows_env, nullptr, 10);
        if (parsed > 0) s.max_rows_per_k = static_cast<uint32_t>(parsed);
    }
    const char* tensors_env = std::getenv("SHELF_LINEAR_INPUT_DUMP_MAX_TENSORS");
    if (tensors_env != nullptr && tensors_env[0] != '\0') {
        const long parsed = std::strtol(tensors_env, nullptr, 10);
        if (parsed > 0) s.max_tensors = static_cast<uint32_t>(parsed);
    }
    std::string base(dir);
    if (!base.empty() && base.back() == '/') base.pop_back();
    s.meta_file = std::fopen((base + "/meta.jsonl").c_str(), "wb");
    if (s.meta_file == nullptr) {
        std::fprintf(stderr, "linear-input: failed to open %s/meta.jsonl\n", base.c_str());
        return;
    }
    s.dir = base;
    s.enabled = true;
    std::fprintf(stderr, "linear-input: dumping activation inputs to %s (max_rows_per_k=%u, max_tensors=%u)\n",
                 base.c_str(), s.max_rows_per_k, s.max_tensors);
}

std::FILE* tensor_file(LinearInputDumpState& s, uint32_t k) {
    auto found = s.files.find(k);
    if (found != s.files.end()) return found->second;
    const std::string path = s.dir + "/act_k" + std::to_string(k) + ".bf16";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    s.files.emplace(k, f);
    return f;
}

// The linear dispatch that consumes the quantized activation, used to name the
// capture by the weight it feeds.
const ::shelf::runtime::DispatchBinding* consuming_linear(
    const ::shelf::runtime::Program& program, std::size_t dispatch_index) {
    for (std::size_t i = dispatch_index + 1u; i < program.dispatches.size(); ++i) {
        const auto& d = program.dispatches[i];
        if (d.kernel_id == ::shelf::runtime::KernelId::LINEAR_MXFP4_W4A8 ||
            d.kernel_id == ::shelf::runtime::KernelId::LINEAR_FP8_FINEGRAINED ||
            d.kernel_id == ::shelf::runtime::KernelId::LINEAR_BF16)
            return &d;
    }
    return nullptr;
}

}  // namespace

void linear_input_dump_maybe(
    const ::shelf::runtime::Program& program,
    std::size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    LinearInputDumpState& s = state();
    if (!s.initialized) initialize(s);
    if (!s.enabled) return;
    if (s.tensors_written >= s.max_tensors) return;
    if (dispatch_index >= program.dispatches.size()) return;
    const auto& b = program.dispatches[dispatch_index];
    if (b.kernel_id != ::shelf::runtime::KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED)
        return;

    PhysicalActivationQuantizeLaunch launch{};
    const Result<PhysicalResolveStatus> resolved =
        resolve_activation_quantize_physical(program, b, ctx, launch);
    if (!resolved.ok() || resolved.value() != PhysicalResolveStatus::Applicable) return;
    if (launch.rows == 0u || launch.k == 0u || launch.input == 0u) return;
    if (launch.input_row_stride < launch.k) return;

    const uint32_t already = s.rows_written[launch.k];
    const uint32_t room = already >= s.max_rows_per_k ? 0u : s.max_rows_per_k - already;
    const uint32_t dump_rows = launch.rows < room ? launch.rows : room;
    if (dump_rows == 0u) return;

    hipError_t herr = hipStreamSynchronize(stream);
    if (herr != hipSuccess) {
        std::fprintf(stderr, "linear-input: stream sync failed: %s\n", hipGetErrorString(herr));
        return;
    }

    std::vector<uint8_t> host(static_cast<std::size_t>(dump_rows) * launch.k * 2u);
    for (uint32_t r = 0; r < dump_rows; ++r) {
        herr = hipMemcpy(host.data() + static_cast<std::size_t>(r) * launch.k * 2u,
                         reinterpret_cast<const uint8_t*>(launch.input) +
                             static_cast<std::size_t>(r) * launch.input_row_stride * 2u,
                         static_cast<std::size_t>(launch.k) * 2u, hipMemcpyDeviceToHost);
        if (herr != hipSuccess) {
            std::fprintf(stderr, "linear-input: copy failed: %s\n", hipGetErrorString(herr));
            return;
        }
    }

    std::FILE* file = tensor_file(s, launch.k);
    if (file == nullptr) {
        std::fprintf(stderr, "linear-input: failed to open the capture file for k=%u\n", launch.k);
        return;
    }
    if (std::fwrite(host.data(), 1u, host.size(), file) != host.size()) {
        std::fprintf(stderr, "linear-input: short write for k=%u\n", launch.k);
        return;
    }
    std::fflush(file);

    const auto* lin = consuming_linear(program, dispatch_index);
    uint32_t weight_index = 0xFFFFFFFFu;
    uint32_t w_rows = 0u, w_cols = 0u, w_kp = 0u, w_enc = 0u, w_group = 0u;
    if (lin != nullptr && lin->weight_index < ctx.weight_count && ctx.weights != nullptr) {
        const ::shelf::runtime::WeightSlot e = ctx.weights[lin->weight_index];
        weight_index = lin->weight_index;
        w_rows = e.rows;
        w_cols = e.cols;
        w_kp = e.k_padded;
        w_enc = e.encoding;
        w_group = e.weight_scale_group;
    }
    std::fprintf(s.meta_file,
                 "{\"dispatch\":%zu,\"k\":%u,\"k_padded\":%u,\"rows\":%u,\"dumped_rows\":%u,"
                 "\"row_offset\":%u,\"weight_index\":%u,\"weight_rows\":%u,\"weight_cols\":%u,"
                 "\"weight_k_padded\":%u,\"weight_encoding\":%u,\"weight_scale_group\":%u}\n",
                 dispatch_index, launch.k, launch.k_padded, launch.rows, dump_rows, already,
                 weight_index, w_rows, w_cols, w_kp, w_enc, w_group);
    std::fflush(s.meta_file);

    s.rows_written[launch.k] = already + dump_rows;
    s.tensors_written += 1u;
}

}  // namespace shelf::execution
