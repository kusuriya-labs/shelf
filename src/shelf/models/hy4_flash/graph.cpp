#include <shelf/models/hy4_flash/ops.h>

namespace shelf::hy4 {
namespace {
void collect(std::string &errors, hipError_t status) {
    if (status != hipSuccess)
        errors += std::string(hipGetErrorString(status)) + "\n";
}
}
void GraphCapture::begin(int rank) {
    checked(hipSetDevice(gpu.physical[rank]));
    checked(hipStreamBeginCapture(gpu.compute[rank], hipStreamCaptureModeThreadLocal));
    active = rank;
}

void GraphCapture::finish(hipGraphExec_t &exec) {
    hipGraph_t graph{};
    auto status = hipStreamEndCapture(gpu.compute[active], &graph);
    active = -1;
    checked(status);
    status = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    checked(hipGraphDestroy(graph));
    checked(status);
}

void GraphCapture::cancel(std::string &errors) {
    if (active < 0)
        return;
    collect(errors, hipSetDevice(gpu.physical[active]));
    hipGraph_t graph{};
    auto status = hipStreamEndCapture(gpu.compute[active], &graph);
    active = -1;
    if (graph)
        collect(errors, hipGraphDestroy(graph));
    if (status != hipErrorStreamCaptureInvalidated)
        collect(errors, status);
}

void GraphPlan::launch(Residency &gpu) const {
    auto root = gpu.compute[0];
    for (const auto &layer : layers) {
        checked(hipGraphLaunch(layer.scatter, root));
        for (int r = 1; r < gpu.ep_size; ++r)
            checked(hipEventRecord(gpu.ready[0][r][0], root));
        for (int r = 1; r < gpu.ep_size; ++r) {
            checked(hipSetDevice(gpu.physical[r]));
            checked(hipStreamWaitEvent(gpu.compute[r], gpu.ready[0][r][0], 0));
            checked(hipGraphLaunch(layer.expert[r], gpu.compute[r]));
            checked(hipEventRecord(gpu.ready[r][0][0], gpu.compute[r]));
        }
        checked(hipSetDevice(gpu.physical[0]));
        checked(hipGraphLaunch(layer.expert[0], root));
        for (int r = 1; r < gpu.ep_size; ++r)
            checked(hipStreamWaitEvent(root, gpu.ready[r][0][0], 0));
    }
    checked(hipGraphLaunch(tail, root));
}

void GraphPlan::release(Residency &gpu, std::string &errors) {
    auto destroy = [&](hipGraphExec_t &exec) {
        if (exec) {
            collect(errors, hipGraphExecDestroy(exec));
            exec = nullptr;
        }
    };
    for (int r = gpu.ranks - 1; r >= 0; --r) {
        collect(errors, hipSetDevice(gpu.physical[r]));
        for (auto &layer : layers) {
            destroy(layer.expert[r]);
            if (!r)
                destroy(layer.scatter);
        }
        if (!r) {
            destroy(tail);
            destroy(entry);
        }
    }
    key = {};
}
}
