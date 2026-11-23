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

void GraphPlan::launch(Residency &gpu) {
    auto root = gpu.compute[0];
    if (!layers.front().scatter) {
        checked(hipGraphLaunch(tail, root));
        return;
    }
    if (!events_ready) {
        for (int slot = 0; slot < 2; ++slot) {
            checked(hipSetDevice(gpu.physical[0]));
            if (!replay_done[slot])
                checked(hipEventCreateWithFlags(&replay_done[slot], hipEventDisableTiming));
            for (auto &layer : layers) {
                if (!layer.scatter_done[slot])
                    checked(hipEventCreateWithFlags(&layer.scatter_done[slot], hipEventDisableTiming | hipEventReleaseToSystem));
                if (layer.local && !layer.local_done[slot])
                    checked(hipEventCreateWithFlags(&layer.local_done[slot], hipEventDisableTiming));
                for (int r = 1; r < gpu.ep_size; ++r) {
                    checked(hipSetDevice(gpu.physical[r]));
                    if (!layer.expert_done[slot][r])
                        checked(hipEventCreateWithFlags(&layer.expert_done[slot][r], hipEventDisableTiming | hipEventReleaseToSystem));
                }
                checked(hipSetDevice(gpu.physical[0]));
            }
        }
        events_ready = true;
    }
    unsigned slot = replay_slot;
    if (replay_recorded[slot]) {
        auto status = hipEventQuery(replay_done[slot]);
        if (status == hipErrorNotReady)
            checked(hipEventSynchronize(replay_done[slot]));
        else
            checked(status);
    }
    for (const auto &layer : layers) {
        checked(hipGraphLaunch(layer.scatter, root));
        if (gpu.ep_size > 1) checked(hipEventRecord(layer.scatter_done[slot], root));
        if (layer.local) {
            checked(hipStreamWaitEvent(gpu.local_expert[0], layer.scatter_done[slot], 0));
            checked(hipGraphLaunch(layer.local, gpu.local_expert[0]));
            checked(hipEventRecord(layer.local_done[slot], gpu.local_expert[0]));
        }
        for (int r = 1; r < gpu.ep_size; ++r) {
            checked(hipSetDevice(gpu.physical[r]));
            checked(hipStreamWaitEvent(gpu.compute[r], layer.scatter_done[slot], 0));
            checked(hipGraphLaunch(layer.expert[r], gpu.compute[r]));
            checked(hipEventRecord(layer.expert_done[slot][r], gpu.compute[r]));
        }
        checked(hipSetDevice(gpu.physical[0]));
        checked(hipGraphLaunch(layer.expert[0], root));
        if (layer.local) checked(hipStreamWaitEvent(root, layer.local_done[slot], 0));
        for (int r = 1; r < gpu.ep_size; ++r)
            checked(hipStreamWaitEvent(root, layer.expert_done[slot][r], 0));
    }
    checked(hipGraphLaunch(tail, root));
    checked(hipEventRecord(replay_done[slot], root));
    replay_recorded[slot] = true;
    replay_slot ^= 1;
}

void GraphPlan::release(Residency &gpu, std::string &errors) {
    auto destroy = [&](hipGraphExec_t &exec) {
        if (exec) {
            collect(errors, hipGraphExecDestroy(exec));
            exec = nullptr;
        }
    };
    auto destroy_event = [&](hipEvent_t &event) {
        if (event) {
            collect(errors, hipEventDestroy(event));
            event = nullptr;
        }
    };
    for (int r = gpu.ranks - 1; r >= 0; --r) {
        collect(errors, hipSetDevice(gpu.physical[r]));
        for (auto &layer : layers) {
            destroy(layer.expert[r]);
            for (int slot = 0; slot < 2; ++slot) {
                destroy_event(layer.expert_done[slot][r]);
                if (!r) {
                    destroy_event(layer.scatter_done[slot]);
                    destroy_event(layer.local_done[slot]);
                }
            }
            if (!r) {
                destroy(layer.scatter);
                destroy(layer.local);
            }
        }
        if (!r) {
            destroy(tail);
            destroy(entry);
            for (auto &event : replay_done) destroy_event(event);
        }
    }
    key = {};
    replay_recorded = {};
    replay_slot = 0;
    events_ready = false;
}
}
