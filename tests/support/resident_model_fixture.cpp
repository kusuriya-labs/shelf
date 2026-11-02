#include "resident_model_fixture.h"

#include <phaseshift/resident/resident_protocol.h>

#include <cstdio>

namespace ps {
namespace resident {

namespace {

void release_attachment(ps::resident::ResidentModelAttachment& attachment) {
    if (!attachment.valid()) {
        return;
    }
    const ps::resident::ResidentModelKey key = attachment.key();
    attachment = ps::resident::ResidentModelAttachment();
    auto path = ps::resident::resident_socket_path(key);
    if (path.ok()) {
        (void)ps::resident::resident_host_release(path.value(), key.to_string());
    }
}

}

void ResidentModelFixture::release() {
    release_attachment(qwen_);
    release_attachment(dflash_);
}

Result<ResidentModelFixture> ResidentModelFixture::acquire(
    const ResidentTestRequest& request) {
    ResidentModelFixture fixture;
    fixture.requested_arena_bytes_ = request.arena_bytes;

    if (resident_model_disabled()) {
        return fixture;
    }

    if (!request.qwen_model_dir.empty()) {
        const ResidentModelKey key = ResidentModelKey::make(
            request.qwen_model_dir, ModelKind::Qwen35Target,
            resident_load_flags(request.qwen_options), request.tp_size,
            request.tp_rank, request.device);
        auto attachment = acquire_resident_model(key, request.device);
        if (!attachment.ok()) {
            std::fprintf(stderr, "resident fixture: qwen35 acquire failed: %s\n",
                         attachment.status().message().c_str());
            return attachment.status();
        }
        fixture.qwen_ = attachment.release();
    }

    if (!request.dflash2_model_dir.empty()) {
        const ResidentModelKey key = ResidentModelKey::make(
            request.dflash2_model_dir, ModelKind::DFlash2Draft,
            request.dflash2_options.preshuffle ? kLoadFlagPreshuffle : 0u, 1, 0,
            request.device);
        auto attachment = acquire_resident_model(key, request.device);
        if (!attachment.ok()) {
            std::fprintf(stderr, "resident fixture: dflash2 acquire failed: %s\n",
                         attachment.status().message().c_str());
            return attachment.status();
        }
        fixture.dflash_ = attachment.release();
    }

    if (request.arena_bytes > 0 && fixture.arena_bytes() == 0) {
        std::fprintf(stderr,
                     "resident fixture: persistent_bytes=%zu does not fit arena budget=%zu\n",
                     fixture.persistent_bytes(), request.arena_bytes);
        return Status::insufficient_memory(
            "resident weights do not fit the requested arena budget", __FILE__, __LINE__);
    }
    return fixture;
}

}
}
