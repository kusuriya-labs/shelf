#include <shelf/models/qwen35/runtime/dflash2_adaptive_batch.h>

#include <cstdio>
#include <memory>
#include <vector>

int main() {
    int failures = 0;
    auto check = [&](bool condition, const char* name) {
        if (!condition) {
            std::printf("FAIL %s\n", name);
            ++failures;
        }
    };
    {
        namespace rt = shelf::qwen35::runtime;
        rt::DFlash2BatcherStats stats;
        rt::DFlash2BatcherBinding binding;
        binding.num_drafts = 2u;
        binding.stats = &stats;
        rt::ContinuousBatcherConfig config;
        std::vector<rt::RuntimeRequest> live(13);
        std::vector<rt::RuntimeRequest*> active;
        for (auto& request : live) {
            request.state = rt::RequestState::Active;
            request.max_new_tokens = 100u;
            request.sequence.position = 20u;
            request.sequence.max_seq_len = 512u;
            request.dflash = std::make_unique<rt::DFlash2RequestSession>();
            request.dflash->ready = true;
            active.push_back(&request);
        }
        uint64_t round = 0;
        rt::select_dflash2_adaptive(active, binding, config, round);
        check(live[0].dflash->adaptive_rows == 0u && live[12].dflash->adaptive_rows == 0u,
              "above the capacity all requests decode normally");
        for (unsigned i = 8; i < 13; ++i) {
            live[i].dflash.reset();
            live[i].pending_decode_token = 1;
        }
        rt::select_dflash2_adaptive(active, binding, config, round);
        check(live[0].dflash->adaptive_rows == 0u,
              "normal decode requests count toward batch capacity");
        active.resize(2);
        live[1].dflash->adaptive_accepted_ema = 0.5;
        rt::select_dflash2_adaptive(active, binding, config, round);
        check(live[0].dflash->adaptive_rows == 3u && live[1].dflash->adaptive_rows == 0u,
              "low acceptance request decodes normally");
        live[1].dflash->adaptive_probe_age = 63;
        rt::select_dflash2_adaptive(active, binding, config, round);
        check(live[1].dflash->adaptive_rows == 3u, "disabled request is probed");
        rt::select_dflash2_adaptive(active, binding, config, round);
        check(live[1].dflash->adaptive_rows == 0u, "probe lasts one round");
        live[1].dflash->adaptive_accepted_ema = 1.5;
        rt::select_dflash2_adaptive(active, binding, config, round);
        check(live[1].dflash->adaptive_rows == 3u, "recovered request resumes speculation");
        active.clear();
        for (auto& request : live) {
            if (!request.dflash) {
                request.dflash = std::make_unique<rt::DFlash2RequestSession>();
                request.dflash->ready = true;
            }
            active.push_back(&request);
        }
        for (auto* request : active) {
            request->dflash->adaptive_accepted_ema = 0.5;
            request->dflash->adaptive_probe_age = 63;
        }
        rt::select_dflash2_adaptive(active, binding, config, round);
        check(live[0].dflash->adaptive_rows == 3u && live[1].dflash->adaptive_rows == 0u,
              "simultaneous probes cannot deadlock above the capacity");
    }

    return failures ? 1 : 0;
}
