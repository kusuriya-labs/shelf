#include <shelf/models/hy4_flash/cli.h>
#include <shelf/models/hy4_flash/residency.h>
#include <shelf/models/hy4_flash/generate.h>
#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>
#include <poll.h>
#include <unistd.h>
#include <shelf/io/token_ids.h>
#include <cstdlib>
#include <cerrno>
#include <fstream>
#include <sstream>
#include <cmath>
#include <charconv>
#include <cstdio>
#include <stdexcept>
#include <string_view>

namespace shelf::hy4 {
namespace {
using Json = nlohmann::json;

void reply(const Json &event) {
    std::cout << event.dump() << std::endl;
}

void reply_stats(const Residency &gpu, size_t active, size_t queued, size_t draft_rounds) {
    size_t free_slots = gpu.max_slots - active;
    reply({{"event", "stats"}, {"max_concurrent_requests", gpu.max_slots}, {"slot_tokens", gpu.tokens},
           {"active_requests", active}, {"queued_requests", queued}, {"batch_draft_rounds", draft_rounds},
           {"kv", {{"page_tokens", gpu.tokens}, {"total_pages", gpu.max_slots}, {"used_pages", active},
                   {"free_pages", free_slots}, {"capacity_tokens", gpu.max_slots * gpu.tokens},
                   {"free_tokens", free_slots * gpu.tokens}, {"active_requests", active},
                   {"queued_requests", queued}, {"banker_safe", true}, {"overcommit_pages", 0}}}});
}

std::array<unsigned long long, 2> page_faults() {
    std::ifstream file("/proc/self/stat");
    std::string line, field;
    if (!std::getline(file, line))
        throw std::runtime_error("/proc/self/stat を取得できません");
    auto close = line.rfind(')');
    if (close == std::string::npos)
        throw std::runtime_error("/proc/self/stat の形式が不正です");
    std::istringstream stream(line.substr(close + 1));
    std::array<unsigned long long, 2> faults{};
    for (int i = 3; i <= 12; ++i) {
        if (!(stream >> field))
            throw std::runtime_error("/proc/self/stat の項目が不足しています");
        if (i == 10) faults[0] = std::stoull(field);
        if (i == 12) faults[1] = std::stoull(field);
    }
    return faults;
}

void serve(Residency &gpu, int chunk) {
    struct Interrupted {};
    bool stopping = false;
    auto control = [&](const Json &request, int active) {
        const auto op = request.at("op").get<std::string>();
        if (op == "ping")
            reply({{"event", "pong"}});
        else if (op == "stats")
            reply_stats(gpu, active >= 0, 0, 0);
        else if (op == "shutdown") {
            stopping = true;
            reply({{"event", "shutdown"}});
            if (active >= 0) throw Interrupted{};
        } else if (op == "cancel") {
            if (active >= 0 && request.at("request_id") == active) throw Interrupted{};
        } else
            return false;
        return true;
    };
    std::setvbuf(stdin, nullptr, _IONBF, 0);
    std::string line;
    while (!stopping && std::getline(std::cin, line)) {
        int id = -1;
        try {
            const auto request = Json::parse(line);
            if (control(request, -1)) continue;
            id = request.at("request_id").get<int>();
            if (request.at("op") != "generate" || request.value("grammar", false) ||
                request.value("reasoning_budget_tokens", 0) || request.value("min_p", 0.f) != 0)
                throw std::invalid_argument("Hy4 の生成経路が対応していない指定です");
            auto ids = request.at("input_ids").get<std::vector<int>>();
            int count = request.at("max_new_tokens").get<int>();
            float temp = request.value("temperature", 0.f);
            if (count <= 0 || ids.empty() || ids.size() + count > gpu.tokens ||
                !std::isfinite(temp) || temp < 0 ||
                std::any_of(ids.begin(), ids.end(), [](int t) { return t < 0 || t >= 120832; }))
                throw std::invalid_argument("入力 ID、生成長、または温度が範囲外です");
            auto emit = [&](int token) {
                pollfd input{STDIN_FILENO, POLLIN, 0};
                while (poll(&input, 1, 0) > 0) {
                    if (!std::getline(std::cin, line)) { stopping = true; throw Interrupted{}; }
                    auto cmd = Json::parse(line);
                    if (!control(cmd, id))
                        reply({{"event", "error"}, {"request_id", cmd.value("request_id", -1)},
                               {"message", "Hy4 は一度に 1 件の生成を処理します"}});
                }
                reply({{"event", "token"}, {"request_id", id}, {"token_id", token}});
            };
            fprintf(stderr, "hy4_request=%d input=%zu limit=%d temperature=%.3f top_p=%.3f top_k=%d\n",
                    id, ids.size(), count, temp, request.value("top_p", 1.f), request.value("top_k", 0));
            const auto profile = std::getenv("SHELF_HY4_PROFILE_EP");
            bool profiling = profile && std::string_view(profile) == "1";
            std::array<unsigned long long, 2> before{};
            if (profiling) {
                gpu.model.report_engram_residency("request_before");
                before = page_faults();
            }
            auto result = generate(gpu, ids, count, chunk, temp, request.value("seed", uint64_t{}), {},
                                   request.value("top_k", 0), request.value("top_p", 1.f), emit);
            if (profiling) {
                auto after = page_faults();
                fprintf(stderr, "hy4_faults request=%d minor=%llu major=%llu\n", id,
                        after[0] - before[0], after[1] - before[1]);
                gpu.model.report_engram_residency("request_after");
            }
            reply({{"event", "done"}, {"request_id", id}, {"generated_ids", result},
                   {"finish_reason", result.back() == 120025 ? "stop" : "length"}});
        } catch (const Interrupted &) {
            reply({{"event", "done"}, {"request_id", id}, {"finish_reason", "cancelled"}});
        } catch (const std::invalid_argument &e) {
            reply({{"event", "error"}, {"request_id", id}, {"code", "invalid_argument"}, {"message", e.what()}});
        } catch (const Json::exception &e) {
            reply({{"event", "error"}, {"request_id", id}, {"code", "invalid_argument"}, {"message", e.what()}});
        }
    }
}
void serve_batch(Residency &gpu, int chunk) {
    bool stopping = false, eof = false;
    std::string pending;
    Hy4BatchSession session(gpu,
        [](int id, int token) { reply({{"event", "token"}, {"request_id", id}, {"token_id", token}}); },
        [](int id, const std::vector<int> &ids, const char *reason) {
            reply({{"event", "done"}, {"request_id", id}, {"generated_ids", ids}, {"finish_reason", reason}});
        });
    auto handle = [&](const std::string &line) {
        int id = -1;
        try {
            auto request = Json::parse(line);
            auto op = request.at("op").get<std::string>();
            if (op == "ping") reply({{"event", "pong"}});
            else if (op == "stats")
                reply_stats(gpu, session.active(), session.queued(), session.draft_rounds());
            else if (op == "shutdown") { stopping = true; reply({{"event", "shutdown"}}); }
            else if (op == "cancel") session.cancel(request.at("request_id").get<int>());
            else {
                id = request.at("request_id").get<int>();
                if (op != "generate" || request.value("grammar", false) || request.value("reasoning_budget_tokens", 0) ||
                    request.value("min_p", 0.f) != 0)
                    throw std::invalid_argument("Hy4 の生成経路が対応していない指定です");
                Hy4Request item{id, request.at("input_ids").get<std::vector<int>>(),
                    request.at("max_new_tokens").get<int>(),
                    {chunk, request.value("temperature", 0.f), request.value("seed", uint64_t{}),
                     request.value("top_k", 0), request.value("top_p", 1.f)}};
                session.admit(std::move(item));
            }
        } catch (const std::invalid_argument &e) {
            reply({{"event", "error"}, {"request_id", id}, {"code", "invalid_argument"}, {"message", e.what()}});
        } catch (const Json::exception &e) {
            reply({{"event", "error"}, {"request_id", id}, {"code", "invalid_argument"}, {"message", e.what()}});
        }
    };
    while (!stopping && !eof) {
        pollfd input{STDIN_FILENO, POLLIN, 0};
        int ready = poll(&input, 1, session.active() || session.queued() ? 0 : -1);
        if (ready < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("標準入力の監視に失敗しました");
        }
        if (ready && (input.revents & (POLLIN | POLLHUP))) {
            char buffer[16384];
            auto count = read(STDIN_FILENO, buffer, sizeof(buffer));
            if (count < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("標準入力を読み取れません");
            }
            if (!count) eof = true;
            else pending.append(buffer, size_t(count));
            size_t end;
            while ((end = pending.find('\n')) != std::string::npos) {
                auto line = pending.substr(0, end);
                pending.erase(0, end + 1);
                handle(line);
                if (stopping) break;
            }
        }
        if (!stopping && !eof && (session.active() || session.queued())) session.step();
    }
}

}
int compute(int argc, char **argv) {
    try {
        if (argc == 1 || std::string_view(argv[1]) == "--help") {
            puts(
                "shelf-compute [hy4] [inspect|reserve|generate|serve] --model-dir PATH --engram-dir PATH --placement "
                "PATH [--dfly-dir PATH] [--dfly-drafts N] [--max-seq-len N] [--input-ids-file PATH] [--max-new-tokens N] "
                "[--chunk N] [--temperature T] [--seed N] [--dump-logits PATH] [--repeats N] "
                "[--expert-parallel N] [--tensor-parallel N] [--max-concurrent-requests N] [--batch-size N] "
                "[--batch-input-files PATH1,PATH2,...] [--batch-max-new-tokens N1,N2,...]");
            return 0;
        }
        bool implicit = std::string_view(argv[1]).starts_with("--");
        std::string_view mode = implicit ? "generate" : argv[1];
        Path model, engram, draft, placement;
        size_t tokens = mode == "reserve" ? 2097152 : 4096;
        Path input, logits_file;
        int count = 32, chunk = 64, repeats = 1, ep = 0, tp = 1, batch_size = 1, max_concurrent = 1, dfly_drafts = 0;
        std::string batch_files, batch_counts;
        float temperature = 0;
        uint64_t seed = 0;
        for (int i = implicit ? 1 : 2; i < argc; i += 2) {
            if (i + 1 >= argc)
                throw std::runtime_error("引数の値がありません");
            std::string_view key = argv[i], value = argv[i + 1];
            if (key == "--model-dir")
                model = value;
            else if (key == "--engram-dir")
                engram = value;
            else if (key == "--dfly-dir")
                draft = value;
            else if (key == "--dfly-drafts")
                dfly_drafts = Residency::parse_draft_tokens(value);
            else if (key == "--temperature") {
                auto [p, err] = std::from_chars(value.data(), value.data() + value.size(), temperature);
                if (p != value.data() + value.size() || err != std::errc{} || !std::isfinite(temperature) ||
                    temperature < 0)
                    throw std::runtime_error("温度が不正です");
            } else if (key == "--seed") {
                auto [p, err] = std::from_chars(value.data(), value.data() + value.size(), seed);
                if (p != value.data() + value.size() || err != std::errc{})
                    throw std::runtime_error("乱数 seed が不正です");
            } else if (key == "--dump-logits")
                logits_file = value;
            else if (key == "--batch-input-files")
                batch_files = value;
            else if (key == "--batch-max-new-tokens")
                batch_counts = value;
            else if (key == "--input-ids-file")
                input = value;
            else if (key == "--expert-parallel" || key == "--tensor-parallel") {
                auto &dest = key == "--tensor-parallel" ? tp : ep;
                auto [p, err] = std::from_chars(value.data(), value.data() + value.size(), dest);
                if (p != value.data() + value.size() || err != std::errc{} || dest < 1 || dest > 4)
                    throw std::runtime_error("EP / TP の値が不正です");
            } else if (key == "--max-new-tokens" || key == "--chunk" || key == "--repeats" || key == "--batch-size" || key == "--max-concurrent-requests") {
                auto &dest = key == "--chunk" ? chunk : key == "--repeats" ? repeats :
                             key == "--batch-size" ? batch_size : key == "--max-concurrent-requests" ? max_concurrent : count;
                auto [p, err] = std::from_chars(value.data(), value.data() + value.size(), dest);
                if (p != value.data() + value.size() || err != std::errc{} || dest <= 0)
                    throw std::runtime_error("長さが不正です");
            } else if (key == "--placement")
                placement = value;
            else if (key == "--max-seq-len") {
                auto end = value.data() + value.size();
                auto [p, err] = std::from_chars(value.data(), end, tokens);
                if (p != end || err != std::errc{} || !tokens || tokens > 2097152)
                    throw std::runtime_error("予約長が不正です");
            } else
                throw std::runtime_error("未知の引数です: " + std::string(key));
        }
        if (mode != "inspect" && mode != "reserve" && mode != "generate" && mode != "serve")
            throw std::runtime_error("未知の動作です");
        if (model.empty() || engram.empty() || placement.empty())
            throw std::runtime_error("重みと配置表が必要です");
        if (batch_size > 16 || max_concurrent > 16 || (max_concurrent != 1 && mode != "serve" && mode != "reserve"))
            throw std::invalid_argument("スロット数または動作の指定が不正です");
        dfly_drafts = Residency::resolve_draft_tokens(dfly_drafts);
        const int slots = mode == "generate" ? batch_size : max_concurrent;
        auto topology = Residency::check_batch_topology(tp, slots);
        if (!topology.ok())
            throw std::runtime_error(topology.message());
        int visible = 0;
        if (hipGetDeviceCount(&visible) != hipSuccess || visible < 1 || visible > 4)
            throw std::runtime_error("R9700 を 1〜4 台指定してください");
        if (!ep)
            ep = visible;
        if (ep * tp != visible)
            throw std::runtime_error("EP × TP と GPU 数が一致しません");
        Checkpoint checkpoint(model, engram, draft, read_owners(placement), ep, tp);
        if (mode == "serve" && (tokens > 2097152 || chunk > max_prefill_chunk))
            throw std::invalid_argument("サーバーの予約長または chunk が範囲外です");
        fprintf(mode == "serve" ? stderr : stdout, "DFly=%d weight_bytes=%zu,%zu,%zu,%zu\n", checkpoint.has_draft, checkpoint.bytes[0],
               checkpoint.bytes[1], checkpoint.bytes[2], checkpoint.bytes[3]);
        if (mode == "inspect")
            return 0;
        if (mode == "generate") {
            auto populate = std::getenv("SHELF_HY4_ENGRAM_POPULATE");
            if (populate && std::string_view(populate) == "1")
                checkpoint.populate_engram(false);
        }
        std::vector<std::vector<int>> prompts;
        std::vector<int> counts;
        auto split = [](const std::string &value) {
            std::vector<std::string> parts;
            std::istringstream stream(value);
            std::string part;
            while (std::getline(stream, part, ',')) {
                if (part.empty()) throw std::invalid_argument("バッチ指定に空の項目があります");
                parts.push_back(part);
            }
            if (!value.empty() && value.back() == ',') throw std::invalid_argument("バッチ指定に空の項目があります");
            return parts;
        };
        if (mode == "generate") {
            if (!batch_files.empty()) {
                for (const auto &file : split(batch_files)) prompts.push_back(io::read_input_ids(file));
                if (batch_size != 1 && size_t(batch_size) != prompts.size())
                    throw std::invalid_argument("バッチサイズと入力ファイル数が一致しません");
                batch_size = int(prompts.size());
                if (batch_size > 16) throw std::invalid_argument("バッチの上限は 16 です");
            } else {
                prompts.assign(batch_size, io::read_input_ids(input.string()));
            }
            counts.assign(batch_size, count);
            if (!batch_counts.empty()) {
                auto parts = split(batch_counts);
                if (parts.size() != counts.size()) throw std::invalid_argument("バッチの生成長数が一致しません");
                for (size_t i = 0; i < parts.size(); ++i) {
                    auto [p, err] = std::from_chars(parts[i].data(), parts[i].data() + parts[i].size(), counts[i]);
                    if (err != std::errc{} || p != parts[i].data() + parts[i].size() || counts[i] <= 0)
                        throw std::invalid_argument("バッチの生成長が不正です");
                }
            }
            if (batch_size > 1 && !logits_file.empty())
                throw std::invalid_argument("バッチ生成では logits 出力を使用できません");
            for (size_t i = 0; i < prompts.size(); ++i)
                if (prompts[i].empty() || prompts[i].size() + counts[i] > tokens || tokens > 1048576 ||
                    chunk > max_prefill_chunk || std::any_of(prompts[i].begin(), prompts[i].end(),
                    [](int id) { return id < 0 || id >= 120832; }))
                    throw std::runtime_error("入力列または生成長が不正です");
        } else if (batch_size != 1 || !batch_files.empty() || !batch_counts.empty())
            throw std::invalid_argument("バッチ指定は generate 専用です");
        for (;;) {
            Residency gpu(checkpoint, tokens, mode == "generate" ? batch_size : max_concurrent, dfly_drafts);
            try {
                auto status = gpu.reserve();
                if (!status.ok())
                    throw std::runtime_error(status.message());
                if (mode == "serve") {
                    gpu.upload();
                    auto populate = std::getenv("SHELF_HY4_ENGRAM_POPULATE");
                    auto locked = std::getenv("SHELF_HY4_ENGRAM_MLOCK");
                    bool lock = locked && std::string_view(locked) == "1";
                    if (lock && populate && std::string_view(populate) == "0")
                        throw std::invalid_argument("Engram の読み込みを無効にしたまま固定できません");
                    if (!populate || std::string_view(populate) != "0")
                        checkpoint.populate_engram(lock);
                    if (auto profile = std::getenv("SHELF_HY4_PROFILE_EP");
                        profile && std::string_view(profile) == "1")
                        checkpoint.report_engram_residency("ready");
                    if (max_concurrent == 1) serve(gpu, chunk);
                    else serve_batch(gpu, chunk);
                } else if (mode == "generate") {
                    gpu.upload();
                    for (int pass = 0; pass < repeats; ++pass) {
                        if (batch_size > 1) {
                            auto results = generate_batch(gpu, prompts, counts, {chunk, temperature, seed});
                            for (size_t slot = 0; slot < results.size(); ++slot) {
                                printf("GENERATED_IDS[%zu]=", slot);
                                for (size_t i = 0; i < results[slot].size(); ++i)
                                    printf("%s%d", i ? "," : "", results[slot][i]);
                                puts("");
                            }
                            continue;
                        }
                        auto result = generate(gpu, prompts.front(), counts.front(), chunk, temperature, seed, pass ? Path{} : logits_file);
                        printf("GENERATED_IDS=");
                        for (size_t i = 0; i < result.size(); ++i)
                            printf("%s%d", i ? "," : "", result[i]);
                        puts("");
                    }
                }
            } catch (...) {
                gpu.shutdown();
                throw;
            }
            gpu.shutdown();
            break;
        }
        if (mode == "reserve")
            printf("reserved_tokens=%zu\n", tokens);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "Hy4-flash: %s\n", e.what());
        return 1;
    }
}
}

#ifdef SHELF_HY4_ONLY
int main(int argc, char **argv) {
    if (argc > 1 && std::string_view(argv[1]) == "hy4")
        return shelf::hy4::compute(argc - 1, argv + 1);
    return shelf::hy4::compute(argc, argv);
}
#endif
