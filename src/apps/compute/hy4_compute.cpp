#include <shelf/models/hy4_flash/cli.h>
#include <shelf/models/hy4_flash/residency.h>
#include <shelf/models/hy4_flash/generate.h>
#include <algorithm>
#include <shelf/io/token_ids.h>
#include <cmath>
#include <charconv>
#include <cstdio>
#include <stdexcept>
#include <string_view>

namespace shelf::hy4 {
int compute(int argc, char **argv) {
    try {
        if (argc == 1 || std::string_view(argv[1]) == "--help") {
            puts(
                "shelf-compute [hy4] [inspect|reserve|generate] --model-dir PATH --engram-dir PATH --placement "
                "PATH [--dfly-dir PATH] [--max-seq-len N] [--input-ids-file PATH] [--max-new-tokens N] "
                "[--chunk N] [--temperature T] [--seed N] [--dump-logits PATH] [--repeats N]");
            return 0;
        }
        bool implicit = std::string_view(argv[1]).starts_with("--");
        std::string_view mode = implicit ? "generate" : argv[1];
        Path model, engram, draft, placement;
        size_t tokens = mode == "reserve" ? 2097152 : 4096;
        Path input, logits_file;
        int count = 32, chunk = 64, repeats = 1;
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
            else if (key == "--input-ids-file")
                input = value;
            else if (key == "--max-new-tokens" || key == "--chunk" || key == "--repeats") {
                auto &dest = key == "--chunk" ? chunk : key == "--repeats" ? repeats : count;
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
        if (mode != "inspect" && mode != "reserve" && mode != "generate")
            throw std::runtime_error("未知の動作です");
        if (model.empty() || engram.empty() || placement.empty())
            throw std::runtime_error("重みと配置表が必要です");
        Checkpoint checkpoint(model, engram, draft, read_owners(placement));
        printf("DFly=%d weight_bytes=%zu,%zu,%zu\n", checkpoint.has_draft, checkpoint.bytes[0],
               checkpoint.bytes[1], checkpoint.bytes[2]);
        if (mode == "inspect")
            return 0;
        std::vector<int> ids;
        if (mode == "generate") {
            ids = io::read_input_ids(input.string());
            if (std::any_of(ids.begin(), ids.end(), [](int id) { return id < 0 || id >= 120832; }))
                throw std::runtime_error("入力 ID が範囲外です");
            if (ids.empty() || ids.size() + count > tokens || tokens > 1048576 || chunk > 128)
                throw std::runtime_error("入力列または生成長が不正です");
        }
        for (;;) {
            Residency gpu(checkpoint, tokens);
            try {
                gpu.reserve();
                if (mode == "generate") {
                    gpu.upload();
                    for (int pass = 0; pass < repeats; ++pass)
                        generate(gpu, ids, count, chunk, temperature, seed, pass ? Path{} : logits_file);
                }
            } catch (const std::bad_alloc &) {
                gpu.shutdown();
                if (mode == "reserve" && tokens == 2097152) {
                    fprintf(stderr, "2Mi の領域が不足しています。1Mi で再試行します。\n");
                    tokens = 1048576;
                    continue;
                }
                throw;
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
