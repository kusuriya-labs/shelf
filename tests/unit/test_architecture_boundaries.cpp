#include <cstdio>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>

#ifndef PHASESHIFT_SOURCE_ROOT
#error "PHASESHIFT_SOURCE_ROOT must be defined"
#endif

namespace fs = std::filesystem;

namespace {

int passed = 0, failed = 0;

void fail(const std::string& msg) {
    ++failed;
    printf("FAIL: %s\n", msg.c_str());
}

bool contains(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string read_file(const fs::path& p) {
    std::string out;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (f == nullptr) return out;
    char buf[8192];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

std::vector<fs::path> scan(const fs::path& root) {
    std::vector<fs::path> files;
    if (!fs::exists(root)) return files;
    for (auto it = fs::recursive_directory_iterator(root);
         it != fs::recursive_directory_iterator(); ++it) {
        if (!it->is_regular_file()) continue;
        const auto ext = it->path().extension().string();
        if (ext == ".h" || ext == ".hpp" || ext == ".cpp" || ext == ".cc" ||
            ext == ".hip") {
            files.push_back(it->path());
        }
    }
    return files;
}

void scan_clean(const fs::path& root, const std::vector<const char*>& forbidden) {
    if (fs::is_regular_file(root)) {
        const std::string text = read_file(root);
        for (const char* needle : forbidden) {
            if (contains(text, needle)) {
                fail(root.string() + std::string(" contains forbidden ") + needle);
            }
        }
        ++passed;
        return;
    }
    for (const auto& file : scan(root)) {
        const std::string text = read_file(file);
        for (const char* needle : forbidden) {
            if (contains(text, needle)) {
                fail(file.string() + std::string(" contains forbidden ") + needle);
            }
        }
        ++passed;
    }
}

void check_correctness_location(const fs::path& root) {
    struct RootPair {
        fs::path search;
        fs::path allowed;
    };
    const std::vector<RootPair> roots = {
        {root / "src/phaseshift/models/qwen35",
         root / "src/phaseshift/kernels/correctness"},
        {root / "include/phaseshift/models/qwen35",
         root / "include/phaseshift/kernels/correctness"},
    };
    for (const auto& rp : roots) {
        for (const auto& file : scan(rp.search)) {
            if (file.filename().string().find("_correctness.") == std::string::npos) {
                continue;
            }
            const std::string path = file.string();
            if (path.rfind(rp.allowed.string(), 0) != 0) {
                fail(path + " must live under kernels/correctness/");
            } else {
                ++passed;
            }
        }
    }
}

void check_correctness_layout(const fs::path& root) {
    const fs::path src = root / "src/phaseshift/kernels/correctness";
    const std::string dispatch = "model_dispatch_correctness.hip";
    for (const auto& file : scan(src)) {
        if (file.extension() != ".hip") continue;
        const std::string rel = fs::relative(file, src).string();
        if (rel == dispatch || rel.rfind("standalone/", 0) == 0) {
            ++passed;
        } else {
            fail(file.string() +
                 " must be model_dispatch_correctness.hip or under correctness/standalone/");
        }
    }
}

void check_no_server_deps(const fs::path& root) {
    const std::vector<const char*> forbidden = {
        "LocalAI", "localai", "protobuf", "Protobuf",
        "gRPC", "grpc", "OpenAI", "openai",
        "xgrammar", "XGrammar",
    };
    scan_clean(root / "src/phaseshift", forbidden);
    scan_clean(root / "include/phaseshift", forbidden);
}

}

int main() {
    const fs::path root(PHASESHIFT_SOURCE_ROOT);

    const std::vector<fs::path> core_dirs = {
        root / "include/phaseshift/runtime",
        root / "src/phaseshift/runtime",
    };
    for (const auto& dir : core_dirs) {
        scan_clean(dir, {"qwen35", "Qwen35", "phaseshift/models/"});
    }

    const std::vector<fs::path> generic_dirs = {
        root / "include/phaseshift/kernels",
        root / "src/phaseshift/kernels",
        root / "include/phaseshift/inference",
        root / "src/phaseshift/inference",
        root / "include/phaseshift/models/common",
        root / "src/phaseshift/models/common",
    };
    for (const auto& dir : generic_dirs) {
        scan_clean(dir, {"qwen35", "Qwen35", "qwen4exp", "Qwen4Exp",
                         "phaseshift/models/qwen"});
    }

    const std::vector<fs::path> weights_dirs = {
        root / "include/phaseshift/weights",
        root / "src/phaseshift/weights",
    };
    for (const auto& dir : weights_dirs) {
        scan_clean(dir, {"qwen35", "Qwen35", "phaseshift/models/"});
    }

    const std::vector<fs::path> format_dirs = {
        root / "include/phaseshift/quantization/manifest",
        root / "include/phaseshift/quantization/bundle",
        root / "include/phaseshift/quantization/codec",
        root / "src/phaseshift/quantization/manifest",
        root / "src/phaseshift/quantization/bundle",
        root / "src/phaseshift/quantization/codec",
        root / "include/phaseshift/quantization/quant_format.h",
        root / "include/phaseshift/quantization/quant_spec.h",
        root / "include/phaseshift/quantization/payload_descriptor.h",
        root / "include/phaseshift/quantization/quant_contract.h",
        root / "include/phaseshift/quantization/quantization_types.h",
        root / "include/phaseshift/quantization/quantized_compute_view.h",
        root / "include/phaseshift/core",
        root / "src/phaseshift/core",
        root / "include/phaseshift/io",
        root / "src/phaseshift/io",
    };
    for (const auto& dir : format_dirs) {
        scan_clean(dir, {"phaseshift/models/"});
    }

    const std::vector<fs::path> model_dirs = {
        root / "include/phaseshift/models/qwen35/model",
        root / "src/phaseshift/models/qwen35/model",
        root / "include/phaseshift/models/qwen35/weights",
        root / "src/phaseshift/models/qwen35/weights",
    };
    for (const auto& dir : model_dirs) {
        scan_clean(dir, {"models/qwen35/runtime", "Executor", "ContinuousBatcher"});
    }

    const std::vector<fs::path> kernel_dirs = {
        root / "include/phaseshift/models/qwen35/kernels",
        root / "src/phaseshift/models/qwen35/kernels",
    };
    for (const auto& dir : kernel_dirs) {
        scan_clean(dir,
                   {"models/qwen35/runtime", "Executor", "HostExecutionContext",
                    "ContinuousBatcher"});
    }


    check_correctness_location(root);
    check_correctness_layout(root);
    check_no_server_deps(root);

    printf("test_architecture_boundaries: passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
