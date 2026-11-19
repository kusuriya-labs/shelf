#include <cstdio>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>

#ifndef SHELF_SOURCE_ROOT
#error "SHELF_SOURCE_ROOT must be defined"
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

// Keeps the code part of a translation unit and drops comments and string / character
// literals. A model name in a help string is not a dependency; a model name in code is.
std::string code_only(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    enum class State { Normal, LineComment, BlockComment, String, Character };
    State state = State::Normal;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        const char next = i + 1 < text.size() ? text[i + 1] : '\0';
        switch (state) {
            case State::Normal:
                if (c == '/' && next == '/') {
                    state = State::LineComment;
                    ++i;
                } else if (c == '/' && next == '*') {
                    state = State::BlockComment;
                    ++i;
                } else if (c == '"') {
                    state = State::String;
                } else if (c == '\'') {
                    state = State::Character;
                } else {
                    out.push_back(c);
                }
                break;
            case State::LineComment:
                if (c == '\n') {
                    out.push_back('\n');
                    state = State::Normal;
                }
                break;
            case State::BlockComment:
                if (c == '*' && next == '/') {
                    state = State::Normal;
                    ++i;
                } else if (c == '\n') {
                    out.push_back('\n');
                }
                break;
            case State::String:
                if (c == '\\') {
                    ++i;
                } else if (c == '"') {
                    state = State::Normal;
                }
                break;
            case State::Character:
                if (c == '\\') {
                    ++i;
                } else if (c == '\'') {
                    state = State::Normal;
                }
                break;
        }
    }
    return out;
}

std::vector<std::string> included_paths(const std::string& text) {
    std::vector<std::string> out;
    std::size_t pos = 0;
    while ((pos = text.find("#include", pos)) != std::string::npos) {
        const std::size_t line_end = text.find('\n', pos);
        const std::string line =
            text.substr(pos, line_end == std::string::npos ? std::string::npos
                                                           : line_end - pos);
        const std::size_t open_angle = line.find('<');
        const std::size_t open_quote = line.find('"');
        std::size_t begin = std::string::npos;
        char close = '\0';
        if (open_angle != std::string::npos) {
            begin = open_angle + 1;
            close = '>';
        } else if (open_quote != std::string::npos) {
            begin = open_quote + 1;
            close = '"';
        }
        if (begin != std::string::npos) {
            const std::size_t end = line.find(close, begin);
            if (end != std::string::npos) out.push_back(line.substr(begin, end - begin));
        }
        pos = line_end == std::string::npos ? text.size() : line_end;
    }
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
        {root / "src/shelf/models/qwen35",
         root / "src/shelf/kernels/correctness"},
        {root / "include/shelf/models/qwen35",
         root / "include/shelf/kernels/correctness"},
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
    const fs::path src = root / "src/shelf/kernels/correctness";
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
    scan_clean(root / "src/shelf", forbidden);
    scan_clean(root / "include/shelf", forbidden);
}

// The libraries one CMake target links, as written in cmake/targets.cmake. The marker ends
// in a space so shelf_qwen35 does not match the shelf_qwen35moe clause.
std::vector<std::string> linked_targets(const std::string& text, const std::string& target) {
    std::vector<std::string> out;
    const std::string marker = "target_link_libraries(" + target + " ";
    std::size_t pos = 0;
    while ((pos = text.find(marker, pos)) != std::string::npos) {
        const std::size_t begin = pos + marker.size();
        const std::size_t end = text.find(')', begin);
        if (end == std::string::npos) break;
        std::string token;
        for (std::size_t i = begin; i < end; ++i) {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            if (std::isspace(c) || c == ';') {
                if (!token.empty()) {
                    out.push_back(token);
                    token.clear();
                }
            } else {
                token.push_back(static_cast<char>(c));
            }
        }
        if (!token.empty()) out.push_back(token);
        pos = end;
    }
    return out;
}

void check_cmake_target_edges(const fs::path& root) {
    const std::string text = read_file(root / "cmake/targets.cmake");
    if (text.empty()) {
        fail("cmake/targets.cmake could not be read");
        return;
    }

    struct EdgeRule {
        const char* target;
        const char* forbidden;
    };
    const std::vector<EdgeRule> forbidden_edges = {
        {"shelf_qwen35", "shelf_qwen35moe"},
        {"shelf_qwen35moe", "shelf_qwen35"},
        {"shelf_qwen_hybrid", "shelf_qwen35"},
        {"shelf_qwen_hybrid", "shelf_qwen35moe"},
        {"shelf_weights", "shelf_qwen35"},
        {"shelf_weights", "shelf_qwen35moe"},
        {"shelf_runtime", "shelf_qwen35"},
        {"shelf_runtime", "shelf_qwen35moe"},
        {"shelf_engine", "shelf_qwen35"},
        {"shelf_engine", "shelf_qwen35_runtime"},
        {"shelf_engine", "shelf_qwen35moe"},
        {"shelf_engine", "shelf_qwen35qsa"},
        {"shelf_engine", "shelf_qwen4exp"},
        {"shelf_engine", "shelf_plamo3"},
        {"shelf_engine", "shelf_hunyuan_dense"},
        {"shelf_mtp", "shelf_qwen35"},
        {"shelf_mtp", "shelf_qwen35_runtime"},
        {"shelf_mtp", "shelf_qwen35moe"},
        {"shelf_mtp", "shelf_qwen35qsa"},
        {"shelf_mtp", "shelf_qwen4exp"},
        {"shelf_mtp", "shelf_plamo3"},
        {"shelf_mtp", "shelf_hunyuan_dense"},
    };
    for (const auto& rule : forbidden_edges) {
        const std::vector<std::string> links = linked_targets(text, rule.target);
        if (links.empty()) {
            fail(std::string("no target_link_libraries clause for ") + rule.target);
            continue;
        }
        for (const auto& lib : links) {
            if (lib == rule.forbidden) {
                fail(std::string(rule.target) + " links " + rule.forbidden);
            }
        }
        ++passed;
    }

    // Both families build on the shared hybrid target rather than on each other.
    for (const char* family : {"shelf_qwen35", "shelf_qwen35moe"}) {
        bool hybrid = false;
        for (const auto& lib : linked_targets(text, family)) {
            if (lib == "shelf_qwen_hybrid") hybrid = true;
        }
        if (!hybrid) fail(std::string(family) + " does not link shelf_qwen_hybrid");
        else ++passed;
    }

    // The compute application depends on the engine, not on a family or on shelf::models
    // directly. The aggregate target is not enough: it would silently propagate every
    // family library.
    const std::string apps = read_file(root / "cmake/apps.cmake");
    if (apps.empty()) {
        fail("cmake/apps.cmake could not be read");
        return;
    }
    bool engine_link = false;
    const std::vector<std::string> app_links = linked_targets(apps, "shelf-compute");
    for (const auto& lib : app_links) {
        if (lib == "shelf_engine") engine_link = true;
        if (lib == "shelf" || lib == "shelf_models" || lib == "shelf_qwen35" ||
            lib == "shelf_qwen35_runtime" || lib == "shelf_qwen35moe" ||
            lib == "shelf_qwen35qsa" || lib == "shelf_qwen4exp" || lib == "shelf_plamo3" ||
            lib == "shelf_hunyuan_dense") {
            fail("shelf-compute links " + lib + " directly");
        }
    }
    if (!engine_link) fail("shelf-compute does not link shelf_engine");
    else ++passed;
}

// Every production .cpp/.hip under the model layer must be listed by a CMake target. A
// file left behind after a move is not compiled, but it is still edited and reviewed as
// if it were live code.
void check_no_orphan_model_sources(const fs::path& root) {
    const std::string cmake = read_file(root / "cmake/targets.cmake");
    if (cmake.empty()) {
        fail("cmake/targets.cmake could not be read");
        return;
    }
    for (const auto& file : scan(root / "src/shelf/models")) {
        const std::string extension = file.extension().string();
        if (extension != ".cpp" && extension != ".hip") continue;
        const std::string relative = fs::relative(file, root).generic_string();
        if (!contains(cmake, relative.c_str())) {
            fail(relative + " is not listed by any CMake target");
        } else {
            ++passed;
        }
    }
}

// src/apps/compute is a token-in / token-out driver. It may print a model name in a help
// string, but it may not include a family header or reach a concrete model / rank runtime.
void check_compute_app_boundaries(const fs::path& root) {
    const std::vector<std::string> forbidden_headers = {
        "shelf/models/qwen35", "shelf/models/qwen35moe", "shelf/models/qwen35qsa",
        "shelf/models/qwen4exp", "shelf/models/plamo3", "shelf/models/hunyuan_dense",
        "shelf/models/loaded_model.h",
        "shelf/models/tp_rank_model.h", "shelf/models/pipeline_stage_model.h",
        "shelf/runtime/executor.h", "shelf/runtime/tp/tensor_parallel.h",
    };
    const std::vector<const char*> forbidden_code = {
        "qwen35::", "Qwen35", "qwen4exp::", "Qwen4Exp", "plamo3::", "Plamo3",
        "hunyuan_dense::", "HunyuanDense",
        "qwen35moe", "qwen35qsa", "runtime::Executor", "runtime::TpCoordinator",
        "TpRankRuntime", "PipelineStageRuntime", "LoadedModel", "hipStream_t",
        "hipMalloc", "hipMemcpy",
    };
    for (const auto& file : scan(root / "src/apps/compute")) {
        const std::string text = read_file(file);
        for (const auto& header : included_paths(text)) {
            for (const auto& forbidden : forbidden_headers) {
                if (header.rfind(forbidden, 0) == 0) {
                    fail(file.string() + " includes " + header);
                }
            }
        }
        const std::string code = code_only(text);
        for (const char* needle : forbidden_code) {
            if (contains(code, needle)) {
                fail(file.string() + std::string(" uses ") + needle);
            }
        }
        ++passed;
    }
}

// The engine may orchestrate models, but it may not include a family header or a serving
// schema. MTP lives in models/common/drafter and DFlash2 lives in shelf_runtime, so the
// engine has no family exception.
void check_engine_boundaries(const fs::path& root) {
    const std::vector<const char*> forbidden_generic = {
        "nlohmann", "OpenAI", "openai", "protobuf", "Protobuf", "gRPC", "grpc",
        "xgrammar", "XGrammar",
    };
    const std::vector<const char*> forbidden_family = {
        "shelf/models/qwen35", "shelf/models/qwen35moe", "shelf/models/qwen35qsa",
        "shelf/models/qwen4exp", "shelf/models/plamo3", "shelf/models/hunyuan_dense",
        "qwen35::", "Qwen35",
        "qwen4exp::", "Qwen4Exp", "plamo3::", "Plamo3",
        "hunyuan_dense::", "HunyuanDense",
    };

    // Public engine headers stay family neutral.
    for (const auto& file : scan(root / "include/shelf/engine")) {
        const std::string text = read_file(file);
        const std::string code = code_only(text);
        for (const char* needle : forbidden_generic) {
            if (contains(code, needle)) fail(file.string() + std::string(" uses ") + needle);
        }
        for (const auto& header : included_paths(text)) {
            for (const char* needle : forbidden_family) {
                if (header.rfind(needle, 0) == 0) {
                    fail(file.string() + " includes " + header);
                }
            }
        }
        ++passed;
    }

    for (const auto& file : scan(root / "src/shelf/engine")) {
        const std::string text = read_file(file);
        const std::string code = code_only(text);
        for (const char* needle : forbidden_generic) {
            if (contains(code, needle)) fail(file.string() + std::string(" uses ") + needle);
        }
        for (const auto& header : included_paths(text)) {
            for (const char* needle : forbidden_family) {
                if (header.rfind(needle, 0) == 0) {
                    fail(file.string() + " includes " + header);
                }
            }
        }
        for (const char* needle : forbidden_family) {
            if (contains(code, needle)) fail(file.string() + std::string(" uses ") + needle);
        }
        ++passed;
    }
}

}

int main() {
    const fs::path root(SHELF_SOURCE_ROOT);

    const std::vector<fs::path> core_dirs = {
        root / "include/shelf/runtime",
        root / "src/shelf/runtime",
    };
    for (const auto& dir : core_dirs) {
        scan_clean(dir, {"qwen35", "Qwen35", "shelf/models/qwen35", "shelf/models/qwen",
                         "hy4_flash", "shelf_hy4_", "Hy4Expert", "Hy4Route"});
    }

    const std::vector<fs::path> generic_dirs = {
        root / "include/shelf/kernels",
        root / "src/shelf/kernels",
        root / "include/shelf/inference",
        root / "src/shelf/inference",
        root / "include/shelf/models/common",
        root / "src/shelf/models/common",
    };
    for (const auto& dir : generic_dirs) {
        scan_clean(dir, {"qwen35", "Qwen35", "shelf/models/qwen",
                         "hy4_flash", "shelf_hy4_", "Hy4Expert", "Hy4Route"});
    }

    for (const auto& dir : {root / "src/shelf/kernels/hy4_flash", root / "include/shelf/kernels/hy4_flash"}) {
        if (fs::exists(dir)) fail(dir.string() + " must be model-local");
        else ++passed;
    }

    const std::vector<fs::path> weights_dirs = {
        root / "include/shelf/weights",
        root / "src/shelf/weights",
    };
    for (const auto& dir : weights_dirs) {
        scan_clean(dir, {"qwen35", "Qwen35", "shelf/models/"});
    }

    const std::vector<fs::path> format_dirs = {
        root / "include/shelf/quantization/manifest",
        root / "include/shelf/quantization/bundle",
        root / "include/shelf/quantization/codec",
        root / "include/shelf/quantization/ngram",
        root / "src/shelf/quantization/manifest",
        root / "src/shelf/quantization/bundle",
        root / "src/shelf/quantization/codec",
        root / "src/shelf/quantization/ngram",
        root / "include/shelf/quantization/quant_format.h",
        root / "include/shelf/quantization/quant_spec.h",
        root / "include/shelf/quantization/payload_descriptor.h",
        root / "include/shelf/quantization/quant_contract.h",
        root / "include/shelf/quantization/quantization_types.h",
        root / "include/shelf/quantization/quantized_compute_view.h",
        root / "include/shelf/core",
        root / "src/shelf/core",
        root / "include/shelf/io",
        root / "src/shelf/io",
    };
    for (const auto& dir : format_dirs) {
        scan_clean(dir, {"shelf/models/"});
    }

    const std::vector<fs::path> model_dirs = {
        root / "include/shelf/models/qwen35/model",
        root / "src/shelf/models/qwen35/model",
        root / "include/shelf/models/qwen35/weights",
        root / "src/shelf/models/qwen35/weights",
    };
    for (const auto& dir : model_dirs) {
        scan_clean(dir, {"models/qwen35/runtime", "Executor", "ContinuousBatcher"});
    }

    // The dense family and the MoE family are siblings: neither names the other. Both use
    // models/common/qwen_hybrid for what they share, and the choice between the two
    // loaders lives in models/model_weights_factory, the only target that links both.
    const std::vector<fs::path> qwen35_dirs = {
        root / "include/shelf/models/qwen35",
        root / "src/shelf/models/qwen35",
    };
    for (const auto& dir : qwen35_dirs) {
        scan_clean(dir, {"qwen35moe", "Qwen35Moe", "MoE", "moe"});
    }
    const std::vector<fs::path> qwen35moe_dirs = {
        root / "include/shelf/models/qwen35moe",
        root / "src/shelf/models/qwen35moe",
    };
    for (const auto& dir : qwen35moe_dirs) {
        scan_clean(dir, {"models/qwen35/", "shelf::qwen35::", "qwen35::", "Qwen35TextConfig",
                         "Qwen35MtpWeights", "DenseFfnConfig", "DenseFfnWeights",
                         "quantized_model_body"});
    }

    // The hybrid module is the shared part, so it may not reach down into either family.
    scan_clean(root / "include/shelf/models/common/qwen_hybrid",
               {"Qwen35Moe", "qwen35moe", "Qwen35Text", "Qwen35Layer", "Qwen35Model",
                "MoeExpert", "MoeFfn", "DenseFfn", "load_qwen35"});
    scan_clean(root / "src/shelf/models/common/qwen_hybrid",
               {"Qwen35Moe", "qwen35moe", "Qwen35Text", "Qwen35Layer", "Qwen35Model",
                "MoeExpert", "MoeFfn", "DenseFfn", "load_qwen35"});

    check_cmake_target_edges(root);
    check_no_orphan_model_sources(root);
    check_compute_app_boundaries(root);
    check_engine_boundaries(root);

    check_correctness_location(root);
    check_correctness_layout(root);
    check_no_server_deps(root);

    printf("test_architecture_boundaries: passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
