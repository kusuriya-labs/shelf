// shelf architecture test — forbidden dependency detection (plan §3.1, §7).
//
// Two mechanisms:
//
//  1. #include-directive scan: each module must not include forbidden
//     module paths (checked on the include path text only, comments are safe).
//  2. raw-content scan: model names (qwen/llama/deepseek...) are banned
//     everywhere except src/model/, so backends/runtime never branch on
//     "if (model == Qwen...)".
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef SHELF_SOURCE_ROOT
#error "SHELF_SOURCE_ROOT must be defined"
#endif

namespace fs = std::filesystem;

namespace {

int passed = 0;
int failed = 0;

void fail(const std::string& msg) {
    ++failed;
    std::printf("FAIL: %s\n", msg.c_str());
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool is_source(const fs::path& p) {
    const auto ext = p.extension().string();
    return ext == ".h" || ext == ".hpp" || ext == ".cpp" || ext == ".cc" ||
           ext == ".cu" || ext == ".hip";
}

std::vector<fs::path> scan(const fs::path& root) {
    std::vector<fs::path> files;
    std::error_code ec;
    if (!fs::exists(root, ec)) return files;  // empty skeleton dirs are fine
    for (auto it = fs::recursive_directory_iterator(root, ec);
         it != fs::recursive_directory_iterator(); ++it) {
        if (it->is_regular_file() && is_source(it->path())) files.push_back(it->path());
    }
    return files;
}

// Extract include-path text from "#include <path>" / "#include \"path\"" lines.
std::vector<std::string> include_targets(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        const auto hash = line.find('#');
        if (hash == std::string::npos) continue;
        const auto inc = line.find("include", hash);
        if (inc == std::string::npos) continue;
        const auto lt = line.find('<', inc);
        const auto qt = line.find('"', inc);
        std::size_t open = std::min(lt == std::string::npos ? std::size_t(-1) : lt,
                                    qt == std::string::npos ? std::size_t(-1) : qt);
        if (open == std::size_t(-1)) continue;
        const char close = line[open] == '<' ? '>' : '"';
        const auto end = line.find(close, open + 1);
        if (end == std::string::npos) continue;
        out.push_back(to_lower(line.substr(open + 1, end - open - 1)));
    }
    return out;
}

struct Rule {
    fs::path dir;                       // scanned recursively
    std::vector<std::string> forbidden; // substrings of include paths
};

void check_includes(const Rule& rule) {
    for (const auto& file : scan(rule.dir)) {
        bool clean = true;
        for (const auto& target : include_targets(read_file(file))) {
            for (const auto& needle : rule.forbidden) {
                if (target.find(needle) != std::string::npos) {
                    fail(file.string() + " includes <" + target +
                         "> — forbidden in " + rule.dir.string() + " (contains \"" + needle + "\")");
                    clean = false;
                }
            }
        }
        if (clean) ++passed;
    }
}

void check_no_model_names(const std::vector<fs::path>& roots,
                          const std::vector<fs::path>& exemptions) {
    const std::vector<std::string> names = {"qwen", "llama", "deepseek", "mistral", "phi-"};
    for (const auto& root : roots) {
        for (const auto& file : scan(root)) {
            bool exempt = false;
            for (const auto& ex : exemptions) {
                auto rel = fs::relative(file, ex);
                if (!rel.empty() && *rel.begin() != "..") { exempt = true; break; }
            }
            if (exempt) continue;
            const std::string text = to_lower(read_file(file));
            for (const auto& name : names) {
                if (text.find(name) != std::string::npos) {
                    fail(file.string() + " mentions model name \"" + name +
                         "\" outside src/model/");
                }
            }
        }
    }
}

} // namespace

int main() {
    const fs::path root(SHELF_SOURCE_ROOT);
    const fs::path src = root / "src";

    // Module include rules (lowercase substrings of include paths).
    // Keep in sync with docs/architecture/dependency_rules.md.
    // HIP types confined to src/backend/hip/.
    const std::vector<Rule> rules = {
        {src / "foundation",
         {"shelf/model/", "shelf/ir/", "shelf/artifact/", "shelf/quant/", "shelf/weights/",
          "shelf/compiler/", "shelf/backend/", "shelf/placement/", "shelf/runtime/",
          "hip/"}},
        {src / "model",
         {"shelf/artifact/", "shelf/quant/", "shelf/backend/", "shelf/compiler/",
          "shelf/runtime/", "shelf/placement/", "hip/"}},
        {src / "ir",
         {"shelf/backend/", "shelf/compiler/", "shelf/runtime/", "shelf/placement/",
          "hip/"}},
        {src / "artifact",
         {"shelf/model/", "shelf/runtime/", "shelf/backend/", "hip/"}},
        {src / "quant",
         {"shelf/model/", "shelf/backend/", "shelf/runtime/", "hip/"}},
        {src / "weights",
         {"shelf/runtime/", "shelf/api/", "hip/"}},
        {src / "compiler",
         {"shelf/runtime/", "shelf/api/", "hip/"}},
        {src / "backend",
         {"shelf/model/", "shelf/compiler/", "shelf/runtime/", "shelf/placement/"}},
        {src / "backend/common", {"hip/"}},
        {src / "backend/cpu",   {"hip/"}},
        {src / "placement",
         {"shelf/model/", "shelf/runtime/", "hip/"}},
        {src / "runtime",
         {"shelf/artifact/", "shelf/api/", "hip/"}},
        {src / "api",
         {"hip/"}},
        {root / "include/shelf",
         {"hip/", "shelf/backend/", "shelf/foundation/"}},
    };
    for (const auto& rule : rules) check_includes(rule);

    // Note: src/backend rule forbids model includes; the raw scan below also
    // forbids model *names* in backend/runtime/etc.

    check_no_model_names({src, root / "include"}, {src / "model"});

    std::printf("test_architecture_boundaries: passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
