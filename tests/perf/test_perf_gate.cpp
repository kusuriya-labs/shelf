// shelf — 性能の回帰ゲート（壁時計、opt-in）。
//
// `shelf-bench kernel gemm` を固定形状で走らせ、`us_p50`（中央値）を基準ファイルと比べる。
// 時間は装置・負荷でぶれるため、**通常の ctest には登録しない**
// （`-DSHELF_ENABLE_PERF_GATE=ON` のときだけ `gate.perf.hip` として登録される）。
//
// 使い方:
//   test_perf_gate <shelf-bench のパス> <基準ファイル>            … 判定
//   test_perf_gate <shelf-bench のパス> <基準ファイル> --dump     … 基準を作り直す
//   SHELF_PERF_TOLERANCE=1.6 … 許容倍率（既定 1.6。回帰のみを拾い、ぶれは許す）
//
// 基準ファイルの書式（1 行 1 形状）:
//   <shape> <us_p50>（中央値）
// 例: 1x4096x4096 52.1
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Row {
    std::string shape;
    double us = 0.0;
};

// shelf-bench の stdout から shape / us_median の組を拾う。
std::vector<Row> run_bench(const std::string& bench, std::vector<std::string>* raw_log) {
    std::vector<Row> rows;
    const std::string cmd = bench +
                            " kernel gemm --rows 1,16,64 --n 4096 --k 4096"
                            " --warmup 3 --samples 5 --launches 8 --gpu 0 2>&1";
    FILE* p = ::popen(cmd.c_str(), "r");
    if (p == nullptr) {
        std::printf("FAIL: shelf-bench を起動できない: %s\n", cmd.c_str());
        return rows;
    }
    std::string shape;
    std::array<char, 512> buf{};
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), p) != nullptr) {
        std::string line(buf.data());
        if (!line.empty() && line.back() == '\n') line.pop_back();
        if (raw_log) raw_log->push_back(line);
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const std::string val = line.substr(eq + 1);
        if (key == "shape") {
            shape = val;
        } else if (key == "us_p50" && !shape.empty()) {
            rows.push_back(Row{shape, std::atof(val.c_str())});
            shape.clear();
        }
    }
    const int rc = ::pclose(p);
    if (rc != 0)
        std::printf("注意: shelf-bench が終了コード %d で終わった\n", rc);
    return rows;
}

std::vector<Row> read_baseline(const std::string& path) {
    std::vector<Row> rows;
    std::ifstream f(path);
    if (!f) {
        std::printf("注意: 基準ファイルが読めない: %s\n", path.c_str());
        return rows;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        // 末尾が数値、その手前までが shape（shape に空白を含むため右から切る）。
        const auto sp = line.find_last_of(' ');
        if (sp == std::string::npos) continue;
        rows.push_back(Row{line.substr(0, sp), std::atof(line.c_str() + sp + 1)});
    }
    return rows;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: %s <shelf-bench> <baseline.txt> [--dump]\n", argv[0]);
        return 2;
    }
    const std::string bench = argv[1];
    const std::string baseline_path = argv[2];
    const bool dump = (argc > 3 && std::string(argv[3]) == "--dump");

    std::vector<std::string> log;
    const auto rows = run_bench(bench, dump ? &log : nullptr);
    if (rows.empty()) {
        std::printf("FAIL: 計測結果が取れない（GPU が空いているか確認）\n");
        for (const auto& l : log) std::printf("  bench: %s\n", l.c_str());
        return 1;
    }

    if (dump) {
        std::ofstream f(baseline_path);
        if (!f) {
            std::printf("FAIL: 書けない: %s\n", baseline_path.c_str());
            return 1;
        }
        f << "# shelf-bench kernel gemm（--rows 1,16,64 --n 4096 --k 4096）の us_p50\n";
        f << "# 取り直し: test_perf_gate <shelf-bench> <this file> --dump（空いた GPU で）\n";
        for (const auto& r : rows) f << r.shape << " " << r.us << "\n";
        std::printf("基準を書き出した: %s（%zu 形状）\n", baseline_path.c_str(), rows.size());
        return 0;
    }

    const auto base = read_baseline(baseline_path);
    if (base.empty()) {
        std::printf("FAIL: 基準が空（先に --dump で作る）: %s\n", baseline_path.c_str());
        return 1;
    }
    double tol = 1.6;
    if (const char* t = std::getenv("SHELF_PERF_TOLERANCE")) tol = std::atof(t);

    std::map<std::string, double> want;
    for (const auto& b : base) want[b.shape] = b.us;

    int failures = 0;
    for (const auto& r : rows) {
        const auto it = want.find(r.shape);
        if (it == want.end()) {
            std::printf("注意: 基準に無い形状（比較しない）: %s\n", r.shape.c_str());
            continue;
        }
        const double limit = it->second * tol;
        const bool ok = r.us <= limit;
        std::printf("%-24s us_p50=%8.3f  基準=%8.3f  上限=%.3f  %s\n", r.shape.c_str(), r.us,
                    it->second, limit, ok ? "OK" : "回帰");
        if (!ok) ++failures;
    }
    std::printf("test_perf_gate: %d 形状中 %d 件が回帰（許容 %.2f 倍）\n",
                static_cast<int>(rows.size()), failures, tol);
    return failures == 0 ? 0 : 1;
}
