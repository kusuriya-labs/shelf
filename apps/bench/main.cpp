// shelf-bench — 計測エントリ（pp / tg / kernel <name>）。
//
// 形は phaseshift-bench に合わせる: サブコマンドのディスパッチと usage のみを
// 持ち、各 runner は独立ファイル。--help は装置初期化の前に必ず exit 0。
#include <cstdio>
#include <string>

#include "bench_common.hpp"

namespace shelf::bench {
int run_pp(int argc, char** argv);
int run_tg(int argc, char** argv);
int run_kernel(int argc, char** argv);
}  // namespace shelf::bench

namespace {

void usage(std::FILE* out) {
    std::fprintf(out,
        "usage: shelf-bench <command> [options]\n"
        "\n"
        "commands:\n"
        "  pp                 prompt processing (prefill) を測る\n"
        "  tg                 token generation (decode) を測る\n"
        "  kernel <name>      1 つのカーネルを測る（名称一覧は kernel --help）\n"
        "\n"
        "common:\n"
        "  --help, -h         このヘルプを出して exit 0（GPU を触らない）\n"
        "\n"
        "各コマンドの詳細: shelf-bench <command> --help\n"
        "exit codes: 0 = ok, 1 = runtime error, 2 = usage error\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(stdout);
        return shelf::bench::kExitUsage;
    }
    const std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        usage(stdout);
        return shelf::bench::kExitOk;
    }
    // 位置引数はサブコマンド直下から（プログラム名とサブコマンド名はスキップ）。
    if (cmd == "pp") return shelf::bench::run_pp(argc - 2, argv + 2);
    if (cmd == "tg") return shelf::bench::run_tg(argc - 2, argv + 2);
    if (cmd == "kernel") return shelf::bench::run_kernel(argc - 2, argv + 2);

    std::fprintf(stderr, "unknown command: %s\n", cmd.c_str());
    usage(stderr);
    return shelf::bench::kExitUsage;
}
