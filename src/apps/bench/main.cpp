#include <cstdio>
#include <cstring>
#include <string>

int run_pp(int argc, char** argv);
int run_tp_reduce(int argc, char** argv);
int run_tg(int argc, char** argv);
int run_quality(int argc, char** argv);
int run_batch(int argc, char** argv);
int run_activation_quantize(int argc, char** argv);
int run_gemm(int argc, char** argv);
int run_rmsnorm(int argc, char** argv);
int run_l2_normalize(int argc, char** argv);
int run_kv_append(int argc, char** argv);
int run_paged_attention(int argc, char** argv);
int run_rope(int argc, char** argv);
int run_gdn_recurrence(int argc, char** argv);
int run_gdn_conv1d(int argc, char** argv);
int run_elementwise(int argc, char** argv);
int run_embedding(int argc, char** argv);
int run_sampling(int argc, char** argv);
int run_gpu_memory(int argc, char** argv);

namespace {

void usage() {
    std::printf("usage:\n");
    std::printf("  shelf-bench <subcommand> [options]\n");
    std::printf("\n");
    std::printf("E2E:\n");
    std::printf("  pp                 prefill benchmark\n");
    std::printf("  tg                 decode / token-generation benchmark\n");
    std::printf("  batch              continuous-batching / prefix-cache benchmark\n");
    std::printf("\n");
    std::printf("Linear:\n");
    std::printf("  activation-quantize A8 activation quantization micro-benchmark\n");
    std::printf("  gemm               linear GEMV/GEMM micro-benchmark\n");
    std::printf("\n");
    std::printf("Normalization:\n");
    std::printf("  rmsnorm            RMSNorm micro-benchmark\n");
    std::printf("  l2-normalize       L2 normalize micro-benchmark\n");
    std::printf("\n");
    std::printf("Attention/KV:\n");
    std::printf("  kv-append          KV append benchmark\n");
    std::printf("  paged-attention    paged-attention benchmark\n");
    std::printf("  rope               RoPE micro-benchmark\n");
    std::printf("\n");
    std::printf("GDN:\n");
    std::printf("  gdn-recurrence     GDN recurrence benchmark\n");
    std::printf("  gdn-conv1d         GDN causal conv1d benchmark\n");
    std::printf("\n");
    std::printf("Auxiliary:\n");
    std::printf("  elementwise        Qwen elementwise benchmark\n");
    std::printf("  embedding          embedding lookup benchmark\n");
    std::printf("  sampling           greedy argmax sampling benchmark\n");
    std::printf("\n");
    std::printf("Memory:\n");
    std::printf("  tp-reduce            tp sum_hidden transport microbenchmark\n");
    std::printf("  gpu-memory         global / LDS crossover sweep\n");
    std::printf("\n");
    std::printf("Quality:\n");
    std::printf("  quality            teacher-forced logits dump / quality comparison\n");
    std::printf("\n");
    std::printf("run 'shelf-bench <subcommand> --help' for options\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string sub = argv[1];
    if (sub == "-h" || sub == "--help") {
        usage();
        return 0;
    }
    if (sub == "pp") return run_pp(argc - 1, argv + 1);
    if (sub == "tg") return run_tg(argc - 1, argv + 1);
    if (sub == "quality") return run_quality(argc - 1, argv + 1);
    if (sub == "batch") return run_batch(argc - 1, argv + 1);
    if (sub == "activation-quantize") return run_activation_quantize(argc - 1, argv + 1);
    if (sub == "gemm") return run_gemm(argc - 1, argv + 1);
    if (sub == "rmsnorm") return run_rmsnorm(argc - 1, argv + 1);
    if (sub == "l2-normalize") return run_l2_normalize(argc - 1, argv + 1);
    if (sub == "kv-append") return run_kv_append(argc - 1, argv + 1);
    if (sub == "paged-attention") return run_paged_attention(argc - 1, argv + 1);
    if (sub == "rope") return run_rope(argc - 1, argv + 1);
    if (sub == "gdn-recurrence") return run_gdn_recurrence(argc - 1, argv + 1);
    if (sub == "gdn-conv1d") return run_gdn_conv1d(argc - 1, argv + 1);
    if (sub == "elementwise") return run_elementwise(argc - 1, argv + 1);
    if (sub == "embedding") return run_embedding(argc - 1, argv + 1);
    if (sub == "sampling") return run_sampling(argc - 1, argv + 1);
    if (sub == "gpu-memory") return run_gpu_memory(argc - 1, argv + 1);
    if (sub == "tp-reduce") return run_tp_reduce(argc - 1, argv + 1);
    std::fprintf(stderr, "unknown subcommand: %s\n", sub.c_str());
    usage();
    return 2;
}
