// Standalone harness: the kernel TU and the bench TU compiled together (fast iteration).
#include KERNEL_SRC
#include "apps/bench/exl3_int8.hip"
int main(int argc, char** argv) { return run_exl3_int8(argc, argv); }
