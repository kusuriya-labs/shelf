// shelf unit test — Paged KV ABI + prefix key contract (PR-007).
#include <cstdio>
#include <vector>

#include <shelf/backend/common/paged_kv.hpp>
#include <shelf/runtime/state/prefix_key.hpp>

namespace {

using namespace shelf;
using namespace shelf::backend::common;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

void test_layout() {
    PagedKVLayout L;
    L.block_size = 16; L.kv_heads = 4; L.head_dim = 256; L.num_blocks = 128;
    L.dtype = DataType::bf16;
    check(L.prepare().ok(), "prepare canonical strides");
    check(L.slot_stride == 512 && L.head_stride == 16 * 512 &&
          L.block_stride == 4 * 16 * 512, "canonical stride values");
    check(L.elements_per_block() == 4 * 16 * 512, "elements_per_block");

    // offset formula: (b,h,s,kind,d)
    const std::size_t o = paged_kv_offset(L, 3, 2, 5, 1, 7);
    check(o == 3 * L.block_stride + 2 * L.head_stride + 5 * L.slot_stride + 256 + 7,
          "offset K-then-V per slot");

    PagedKVLayout bad = L;
    bad.slot_stride = 10;
    check(!bad.check().ok(), "slot_stride too small rejected");
    bad = L;
    bad.head_stride = 1;
    check(!bad.check().ok(), "head_stride overlap rejected");
    bad = L;
    bad.block_stride = 0;
    check(!bad.check().ok(), "unset strides need prepare");
}

void test_args() {
    PagedKVLayout L;
    L.block_size = 16; L.kv_heads = 2; L.head_dim = 8; L.num_blocks = 4;
    L.prepare();
    // seq0: 20 tokens -> 2 blocks; seq1: 3 tokens -> 1 block
    std::vector<std::int32_t> table = {0, 2, -1,
                                       1, -1, -1};
    std::vector<std::int32_t> used = {20, 3};
    PagedKVArgs A;
    A.layout = L;
    A.table = {table.data(), 2, 3, 3};
    A.used = {used.data(), 2};
    check(A.validate().ok(), "valid paged args");

    auto A2 = A;
    A2.table = {table.data(), 2, 3, 2};
    check(!A2.validate().ok(), "row_stride < max_blocks rejected");

    auto A3 = A;
    std::vector<std::int32_t> bad_table = {0, 9, -1, 1, -1, -1};  // id outside pool
    A3.table = {bad_table.data(), 2, 3, 3};
    check(!A3.validate().ok(), "block id outside pool rejected");

    auto A4 = A;
    std::vector<std::int32_t> used_big = {64, 3};  // needs 4 blocks > width 3
    A4.used = {used_big.data(), 2};
    check(!A4.validate().ok(), "seqused exceeds table width rejected");
}

void test_prefix_key() {
    using namespace shelf::runtime::state;
    const std::int32_t a[] = {1, 2, 3, 4};
    const std::int32_t b[] = {1, 2, 3, 5};
    const std::int32_t a_short[] = {1, 2, 3};

    BlockKey ka = chain_block_key(kRootBlockKey, a, 4);
    BlockKey kb = chain_block_key(kRootBlockKey, b, 4);
    BlockKey ka2 = chain_block_key(kRootBlockKey, a, 4);
    check(ka.valid() && ka == ka2, "deterministic");
    check(ka != kb, "different tokens -> different key");

    // same tokens but different PREFIX (parent) -> different identity
    BlockKey pa = chain_block_key(ka, a, 4);
    check(pa != ka && pa.valid(), "prefix-chained identity differs");

    // prefix relation must not collide with shorter block at root
    check(chain_block_key(kRootBlockKey, a_short, 3) != ka, "length participates");

    // long random-ish stream stability
    std::vector<std::int32_t> big(64);
    for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<std::int32_t>(i * 1000003);
    BlockKey k1 = chain_block_key(ka, big.data(), big.size());
    BlockKey k2 = chain_block_key(ka, big.data(), big.size());
    check(k1 == k2, "long chain deterministic");
}

} // namespace

int main() {
    test_layout();
    test_args();
    test_prefix_key();
    std::printf("test_paged_kv_abi: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
