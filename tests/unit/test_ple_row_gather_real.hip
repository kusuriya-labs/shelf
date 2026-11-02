#include <phaseshift/models/qwen4exp/ple/ple_row_gather.h>
#include <phaseshift/weights/weight_loader.h>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <map>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace ple = ps::qwen4exp::ple;
namespace fw = ps::weights;

int g_fail = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

}

int main(int argc, char** argv) {
    const char* model_dir = argc > 1
        ? argv[1]
        : "/opt/zen/work/models/Qwen3.8-Flash-Next-Quark-AWQ-PSQ";

    std::printf("test_ple_row_gather_real\n");
    std::printf("model: %s\n", model_dir);

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    auto collection_result = fw::SafetensorsCollection::open(model_dir);
    if (!collection_result.ok()) {
        std::printf("FAIL: open collection: %s\n",
                    collection_result.status().message().c_str());
        return 1;
    }
    const fw::SafetensorsCollection& collection = collection_result.value();

    const std::string shard_prefix =
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";
    auto layout_result = ple::table_layout(collection, shard_prefix);
    if (!layout_result.ok()) {
        std::printf("FAIL: table_layout: %s\n", layout_result.status().message().c_str());
        return 1;
    }
    const ple::TableLayout layout = layout_result.value();
    const std::uint64_t total_rows =
        static_cast<std::uint64_t>(layout.shard_count) * layout.rows_per_shard;
    std::printf("shards=%u rows_per_shard=%llu row_bytes=%llu total_rows=%llu\n",
                layout.shard_count,
                static_cast<unsigned long long>(layout.rows_per_shard),
                static_cast<unsigned long long>(layout.row_bytes),
                static_cast<unsigned long long>(total_rows));

    std::vector<std::uint64_t> ids;
    ids.push_back(0);
    ids.push_back(1);
    ids.push_back(layout.rows_per_shard - 1);
    ids.push_back(layout.rows_per_shard);
    ids.push_back(layout.rows_per_shard + 1);
    ids.push_back(total_rows - 1);
    for (std::uint64_t seed = 7; ids.size() < 300; seed += 13) {
        const std::uint64_t r = mix(seed) % total_rows;
        ids.push_back(r);
    }
    const std::uint32_t count = static_cast<std::uint32_t>(ids.size());
    const std::size_t bytes = static_cast<std::size_t>(count) * layout.row_bytes;
    std::printf("sampling %u rows (%zu bytes)\n", count, bytes);

    std::vector<ple::RowLocation> locs;
    const ps::Status loc_st = ple::locate_rows(layout, collection, ids.data(), count, locs);
    check(loc_st.ok(), "locate_rows on real table");
    if (!loc_st.ok()) {
        std::printf("  reason: %s\n", loc_st.message().c_str());
        return 1;
    }
    check(locs.size() == count, "locate returns one location per id");
    check(locs[0].absolute_offset > 0, "locate offset non-zero");
    check(locs[0].byte_length == layout.row_bytes, "locate row byte length");
    check(locs[3].row_in_shard == 0, "locate shard boundary row starts a shard");
    check(locs[3].global_row == layout.rows_per_shard, "locate shard boundary row id");
    check(locs[5].global_row == total_rows - 1, "locate last row");
    check(locs[5].row_in_shard == layout.rows_per_shard - 1, "locate last row in shard");
    check(locs[0].global_row == 0 && locs[2].global_row == layout.rows_per_shard - 1,
          "locate boundary rows");

    std::vector<std::uint8_t> expect(bytes);
    bool expect_ok = true;
    std::map<std::string, const std::uint8_t*> shard_base;
    std::map<std::string, std::size_t> shard_bytes;
    for (std::uint32_t i = 0; i < count; ++i) {
        auto it = shard_base.find(locs[i].tensor_name);
        if (it == shard_base.end()) {
            std::size_t tensor_bytes = 0;
            auto data = collection.tensor_data(locs[i].tensor_name, tensor_bytes);
            if (!data.ok()) {
                std::printf("FAIL: tensor_data %s: %s\n", locs[i].tensor_name.c_str(),
                            data.status().message().c_str());
                ++g_fail;
                expect_ok = false;
                break;
            }
            shard_bytes[locs[i].tensor_name] = tensor_bytes;
            shard_base[locs[i].tensor_name] =
                static_cast<const std::uint8_t*>(data.value());
        }
        const std::size_t base = static_cast<std::size_t>(locs[i].row_in_shard) *
                                 static_cast<std::size_t>(layout.row_bytes);
        if (base + layout.row_bytes > shard_bytes[locs[i].tensor_name]) {
            std::printf("FAIL: row %u base=%zu exceeds tensor bytes=%zu\n", i, base,
                        shard_bytes[locs[i].tensor_name]);
            ++g_fail;
            expect_ok = false;
            break;
        }
        std::memcpy(expect.data() + static_cast<std::size_t>(i) * layout.row_bytes,
                    shard_base[locs[i].tensor_name] + base, layout.row_bytes);
    }
    check(expect_ok, "expected rows materialized from shard tensors");

    void* dev_dst = nullptr;
    if (hipMalloc(&dev_dst, bytes) != hipSuccess) {
        std::printf("FAIL: hipMalloc\n");
        return 1;
    }

    ple::SsdGatherStats stats;
    const ps::Status ssd_st =
        ple::gather_rows_ssd(layout, collection, ids.data(), count, dev_dst, nullptr, stats,
                             true);
    check(ssd_st.ok(), "gather_rows_ssd on real table");
    if (!ssd_st.ok()) {
        std::printf("  reason: %s\n", ssd_st.message().c_str());
        (void)hipFree(dev_dst);
        return 1;
    }
    std::printf("gather: path=%d logical=%llu physical=%llu blocks=%llu "
                "hipfile=%llu posix=%llu eligible=%d short_read=%d\n",
                static_cast<int>(stats.path),
                static_cast<unsigned long long>(stats.logical_bytes),
                static_cast<unsigned long long>(stats.physical_bytes),
                static_cast<unsigned long long>(stats.block_count),
                static_cast<unsigned long long>(stats.hipfile_blocks),
                static_cast<unsigned long long>(stats.posix_blocks),
                stats.fastpath_eligible ? 1 : 0, stats.short_read ? 1 : 0);
    check(stats.path != ps::io::SsdReadPath::none, "gather used a storage path");
    check(stats.logical_bytes == bytes, "gather logical byte count");
    check(stats.block_count > 0, "gather touched blocks");
    check(!stats.short_read, "gather had no short read");
    check(stats.hipfile_blocks > 0, "gather read through hipFile");

    std::vector<std::uint8_t> got(bytes);
    const bool copied = hipMemcpy(got.data(), dev_dst, bytes, hipMemcpyDeviceToHost) ==
                        hipSuccess;
    check(copied, "gather result readback");

    if (copied && g_fail == 0) {
        std::uint32_t mismatch = 0;
        std::uint32_t first_bad = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::size_t off = static_cast<std::size_t>(i) * layout.row_bytes;
            if (std::memcmp(got.data() + off, expect.data() + off, layout.row_bytes) != 0) {
                if (mismatch == 0) first_bad = i;
                ++mismatch;
            }
        }
        check(mismatch == 0, "gathered rows are bitwise identical to shard contents");
        if (mismatch != 0) {
            std::printf("  mismatched rows: %u first_bad=%u (row_id=%llu)\n", mismatch,
                        first_bad,
                        static_cast<unsigned long long>(ids[first_bad]));
        }
        std::uint64_t nonzero = 0;
        for (std::uint8_t b : got) {
            if (b != 0) ++nonzero;
        }
        std::printf("nonzero bytes: %llu / %zu\n",
                    static_cast<unsigned long long>(nonzero), got.size());
    }

    ple::SsdGatherStats ram_stats;
    const ps::Status ram_st =
        ple::gather_rows_ram(layout, collection, ids.data(), count, dev_dst, nullptr);
    check(ram_st.ok(), "gather_rows_ram on real table");
    (void)ram_stats;

    (void)hipFree(dev_dst);

    if (g_fail != 0) {
        std::printf("=== test_ple_row_gather_real: %d FAIL ===\n", g_fail);
        return 1;
    }
    std::printf("=== test_ple_row_gather_real: PASS ===\n");
    return 0;
}
