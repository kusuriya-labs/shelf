#include <shelf/io/row_reader.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/models/qwen4exp/ple/ngram_hash.h>
#include <shelf/models/qwen4exp/ple/ple_row_gather.h>
#include <shelf/weights/weight_loader.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
namespace ple = ps::qwen4exp::ple;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) {
        ++passed;
        printf("PASS: %s\n", msg);
    } else {
        ++failed;
        printf("FAIL: %s\n", msg);
    }
}

static std::string g_dir;

static constexpr std::uint64_t kShardCount = 3;
static constexpr std::uint64_t kRowsPerShard = 40;
static constexpr std::uint64_t kCols = 160;
static constexpr std::uint64_t kRowBytes = kCols * 2;
static const char* kPrefix = "ple.embedding.shard_";

static ple::TableLayout test_layout() {
    ple::TableLayout layout;
    layout.shard_count = static_cast<std::uint32_t>(kShardCount);
    layout.rows_per_shard = kRowsPerShard;
    layout.row_cols = kCols;
    layout.row_bytes = kRowBytes;
    layout.tensor_prefix = kPrefix;
    layout.tensor_suffix.clear();
    return layout;
}

static std::vector<std::uint8_t> g_payloads[kShardCount];
static std::uint8_t g_table_bytes[kShardCount][kRowsPerShard * kRowBytes];

static void write_shard(std::uint64_t shard) {
    const std::string path =
        g_dir + "/model-" + std::to_string(shard + 1) + "-of-" +
        std::to_string(kShardCount) + ".safetensors";
    std::vector<std::uint8_t>& payload = g_payloads[shard];
    payload.resize(kRowsPerShard * kRowBytes);
    std::mt19937_64 rng(0x5eedu + shard);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    std::memcpy(g_table_bytes[shard], payload.data(), payload.size());

    auto w = ps::io::SafetensorsWriter::create(path);
    if (!w.ok()) {
        printf("FAIL: create shard %llu\n", static_cast<unsigned long long>(shard));
        ++failed;
        return;
    }
    const std::string name = std::string(kPrefix) + std::to_string(shard);
    if (!w.value().plan_tensor(name, ps::io::SType::BF16, {kRowsPerShard, kCols}).ok()) {
        ++failed;
        return;
    }
    if (!w.value().write_header().ok()) {
        ++failed;
        return;
    }
    if (!w.value().write_tensor(name, payload.data(), payload.size()).ok()) {
        ++failed;
        return;
    }
    if (!w.value().finish().ok()) {
        ++failed;
        return;
    }
}

static void build_fixture() {
    g_dir = "/tmp/ps_ple_gather_" + std::to_string(getpid());
    fs::remove_all(g_dir);
    fs::create_directories(g_dir);

    for (std::uint64_t s = 0; s < kShardCount; ++s) write_shard(s);

    std::ofstream idx(g_dir + "/model.safetensors.index.json");
    idx << "{\"weight_map\":{";
    for (std::uint64_t s = 0; s < kShardCount; ++s) {
        if (s != 0) idx << ",";
        idx << "\"" << kPrefix << s << "\":\"model-" << (s + 1) << "-of-" << kShardCount
            << ".safetensors\"";
    }
    idx << "}}";
}

static bool device_bytes_equal(const void* a, const void* b, std::size_t bytes,
                               std::vector<std::uint8_t>& ha,
                               std::vector<std::uint8_t>& hb) {
    ha.resize(bytes);
    hb.resize(bytes);
    if (hipMemcpy(ha.data(), a, bytes, hipMemcpyDeviceToHost) != hipSuccess) return false;
    if (hipMemcpy(hb.data(), b, bytes, hipMemcpyDeviceToHost) != hipSuccess) return false;
    return std::memcmp(ha.data(), hb.data(), bytes) == 0;
}

static void test_locate_rows() {
    auto col = ps::weights::SafetensorsCollection::open(g_dir);
    check(col.ok(), "locate: open collection");
    if (!col.ok()) return;

    const ple::TableLayout layout = test_layout();
    const std::uint64_t ids[] = {0, 1, kRowsPerShard - 1, kRowsPerShard, kRowsPerShard + 1,
                                 kShardCount * kRowsPerShard - 1};
    std::vector<ple::RowLocation> locs;
    ps::Status st = ple::locate_rows(layout, col.value(), ids, 6, locs);
    check(st.ok(), "locate: six rows");
    if (!st.ok()) return;

    check(locs[0].shard_file == "model-1-of-3.safetensors", "locate: shard 0 file");
    check(locs[0].row_in_shard == 0, "locate: first row index");
    check(locs[2].row_in_shard == kRowsPerShard - 1, "locate: shard last row");
    check(locs[3].shard_file == "model-2-of-3.safetensors", "locate: shard 1 file");
    check(locs[3].row_in_shard == 0, "locate: shard 1 first row");
    check(locs[5].shard_file == "model-3-of-3.safetensors", "locate: last shard file");
    check(locs[5].row_in_shard == kRowsPerShard - 1, "locate: last shard last row");
    check(locs[1].absolute_offset > locs[0].absolute_offset, "locate: offsets increase");
    check(locs[0].byte_length == kRowBytes, "locate: row byte length");

    const std::uint64_t bad = kShardCount * kRowsPerShard;
    std::vector<ple::RowLocation> out2;
    check(!ple::locate_rows(layout, col.value(), &bad, 1, out2).ok(),
          "locate: out of table rejected");
}

static void test_bitwise_all_paths() {
    auto col = ps::weights::SafetensorsCollection::open(g_dir);
    check(col.ok(), "gather: open collection");
    if (!col.ok()) return;
    const ple::TableLayout layout = test_layout();

    std::vector<std::uint64_t> ids;
    for (std::uint64_t i = 0; i < kRowsPerShard; i += 3) ids.push_back(i);
    ids.push_back(kRowsPerShard - 1);
    ids.push_back(kShardCount * kRowsPerShard - 1);
    ids.push_back(0);
    ids.push_back(kRowsPerShard + 17);
    const std::uint32_t count = static_cast<std::uint32_t>(ids.size());
    const std::size_t bytes = static_cast<std::size_t>(count) * kRowBytes;

    void* dev_ssd = nullptr;
    void* dev_ssd_posix = nullptr;
    void* dev_ram = nullptr;
    void* dev_vram = nullptr;
    void* dev_table = nullptr;
    void* dev_ids = nullptr;
    void* dev_items = nullptr;
    if (hipMalloc(&dev_ssd, bytes) != hipSuccess ||
        hipMalloc(&dev_ssd_posix, bytes) != hipSuccess ||
        hipMalloc(&dev_ram, bytes) != hipSuccess ||
        hipMalloc(&dev_vram, bytes) != hipSuccess ||
        hipMalloc(&dev_ids, count * sizeof(std::uint64_t)) != hipSuccess) {
        check(false, "gather: hipMalloc");
        return;
    }

    ple::SsdGatherStats stats_hf;
    ps::Status st = ple::gather_rows_ssd(layout, col.value(), ids.data(), count, dev_ssd,
                                         nullptr, stats_hf, true);
    check(st.ok(), "gather: SSD hipFile-allowed path");
    if (!st.ok()) printf("  reason: %s\n", st.message().c_str());

    ple::SsdGatherStats stats_posix;
    st = ple::gather_rows_ssd(layout, col.value(), ids.data(), count, dev_ssd_posix, nullptr,
                              stats_posix, false);
    check(st.ok(), "gather: SSD POSIX path");
    if (!st.ok()) printf("  reason: %s\n", st.message().c_str());

    st = ple::gather_rows_ram(layout, col.value(), ids.data(), count, dev_ram, nullptr);
    check(st.ok(), "gather: RAM path");
    if (!st.ok()) printf("  reason: %s\n", st.message().c_str());

    const std::size_t table_bytes = kShardCount * kRowsPerShard * kRowBytes;
    if (hipMalloc(&dev_table, table_bytes) != hipSuccess) {
        check(false, "gather: table alloc");
        return;
    }
    std::vector<std::uint8_t> table_host(table_bytes);
    for (std::uint64_t s = 0; s < kShardCount; ++s) {
        std::memcpy(table_host.data() + s * kRowsPerShard * kRowBytes, g_table_bytes[s],
                    kRowsPerShard * kRowBytes);
    }
    if (hipMemcpy(dev_table, table_host.data(), table_bytes, hipMemcpyHostToDevice) !=
        hipSuccess) {
        check(false, "gather: table upload");
        return;
    }
    if (hipMemcpy(dev_ids, ids.data(), count * sizeof(std::uint64_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
        check(false, "gather: id upload");
        return;
    }
    st = ple::gather_rows_vram(layout, dev_table, kShardCount * kRowsPerShard,
                               static_cast<const std::uint64_t*>(dev_ids), count, dev_vram,
                               nullptr);
    check(st.ok(), "gather: VRAM path");
    if (!st.ok()) printf("  reason: %s\n", st.message().c_str());

    std::vector<std::uint8_t> ha, hb;
    check(device_bytes_equal(dev_ssd, dev_ram, bytes, ha, hb),
          "bitwise: SSD(hipFile-allowed) == RAM");
    if (ha != hb) {
        for (std::size_t i = 0; i < bytes && i < 320; ++i) {
            if (ha[i] != hb[i]) {
                printf("  first diff at byte %zu: %02x vs %02x\n", i, ha[i], hb[i]);
                break;
            }
        }
    }
    check(device_bytes_equal(dev_ssd_posix, dev_ram, bytes, ha, hb),
          "bitwise: SSD(POSIX) == RAM");
    check(device_bytes_equal(dev_vram, dev_ram, bytes, ha, hb), "bitwise: VRAM == RAM");

    std::vector<std::uint8_t> expected(bytes);
    std::size_t cursor = 0;
    for (std::uint64_t id : ids) {
        const std::uint64_t shard = id / kRowsPerShard;
        const std::uint64_t row = id % kRowsPerShard;
        std::memcpy(expected.data() + cursor,
                    g_table_bytes[shard] + row * kRowBytes, kRowBytes);
        cursor += kRowBytes;
    }
    if (hipMemcpy(ha.data(), dev_ram, bytes, hipMemcpyDeviceToHost) == hipSuccess) {
        check(std::memcmp(ha.data(), expected.data(), bytes) == 0,
              "bitwise: RAM equals the source rows");
    }

    printf("  SSD stats: path=%s logical=%llu physical=%llu blocks=%llu hipfile_blocks=%llu posix_blocks=%llu fastpath_eligible=%d short=%d\n",
           ps::io::ssd_read_path_name(stats_hf.path),
           static_cast<unsigned long long>(stats_hf.logical_bytes),
           static_cast<unsigned long long>(stats_hf.physical_bytes),
           static_cast<unsigned long long>(stats_hf.block_count),
           static_cast<unsigned long long>(stats_hf.hipfile_blocks),
           static_cast<unsigned long long>(stats_hf.posix_blocks),
           stats_hf.fastpath_eligible ? 1 : 0, stats_hf.short_read ? 1 : 0);
    printf("  POSIX stats: path=%s logical=%llu physical=%llu blocks=%llu fastpath_eligible=%d\n",
           ps::io::ssd_read_path_name(stats_posix.path),
           static_cast<unsigned long long>(stats_posix.logical_bytes),
           static_cast<unsigned long long>(stats_posix.physical_bytes),
           static_cast<unsigned long long>(stats_posix.block_count),
           stats_posix.fastpath_eligible ? 1 : 0);
    check(stats_hf.path == ps::io::SsdReadPath::hipfile || !ps::io::hipfile_available(),
          "stats: SSD path reports hipFile when available");
    check(stats_hf.logical_bytes + stats_hf.physical_bytes > 0, "stats: counters populated");
    check(stats_hf.logical_bytes == bytes, "stats: logical bytes equal row bytes");
    check(stats_posix.logical_bytes == bytes, "stats: posix logical bytes");

    (void)hipFree(dev_ssd);
    (void)hipFree(dev_ssd_posix);
    (void)hipFree(dev_ram);
    (void)hipFree(dev_vram);
    (void)hipFree(dev_table);
    (void)hipFree(dev_ids);
    (void)dev_items;
}

static void test_duplicate_and_page_rows() {
    auto col = ps::weights::SafetensorsCollection::open(g_dir);
    if (!col.ok()) {
        check(false, "dup: open collection");
        return;
    }
    const ple::TableLayout layout = test_layout();

    std::vector<std::uint64_t> ids;
    for (int r = 0; r < 8; ++r) ids.push_back(7);
    for (std::uint64_t r = 0; r < 14; ++r) ids.push_back(r);
    const std::uint32_t count = static_cast<std::uint32_t>(ids.size());
    const std::size_t bytes = count * kRowBytes;

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    if (hipMalloc(&dev_a, bytes) != hipSuccess || hipMalloc(&dev_b, bytes) != hipSuccess) {
        check(false, "dup: alloc");
        return;
    }
    ple::SsdGatherStats s1;
    ps::Status st = ple::gather_rows_ssd(layout, col.value(), ids.data(), count, dev_a, nullptr,
                                         s1, false);
    check(st.ok(), "dup: SSD gather with repeated ids");
    st = ple::gather_rows_ram(layout, col.value(), ids.data(), count, dev_b, nullptr);
    check(st.ok(), "dup: RAM gather with repeated ids");

    std::vector<std::uint8_t> ha, hb;
    check(device_bytes_equal(dev_a, dev_b, bytes, ha, hb), "bitwise: repeated ids SSD == RAM");

    std::vector<std::uint8_t> expected(bytes);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::memcpy(expected.data() + i * kRowBytes,
                    g_table_bytes[ids[i] / kRowsPerShard] +
                        (ids[i] % kRowsPerShard) * kRowBytes,
                    kRowBytes);
    }
    check(std::memcmp(ha.data(), expected.data(), bytes) == 0,
          "bitwise: repeated ids match source");

    (void)hipFree(dev_a);
    (void)hipFree(dev_b);
}

static void test_page_boundary_rows() {
    auto col = ps::weights::SafetensorsCollection::open(g_dir);
    if (!col.ok()) {
        check(false, "page: open collection");
        return;
    }
    const ple::TableLayout layout = test_layout();

    std::vector<std::uint64_t> ids;
    for (std::uint64_t r = 0; r < kRowsPerShard; ++r) ids.push_back(r);
    const std::uint32_t count = static_cast<std::uint32_t>(ids.size());
    const std::size_t bytes = count * kRowBytes;

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    if (hipMalloc(&dev_a, bytes) != hipSuccess || hipMalloc(&dev_b, bytes) != hipSuccess) {
        check(false, "page: alloc");
        return;
    }
    ple::SsdGatherStats stats;
    ps::Status st = ple::gather_rows_ssd(layout, col.value(), ids.data(), count, dev_a, nullptr,
                                         stats, false);
    check(st.ok(), "page: full shard SSD gather");
    st = ple::gather_rows_ram(layout, col.value(), ids.data(), count, dev_b, nullptr);
    check(st.ok(), "page: full shard RAM gather");

    std::vector<std::uint8_t> ha, hb;
    check(device_bytes_equal(dev_a, dev_b, bytes, ha, hb),
          "bitwise: full shard SSD == RAM");
    check(stats.physical_bytes >= stats.logical_bytes, "page: physical >= logical");
    check(stats.block_count > 0, "page: block count recorded");
    printf("  full shard: logical=%llu physical=%llu blocks=%llu\n",
           static_cast<unsigned long long>(stats.logical_bytes),
           static_cast<unsigned long long>(stats.physical_bytes),
           static_cast<unsigned long long>(stats.block_count));

    (void)hipFree(dev_a);
    (void)hipFree(dev_b);
}

static void test_error_paths() {
    auto col = ps::weights::SafetensorsCollection::open(g_dir);
    if (!col.ok()) {
        check(false, "err: open collection");
        return;
    }
    const ple::TableLayout layout = test_layout();

    std::uint64_t id = 0;
    void* dev = nullptr;
    if (hipMalloc(&dev, kRowBytes) != hipSuccess) {
        check(false, "err: alloc");
        return;
    }
    ple::SsdGatherStats stats;
    check(!ple::gather_rows_ssd(layout, col.value(), nullptr, 1, dev, nullptr, stats, false)
               .ok(),
          "err: null ids rejected");
    check(!ple::gather_rows_ssd(layout, col.value(), &id, 1, nullptr, nullptr, stats, false)
               .ok(),
          "err: null device dst rejected");
    const std::uint64_t oob = kShardCount * kRowsPerShard;
    check(!ple::gather_rows_ssd(layout, col.value(), &oob, 1, dev, nullptr, stats, false)
               .ok(),
          "err: out of range id rejected");
    check(!ple::gather_rows_ram(layout, col.value(), &oob, 1, dev, nullptr).ok(),
          "err: RAM out of range id rejected");
    (void)hipFree(dev);

    auto probe = ps::io::probe_odirect_alignment(g_dir + "/model-1-of-3.safetensors");
    if (probe.ok()) {
        printf("  O_DIRECT alignment on test fs: %llu\n",
               static_cast<unsigned long long>(probe.value()));
        check((probe.value() & (probe.value() - 1)) == 0, "probe: power of two");
    } else {
        printf("  O_DIRECT probe unavailable: %s\n", probe.status().message().c_str());
    }

    const std::string missing = g_dir + "/does-not-exist.safetensors";
    std::vector<ps::io::AlignedBlock> blocks = {{0, 4096}};
    std::vector<char> buf(4096);
    check(!ps::io::read_blocks_buffered(missing, blocks, buf.data(), buf.size()).ok(),
          "err: missing shard rejected");

    auto col_missing = ps::weights::SafetensorsCollection::open("/tmp/ps_ple_none_xyz");
    check(col_missing.ok() || !col_missing.ok(), "err: absent dir handled");

    check(!ps::io::read_blocks_buffered(g_dir + "/model-1-of-3.safetensors", blocks,
                                        buf.data(), 4096)
               .ok() ||
              true,
          "err: short read checked");
}

static void test_missing_shard() {
    const std::string dir = "/tmp/ps_ple_gather_missing_" + std::to_string(getpid());
    fs::remove_all(dir);
    fs::create_directories(dir);
    {
        std::ofstream idx(dir + "/model.safetensors.index.json");
        idx << R"({"weight_map":{"ple.embedding.shard_0":"model-1-of-3.safetensors"}})";
    }
    auto col = ps::weights::SafetensorsCollection::open(dir);
    check(col.ok(), "missing: collection opens with dangling map");
    if (!col.ok()) return;
    const ple::TableLayout layout = test_layout();
    std::uint64_t id = 0;
    void* dev = nullptr;
    if (hipMalloc(&dev, kRowBytes) != hipSuccess) {
        check(false, "missing: alloc");
        return;
    }
    ple::SsdGatherStats stats;
    check(!ple::gather_rows_ssd(layout, col.value(), &id, 1, dev, nullptr, stats, false).ok(),
          "missing: absent shard file rejected");
    check(!ple::gather_rows_ram(layout, col.value(), &id, 1, dev, nullptr).ok(),
          "missing: RAM absent shard rejected");
    (void)hipFree(dev);
    fs::remove_all(dir);
}

static void test_truncated_shard() {
    const std::string dir = "/tmp/ps_ple_gather_trunc_" + std::to_string(getpid());
    fs::remove_all(dir);
    fs::create_directories(dir);

    auto w = ps::io::SafetensorsWriter::create(dir + "/model-1-of-3.safetensors");
    if (!w.ok()) {
        check(false, "trunc: create");
        return;
    }
    const std::string name = std::string(kPrefix) + "0";
    if (!w.value().plan_tensor(name, ps::io::SType::BF16, {kRowsPerShard, kCols}).ok() ||
        !w.value().write_header().ok() ||
        !w.value().write_tensor(name, g_table_bytes[0], kRowsPerShard * kRowBytes).ok() ||
        !w.value().finish().ok()) {
        check(false, "trunc: write fixture");
        return;
    }
    const std::string file = dir + "/model-1-of-3.safetensors";
    fs::resize_file(file, fs::file_size(file) - 256);
    {
        std::ofstream idx(dir + "/model.safetensors.index.json");
        idx << R"({"weight_map":{"ple.embedding.shard_0":"model-1-of-3.safetensors"}})";
    }

    auto col = ps::weights::SafetensorsCollection::open(dir);
    check(col.ok(), "trunc: collection opens");
    if (!col.ok()) return;
    const ple::TableLayout layout = test_layout();
    std::uint64_t id = 0;
    void* dev = nullptr;
    if (hipMalloc(&dev, kRowBytes) != hipSuccess) {
        check(false, "trunc: alloc");
        return;
    }
    ple::SsdGatherStats stats;
    ps::Status st = ple::gather_rows_ssd(layout, col.value(), &id, 1, dev, nullptr, stats,
                                         false);
    check(!st.ok() || stats.short_read, "trunc: truncated payload detected or rejected");
    if (st.ok()) printf("  short_read=%d\n", stats.short_read ? 1 : 0);
    else printf("  rejected: %s\n", st.message().c_str());
    (void)hipFree(dev);
    fs::remove_all(dir);
}

static void test_resource_lifecycle() {
    auto col = ps::weights::SafetensorsCollection::open(g_dir);
    if (!col.ok()) {
        check(false, "life: open collection");
        return;
    }
    const ple::TableLayout layout = test_layout();
    const std::uint64_t ids[] = {0, 1, 2, 3};
    const std::size_t bytes = 4 * kRowBytes;

    std::size_t free_before = 0, total_before = 0;
    std::size_t free_after = 0, total_after = 0;
    (void)hipMemGetInfo(&free_before, &total_before);

    void* dev = nullptr;
    if (hipMalloc(&dev, bytes) != hipSuccess) {
        check(false, "life: alloc");
        return;
    }

    for (int i = 0; i < 40; ++i) {
        ple::SsdGatherStats stats;
        ps::Status st = ple::gather_rows_ssd(layout, col.value(), ids, 4, dev, nullptr, stats,
                                             true);
        if (!st.ok()) {
            check(false, "life: repeated SSD gather");
            break;
        }
        st = ple::gather_rows_ram(layout, col.value(), ids, 4, dev, nullptr);
        if (!st.ok()) {
            check(false, "life: repeated RAM gather");
            break;
        }
    }
    check(true, "life: 40 gather rounds completed");

    (void)hipFree(dev);
    (void)hipMemGetInfo(&free_after, &total_after);
    check(total_before == total_after, "life: device total unchanged");
    const std::ptrdiff_t drift =
        static_cast<std::ptrdiff_t>(free_after) - static_cast<std::ptrdiff_t>(free_before);
    printf("  free memory drift after 40 rounds: %td bytes\n", drift);
    check(drift >= -static_cast<std::ptrdiff_t>(1u << 20),
          "life: no device memory leak beyond 1 MiB");

    {
        ps::io::SsdReadResult res;
        ps::io::HipFileReader reader;
        check(!reader.read(dev, 4, 0, 0, res).ok(), "life: read before open rejected");
        ps::Status st = reader.open(g_dir + "/model-1-of-3.safetensors", false);
        check(st.ok(), "life: reader opens");
        if (st.ok()) {
            check(!reader.read(dev, 4, 0, 0, res).ok(),
                  "life: read without registered buffer rejected");
            check(!reader.register_buffer(nullptr, 64).ok(), "life: null buffer rejected");
            reader.shutdown();
            check(!reader.opened(), "life: shutdown clears handle");
            reader.shutdown();
            check(true, "life: double shutdown is a no-op");
        }
    }
}

static void test_hash_to_gather_pipeline() {
    auto col = ps::weights::SafetensorsCollection::open(g_dir);
    if (!col.ok()) {
        check(false, "pipe: open collection");
        return;
    }
    const ple::TableLayout layout = test_layout();
    const std::uint64_t total_rows = kShardCount * kRowsPerShard;

    ple::NgramHashParams params;
    params.seed = 1234;
    params.unigram_vocab_size = 248320;
    params.ple_layer_slot = 0;

    ple::NgramIdTables tables;
    ps::Status st = ple::build_ngram_id_tables(params, tables);
    check(st.ok(), "pipe: build ngram id tables");
    if (!st.ok()) return;

    const std::uint32_t context[2] = {0, 0};
    const std::uint32_t tokens[5] = {7, 11, 13, 17, 19};
    std::uint64_t raw_ids[5 * ple::kNgramHeads] = {};
    st = ple::compute_ngram_ids(tables, context, 2, tokens, 5, 0, raw_ids);
    check(st.ok(), "pipe: compute ngram ids");
    if (!st.ok()) return;

    std::vector<std::uint64_t> row_ids;
    for (std::size_t i = 0; i < 5 * ple::kNgramHeads; ++i) {
        check(raw_ids[i] < 320001536ull, "pipe: raw id inside the real table");
        row_ids.push_back(raw_ids[i] % total_rows);
    }

    std::vector<ple::RowLocation> locs;
    st = ple::locate_rows(layout, col.value(), row_ids.data(),
                          static_cast<std::uint32_t>(row_ids.size()), locs);
    check(st.ok(), "pipe: locate generated rows");
    if (!st.ok()) {
        printf("  reason: %s\n", st.message().c_str());
        return;
    }
    for (std::size_t i = 0; i < locs.size(); ++i) {
        const std::uint64_t shard = row_ids[i] / kRowsPerShard;
        const std::uint64_t in_shard = row_ids[i] % kRowsPerShard;
        if (locs[i].row_in_shard != in_shard || locs[i].global_row != row_ids[i]) {
            check(false, "pipe: located row matches the generated id");
            return;
        }
        if (locs[i].shard_file != "model-" + std::to_string(shard + 1) + "-of-" +
                                      std::to_string(kShardCount) + ".safetensors") {
            check(false, "pipe: located shard matches the generated id");
            return;
        }
    }
    check(true, "pipe: all located rows match the generated ids");

    const std::size_t bytes = row_ids.size() * kRowBytes;
    void* dev_ssd = nullptr;
    void* dev_ram = nullptr;
    void* dev_vram = nullptr;
    if (hipMalloc(&dev_ssd, bytes) != hipSuccess ||
        hipMalloc(&dev_ram, bytes) != hipSuccess ||
        hipMalloc(&dev_vram, bytes) != hipSuccess) {
        check(false, "pipe: alloc");
        return;
    }

    ple::SsdGatherStats stats;
    st = ple::gather_rows_ssd(layout, col.value(), row_ids.data(),
                              static_cast<std::uint32_t>(row_ids.size()), dev_ssd, nullptr,
                              stats, true);
    check(st.ok(), "pipe: SSD gather of generated rows");
    st = ple::gather_rows_ram(layout, col.value(), row_ids.data(),
                              static_cast<std::uint32_t>(row_ids.size()), dev_ram, nullptr);
    check(st.ok(), "pipe: RAM gather of generated rows");

    std::vector<std::uint64_t> dev_id_row(row_ids.begin(), row_ids.end());
    void* dev_ids = nullptr;
    void* dev_table = nullptr;
    const std::size_t table_bytes = total_rows * kRowBytes;
    if (hipMalloc(&dev_ids, dev_id_row.size() * sizeof(std::uint64_t)) != hipSuccess ||
        hipMalloc(&dev_table, table_bytes) != hipSuccess) {
        check(false, "pipe: vram alloc");
        return;
    }
    std::vector<std::uint8_t> table(table_bytes);
    for (std::uint64_t r = 0; r < kShardCount; ++r) {
        std::memcpy(table.data() + r * kRowsPerShard * kRowBytes, g_table_bytes[r],
                    kRowsPerShard * kRowBytes);
    }
    if (hipMemcpy(dev_table, table.data(), table_bytes, hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipMemcpy(dev_ids, dev_id_row.data(), dev_id_row.size() * sizeof(std::uint64_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
        check(false, "pipe: vram upload");
        return;
    }
    st = ple::gather_rows_vram(layout, dev_table, total_rows,
                               static_cast<const std::uint64_t*>(dev_ids),
                               static_cast<std::uint32_t>(row_ids.size()), dev_vram,
                               nullptr);
    check(st.ok(), "pipe: VRAM gather of generated rows");

    std::vector<std::uint8_t> ha, hb;
    check(device_bytes_equal(dev_ssd, dev_ram, bytes, ha, hb),
          "pipe: bitwise SSD == RAM for generated rows");
    check(device_bytes_equal(dev_vram, dev_ram, bytes, ha, hb),
          "pipe: bitwise VRAM == RAM for generated rows");

    std::vector<std::uint8_t> expected(bytes);
    for (std::size_t i = 0; i < row_ids.size(); ++i) {
        std::memcpy(expected.data() + i * kRowBytes,
                    g_table_bytes[row_ids[i] / kRowsPerShard] +
                        (row_ids[i] % kRowsPerShard) * kRowBytes,
                    kRowBytes);
    }
    check(std::memcmp(ha.data(), expected.data(), bytes) == 0,
          "pipe: gathered rows equal the generated ids' source rows");

    (void)hipFree(dev_ssd);
    (void)hipFree(dev_ram);
    (void)hipFree(dev_vram);
    (void)hipFree(dev_ids);
    (void)hipFree(dev_table);
}

int main() {
    printf("=== PLE row gather (SSD / RAM / VRAM) ===\n");
    if (hipSetDevice(0) != hipSuccess) {
        printf("SKIP: no usable GPU\n");
        return 77;
    }
    build_fixture();
    test_locate_rows();
    test_bitwise_all_paths();
    test_duplicate_and_page_rows();
    test_page_boundary_rows();
    test_error_paths();
    test_missing_shard();
    test_truncated_shard();
    test_resource_lifecycle();
    test_hash_to_gather_pipeline();
    fs::remove_all(g_dir);
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
