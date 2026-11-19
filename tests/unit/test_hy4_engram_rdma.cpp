#include <atomic>
#include <chrono>
#include <thread>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <shelf/models/hy4_flash/rdma/client.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
using namespace shelf::hy4::rdma;
struct File {
    int fd;
    size_t size;
    uint8_t *data;
    const uint8_t *bits, *scale;
    uint64_t rows;
    File(int g, const std::string &directory) {
        std::string key = "model.layers." + std::to_string(1 + g * 10) +
                          ".engram_layer.multi_head_embedding.embedding.weight";
        auto path = directory + "/" + key + ".fp8.safetensors";
        fd = open(path.c_str(), O_RDONLY);
        struct stat st{};
        if (fd < 0 || fstat(fd, &st))
            throw std::runtime_error("open");
        size = st.st_size;
        data = (uint8_t *)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED)
            throw std::runtime_error("mmap");
        uint64_t n;
        std::memcpy(&n, data, 8);
        auto h = nlohmann::json::parse(data + 8, data + 8 + n);
        rows = h[key]["shape"][0].get<uint64_t>();
        bits = data + 8 + n + h[key]["data_offsets"][0].get<uint64_t>();
        scale = data + 8 + n + h[key + ".scale"]["data_offsets"][0].get<uint64_t>();
    }
    ~File() {
        munmap(data, size);
        close(fd);
    }
};
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "使用方法: test_hy4_engram_rdma 表のディレクトリ\n";
        return 2;
    }
    File a(0, argv[1]), b(1, argv[1]), c(2, argv[1]), d(3, argv[1]);
    File *files[] = {&a, &b, &c, &d};
    std::mt19937_64 rng(763210);
    Client client;
    for (int start = 0; start < 250000; start += 4096) {
        Indices ix;
        int count = std::min(4096, 250000 - start);
        for (int g = 0; g < 4; ++g) {
            for (int i = 0; i < count; ++i)
                ix[g].push_back(rng() % files[g]->rows);
            if (!start) {
                ix[g].push_back(0);
                ix[g].push_back(files[g]->rows - 1);
                ix[g].push_back(0);
                ix[g].push_back(1);
            }
        }
        auto out = client.read(ix);
        std::atomic<bool> mismatch{false};
        std::array<std::thread, 4> workers;
        for (int g = 0; g < 4; ++g) {
            workers[g] = std::thread([&, g] {
                for (size_t i = 0; i < ix[g].size(); ++i) {
                    auto idx = ix[g][i];
                    auto p = out[g].data() + i * 98;
                    if (memcmp(p, files[g]->bits + idx * 96, 96) ||
                        memcmp(p + 96, files[g]->scale + idx * 2, 2)) {
                        std::cerr << "不一致: 表 " << g << " 行 " << idx << "\n";
                        mismatch = true;
                        return;
                    }
                }
            });
        }
        for (auto &worker : workers)
            worker.join();
        if (mismatch)
            return 1;
        if (start % 40960 == 0)
            std::cerr << start << "\n";
    }
    std::cout << "4 表合計 100 万行と端の行が一致\n";
}
