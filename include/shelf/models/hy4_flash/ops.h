#pragma once

#include <shelf/models/hy4_flash/residency.h>
#include <shelf/models/hy4_flash/linear.h>
#include <future>
#include <optional>

namespace shelf::hy4 {
struct Mat {
    uint16_t *p;
    int rows, cols;
    size_t bytes() const { return size_t(rows) * cols * 2; }
    Mat row(int i, int count = 1) const { return {p + size_t(i) * cols, count, cols}; }
};
struct Param {
    const void *p;
    io::SType type;
};

std::vector<uint8_t> gather_engram(const Checkpoint &model, int layer, int rows, const std::vector<int> &ids,
                                   const std::vector<int> &prior);

class Ops {
  public:
    Residency &gpu;
    int rank;
    Region &arena;
    std::future<std::array<std::vector<uint8_t>, 4>> engram_future;
    std::optional<std::array<std::vector<uint8_t>, 4>> engram_data;
    hipStream_t stream = nullptr;
    const Hy4StepParams *params = nullptr;
    GraphCapture *capture = nullptr;
    Mat embed_device(const int *ids, int rows, bool expand);
    int max_kv_len = 0;
    explicit Ops(Residency &g, int r = 0) : gpu(g), rank(r), arena(g.workspace.at(r)), stream(g.compute.at(r)) {}
    void prefetch_engram(const std::vector<int> &ids, const std::vector<int> &prior);
    std::vector<uint8_t> take_engram(int group);
    Mat mat(int rows, int cols);
    Param param(const std::string &key) const;
    Mat linear(const std::string &name, Mat input);
    Mat norm(Mat x, const std::string &key, float eps = 1e-5f);
    Mat norm_from(Mat x, int first, int width, const std::string &key, float eps = 1e-5f);
    Mat norm_bias(Mat x, const uint16_t *bias, const std::string &key, float eps = 1e-5f);
    Mat norm_slice(Mat x, int first, int width, const std::string &key, float eps = 1e-5f);
    Mat slice(Mat x, int first, int width);
    Mat join(Mat a, Mat b);
    Mat add(Mat a, Mat b);
    void add_into(Mat a, Mat b, Mat out);
    Mat swiglu(Mat gate, Mat up, bool clip);
    Mat embed(const std::vector<int> &ids, bool expand);
    Mat collapse(Mat x, const std::string &prefix, float *&post);
    Mat collapse_norm(Mat x, const std::string &prefix, float *&post, const std::string &key, float eps = 1e-5f);
    Mat expand(Mat x, Mat y, const float *post);
    Mat rope(Mat x, int heads, int start, float theta);
    Mat rope_from(Mat x, int first, int width, int heads, int start, float theta);
    void copy(void *dest, const void *src, size_t bytes, hipMemcpyKind kind);
    int *indices(const std::vector<int> &ids);
    void *reserve_persistent(size_t bytes);
    int sparse_last_layer[42]{};
};
void checked(hipError_t status);
}
