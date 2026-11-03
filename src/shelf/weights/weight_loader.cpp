#include <shelf/core/memory/size_format.h>
#include <shelf/weights/weight_loader.h>

#include <shelf/core/memory/types.h>
#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/manifest/runtime_resolution.h>
#include <shelf/quantization/fp8/fp8_native.h>
#include <shelf/quantization/mxfp4/mxfp4.h>
#include <shelf/weights/canonical_partition.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace shelf::weights {
namespace {

using shelf::quantization::manifest::QuantizedEncoding;
using shelf::quantization::bundle::QuantizedModelReader;
using shelf::quantization::bundle::QuantizedTensorView;
using Tensor = gpu::Tensor;
namespace fs = std::filesystem;

Tensor alloc_tensor(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& strides, gpu::GpuArena& arena) {
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    auto v = arena.allocate_aligned(n * sizeof(std::uint16_t), 16);
    if (!v.ok()) return {};
    auto t = Tensor::create<std::uint16_t>(v.release(), shape, strides);
    if (!t.ok()) return {};
    return t.release();
}

Tensor alloc_tensor_f32(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& strides, gpu::GpuArena& arena) {
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    auto v = arena.allocate_aligned(n * sizeof(float), 16);
    if (!v.ok()) return {};
    auto t = Tensor::create<float>(v.release(), shape, strides);
    if (!t.ok()) return {};
    return t.release();
}

gpu::Tensor alloc_bytes(
    std::size_t bytes,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides,
    gpu::GpuArena& arena)
{
    auto v = arena.allocate_aligned(bytes, 16);
    if (!v.ok()) return {};
    auto t = gpu::Tensor::create<uint8_t>(v.release(), shape, strides);
    if (!t.ok()) return {};
    return t.release();
}

gpu::Tensor alloc_bytes16(
    std::size_t bytes,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides,
    gpu::GpuArena& arena)
{
    auto v = arena.allocate_aligned(bytes, 16);
    if (!v.ok()) return {};
    auto t = gpu::Tensor::create<std::uint16_t>(v.release(), shape, strides);
    if (!t.ok()) return {};
    return t.release();
}

Status copy_to_device(void* dst, const void* src, std::size_t bytes, hipStream_t stream) {
    hipError_t e = hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) {
        return Status::hip_error("hipMemcpyAsync", hipGetErrorString(e), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status expect_bytes(const std::string& name, const quantization::bundle::ByteSpan& span, std::size_t expected) {
    if ((expected > 0 && span.data == nullptr) || span.size != expected) {
        return Status::invalid_argument(
            ("quantized tensor byte count mismatch: " + name).c_str(), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

std::vector<std::size_t> contiguous_strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t i = shape.size(); i-- > 1u;) strides[i - 1u] = strides[i] * shape[i];
    return strides;
}

Status upload_tensor_direct(const SafetensorsCollection& reader, const std::string& name, void* dst, hipStream_t stream) {
    std::size_t bytes = 0;
    auto ptr_result = reader.tensor_data(name, bytes);
    if (!ptr_result.ok()) return ptr_result.status();
    const void* src = ptr_result.value();

    hipError_t e = hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) {
        return Status::hip_error("hipMemcpyAsync", hipGetErrorString(e), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status upload_tensor_f32_to_bf16(const SafetensorsCollection& reader, const std::string& name, void* dst, std::size_t n, hipStream_t stream) {
    std::size_t bytes = 0;
    auto ptr_result = reader.tensor_data(name, bytes);
    if (!ptr_result.ok()) return ptr_result.status();
    const void* src = ptr_result.value();

    std::vector<bf16_t> converted(n);
    const float* src_f = static_cast<const float*>(src);
    for (std::size_t i = 0; i < n; ++i) {
        converted[i].data = f32_to_bf16_rne(src_f[i]);
    }

    hipError_t e = hipMemcpyAsync(dst, converted.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) {
        return Status::hip_error("hipMemcpyAsync", hipGetErrorString(e), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Result<MatrixWeight> load_bf16_matrix_raw(
    const SafetensorsCollection& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream)
{
    auto spec_result = reader.tensor_spec(name);
    if (!spec_result.ok()) return spec_result.status();
    const auto& spec = spec_result.value();
    const auto shape = spec.shape;
    if (shape.size() != 2) {
        return Status::invalid_argument(
            ("matrix weight must be 2D: " + name).c_str(), __FILE__, __LINE__);
    }

    MatrixWeight w;
    w.encoding = MatrixEncoding::Bf16;
    w.rows = static_cast<uint32_t>(shape[0]);
    w.cols = static_cast<uint32_t>(shape[1]);
    w.k_padded = w.cols;

    std::vector<std::size_t> strides = {shape[1], 1};
    const std::size_t arena_used_before = arena.used();
    const std::size_t want_bytes = static_cast<std::size_t>(shape[0]) * shape[1] * sizeof(bf16_t);
    w.bf16 = alloc_tensor(shape, strides, arena);
    if (w.bf16.ndim() == 0) {
        return Status::insufficient_memory(
            "matrix allocation failed: " + name + " want " + gpu::format_gib(want_bytes) +
                " arena used " + gpu::format_gib(arena_used_before) + " / capacity " +
                gpu::format_gib(arena.capacity()),
            __FILE__, __LINE__);
    }
    if (!upload_tensor_direct(reader, name, w.bf16.data<void>(), stream).ok()) {
        return Status::invalid_state(
            ("matrix upload failed: " + name).c_str(), __FILE__, __LINE__);
    }
    return w;
}

Result<MatrixWeight> load_quantized_view(
    const QuantizedTensorView& v,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options)
{
    const std::string& name = v.name;

    if (v.logical_shape.size() != 2) {
        return Status::invalid_argument(
            ("quantized matrix weight must be 2D: " + name).c_str(), __FILE__, __LINE__);
    }
    const uint64_t m = static_cast<uint64_t>(v.logical_shape[0]);
    const uint64_t k = static_cast<uint64_t>(v.logical_shape[1]);
    if (m == 0 || k == 0) {
        return Status::invalid_argument(
            ("quantized matrix weight must be non-empty: " + name).c_str(), __FILE__, __LINE__);
    }
    const uint64_t kp = static_cast<uint64_t>(v.k_padded);

    MatrixWeight w;
    w.rows = static_cast<uint32_t>(m);
    w.cols = static_cast<uint32_t>(k);

    if (v.encoding == QuantizedEncoding::Bf16) {
        if (v.codes.data != nullptr || v.metadata1.data != nullptr) {
            return Status::invalid_argument(
                ("quantized bf16 tensor must not carry codes/scales: " + name).c_str(), __FILE__, __LINE__);
        }
        auto st = expect_bytes(name + " data", v.data, m * k * 2);
        if (!st.ok()) return st;
        w.encoding = MatrixEncoding::Bf16;
        w.k_padded = static_cast<uint32_t>(k);
        std::vector<std::size_t> shape = {m, k};
        std::vector<std::size_t> strides = {k, 1};
        w.bf16 = alloc_bytes(m * k * 2, shape, strides, arena);
        if (w.bf16.ndim() == 0) {
            return Status::insufficient_memory(
                ("quantized matrix allocation failed: " + name).c_str(), __FILE__, __LINE__);
        }
        st = copy_to_device(w.bf16.data<void>(), v.data.data, v.data.size, stream);
        if (!st.ok()) return st;
        apply_default_compute_spec(w);
        return w;
    }

    if (kp < k || kp % 32 != 0) {
        return Status::invalid_argument(
            ("quantized matrix k_padded invalid: " + name).c_str(), __FILE__, __LINE__);
    }
    if (v.data.data != nullptr) {
        return Status::invalid_argument(
            ("quantized tensor must not carry bf16 data: " + name).c_str(), __FILE__, __LINE__);
    }
    w.k_padded = static_cast<uint32_t>(kp);

    if (v.encoding == QuantizedEncoding::Fp8) {
        if ((kp % 128u) != 0u) {
            return Status::invalid_argument(
                ("fp8 block128 k_padded must be a multiple of 128: " + name).c_str(),
                __FILE__, __LINE__);
        }
        const uint64_t scale_block_k = v.scale_block_k != 0 ? v.scale_block_k : 128u;
        const uint64_t scale_n = (m + 127u) / 128u;
        const uint64_t scale_k = kp / scale_block_k;
        auto st = expect_bytes(name + " codes", v.codes, m * kp);
        if (!st.ok()) return st;
        st = expect_bytes(name + " scales", v.metadata1, scale_n * scale_k * sizeof(float));
        if (!st.ok()) return st;

        quantization::fp8::Fp8BlockCanonicalView cv;
        cv.batch = 1;
        cv.n = m;
        cv.padded_k = kp;
        cv.codes = static_cast<const uint8_t*>(v.codes.data);
        cv.codes_bytes = v.codes.size;
        cv.scales = reinterpret_cast<const float*>(v.metadata1.data);
        cv.scales_bytes = v.metadata1.size;
        cv.scale_n = scale_n;
        cv.scale_k = scale_k;
        cv.scale_block_k = scale_block_k;
        if (!cv.validate()) {
            return Status::invalid_argument(
                ("fp8 block128 canonical payload invalid: " + name).c_str(), __FILE__, __LINE__);
        }
        w.encoding = MatrixEncoding::Fp8Finegrained;
        w.weight_scale_group = static_cast<uint32_t>(scale_block_k);

        if (!options.preshuffle) {
            std::vector<std::size_t> cshape = {m, kp};
            std::vector<std::size_t> cstrides = {kp, 1};
            w.codes = alloc_bytes(v.codes.size, cshape, cstrides, arena);
            if (w.codes.ndim() == 0) {
                return Status::insufficient_memory(
                    ("quantized matrix codes allocation failed: " + name).c_str(), __FILE__, __LINE__);
            }
            st = copy_to_device(w.codes.data<void>(), v.codes.data, v.codes.size, stream);
            if (!st.ok()) return st;
            std::vector<std::size_t> sshape = {scale_n, scale_k};
            std::vector<std::size_t> sstrides = {scale_k, 1};
            w.scales = alloc_tensor_f32(sshape, sstrides, arena);
            if (w.scales.ndim() == 0) {
                return Status::insufficient_memory(
                    ("quantized matrix scales allocation failed: " + name).c_str(), __FILE__, __LINE__);
            }
            st = copy_to_device(w.scales.data<void>(), v.metadata1.data, v.metadata1.size, stream);
            if (!st.ok()) return st;
            w.storage_scale_stride_bytes = static_cast<uint32_t>(scale_k * sizeof(float));
            w.codes_row_stride_bytes = static_cast<uint32_t>(kp);
            w.scale_row_stride_bytes = static_cast<uint32_t>(scale_k * sizeof(float));
            apply_default_compute_spec(w);
            return w;
        }

        static quantization::fp8::Fp8NativeHost native;
        if (!quantization::fp8::preshuffle_fp8_native(cv, native)) {
            return Status::invalid_argument(
                ("fp8 block128 payload not supported for arch: " + name).c_str(), __FILE__, __LINE__);
        }
        std::vector<std::size_t> cshape = {native.rows / 16u, native.codes_row_stride_bytes};
        std::vector<std::size_t> cstrides = {native.codes_row_stride_bytes, 1};
        w.codes = alloc_bytes(native.codes_bytes, cshape, cstrides, arena);
        if (w.codes.ndim() == 0) {
            return Status::insufficient_memory(
                ("quantized matrix codes allocation failed: " + name).c_str(), __FILE__, __LINE__);
        }
        st = copy_to_device(w.codes.data<void>(), native.codes.get(), native.codes_bytes, stream);
        if (!st.ok()) return st;
        std::vector<std::size_t> sshape = {native.scale_n, native.scale_k};
        std::vector<std::size_t> sstrides = {native.scale_k, 1};
        w.scales = alloc_tensor_f32(sshape, sstrides, arena);
        if (w.scales.ndim() == 0) {
            return Status::insufficient_memory(
                ("quantized matrix scales allocation failed: " + name).c_str(), __FILE__, __LINE__);
        }
        st = copy_to_device(w.scales.data<void>(), native.scales.get(),
                            native.scale_count * sizeof(float), stream);
        if (!st.ok()) return st;
        w.storage_scale_stride_bytes = static_cast<uint32_t>(native.scale_k * sizeof(float));
        w.codes_row_stride_bytes = static_cast<uint32_t>(native.codes_row_stride_bytes);
        w.scale_row_stride_bytes = static_cast<uint32_t>(native.scale_k * sizeof(float));
        w.preshuffled = true;
        apply_default_compute_spec(w);
        return w;
    }

    if (v.encoding == QuantizedEncoding::Mxfp4) {
        if ((kp % 64u) != 0u) {
            return Status::unsupported(
                ("mxfp4 matrix weight k_padded must be a multiple of 64: " + name).c_str(),
                __FILE__, __LINE__);
        }
        if ((m % 16u) != 0u) {
            return Status::unsupported(
                ("mxfp4 matrix weight rows must be a multiple of 16: " + name).c_str(),
                __FILE__, __LINE__);
        }
        const uint64_t packed_k = kp / quantization::kMxfp4WeightsPerCode;
        const uint64_t scale_k = kp / quantization::kMxfp4ScaleBlockK;
        auto st = expect_bytes(name + " codes", v.codes, m * packed_k);
        if (!st.ok()) return st;
        st = expect_bytes(name + " scales", v.metadata1, m * scale_k);
        if (!st.ok()) return st;

        quantization::mxfp4::Mxfp4CanonicalView cv;
        cv.batch = 1;
        cv.n = m;
        cv.padded_k = kp;
        cv.codes = static_cast<const uint8_t*>(v.codes.data);
        cv.codes_bytes = v.codes.size;
        cv.scales = static_cast<const uint8_t*>(v.metadata1.data);
        cv.scales_bytes = v.metadata1.size;
        if (!cv.validate()) {
            return Status::invalid_argument(
                ("mxfp4 canonical payload invalid: " + name).c_str(), __FILE__, __LINE__);
        }
        const uint64_t scale_count = m * scale_k;
        for (uint64_t i = 0; i < scale_count; ++i) {
            if (cv.scales[i] == quantization::mxfp4::kE8m0NanByte) {
                return Status::invalid_argument(
                    ("mxfp4 scale stream carries an E8M0 NaN byte: " + name).c_str(),
                    __FILE__, __LINE__);
            }
        }

        w.encoding = MatrixEncoding::Mxfp4;
        w.weight_scale_group = quantization::kMxfp4ScaleBlockK;

        std::vector<std::size_t> cshape = {m, packed_k};
        std::vector<std::size_t> cstrides = {packed_k, 1};
        w.codes = alloc_bytes(v.codes.size, cshape, cstrides, arena);
        if (w.codes.ndim() == 0) {
            return Status::insufficient_memory(
                ("mxfp4 codes allocation failed: " + name).c_str(), __FILE__, __LINE__);
        }
        st = copy_to_device(w.codes.data<void>(), v.codes.data, v.codes.size, stream);
        if (!st.ok()) return st;

        std::vector<std::size_t> sshape = {m, scale_k};
        std::vector<std::size_t> sstrides = {scale_k, 1};
        w.scales = alloc_bytes(v.metadata1.size, sshape, sstrides, arena);
        if (w.scales.ndim() == 0) {
            return Status::insufficient_memory(
                ("mxfp4 scales allocation failed: " + name).c_str(), __FILE__, __LINE__);
        }
        st = copy_to_device(w.scales.data<void>(), v.metadata1.data, v.metadata1.size, stream);
        if (!st.ok()) return st;

        w.codes_row_stride_bytes = static_cast<uint32_t>(packed_k);
        w.scale_row_stride_bytes = static_cast<uint32_t>(scale_k);
        w.storage_scale_stride_bytes = static_cast<uint32_t>(scale_k);
        w.preshuffled = options.preshuffle;
        apply_default_compute_spec(w);
        return w;
    }

    return Status::invalid_argument(
        ("quantized matrix unknown encoding: " + name).c_str(), __FILE__, __LINE__);
}

Result<MatrixWeight> load_quantized_matrix_impl(
    const QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options)
{
    auto view_result = reader.resolve(name);
    if (!view_result.ok()) return view_result.status();

    OwnedCanonicalTensor partitioned_payload;
    QuantizedTensorView partitioned_view;
    const QuantizedTensorView* resolved = &view_result.value();
    const TensorPartitionDesc* applied_partition = nullptr;
    if (options.partition_plan != nullptr) {
        auto it = options.partition_plan->tensors.find(resolved->name);
        if (it != options.partition_plan->tensors.end()) {
            Status partition_status = materialize_rank_local_canonical(
                it->second, *resolved, partitioned_payload, partitioned_view);
            if (!partition_status.ok()) return partition_status;
            resolved = &partitioned_view;
            applied_partition = &it->second;
        }
    }

    Result<MatrixWeight> loaded = load_quantized_view(*resolved, arena, stream, options);
    if (!loaded.ok()) return loaded.status();
    if (applied_partition != nullptr) loaded.value().partition = *applied_partition;
    return loaded;
}

}

Result<SafetensorsCollection> SafetensorsCollection::open(const std::string& model_dir) {
    SafetensorsCollection col;
    col.model_dir_ = model_dir;

    std::string index_path = model_dir + "/model.safetensors.index.json";
    std::ifstream index_f(index_path);
    bool has_index = index_f.is_open();
    if (has_index) {
        nlohmann::json index_json;
        try {
            index_f >> index_json;
        } catch (const nlohmann::json::exception& e) {
            return Status::invalid_argument(
                ("bad safetensors index: " + std::string(e.what())).c_str(),
                __FILE__, __LINE__);
        }
        if (!index_json.contains("weight_map") || !index_json["weight_map"].is_object()) {
            return Status::invalid_argument(
                "bad safetensors index: missing weight_map", __FILE__, __LINE__);
        }
        for (auto it = index_json["weight_map"].begin();
             it != index_json["weight_map"].end(); ++it) {
            col.weight_map_[it.key()] = it.value().get<std::string>();
        }
    } else {
        const fs::path single = fs::path(model_dir) / "model.safetensors";
        if (fs::exists(single)) {
            col.single_shard_ = "model.safetensors";
        }
    }

    return col;
}

Result<io::StTensorSpec> SafetensorsCollection::tensor_spec(const std::string& name) const {
    auto reader_result = reader_for(name);
    if (!reader_result.ok()) return reader_result.status();
    return reader_result.value()->tensor_spec(name);
}

Result<const void*> SafetensorsCollection::tensor_data(const std::string& name, std::size_t& out_bytes) const {
    auto reader_result = reader_for(name);
    if (!reader_result.ok()) return reader_result.status();
    return reader_result.value()->tensor_data(name, out_bytes);
}

namespace {

bool safe_shard_name(const std::string& shard) {
    if (shard.empty() || shard.front() == '/') return false;
    if (shard.find("..") != std::string::npos) return false;
    if (shard.find('\\') != std::string::npos) return false;
    return shard.find('/') == std::string::npos;
}

}  // namespace

Result<std::string> SafetensorsCollection::tensor_shard(const std::string& name) const {
    std::string shard = shard_for(name);
    if (shard.empty()) {
        return Status::invalid_argument(
            ("tensor not found: " + name).c_str(), __FILE__, __LINE__);
    }
    if (!safe_shard_name(shard)) {
        return Status::invalid_argument(
            ("unsafe shard name: " + shard).c_str(), __FILE__, __LINE__);
    }
    return shard;
}

Result<io::StAxisSpan> SafetensorsCollection::tensor_span(const std::string& name) const {
    auto reader_result = reader_for(name);
    if (!reader_result.ok()) return reader_result.status();
    return reader_result.value()->tensor_span(name);
}

Result<io::StAxisSpan> SafetensorsCollection::axis_span(const std::string& name,
                                                         std::uint64_t unit_begin,
                                                         std::uint64_t unit_count) const {
    auto reader_result = reader_for(name);
    if (!reader_result.ok()) return reader_result.status();
    return reader_result.value()->axis_span(name, unit_begin, unit_count);
}

Result<std::uint64_t> SafetensorsCollection::shard_file_size(const std::string& shard) const {
    if (!safe_shard_name(shard)) {
        return Status::invalid_argument(
            ("unsafe shard name: " + shard).c_str(), __FILE__, __LINE__);
    }
    auto reader_result = open_shard(shard);
    if (!reader_result.ok()) return reader_result.status();
    return reader_result.value()->file_size();
}

std::string SafetensorsCollection::shard_for(const std::string& name) const {
    if (weight_map_.empty()) {
        if (!single_shard_.empty()) return single_shard_;
        return "model.safetensors-00001-of-00001.safetensors";
    }
    auto it = weight_map_.find(name);
    if (it == weight_map_.end()) return "";
    return it->second;
}

Result<const io::SafetensorsReader*> SafetensorsCollection::reader_for(const std::string& name) const {
    std::string shard = shard_for(name);
    if (shard.empty()) {
        return Status::invalid_argument(
            ("tensor not found: " + name).c_str(), __FILE__, __LINE__);
    }
    return open_shard(shard);
}

Result<const io::SafetensorsReader*> SafetensorsCollection::open_shard(const std::string& shard) const {
    auto existing = shard_readers_.find(shard);
    if (existing != shard_readers_.end()) {
        return &existing->second;
    }

    std::string path = model_dir_ + "/" + shard;
    auto reader_result = io::SafetensorsReader::open(path);
    if (!reader_result.ok()) return reader_result.status();

    auto inserted = shard_readers_.emplace(shard, reader_result.release());
    return &inserted.first->second;
}

Result<std::vector<std::string>> SafetensorsCollection::tensor_names() const {
    std::vector<std::string> shards;
    if (weight_map_.empty()) {
        shards.push_back(single_shard_.empty()
                             ? "model.safetensors-00001-of-00001.safetensors"
                             : single_shard_);
    } else {
        std::map<std::string, bool> seen;
        for (const auto& kv : weight_map_) seen[kv.second] = true;
        shards.reserve(seen.size());
        for (const auto& kv : seen) shards.push_back(kv.first);
    }

    std::vector<std::string> names;
    for (const auto& shard : shards) {
        auto reader_result = open_shard(shard);
        if (!reader_result.ok()) return reader_result.status();
        auto list = reader_result.value()->list_tensors();
        if (!list.ok()) return list.status();
        for (auto& n : list.value()) names.push_back(std::move(n));
    }
    return names;
}

bool is_quantized_model_dir(const std::string& model_dir) {
    const fs::path dir(model_dir);
    if (fs::exists(dir / quantization::manifest::kQuantizedMetadataFile)) {
        return true;
    }
    const fs::path config_path = dir / "config.json";
    if (fs::exists(config_path)) {
        std::ifstream cf(config_path, std::ios::binary);
        if (cf) {
            nlohmann::json config;
            try {
                cf >> config;
            } catch (const nlohmann::json::exception&) {
                return false;
            }
            if (config.is_object() && config.contains("shelf_quantization")) {
                return true;
            }
        }
    }
    return quantization::bundle::is_hf_quantized_model_dir(model_dir);
}

Status validate_quantized_model(const QuantizedModelReader& reader) {
    using quantization::manifest::QuantizedManifest;
    using quantization::manifest::validate_quantized_manifest_contract;

    const QuantizedManifest& manifest = reader.manifest();

    Status contract = validate_quantized_manifest_contract(manifest);
    if (!contract.ok()) {
        return contract;
    }

    for (const auto& kv : manifest.aliases) {
        auto target = reader.resolve_alias(kv.first);
        if (!target.ok()) {
            return target.status();
        }
        if (manifest.tensors.find(target.value()) == manifest.tensors.end()) {
            return Status::invalid_argument(
                ("quantized manifest alias target missing: " + target.value()).c_str(),
                __FILE__, __LINE__);
        }
    }

    return Status::make_ok();
}

Result<Tensor> load_bf16_tensor(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream)
{
    auto spec_result = collection.tensor_spec(name);
    if (!spec_result.ok()) return spec_result.status();
    const auto& spec = spec_result.value();
    const auto shape = spec.shape;
    const auto strides = contiguous_strides(shape);

    std::size_t n = 1;
    for (auto d : shape) n *= d;
    Tensor t = alloc_tensor(shape, strides, arena);
    if (t.ndim() == 0) {
        return Status::insufficient_memory(
            ("tensor allocation failed: " + name).c_str(), __FILE__, __LINE__);
    }
    Status st = (spec.dtype == io::SType::F32)
        ? upload_tensor_f32_to_bf16(collection, name, t.data<void>(), n, stream)
        : upload_tensor_direct(collection, name, t.data<void>(), stream);
    if (!st.ok()) {
        return Status::invalid_state(
            ("tensor upload failed: " + name).c_str(), __FILE__, __LINE__);
    }
    return t;
}

Result<Tensor> load_bf16_or_f32_tensor(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream)
{
    auto spec_result = collection.tensor_spec(name);
    if (!spec_result.ok()) return spec_result.status();
    const auto& spec = spec_result.value();
    const auto shape = spec.shape;
    const auto strides = contiguous_strides(shape);

    Tensor t = (spec.dtype == io::SType::BF16 || spec.dtype == io::SType::F16)
        ? alloc_tensor(shape, strides, arena)
        : alloc_tensor_f32(shape, strides, arena);
    if (t.ndim() == 0) {
        return Status::insufficient_memory(
            ("tensor allocation failed: " + name).c_str(), __FILE__, __LINE__);
    }
    Status st = upload_tensor_direct(collection, name, t.data<void>(), stream);
    if (!st.ok()) {
        return Status::invalid_state(
            ("tensor upload failed: " + name).c_str(), __FILE__, __LINE__);
    }
    return t;
}

Result<MatrixWeight> load_bf16_matrix(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options)
{
    if (options.partition_plan == nullptr) {
        return load_bf16_matrix_raw(collection, name, arena, stream);
    }
    auto it = options.partition_plan->tensors.find(name);
    if (it == options.partition_plan->tensors.end()) {
        return load_bf16_matrix_raw(collection, name, arena, stream);
    }
    const shelf::weights::TensorPartitionDesc& partition = it->second;
    auto spec_result = collection.tensor_spec(name);
    if (!spec_result.ok()) return spec_result.status();
    const io::StTensorSpec& spec = spec_result.value();
    if (spec.shape.size() != 2) {
        return Status::invalid_argument(
            ("partitioned matrix weight must be 2D: " + name).c_str(), __FILE__, __LINE__);
    }
    auto local_result = shelf::weights::tensor_partition_local_shape(partition);
    if (!local_result.ok()) return local_result.status();
    const std::vector<int64_t>& local = local_result.value();

    std::size_t bytes = 0;
    auto data_result = collection.tensor_data(name, bytes);
    if (!data_result.ok()) return data_result.status();
    const void* src = data_result.value();

    const std::size_t rows = spec.shape[0];
    const std::size_t cols = spec.shape[1];
    constexpr std::size_t kElem = sizeof(bf16_t);
    if (bytes != rows * cols * kElem) {
        return Status::invalid_state(
            ("partitioned matrix payload size mismatch: " + name).c_str(), __FILE__,
            __LINE__);
    }

    MatrixWeight w;
    w.encoding = MatrixEncoding::Bf16;
    w.rows = static_cast<uint32_t>(local[0]);
    w.cols = static_cast<uint32_t>(local[1]);
    w.k_padded = w.cols;
    std::vector<std::size_t> strides = {
        static_cast<std::size_t>(local[1]), 1};
    w.bf16 = alloc_tensor(
        {static_cast<std::size_t>(local[0]), static_cast<std::size_t>(local[1])},
        strides, arena);
    if (w.bf16.ndim() == 0) {
        return Status::insufficient_memory(
            ("partitioned matrix allocation failed: " + name).c_str(), __FILE__, __LINE__);
    }
    w.partition = partition;

    auto* dst = static_cast<char*>(w.bf16.data<void>());
    const auto* src_base = static_cast<const char*>(src);
    if (partition.axis == 0) {
        std::size_t dst_row = 0;
        for (const shelf::weights::TensorPartitionRange& range : partition.ranges) {
            const char* row_src = src_base + range.global_offset * cols * kElem;
            const std::size_t copy_bytes = range.extent * cols * kElem;
            const hipError_t err = hipMemcpyAsync(
                dst + dst_row * cols * kElem, row_src, copy_bytes, hipMemcpyHostToDevice,
                stream);
            if (err != hipSuccess) {
                return Status::hip_error("hipMemcpyAsync partitioned matrix rows",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
            dst_row += range.extent;
        }
    } else if (partition.axis == 1) {
        std::size_t dst_col = 0;
        const std::size_t dst_pitch = static_cast<std::size_t>(local[1]) * kElem;
        for (const shelf::weights::TensorPartitionRange& range : partition.ranges) {
            const char* col_src = src_base + range.global_offset * kElem;
            const hipError_t err = hipMemcpy2DAsync(
                dst + dst_col * kElem, dst_pitch, col_src, cols * kElem,
                range.extent * kElem, rows, hipMemcpyHostToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error("hipMemcpy2DAsync partitioned matrix cols",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
            dst_col += range.extent;
        }
    } else {
        return Status::unsupported("partition axis for raw bf16 matrix must be 0 or 1",
                                   __FILE__, __LINE__);
    }
    return w;
}

Result<MatrixWeight> load_quantized_matrix(
    const QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options)
{
    return load_quantized_matrix_impl(reader, name, arena, stream, options);
}

Result<gpu::Tensor> load_quantized_small(
    const QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options)
{
    auto view_result = reader.resolve(name);
    if (!view_result.ok()) return view_result.status();

    OwnedCanonicalTensor partitioned_payload;
    QuantizedTensorView partitioned_view;
    const QuantizedTensorView* resolved = &view_result.value();
    if (options.partition_plan != nullptr) {
        auto it = options.partition_plan->tensors.find(resolved->name);
        if (it != options.partition_plan->tensors.end()) {
            Status partition_status = materialize_rank_local_canonical(
                it->second, *resolved, partitioned_payload, partitioned_view);
            if (!partition_status.ok()) return partition_status;
            resolved = &partitioned_view;
        }
    }
    const QuantizedTensorView& v = *resolved;

    if (v.encoding != QuantizedEncoding::Bf16) {
        return Status::invalid_argument(
            ("non-matrix tensor must be bf16: " + name).c_str(), __FILE__, __LINE__);
    }
    if (v.codes.data != nullptr || v.metadata1.data != nullptr) {
        return Status::invalid_argument(
            ("non-matrix tensor must not carry codes/scales: " + name).c_str(), __FILE__, __LINE__);
    }
    if (v.logical_shape.size() < 1 || v.logical_shape.size() > 3) {
        return Status::invalid_argument(
            ("non-matrix tensor ndim unsupported: " + name).c_str(), __FILE__, __LINE__);
    }

    std::size_t n = 1;
    for (auto d : v.logical_shape) {
        if (d <= 0) {
            return Status::invalid_argument(
                ("non-matrix tensor shape invalid: " + name).c_str(), __FILE__, __LINE__);
        }
        n *= static_cast<std::size_t>(d);
    }
    auto st = expect_bytes(name + " data", v.data, n * 2);
    if (!st.ok()) return st;

    std::vector<std::size_t> shape;
    shape.reserve(v.logical_shape.size());
    for (auto d : v.logical_shape) shape.push_back(static_cast<std::size_t>(d));
    std::vector<std::size_t> strides = contiguous_strides(shape);

    gpu::Tensor t = alloc_bytes16(n * 2, shape, strides, arena);
    if (t.ndim() == 0) {
        return Status::insufficient_memory(
            ("non-matrix tensor allocation failed: " + name).c_str(), __FILE__, __LINE__);
    }
    st = copy_to_device(t.data<void>(), v.data.data, v.data.size, stream);
    if (!st.ok()) return st;
    return t;
}

namespace {

Result<const std::uint8_t*> slice_payload_pointer(
    const SafetensorsCollection& collection,
    const std::string& name,
    const io::StAxisSpan& span)
{
    std::size_t total_bytes = 0;
    auto data_result = collection.tensor_data(name, total_bytes);
    if (!data_result.ok()) return data_result.status();
    auto base_result = collection.tensor_span(name);
    if (!base_result.ok()) return base_result.status();
    const std::uint64_t base_offset = base_result.value().absolute_offset;
    const std::uint64_t total = static_cast<std::uint64_t>(total_bytes);
    if (span.absolute_offset < base_offset || span.byte_length > total ||
        span.absolute_offset - base_offset > total - span.byte_length) {
        return Status::invalid_argument(
            ("axis slice payload out of range: " + name).c_str(), __FILE__, __LINE__);
    }
    return static_cast<const std::uint8_t*>(data_result.value()) +
           (span.absolute_offset - base_offset);
}

}

Result<MatrixWeight> load_bf16_matrix_axis_slice(
    const SafetensorsCollection& collection,
    const std::string& name,
    std::uint64_t unit_begin,
    std::uint64_t unit_count,
    gpu::GpuArena& arena,
    hipStream_t stream)
{
    auto spec_result = collection.tensor_spec(name);
    if (!spec_result.ok()) return spec_result.status();
    const io::StTensorSpec& spec = spec_result.value();
    if (spec.dtype != io::SType::BF16 || spec.shape.size() != 3) {
        return Status::invalid_argument(
            ("bf16 axis slice requires a rank-3 BF16 tensor: " + name).c_str(),
            __FILE__, __LINE__);
    }
    const std::uint64_t experts = spec.shape[0];
    if (unit_count == 0 || unit_begin >= experts || unit_count > experts - unit_begin) {
        return Status::invalid_argument(
            ("bf16 axis slice range out of bounds: " + name).c_str(), __FILE__, __LINE__);
    }
    const std::uint64_t rows = unit_count * spec.shape[1];
    const std::uint64_t cols = spec.shape[2];
    if (rows > UINT32_MAX || cols > UINT32_MAX) {
        return Status::invalid_argument(
            ("bf16 axis slice geometry too large: " + name).c_str(), __FILE__, __LINE__);
    }

    auto span_result = collection.axis_span(name, unit_begin, unit_count);
    if (!span_result.ok()) return span_result.status();
    const io::StAxisSpan& span = span_result.value();
    if (span.unit_bytes != spec.shape[1] * cols * sizeof(std::uint16_t) ||
        span.byte_length != rows * cols * sizeof(std::uint16_t)) {
        return Status::invalid_argument(
            ("bf16 axis slice byte count mismatch: " + name).c_str(), __FILE__, __LINE__);
    }
    auto src_result = slice_payload_pointer(collection, name, span);
    if (!src_result.ok()) return src_result.status();

    MatrixWeight w;
    w.encoding = MatrixEncoding::Bf16;
    w.rows = static_cast<uint32_t>(rows);
    w.cols = static_cast<uint32_t>(cols);
    w.k_padded = static_cast<uint32_t>(cols);
    std::vector<std::size_t> shape = {
        static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)};
    std::vector<std::size_t> strides = {static_cast<std::size_t>(cols), 1};
    w.bf16 = alloc_tensor(shape, strides, arena);
    if (w.bf16.ndim() == 0) {
        return Status::insufficient_memory(
            ("bf16 axis slice allocation failed: " + name).c_str(), __FILE__, __LINE__);
    }
    Status st = copy_to_device(w.bf16.data<void>(), src_result.value(),
                              static_cast<std::size_t>(span.byte_length), stream);
    if (!st.ok()) return st;
    apply_default_compute_spec(w);
    return w;
}

Result<MatrixWeight> load_fp8_matrix_axis_slice(
    const SafetensorsCollection& collection,
    const std::string& codes_name,
    const std::string& scale_name,
    std::uint64_t unit_begin,
    std::uint64_t unit_count,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options)
{
    auto codes_spec_result = collection.tensor_spec(codes_name);
    if (!codes_spec_result.ok()) return codes_spec_result.status();
    const io::StTensorSpec& codes_spec = codes_spec_result.value();
    if ((codes_spec.dtype != io::SType::F8_E4M3 && codes_spec.dtype != io::SType::U8) ||
        codes_spec.shape.size() != 3) {
        return Status::invalid_argument(
            ("fp8 axis slice requires a rank-3 F8_E4M3 codes tensor: " + codes_name).c_str(),
            __FILE__, __LINE__);
    }
    const std::uint64_t experts = codes_spec.shape[0];
    const std::uint64_t n = codes_spec.shape[1];
    const std::uint64_t k = codes_spec.shape[2];
    if (n == 0 || k == 0 || (k % quantization::kFp8ScaleBlockK) != 0u) {
        return Status::invalid_argument(
            ("fp8 axis slice K is not a multiple of 128: " + codes_name).c_str(),
            __FILE__, __LINE__);
    }
    if (unit_count == 0 || unit_begin >= experts || unit_count > experts - unit_begin) {
        return Status::invalid_argument(
            ("fp8 axis slice range out of bounds: " + codes_name).c_str(),
            __FILE__, __LINE__);
    }

    auto scale_spec_result = collection.tensor_spec(scale_name);
    if (!scale_spec_result.ok()) return scale_spec_result.status();
    const io::StTensorSpec& scale_spec = scale_spec_result.value();
    const std::uint64_t scale_n =
        (n + quantization::kFp8ScaleBlockN - 1u) / quantization::kFp8ScaleBlockN;
    const std::uint64_t scale_k = k / quantization::kFp8ScaleBlockK;
    if (scale_spec.dtype != io::SType::BF16 || scale_spec.shape.size() != 3 ||
        scale_spec.shape[0] != experts || scale_spec.shape[1] != scale_n ||
        scale_spec.shape[2] != scale_k) {
        return Status::invalid_argument(
            ("fp8 axis slice scale grid mismatch: " + scale_name).c_str(),
            __FILE__, __LINE__);
    }

    auto codes_span_result = collection.axis_span(codes_name, unit_begin, unit_count);
    if (!codes_span_result.ok()) return codes_span_result.status();
    const io::StAxisSpan& codes_span = codes_span_result.value();
    if (codes_span.byte_length != unit_count * n * k) {
        return Status::invalid_argument(
            ("fp8 axis slice byte count mismatch: " + codes_name).c_str(),
            __FILE__, __LINE__);
    }
    auto scale_span_result = collection.axis_span(scale_name, unit_begin, unit_count);
    if (!scale_span_result.ok()) return scale_span_result.status();
    const io::StAxisSpan& scale_span = scale_span_result.value();
    if (scale_span.byte_length != unit_count * scale_n * scale_k * sizeof(std::uint16_t)) {
        return Status::invalid_argument(
            ("fp8 axis slice scale byte count mismatch: " + scale_name).c_str(),
            __FILE__, __LINE__);
    }

    auto codes_src = slice_payload_pointer(collection, codes_name, codes_span);
    if (!codes_src.ok()) return codes_src.status();
    auto scale_src = slice_payload_pointer(collection, scale_name, scale_span);
    if (!scale_src.ok()) return scale_src.status();

    const std::size_t scale_count = static_cast<std::size_t>(unit_count * scale_n * scale_k);
    std::vector<float> scale_staging(scale_count);
    const auto* scale_bf16 = reinterpret_cast<const std::uint16_t*>(scale_src.value());
    for (std::size_t i = 0; i < scale_count; ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(scale_bf16[i]) << 16;
        std::memcpy(&scale_staging[i], &bits, sizeof(bits));
    }

    QuantizedTensorView v;
    v.name = codes_name;
    v.encoding = QuantizedEncoding::Fp8;
    v.logical_shape = {static_cast<int64_t>(unit_count * n), static_cast<int64_t>(k)};
    v.k_padded = static_cast<int64_t>(k);
    v.scale_block_n = quantization::kFp8ScaleBlockN;
    v.scale_block_k = quantization::kFp8ScaleBlockK;
    v.codes = quantization::bundle::ByteSpan{
        codes_src.value(), static_cast<std::size_t>(codes_span.byte_length)};
    v.metadata1 = quantization::bundle::ByteSpan{
        scale_staging.data(), scale_staging.size() * sizeof(float)};

    Result<MatrixWeight> loaded = load_quantized_view(v, arena, stream, options);
    if (!loaded.ok()) return loaded.status();
    if (!options.preshuffle) {
        const hipError_t synced = hipStreamSynchronize(stream);
        if (synced != hipSuccess) {
            return Status::hip_error("hipStreamSynchronize", hipGetErrorString(synced),
                                     __FILE__, __LINE__);
        }
    }
    return loaded;
}

}  // namespace shelf::weights
