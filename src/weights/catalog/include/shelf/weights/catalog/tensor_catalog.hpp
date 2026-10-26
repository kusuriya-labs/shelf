// shelf weights/catalog — Canonical Tensor Catalog (plan §4.2 path).
//
//   Raw Tensor Catalog (per shard/container) -> Canonical Tensor Catalog
//
// Resolves logical names to stable TensorIds across shards, validates specs,
// and rejects duplicate identities. Binding/materialization come in later
// phases; this catalog never touches execution or devices.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <shelf/artifact/common/raw_catalog.hpp>
#include <shelf/foundation/tensor_spec.hpp>
#include <shelf/status.hpp>

namespace shelf::weights {

class TensorCatalog {
public:
    static Result<TensorCatalog> from_raw(const std::vector<artifact::RawTensorEntry>& raw) {
        TensorCatalog cat;
        for (const auto& entry : raw) {
            SHELF_RETURN_IF_ERROR(cat.add_raw(entry));
        }
        SHELF_RETURN_IF_ERROR(validate_unique_ids(cat.specs_));
        return cat;
    }

    std::size_t size() const noexcept { return specs_.size(); }
    const std::vector<TensorSpec>& specs() const noexcept { return specs_; }

    const TensorSpec* find(const std::string& name) const {
        auto it = by_name_.find(name);
        return it == by_name_.end() ? nullptr : &specs_[it->second];
    }

    TensorId id_of(const std::string& name) const {
        const TensorSpec* spec = find(name);
        return spec ? spec->id : TensorId{};
    }

private:
    Status add_raw(const artifact::RawTensorEntry& entry) {
        SHELF_RETURN_IF_ERROR(artifact::validate_entry(entry));
        if (by_name_.count(entry.name)) {
            return Status::invalid_state("duplicate tensor name across shards: " + entry.name,
                                         __FILE__, __LINE__);
        }
        auto next = TensorId::parse(static_cast<std::uint32_t>(specs_.size()));
        if (!next.ok()) return next.status();  // catalog overflow (reserved id reached)

        TensorSpec spec;
        spec.id = next.value();
        spec.shape = entry.shape;
        spec.logical_dtype = entry.dtype;
        spec.role = TensorRole::weight;
        SHELF_RETURN_IF_ERROR(spec.validate());

        by_name_.emplace(entry.name, specs_.size());
        specs_.push_back(std::move(spec));
        return Status::make_ok();
    }

    std::vector<TensorSpec> specs_;
    std::unordered_map<std::string, std::size_t> by_name_;
};

} // namespace shelf::weights
