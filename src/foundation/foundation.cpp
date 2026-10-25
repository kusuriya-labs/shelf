// shelf foundation — translation unit so the static library has an object.
//
// All foundation contracts are header-only for now; this TU verifies the
// headers compile standalone and gives shelf_foundation a linkable object.
#include <shelf/foundation/dtype.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/foundation/shape.hpp>
#include <shelf/foundation/tensor_id.hpp>
#include <shelf/foundation/tensor_spec.hpp>

namespace shelf {
namespace foundation {

// Anchor symbol for the foundation module.
int module_version() noexcept { return 1; }

} // namespace foundation
} // namespace shelf
