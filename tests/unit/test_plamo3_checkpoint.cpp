#include <shelf/models/model_weights_factory.h>
#include <shelf/models/plamo3/model/config.h>
#include <shelf/models/plamo3/weights/model_weights.h>
#include <shelf/weights/weight_loader.h>
#include <cstdio>
#include <cstdlib>

int main() {
    const char* dir = std::getenv("SHELF_PLAMO3_MODEL_DIR");
    if (!dir) { std::puts("SKIP: SHELF_PLAMO3_MODEL_DIR is not set"); return 77; }
    auto family = shelf::models::probe_model_dir(dir);
    auto config = shelf::plamo3::read_config(dir);
    auto collection = shelf::weights::SafetensorsCollection::open(dir);
    if (!family.ok() || family.value() != shelf::models::ModelFamily::Plamo3 ||
        !config.ok() || !collection.ok()) return 1;
    auto valid = shelf::plamo3::validate_checkpoint(collection.value(), config.value());
    if (!valid.ok()) { std::fprintf(stderr, "%s\n", valid.message().c_str()); return 1; }
    auto wrong = config.value();
    ++wrong.head_dim;
    if (shelf::plamo3::validate_checkpoint(collection.value(), wrong).ok()) {
        std::fprintf(stderr, "accepted incompatible attention shape\n"); return 1;
    }
    std::puts("PLaMo checkpoint names, BF16 dtypes, shapes, and family verified");
    return 0;
}
