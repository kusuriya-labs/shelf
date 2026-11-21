#include <shelf/io/token_ids.h>
#include <cstdio>
#include <cstring>

namespace shelf::io {
std::vector<int32_t> read_input_ids(const std::string &path) {
    std::vector<int32_t> ids;
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr)
        return ids;
    char magic[8]{};
    const std::size_t prefix_size = std::fread(magic, 1, sizeof(magic), f);
    if (prefix_size == sizeof(magic) && std::memcmp(magic, "PSKLDTOK", sizeof(magic)) == 0) {
        uint32_t version = 0;
        uint64_t count = 0;
        bool ok = std::fread(&version, sizeof(version), 1, f) == 1 && version == 1u &&
                  std::fread(&count, sizeof(count), 1, f) == 1 && count > 0 && count < (1u << 22);
        if (ok) {
            ids.resize(static_cast<std::size_t>(count));
            ok = std::fread(ids.data(), sizeof(int32_t), ids.size(), f) == ids.size();
        }
        std::fclose(f);
        if (!ok)
            ids.clear();
        return ids;
    }
    for (std::size_t i = prefix_size; i > 0; --i) {
        if (std::ungetc(static_cast<unsigned char>(magic[i - 1]), f) == EOF) {
            std::fclose(f);
            return {};
        }
    }
    long v = 0;
    while (std::fscanf(f, "%ld", &v) == 1) {
        ids.push_back(static_cast<int32_t>(v));
    }
    std::fclose(f);
    return ids;
}

}
