#include <shelf/io/safetensors_reader.h>
#include <shelf/io/safetensors_writer.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/wait.h>

int main() {
    auto path = std::string("/tmp/shelf_population_") + std::to_string(getpid());
    std::array<uint8_t, 8192> expected{};
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = static_cast<uint8_t>(i * 17);
    auto writer = shelf::io::SafetensorsWriter::create(path);
    if (!writer.ok() || !writer.value().plan_tensor("x", shelf::io::SType::U8, {expected.size()}).ok() ||
        !writer.value().write_header().ok() ||
        !writer.value().write_tensor("x", expected.data(), expected.size()).ok() || !writer.value().finish().ok())
        return 1;
    auto opened = shelf::io::SafetensorsReader::open(path);
    if (!opened.ok()) return 2;
    auto reader = opened.release();
    if (!reader.populate_read().ok()) return 3;
    pid_t child = fork();
    if (child < 0) return 7;
    if (child == 0) {
        rlimit limit{0, 0};
        if (setrlimit(RLIMIT_MEMLOCK, &limit) != 0) _exit(8);
        _exit(reader.lock_pages().ok() ? 9 : 0);
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) return 10;
    if (!reader.lock_pages().ok()) return 6;
    auto resident = reader.resident_pages();
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0 || !resident.ok() || resident.value() != (reader.file_size() + page - 1) / page) return 4;
    size_t length = 0;
    auto data = reader.tensor_data("x", length);
    if (!data.ok() || length != expected.size() || std::memcmp(data.value(), expected.data(), length)) return 5;
    std::remove(path.c_str());
    return 0;
}
