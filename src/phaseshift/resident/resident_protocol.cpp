#include <phaseshift/resident/resident_protocol.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace ps {
namespace resident {
namespace {

constexpr std::uint32_t kKeyMagic = 0x314b5350u;

#pragma pack(push, 1)
struct KeyHeader {
    std::uint32_t magic = kKeyMagic;
    std::uint32_t kind = 0;
    std::uint32_t load_flags = 0;
    std::uint32_t tp_size = 0;
    std::uint32_t tp_rank = 0;
    std::int32_t device = 0;
    std::uint64_t fingerprint = 0;
    std::uint64_t dir_bytes = 0;
};
#pragma pack(pop)

static_assert(sizeof(KeyHeader) == 40, "KeyHeader layout");

std::string& error_buffer() {
    thread_local std::string buffer;
    return buffer;
}

Status errno_status(const char* what) {
    error_buffer() = std::string(what) + ": " + std::strerror(errno);
    return Status::invalid_state(error_buffer().c_str(), __FILE__, __LINE__);
}

}

Status write_bytes(int fd, const void* data, std::size_t size) {
    const auto* p = static_cast<const std::byte*>(data);
    std::size_t done = 0;
    while (done < size) {
        const ssize_t n = ::write(fd, p + done, size - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno_status("resident socket write");
        }
        if (n == 0) {
            return Status::invalid_state("resident socket write: short write", __FILE__, __LINE__);
        }
        done += static_cast<std::size_t>(n);
    }
    return Status::make_ok();
}

Status read_bytes(int fd, void* data, std::size_t size) {
    auto* p = static_cast<std::byte*>(data);
    std::size_t done = 0;
    while (done < size) {
        const ssize_t n = ::read(fd, p + done, size - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno_status("resident socket read");
        }
        if (n == 0) {
            return Status::invalid_state("resident socket read: peer closed", __FILE__, __LINE__);
        }
        done += static_cast<std::size_t>(n);
    }
    return Status::make_ok();
}

Status write_frame(int fd, Command cmd, StatusCode status, const void* payload,
                   std::size_t size) {
    WireHeader header;
    header.cmd = static_cast<std::uint32_t>(cmd);
    header.status = static_cast<std::uint32_t>(status);
    header.payload_size = size;
    Status st = write_bytes(fd, &header, sizeof(header));
    if (!st.ok()) return st;
    if (size == 0) return Status::make_ok();
    return write_bytes(fd, payload, size);
}

Result<WireHeader> read_frame(int fd, std::vector<std::byte>& payload) {
    payload.clear();
    WireHeader header;
    Status st = read_bytes(fd, &header, sizeof(header));
    if (!st.ok()) return st;
    if (header.magic != kWireMagic) {
        return Status::invalid_state("resident protocol magic", __FILE__, __LINE__);
    }
    if (header.version != kWireVersion) {
        return Status::unsupported("resident protocol version", __FILE__, __LINE__);
    }
    if (header.payload_size > kMaxFrameBytes) {
        return Status::invalid_state("resident frame too large", __FILE__, __LINE__);
    }
    if (header.payload_size > 0) {
        payload.resize(static_cast<std::size_t>(header.payload_size));
        st = read_bytes(fd, payload.data(), payload.size());
        if (!st.ok()) return st;
    }
    return header;
}

std::vector<std::byte> serialize_key(const ResidentModelKey& key) {
    KeyHeader header;
    header.kind = static_cast<std::uint32_t>(key.kind);
    header.load_flags = key.load_flags;
    header.tp_size = key.tp_size;
    header.tp_rank = key.tp_rank;
    header.device = key.device;
    header.fingerprint = key.fingerprint;
    header.dir_bytes = key.model_dir.size();
    std::vector<std::byte> out(sizeof(header) + key.model_dir.size());
    std::memcpy(out.data(), &header, sizeof(header));
    std::memcpy(out.data() + sizeof(header), key.model_dir.data(), key.model_dir.size());
    return out;
}

Result<ResidentModelKey> parse_key(const std::byte* data, std::size_t size) {
    if (data == nullptr || size < sizeof(KeyHeader)) {
        return Status::invalid_argument("resident key payload", __FILE__, __LINE__);
    }
    KeyHeader header;
    std::memcpy(&header, data, sizeof(header));
    if (header.magic != kKeyMagic) {
        return Status::invalid_state("resident key magic", __FILE__, __LINE__);
    }
    if (header.dir_bytes > 4096 ||
        size != sizeof(header) + static_cast<std::size_t>(header.dir_bytes)) {
        return Status::invalid_state("resident key size", __FILE__, __LINE__);
    }
    ResidentModelKey key;
    key.model_dir.assign(
        reinterpret_cast<const char*>(data + sizeof(header)),
        static_cast<std::size_t>(header.dir_bytes));
    key.kind = static_cast<ModelKind>(header.kind);
    key.load_flags = header.load_flags;
    key.tp_size = header.tp_size;
    key.tp_rank = header.tp_rank;
    key.device = header.device;
    key.fingerprint = header.fingerprint;
    if (key.tp_size == 0) {
        return Status::invalid_state("resident key tp_size", __FILE__, __LINE__);
    }
    if (key.fingerprint != fingerprint_key(key)) {
        return Status::invalid_state("resident key fingerprint", __FILE__, __LINE__);
    }
    return key;
}

std::string default_socket_path() {
    const char* dir = std::getenv("TMPDIR");
    std::string base = (dir != nullptr && dir[0] != '\0') ? dir : "/tmp";
    return base + "/phaseshift-resident-" + std::to_string(::getpid()) + ".sock";
}

}
}
