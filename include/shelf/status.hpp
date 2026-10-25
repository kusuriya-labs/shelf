// shelf — public facade: Status / Result / checked arithmetic.
//
// Backend-specific device error codes belong to the backend that can
// produce them, not here.
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

namespace shelf {

class Status {
public:
    enum class Code {
        ok,
        invalid_argument,
        out_of_range,
        overflow,
        insufficient_memory,
        invalid_state,
        not_found,
        unsupported,
        device_error,
    };

    static Status make_ok();
    static Status invalid_argument(const char* msg, const char* file, int line);
    static Status out_of_range(const char* msg, const char* file, int line);
    static Status overflow(const char* msg, const char* file, int line);
    static Status insufficient_memory(const char* msg, const char* file, int line);
    static Status invalid_state(const char* msg, const char* file, int line);
    static Status not_found(const char* msg, const char* file, int line);
    static Status unsupported(const char* msg, const char* file, int line);
    // Backend-neutral device API failure. Message must be produced by the
    // backend that can interpret the native error code (HIP/SYCL/etc.).
    static Status device_error(const char* expr, const char* msg, const char* file, int line);

    bool ok() const noexcept { return code_ == Code::ok; }
    Code code() const noexcept { return code_; }
    const std::string& message() const noexcept { return message_; }

    explicit operator bool() const noexcept { return ok(); }

private:
    Status(Code code, std::string msg)
        : code_(code), message_(std::move(msg)) {}

    Code code_;
    std::string message_;
};

template <typename T>
class Result {
public:
    Result(T value)
        : status_(Status::make_ok()),
          value_(std::move(value)) {}

    Result(Status status)
        : status_(std::move(status)),
          value_(std::nullopt) {
        assert(!status_.ok() && "cannot construct error Result from ok Status");
    }

    bool ok() const noexcept { return value_.has_value(); }
    const Status& status() const noexcept { return status_; }
    Status::Code code() const noexcept { return status_.code(); }

    const T& value() const {
        assert(value_.has_value());
        return *value_;
    }

    T& value() {
        assert(value_.has_value());
        return *value_;
    }

    T release() {
        assert(value_.has_value());
        T result = std::move(*value_);
        value_.reset();
        return result;
    }
    // DO NOT reuse this Result after calling release().

private:
    Status status_;
    std::optional<T> value_;
};

inline Status Status::make_ok() {
    return Status{Code::ok, ""};
}

inline Status Status::invalid_argument(const char* msg, const char* file, int line) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%d: invalid argument: %s", file, line, msg);
    return Status{Code::invalid_argument, std::string(buf)};
}

inline Status Status::out_of_range(const char* msg, const char* file, int line) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%d: out of range: %s", file, line, msg);
    return Status{Code::out_of_range, std::string(buf)};
}

inline Status Status::overflow(const char* msg, const char* file, int line) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%d: overflow: %s", file, line, msg);
    return Status{Code::overflow, std::string(buf)};
}

inline Status Status::insufficient_memory(const char* msg, const char* file, int line) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%d: insufficient memory: %s", file, line, msg);
    return Status{Code::insufficient_memory, std::string(buf)};
}

inline Status Status::invalid_state(const char* msg, const char* file, int line) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%d: invalid state: %s", file, line, msg);
    return Status{Code::invalid_state, std::string(buf)};
}

inline Status Status::not_found(const char* msg, const char* file, int line) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%d: not found: %s", file, line, msg);
    return Status{Code::not_found, std::string(buf)};
}

inline Status Status::unsupported(const char* msg, const char* file, int line) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%d: unsupported: %s", file, line, msg);
    return Status{Code::unsupported, std::string(buf)};
}

inline Status Status::device_error(const char* expr, const char* msg, const char* file, int line) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%d: device error (%s): %s", file, line, expr, msg);
    return Status{Code::device_error, std::string(buf)};
}

// ---- checked arithmetic (overflow-detecting helpers) ----

inline Status checked_add(std::size_t a, std::size_t b, const char* file, int line) {
    if (a > std::size_t(-1) - b) {
        return Status::overflow("checked_add overflow", file, line);
    }
    return Status::make_ok();
}

inline Status checked_mul(std::size_t a, std::size_t b, std::size_t& out, const char* file, int line) {
    if (b != 0 && a > std::size_t(-1) / b) {
        return Status::overflow("checked_mul overflow", file, line);
    }
    out = a * b;
    return Status::make_ok();
}

inline Status checked_align_up(std::size_t value, std::size_t alignment, std::size_t& out, const char* file, int line) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return Status::invalid_argument("alignment must be power of 2", file, line);
    }
    std::size_t mask = alignment - 1;
    if (value > std::size_t(-1) - mask) {
        return Status::overflow("checked_align_up overflow", file, line);
    }
    out = (value + mask) & ~mask;
    return Status::make_ok();
}

// ---- convenience macros ----

#define SHELF_RETURN_IF_ERROR(status_expr)                     \
    do {                                                       \
        ::shelf::Status _shelf_st = (status_expr);             \
        if (!_shelf_st.ok()) return _shelf_st;                 \
    } while (false)

#define SHELF_TRY(name, expr)                                  \
    auto&& _shelf_res_##name = (expr);                         \
    if (!_shelf_res_##name.ok()) return _shelf_res_##name.status(); \
    auto& name = _shelf_res_##name.value()

} // namespace shelf
