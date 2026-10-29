// shelf unit test — foundation contracts (CPU-only, no external framework).
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <shelf/foundation/dtype.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/foundation/shape.hpp>
#include <shelf/foundation/tensor_id.hpp>
#include <shelf/foundation/tensor_spec.hpp>

namespace {

int failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

void test_status_result() {
    auto ok = shelf::Status::make_ok();
    check(ok.ok(), "make_ok");

    auto err = shelf::Status::invalid_argument("boom", __FILE__, __LINE__);
    check(!err.ok(), "invalid_argument not ok");
    check(err.code() == shelf::Status::Code::invalid_argument, "code matches");
    check(err.message().find("boom") != std::string::npos, "message contains text");

    shelf::Result<int> good{42};
    check(good.ok() && good.value() == 42, "Result value path");

    shelf::Result<int> bad{shelf::Status::unsupported("nope", __FILE__, __LINE__)};
    check(!bad.ok(), "Result error path");
    check(bad.status().code() == shelf::Status::Code::unsupported, "Result error code");
}

void test_checked_arithmetic() {
    std::size_t out = 0;
    check(shelf::checked_mul(std::size_t(1) << 40, std::size_t(1) << 40, out, __FILE__, __LINE__)
              .code() == shelf::Status::Code::overflow,
          "checked_mul overflow");
    check(shelf::checked_mul(7, 6, out, __FILE__, __LINE__).ok() && out == 42, "checked_mul ok");
    check(shelf::checked_add(std::size_t(-1), 1, __FILE__, __LINE__).code() ==
              shelf::Status::Code::overflow,
          "checked_add overflow");
    check(shelf::checked_align_up(5, 8, out, __FILE__, __LINE__).ok() && out == 8, "align_up 5->8");
    check(shelf::checked_align_up(8, 3, out, __FILE__, __LINE__).code() ==
              shelf::Status::Code::invalid_argument,
          "align_up non power-of-two rejected");
}

void test_shape_validation() {
    auto good = shelf::TensorShape::create({2, 3, 4});
    check(good.ok() && good.value().rank() == 3, "shape create ok");
    check(good.value().dim(1) == 3, "shape dim access");
    auto n = good.value().num_elements();
    check(n.ok() && n.value() == 24, "num_elements 24");

    check(shelf::TensorShape::create({}).code() == shelf::Status::Code::out_of_range,
          "rank 0 rejected");
    check(shelf::TensorShape::create({1, 1, 1, 1, 1, 1, 1, 1, 1}).code() ==
              shelf::Status::Code::out_of_range,
          "rank 9 rejected");
    check(shelf::TensorShape::create({4, 0}).code() == shelf::Status::Code::invalid_argument,
          "zero dimension rejected");

    // overflow: 2^40 * 2^40 does not fit in 64-bit element count
    auto big = shelf::TensorShape::create({std::size_t(1) << 40, std::size_t(1) << 40});
    check(big.ok(), "shape of huge dims constructible");
    check(big.value().num_elements().code() == shelf::Status::Code::overflow,
          "num_elements overflow detected");

    check(good.value() == shelf::TensorShape::create({2, 3, 4}).value(), "shape equality");
}

void test_dtype() {
    using shelf::DataType;
    check(shelf::element_size(DataType::bf16) == 2, "bf16 size");
    check(shelf::element_size(DataType::f32) == 4, "f32 size");
    check(shelf::element_size(DataType::fp8_e4m3) == 1, "fp8_e4m3 size 1");
    check(shelf::element_size(DataType::unknown) == 0, "unknown size 0");
    check(shelf::is_floating_point(DataType::fp16), "fp16 is float");
    check(shelf::is_floating_point(DataType::fp8_e4m3), "fp8_e4m3 is float");
    check(!shelf::is_floating_point(DataType::i32), "i32 not float");
    check(!shelf::is_floating_point(DataType::u8), "u8 not float");
}

void test_numeric_conversions() {
    using namespace shelf;
    check(f32_to_bf16_rne(1.0f) == 0x3F80u, "f32->bf16 bits for 1.0");
    check(bf16_to_f32(bf16_t{0x3F80u}) == 1.0f, "bf16->f32 for 1.0");
    // 1.5 rounds exactly; NaN gets quieted; inf preserved
    check(bf16_to_f32(f32_to_bf16(1.5f)) == 1.5f, "bf16 round-trip 1.5");
    auto nan = f32_to_bf16(std::numeric_limits<float>::quiet_NaN());
    check((nan.data & 0x7F80u) == 0x7F80u && (nan.data & 0x007Fu) != 0u, "bf16 NaN quieted");
    check(f32_to_bf16_rne(2.0f) == 0x4000u, "f32->bf16 bits for 2.0");
    // round-to-nearest-even: bf16 has 7 explicit mantissa bits, so ulp(1.0)=2^-7.
    // Exact midpoint (1+2^-8) must tie down to even 0x3F80; just above rounds to 0x3F81.
    check(f32_to_bf16_rne(1.00390625f) == 0x3F80u, "RNE ties to even");
    check(f32_to_bf16_rne(1.0f + 0.00390625f + 0.00048828125f) == 0x3F81u,
          "RNE rounds up past midpoint");
    check(f32_to_bf16_rne(1.001953125f) == 0x3F80u, "below midpoint rounds down");
    fp16_t h = f32_to_fp16(2.0f);
    check(fp16_to_f32(h) == 2.0f, "fp16 round-trip");
}

void test_tensor_id() {
    using shelf::TensorId;
    check(!TensorId{}.valid(), "default TensorId invalid");
    auto parsed = TensorId::parse(TensorId::kInvalidValue);
    check(!parsed.ok(), "reserved value rejected");
    auto a = TensorId::from_uint32(7);
    auto b = TensorId::from_uint32(7);
    auto c = TensorId::from_uint32(8);
    check(a.valid() && a == b && a < c, "TensorId comparison");
}

void test_tensor_spec() {
    using namespace shelf;
    TensorSpec spec{TensorId::from_uint32(1), TensorShape::create({4, 4}).release(),
                    DataType::bf16, TensorRole::weight};
    check(spec.validate().ok(), "valid spec");

    TensorSpec bad_id{TensorId{}, spec.shape, DataType::bf16, TensorRole::weight};
    check(bad_id.validate().code() == Status::Code::invalid_argument, "invalid id detected");

    TensorSpec bad_dtype{TensorId::from_uint32(2), spec.shape, DataType::unknown, TensorRole::state};
    check(bad_dtype.validate().code() == Status::Code::invalid_argument, "unknown dtype detected");

    std::vector<TensorSpec> catalog = {
        spec,
        TensorSpec{TensorId::from_uint32(2), TensorShape::create({16}).release(), DataType::f32,
                   TensorRole::activation},
    };
    check(validate_unique_ids(catalog).ok(), "unique catalog");

    catalog.push_back(TensorSpec{TensorId::from_uint32(1), TensorShape::create({8}).release(),
                                 DataType::f32, TensorRole::workspace});
    check(validate_unique_ids(catalog).code() == Status::Code::invalid_state,
          "duplicate TensorId detected");
}

} // namespace

int main() {
    test_status_result();
    test_checked_arithmetic();
    test_shape_validation();
    test_dtype();
    test_numeric_conversions();
    test_tensor_id();
    test_tensor_spec();

    std::printf("test_foundation: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
