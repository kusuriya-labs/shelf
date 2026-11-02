#include <phaseshift/quantization/quant_spec.h>
#include <phaseshift/runtime/program/activation_workspace_layout.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>

#include <cstdint>
#include <cstdio>
#include <string>

namespace rt = ps::runtime;
namespace quant = ps::quantization;

namespace {

int passed = 0;
int failed = 0;

void check(bool condition, const std::string& message) {
    if (condition) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", message.c_str());
    }
}

void expect_identical(uint32_t k, uint32_t rows, const std::string& tag) {
    const rt::Int8ActivationWorkspaceLayout legacy =
        rt::Int8ActivationWorkspaceLayout::make(k, rows);
    rt::Int8ActivationWorkspaceLayout adapter;
    const ps::Status status =
        rt::make_activation_workspace_layout(quant::kE4m3ActivationQuantSpec, k, rows, adapter);
    check(status.ok(), tag + ": adapter resolves");
    if (!status.ok()) return;
    check(adapter.codes_offset == legacy.codes_offset, tag + ": codes_offset");
    check(adapter.scales_offset == legacy.scales_offset, tag + ": scales_offset");
    check(adapter.total_bytes == legacy.total_bytes, tag + ": total_bytes");
    check(adapter.k_padded == legacy.k_padded, tag + ": k_padded");
    check(adapter.max_rows == legacy.max_rows, tag + ": max_rows");
    check(adapter.code_row_stride_bytes == legacy.code_row_stride_bytes,
          tag + ": code_row_stride_bytes");
    check(adapter.scale_row_stride_bytes == legacy.scale_row_stride_bytes,
          tag + ": scale_row_stride_bytes");
    check(adapter.codes_bytes() == legacy.codes_bytes(), tag + ": codes_bytes");
    check(adapter.scales_bytes() == legacy.scales_bytes(), tag + ": scales_bytes");
}

void expect_unsupported(const quant::ActivationQuantSpec& spec, const std::string& tag) {
    rt::Int8ActivationWorkspaceLayout layout;
    const ps::Status status = rt::make_activation_workspace_layout(spec, 2560u, 128u, layout);
    check(!status.ok(), tag + ": rejected");
    if (status.ok()) return;
    check(status.code() == ps::Status::Code::unsupported ||
              status.code() == ps::Status::Code::invalid_argument,
          tag + ": explicit failure status");
}

}

int main() {
    expect_identical(32u, 1u, "k32 rows1");
    expect_identical(100u, 3u, "k100 rows3");
    expect_identical(2560u, 128u, "k2560 rows128");
    expect_identical(5120u, 4u, "k5120 rows4");
    expect_identical(17408u, 1u, "k17408 rows1");

    {
        rt::Int8ActivationWorkspaceLayout i8_layout;
        const ps::Status i8_status =
            rt::make_activation_workspace_layout(quant::kI8ActivationQuantSpec, 2560u, 8u,
                                                 i8_layout);
        check(i8_status.ok(), "i8 group-128 spec: accepted");
        if (i8_status.ok()) {
            check(i8_layout.scale_row_stride_bytes ==
                      rt::Int8ActivationWorkspaceLayout::make(2560u, 8u).scale_row_stride_bytes,
                  "i8 group-128 spec: stride matches layout");
        }
    }
    {
        quant::ActivationQuantSpec grouped = quant::kE4m3ActivationQuantSpec;
        grouped.group_k = 64u;
        expect_unsupported(grouped, "group-64 e4m3 spec");
    }
    {
        quant::ActivationQuantSpec bf16_scale = quant::kE4m3ActivationQuantSpec;
        bf16_scale.scale_dtype = quant::QuantScalarType::BF16;
        expect_unsupported(bf16_scale, "bf16 scale spec");
    }
    {
        quant::ActivationQuantSpec bf16_codes = quant::kE4m3ActivationQuantSpec;
        bf16_codes.codes_dtype = quant::QuantScalarType::BF16;
        expect_unsupported(bf16_codes, "bf16 codes spec");
    }
    {
        rt::Int8ActivationWorkspaceLayout layout;
        const ps::Status status = rt::make_activation_workspace_layout(
            quant::kE4m3ActivationQuantSpec, 2560u, 0u, layout);
        check(!status.ok(), "rows 0: rejected");
        check(status.code() == ps::Status::Code::invalid_argument,
              "rows 0: invalid argument status");
    }

    std::printf("test_activation_workspace_layout: passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
