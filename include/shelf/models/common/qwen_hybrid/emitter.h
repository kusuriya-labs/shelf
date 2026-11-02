#pragma once

#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/graph/primitive_node.h>
#include <shelf/weights/matrix_weight.h>

#include <vector>

namespace ps {
namespace qwen_hybrid {

namespace RT = ::ps::runtime;

struct Emitter {
    RT::PrimitiveGraph& pg;

    RT::ValueId value(uint32_t features,
                      RT::ValueDType dtype = RT::ValueDType::BF16,
                      RT::ValueRowDomain rows = RT::ValueRowDomain::TOKEN_ROWS) {
        return pg.alloc_value(features, dtype, rows);
    }

    void emit(const RT::PrimitiveNode& node, std::vector<RT::ValueId> ins,
              std::vector<RT::ValueId> outs,
              std::vector<RT::StateId> state_ins = {},
              std::vector<RT::StateId> state_outs = {},
              std::vector<RT::ValueId> keep_alive = {}) {
        auto& gn = pg.add_node(node);
        gn.inputs = std::move(ins);
        gn.outputs = std::move(outs);
        gn.state_inputs = std::move(state_ins);
        gn.state_outputs = std::move(state_outs);
        gn.keep_alive = std::move(keep_alive);
    }

    RT::ValueId linear(const ::ps::weights::MatrixWeight& w,
                       uint32_t weight_index, RT::ValueId x,
                       RT::ValueDType out_dtype = RT::ValueDType::BF16) {
        RT::LinearNode n;
        n.shape = RT::MatrixwiseShapeKey{w.rows, w.cols};
        n.weight_index = weight_index;
        n.out_dtype = out_dtype;
        RT::ValueId y = value(w.rows, out_dtype, pg.value_row_domain(x));
        emit(RT::PrimitiveNode{n}, {x}, {y});
        return y;
    }

    RT::ValueId unary(const RT::PrimitiveNode& node, RT::ValueId x,
                      uint32_t out_features,
                      RT::ValueDType dtype = RT::ValueDType::BF16) {
        RT::ValueId y = value(out_features, dtype);
        emit(node, {x}, {y});
        return y;
    }
};

}
}
