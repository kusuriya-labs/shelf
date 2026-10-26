// shelf unit test — IR graph validation (PR-005).
#include <cstdio>
#include <vector>

#include <shelf/ir/graph.hpp>

namespace {

using namespace shelf;
using namespace shelf::ir;

int failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

ValueDesc vd(std::uint32_t id, std::initializer_list<std::size_t> dims,
             DataType dt = DataType::f32) {
    return ValueDesc{ValueId::from_uint32(id), TensorShape::create(dims).value(), dt};
}

// Fully valid small graph: embedding -> linear.
void test_valid_graph() {
    Graph g;
    check(g.declare_input(vd(0, {1}, DataType::i32)).ok(), "declare tokens input");
    check(g.declare_weight(TensorId::parse(1).value(), TensorShape::create({8, 4}).value()).ok(),
          "declare table");
    check(g.declare_weight(TensorId::parse(2).value(), TensorShape::create({6, 4}).value()).ok(),
          "declare proj");

    Primitive emb;
    emb.kind = Kind::embedding;
    emb.inputs = {ValueId::from_uint32(0)};
    emb.weights = {TensorId::parse(1).value()};
    emb.output = vd(10, {1, 4});
    check(g.add_primitive(emb).ok(), "embedding added");

    Primitive lin;
    lin.kind = Kind::linear;
    lin.inputs = {ValueId::from_uint32(10)};
    lin.weights = {TensorId::parse(2).value()};
    lin.output = vd(11, {1, 6});
    check(g.add_primitive(lin).ok(), "linear added");
    check(g.validate().ok(), "graph validate");
    check(g.size() == 2, "two primitives");
}

void test_rejections() {
    {   // undefined input
        Graph g;
        Primitive lin;
        lin.kind = Kind::linear;
        lin.inputs = {ValueId::from_uint32(99)};
        lin.weights = {TensorId::parse(2).value()};
        lin.output = vd(11, {1, 6});
        check(g.add_primitive(lin).code() == Status::Code::invalid_argument,
              "undefined input rejected");
        check(g.size() == 0, "failed add leaves graph untouched");
    }
    {   // duplicate ValueId (input vs input / input vs output)
        Graph g;
        check(g.declare_input(vd(0, {1}, DataType::i32)).ok(), "input 0");
        check(g.declare_input(vd(0, {1}, DataType::i32)).code() == Status::Code::invalid_argument,
              "duplicate input id rejected");
        Primitive emb;
        emb.kind = Kind::embedding;
        emb.inputs = {ValueId::from_uint32(0)};
        emb.weights = {TensorId::parse(1).value()};
        emb.output = vd(10, {1, 4});
        g.declare_weight(TensorId::parse(1).value(), TensorShape::create({8, 4}).value());
        check(g.add_primitive(emb).ok(), "emb ok");
        check(g.add_primitive(emb).code() == Status::Code::invalid_argument,
              "duplicate output id rejected");
    }
    {   // linear K mismatch: input [1,4] vs weight [6,9]
        Graph g;
        g.declare_input(vd(0, {1}, DataType::i32));
        g.declare_weight(TensorId::parse(1).value(), TensorShape::create({8, 4}).value());
        g.declare_weight(TensorId::parse(2).value(), TensorShape::create({6, 9}).value());
        Primitive emb;
        emb.kind = Kind::embedding;
        emb.inputs = {ValueId::from_uint32(0)};
        emb.weights = {TensorId::parse(1).value()};
        emb.output = vd(10, {1, 4});
        g.add_primitive(emb);
        Primitive lin;
        lin.kind = Kind::linear;
        lin.inputs = {ValueId::from_uint32(10)};
        lin.weights = {TensorId::parse(2).value()};
        lin.output = vd(11, {1, 6});
        check(g.add_primitive(lin).code() == Status::Code::invalid_argument,
              "linear K mismatch rejected");
    }
    {   // embedding tokens must be i32 rank1
        Graph g;
        g.declare_input(vd(0, {1}, DataType::f32));
        g.declare_weight(TensorId::parse(1).value(), TensorShape::create({8, 4}).value());
        Primitive emb;
        emb.kind = Kind::embedding;
        emb.inputs = {ValueId::from_uint32(0)};
        emb.weights = {TensorId::parse(1).value()};
        emb.output = vd(10, {1, 4});
        check(g.add_primitive(emb).code() == Status::Code::invalid_argument,
              "embedding f32 tokens rejected");
    }
    {   // attention needs declared KV state; rope needs matching cos/sin
        Graph g;
        Primitive att;
        att.kind = Kind::attention_decode;
        att.inputs = {ValueId::from_uint32(1), ValueId::from_uint32(2), ValueId::from_uint32(3)};
        att.state_read = {StateId::from_uint32(0)};
        att.state_write = {StateId::from_uint32(0)};
        att.output = vd(4, {4});
        check(g.add_primitive(att).code() == Status::Code::invalid_argument,
              "undeclared state rejected");

        Graph g2;
        g2.declare_input(vd(1, {4}));
        g2.declare_input(vd(2, {4}));
        g2.declare_input(vd(3, {4}));
        g2.declare_state(StateDesc{StateId::from_uint32(0), StateKind::kv_cache,
                                    TensorShape::create({4, 4}).value(), DataType::f32});
        check(g2.add_primitive(att).ok(), "attention valid");
        check(g2.declare_state(StateDesc{StateId::from_uint32(0), StateKind::kv_cache,
                                          TensorShape::create({4, 4}).value(), DataType::f32})
                  .code() == Status::Code::invalid_argument,
              "duplicate state rejected");

        Graph g3;
        g3.declare_input(vd(5, {1, 4}));
        g3.declare_input(vd(6, {1, 3}));  // wrong cos width
        g3.declare_input(vd(7, {1, 4}));
        Primitive rope;
        rope.kind = Kind::rope;
        rope.inputs = {ValueId::from_uint32(5), ValueId::from_uint32(6), ValueId::from_uint32(7)};
        rope.output = vd(8, {1, 4});
        check(g3.add_primitive(rope).code() == Status::Code::invalid_argument,
              "rope cos/sin shape mismatch rejected");
    }
}

} // namespace

int main() {
    test_valid_graph();
    test_rejections();
    std::printf("test_ir_graph_validation: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
