#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <string>
#include "cli.hpp"
#include "factory.hpp"
#include "tensor/tensor.hpp"
#include "cgraph.hpp"
#include "nodes/node.hpp"
#include "nodes/operations.hpp"
#include "nodes/tensor_node.hpp"

using namespace tensor_compiler;

class TestTensorCompiler : public ::testing::Test {
protected:
    ComputeGraph graph;
};


TEST_F(TestTensorCompiler, TensorAllocation) {
    Shape shape = {2, 3};
    Tensor t(shape, DataType::FLOAT32);
    
    EXPECT_EQ(t.size(), 6);
    EXPECT_TRUE(t.empty());
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    t.set_data(data);
    
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.bytes(), 6 * sizeof(float));
    
    const float* raw_ptr = t.data<float>();
    EXPECT_FLOAT_EQ(raw_ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(raw_ptr[5], 6.0f);
}

TEST_F(TestTensorCompiler, TensorReshape) {
    Tensor t({4, 4}, DataType::INT32);
    std::vector<int32_t> data(16, 1);
    t.set_data(data);

    EXPECT_NO_THROW(t.reshape({2, 8}));
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 8);
    
    EXPECT_THROW(t.reshape({2, 2}), std::invalid_argument);
}

TEST_F(TestTensorCompiler, TensorTypeMismatch) {
    Tensor t({1}, DataType::FLOAT32);
    std::vector<int32_t> wrong_data = {1};

    EXPECT_THROW(t.set_data(wrong_data), std::runtime_error);
}

TEST_F(TestTensorCompiler, GraphConnectivity) {
    Shape in_shape {1, 2};
    Tensor* in_t = graph.add_tensor(in_shape, DataType::FLOAT32);
    TensorNode * in_node = graph.add_node<TensorNode>("input_node", in_t);
    
    auto relu = graph.add_node<ReluNode>("relu_op", in_node);
    
    in_node->add_output(relu);
    
    Shape out_shape {1, 2};
    Tensor* out_t = graph.add_tensor(out_shape, DataType::FLOAT32);
    TensorNode * out_node = graph.add_node<TensorNode>("output_node", out_t);
    
    relu->set_out_tensor(out_node);
    out_node->set_input(relu);
    
    EXPECT_EQ(in_node->output().size(), 1);
    EXPECT_EQ(*(in_node->output().begin()), relu);
    EXPECT_EQ(out_node->input(), relu);
    EXPECT_EQ(relu->name(), "relu_op");
}

TEST_F(TestTensorCompiler, FactoryOnnxLoadingFailure) {
    ComputeGraphFactory factory;
    EXPECT_THROW(factory.from_onnx("non_existent_model.onnx"), std::runtime_error);
}

namespace {

std::vector<char *> argv_from(std::vector<std::string> &storage) {
    std::vector<char *> v;
    for (auto &s : storage)
        v.push_back(s.data());
    return v;
}

} // namespace

TEST(CliParse, Help) {
    std::vector<std::string> s = {"tensor-compiler", "--help"};
    auto av = argv_from(s);
    tensor_compiler::ParsedCli p;
    ASSERT_TRUE(tensor_compiler::parse_cli(static_cast<int>(av.size()), av.data(), p));
    EXPECT_TRUE(p.help);
}

TEST(CliParse, PositionalObjectAndFlags) {
    std::vector<std::string> s = {"tensor-compiler", "-S", "-G", "m.onnx", "out.o"};
    auto av = argv_from(s);
    tensor_compiler::ParsedCli p;
    ASSERT_TRUE(tensor_compiler::parse_cli(static_cast<int>(av.size()), av.data(), p));
    EXPECT_EQ(p.input_onnx, "m.onnx");
    EXPECT_EQ(p.output_path, "out.o");
    EXPECT_TRUE(p.emit_assembly_extra);
    ASSERT_TRUE(p.graph_output.has_value());
    EXPECT_EQ(*p.graph_output, "results/graph.png");
}

TEST(CliUtil, DeriveAssemblyPath) {
    EXPECT_EQ(tensor_compiler::derive_assembly_path("dir/foo.o"), "dir/foo.s");
    EXPECT_EQ(tensor_compiler::infer_output_format("x.o", nullptr), tensor_compiler::OutputFormat::Object);
    EXPECT_EQ(tensor_compiler::infer_output_format("x.s", nullptr), tensor_compiler::OutputFormat::Assembly);
}

TEST_F(TestTensorCompiler, AddGraphStructure) {
    Shape sh{2, 2};
    Tensor *ta = graph.add_tensor(sh, DataType::FLOAT32);
    Tensor *tb = graph.add_tensor(sh, DataType::FLOAT32);
    auto *a = graph.add_node<TensorNode>("a", ta);
    auto *b = graph.add_node<TensorNode>("b", tb);
    Tensor *out = graph.add_tensor(sh, DataType::FLOAT32);
    auto *out_n = graph.add_node<TensorNode>("out", out);
    auto *add = graph.add_node<AddNode>("add", a, b);
    a->add_output(add);
    b->add_output(add);
    add->set_out_tensor(out_n);
    out_n->set_input(add);

    EXPECT_EQ(graph.nodes().size(), 5u);
    EXPECT_EQ(add->inputs().size(), 2u);
    EXPECT_EQ(add->output(), out_n);
}
