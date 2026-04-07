#pragma once

#include "cgraph.hpp"
#include "node.hpp"
#include "operations.hpp"
#include "tensor.hpp"
#include "tensor_node.hpp"
#include <cassert>
#include <fstream>
#include <memory>
#include <onnx/onnx_pb.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace tensor_compiler {

class ComputeGraphFactory {
    using TNode = TensorNode<Node *, Tensor *>;
    using TNodePtr = TNode *;
    using OpNodePtr = OpNode<TNodePtr> *;

    static DataType onnx_dtype(int32_t onnx_type) {
        switch (onnx_type) {
        case onnx::TensorProto::FLOAT:
            return DataType::FLOAT32;
        case onnx::TensorProto::DOUBLE:
            return DataType::FLOAT64;
        case onnx::TensorProto::INT32:
            return DataType::INT32;
        case onnx::TensorProto::INT64:
            return DataType::INT64;
        case onnx::TensorProto::INT8:
            return DataType::INT8;
        case onnx::TensorProto::UINT8:
            return DataType::UINT8;
        case onnx::TensorProto::BOOL:
            return DataType::BOOL;
        default:
            throw std::runtime_error("ONNX data type is not supported: " +
                                     std::to_string(onnx_type));
        }
    }

    template <typename T, typename ProtobufField>
    void fill_tensor_data_impl(Tensor &tensor, const ProtobufField &field) {
        std::vector<T> data(field.begin(), field.end());
        tensor.set_data(data);
    }

    void fill_tensor(Tensor &tensor, const onnx::TensorProto &initializer) {
        DataType dtype = onnx_dtype(initializer.data_type());
        if (dtype != tensor.dtype())
            throw std::runtime_error("Tensor data type mismatch");

        if (initializer.has_raw_data()) {
            const std::string &raw = initializer.raw_data();
            // std::vector<uint8_t> bytes(raw.begin(), raw.end());
            tensor.set_data<uint8_t>(raw.begin(), raw.end());
            return;
        }

        switch (dtype) {
        case DataType::FLOAT32:
            fill_tensor_data_impl<float>(tensor, initializer.float_data());
            break;
        case DataType::FLOAT64:
            fill_tensor_data_impl<double>(tensor, initializer.double_data());
            break;
        case DataType::INT32:
            fill_tensor_data_impl<int32_t>(tensor, initializer.int32_data());
            break;
        case DataType::INT64:
            fill_tensor_data_impl<int64_t>(tensor, initializer.int64_data());
            break;
        default:
            throw std::runtime_error("Tensor data type is not supported");
        }
    }

    OpNodePtr handle_conv(ComputeGraph &cgraph, const onnx::NodeProto &node_proto,
                          const std::string &node_name, const std::vector<TNodePtr > &inputs) {
        if (inputs.size() < 2)
            throw std::runtime_error("Conv: need at least X and W inputs");

        std::vector<uint64_t> kernel_shape, strides, dilations, pads;
        uint64_t group = 1;

        for (const auto &attr : node_proto.attribute()) {
            auto fill_vec = [](std::vector<uint64_t> &vec, const auto &attr_proto) {
                vec.reserve(static_cast<size_t>(attr_proto.ints_size()));
                for (auto v : attr_proto.ints())
                    vec.push_back(static_cast<uint64_t>(v));
            };
            if (attr.name() == "kernel_shape")
                fill_vec(kernel_shape, attr);
            else if (attr.name() == "strides")
                fill_vec(strides, attr);
            else if (attr.name() == "dilations")
                fill_vec(dilations, attr);
            else if (attr.name() == "pads")
                fill_vec(pads, attr);
            else if (attr.name() == "group")
                group = static_cast<uint64_t>(attr.i());
        }

        TNodePtr X = inputs[0];
        TNodePtr W = inputs[1];
        TNodePtr B = (inputs.size() > 2) ? inputs[2] : nullptr;

        return cgraph.add_node<ConvNode<TNodePtr >>(node_name, std::move(kernel_shape), X, W, B,
                                                  std::move(strides), std::move(dilations),
                                                  std::move(pads), group);
    }

    OpNodePtr handle_gemm(ComputeGraph &cgraph, const onnx::NodeProto &node_proto,
                          const std::string &node_name, const std::vector<TNodePtr > &inputs) {
        double alpha = 1.0, beta = 1.0;
        bool transA = false, transB = false;

        for (const auto &attr : node_proto.attribute()) {
            if (attr.name() == "alpha")
                alpha = attr.f();
            else if (attr.name() == "beta")
                beta = attr.f();
            else if (attr.name() == "transA")
                transA = (attr.i() != 0);
            else if (attr.name() == "transB")
                transB = (attr.i() != 0);
        }

        return cgraph.add_node<GemmNode<TNodePtr >>(node_name, inputs[0], inputs[1],
                                                  (inputs.size() > 2 ? inputs[2] : nullptr), alpha,
                                                  beta, transA, transB);
    }

    OpNodePtr handle_matmul(ComputeGraph &cgraph, const onnx::NodeProto &,
                            const std::string &node_name, const std::vector<TNodePtr > &inputs) {
        return cgraph.add_node<MatMulNode<TNodePtr >>(node_name, inputs[0], inputs[1]);
    }

    OpNodePtr handle_relu(ComputeGraph &cgraph, const onnx::NodeProto &,
                          const std::string &node_name, const std::vector<TNodePtr > &inputs) {
        return cgraph.add_node<ReluNode<TNodePtr >>(node_name, inputs[0]);
    }

    OpNodePtr handle_add(ComputeGraph &cgraph, const onnx::NodeProto &,
                         const std::string &node_name, const std::vector<TNodePtr > &inputs) {
        return cgraph.add_node<AddNode<TNodePtr >>(node_name, inputs[0], inputs[1]);
    }

    OpNodePtr handle_mul(ComputeGraph &cgraph, const onnx::NodeProto &,
                         const std::string &node_name, const std::vector<TNodePtr > &inputs) {
        return cgraph.add_node<MulNode<TNodePtr >>(node_name, inputs[0], inputs[1]);
    }

    OpNodePtr create_op_node(ComputeGraph &cgraph, const onnx::NodeProto &node_proto,
                             const std::string &node_name, const std::vector<TNodePtr > &inputs) {
        const std::string &op_type = node_proto.op_type();
        if (op_type == "Conv")
            return handle_conv(cgraph, node_proto, node_name, inputs);
        if (op_type == "Gemm")
            return handle_gemm(cgraph, node_proto, node_name, inputs);
        if (op_type == "MatMul")
            return handle_matmul(cgraph, node_proto, node_name, inputs);
        if (op_type == "Add")
            return handle_add(cgraph, node_proto, node_name, inputs);
        if (op_type == "Mul")
            return handle_mul(cgraph, node_proto, node_name, inputs);
        if (op_type == "Relu")
            return handle_relu(cgraph, node_proto, node_name, inputs);
        throw std::runtime_error("Unsupported operation: " + op_type);
    }

    TNodePtr process_onnx_valueinfo(ComputeGraph &cgraph, const onnx::ValueInfoProto &valueinfo_proto,
                                  const std::string &name) {
        Shape shape{};
        const auto &type = valueinfo_proto.type().tensor_type();
        for (const auto &dim : type.shape().dim()) {
            shape.push_back(dim.has_dim_value() ? static_cast<dim_t>(dim.dim_value())
                                                : DYNAMIC_DIM);
        }
        Tensor *tensor = cgraph.add_tensor(shape, onnx_dtype(type.elem_type()));
        return cgraph.add_node<TNode>(name, tensor);
    }

    TNodePtr process_onnx_tensor(ComputeGraph &cgraph, const onnx::TensorProto &tensor_proto,
                               const std::string &name) {
        Shape shape{};
        for (auto dim : tensor_proto.dims())
            shape.push_back(static_cast<dim_t>(dim));
        Tensor *tensor = cgraph.add_tensor(shape, onnx_dtype(tensor_proto.data_type()));
        fill_tensor(*tensor, tensor_proto);
        return cgraph.add_node<TNode>(name, tensor);
    }

public:
    std::unique_ptr<ComputeGraph> from_onnx(const std::string &filename) {
        onnx::ModelProto model;
        std::fstream input(filename, std::ios::in | std::ios::binary);
        if (!input || !model.ParseFromIstream(&input))
            throw std::runtime_error("Failed to load ONNX");

        const auto &graph = model.graph();
        auto cgraph = std::make_unique<ComputeGraph>();
        std::unordered_map<std::string, TNodePtr> tensor_map;

        for (const auto &init : graph.initializer())
            tensor_map[init.name()] = process_onnx_tensor(*cgraph, init, init.name());

        for (const auto &input : graph.input())
            if (tensor_map.find(input.name()) == tensor_map.end())
                tensor_map[input.name()] = process_onnx_valueinfo(*cgraph, input, input.name());

        for (const auto &node_proto : graph.node()) {
            std::string node_name =
                node_proto.name().empty()
                    ? node_proto.op_type() + "_" + std::to_string(tensor_map.size())
                    : node_proto.name();

            std::vector<TNodePtr> inputs;
            for (const auto &in_name : node_proto.input())
                inputs.push_back(tensor_map.at(in_name));

            OpNodePtr op_node = create_op_node(*cgraph, node_proto, node_name, inputs);
            for (auto in : inputs)
                in->add_output(op_node);

            const std::string &out_name = node_proto.output(0);
            if (tensor_map.find(out_name) == tensor_map.end()) {
                Tensor *out_tensor =
                    cgraph->add_tensor(op_node->get_out_shape(), inputs[0]->dtype());
                tensor_map[out_name] = cgraph->add_node<TNode>(out_name, out_tensor);
            }
            op_node->set_out_tensor(tensor_map[out_name]);
            tensor_map.at(out_name)->set_input(op_node);
        }
        return cgraph;
    }
};

} // namespace tensor_compiler