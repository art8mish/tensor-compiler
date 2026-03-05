
#pragma once

#include "cgraph.hpp"
#include "tensor.hpp"
#include <fstream>
#include <memory>
#include <onnx/onnx_pb.h>
#include <stdexcept>
#include <unordered_map>

namespace tensor_compiler {

class ComputeGraphFactory {

    Node *tensor_map(const std::string &name) {
        switch (type) {
        case Type::CONV:
            return "Conv";
        case Type::GEMM:
            return "Gemm";
        case Type::MATMUL:
            return "MatMul";
        case Type::ADD:
            return "Add";
        case Type::MUL:
            return "Mul";
        case Type::RELU:
            return "ReLU";
        case Type::TENSOR:
            return "Tensor";
        default:
            return "?";
        }
    }

    DataType onnx_dtype(int32_t onnx_type) {
        switch (onnx_type) {
        case onnx::TensorProto::FLOAT:
            return DataType::FLOAT32;
        case onnx::TensorProto::DOUBLE:
            return DataType::FLOAT64;
        case onnx::TensorProto::INT32:
            return DataType::INT32;
        case onnx::TensorProto::INT64:
            return DataType::INT64;
        default:
            throw std::runtime_error("ONNX data type is not supported: " +
                                     std::to_string(onnx_type));
        }
    }

    template <typename T, typename Data> void fill_tensor_data(Tensor &tensor, const Data &data) {
        tensor.set_values(data.begin(), data.end());
    }

    void fill_tensor(Tensor &tensor, const onnx::TensorProto &initializer) {
        DataType dtype = onnx_dtype(initializer.data_type());
        if (dtype != tensor.dtype())
            throw std::runtime_error("Tensor data type mismatch");

        // if (initializer.has_raw_data()) {
        //     const std::string &raw = initializer.raw_data();
        //     if (raw.size() != tensor.bytes())
        //         throw std::runtime_error("Tensor raw data size mismatch");
        //     std::memcpy(const_cast<uint8_t *>(tensor.data<uint8_t>()), raw.data(), raw.size());
        // }

        size_t num_elements = tensor.size();
        switch (dtype) {
        case DataType::FLOAT32:
            if (initializer.float_data_size() != static_cast<int>(num_elements))
                throw std::runtime_error("Tensor float32 data size mismatch");
            fill_tensor_data<float>(tensor, initializer.float_data());
            break;
        case DataType::FLOAT64:
            if (initializer.double_data_size() != static_cast<int>(num_elements))
                throw std::runtime_error("Tensor float64 data size mismatch");
            fill_tensor_data<double>(tensor, initializer.double_data());
            break;
        case DataType::INT32:
            if (initializer.int32_data_size() != static_cast<int>(num_elements))
                throw std::runtime_error("Tensor int32 data size mismatch");
            fill_tensor_data<int32_t>(tensor, initializer.int32_data());
            break;
        case DataType::INT64:
            if (initializer.int64_data_size() != static_cast<int>(num_elements))
                throw std::runtime_error("Tensor int64 data size mismatch");
            fill_tensor_data<int64_t>(tensor, initializer.int64_data());
            break;
        default:
            throw std::runtime_error("Tensor data type is not supported");
        }
    }

public:
    std::unique_ptr<ComputeGraph> from_onnx(const std::string &filename) {
        onnx::ModelProto model;
        std::fstream input(filename, std::ios::in | std::ios::binary);
        if (!input.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        if (!model.ParseFromIstream(&input)) {
            throw std::runtime_error("Failed to parse ONNX model");
        }

        const auto &graph = model.graph();
        auto cgraph = std::make_unique<ComputeGraph>();

        std::unordered_map<std::string, TensorNode *> tensor_map;

        for (const auto &initializer : graph.initializer()) {
            const std::string &name = initializer.name();
            Shape shape{};
            for (auto dim : initializer.dims()) {
                if (dim <= 0)
                    throw std::runtime_error("Tensor dimension value is invalid");
                // dim_t d = (dim == -1) ? 0 : static_cast<dim_t>(dim);
                shape.push_back(static_cast<dim_t>(dim));
            }

            DataType dtype = onnx_dtype(initializer.data_type());

            Tensor *tensor = cgraph->add_tensor(shape, dtype);
            fill_tensor(*tensor, initializer);

            TensorNode *tensor_node = cgraph->add_node<TensorNode<Node *, Tensor *>>(name, tensor);
            tensor_map[name] = tensor_node;
        }

        for (const auto &input : graph.input()) {
            const std::string &name = input.name();
            if (tensor_map.find(name) != tensor_map.end())
                continue;
        
            Shape shape {};
            const auto &type = input.type().tensor_type();
            for (const auto &dim : type.shape().dim()) {
                if (dim.has_dim_value()) {
                    shape.push_back(static_cast<dim_t>(dim.dim_value()));
                } else {
                    shape.push_back(0);
                }
            }
            DataType dtype = onnx_dtype(type.elem_type());

            // Создаём пустой тензор (данных нет)
            Tensor *tensor = cgraph->add_tensor(shape, dtype);
            auto *tensor_node = cgraph->add_node<TensorNode<Node *, Tensor *>>(
                name, tensor, nullptr, std::vector<Node *>{});
            tensor_map[name] = tensor_node;
        }

        // 3. Обрабатываем узлы операций
        for (const auto &node_proto : graph.node()) {
            const std::string &op_type = node_proto.op_type();
            std::string node_name = node_proto.name();
            if (node_name.empty()) {
                node_name = op_type + "_" + std::to_string(tensor_map.size());
            }

            // Собираем входные тензоры
            std::vector<Node *> inputs;
            for (const auto &input_name : node_proto.input()) {
                auto it = tensor_map.find(input_name);
                if (it == tensor_map.end()) {
                    throw std::runtime_error("Input tensor not found: " + input_name);
                }
                inputs.push_back(it->second);
            }

            Node *op_node = nullptr;

            // --- Conv ---
            if (op_type == "Conv") {
                std::vector<uint64_t> kernel_shape;
                std::vector<uint64_t> strides;
                std::vector<uint64_t> dilations;
                std::vector<uint64_t> pads;
                uint64_t group = 1;

                for (const auto &attr : node_proto.attribute()) {
                    if (attr.name() == "kernel_shape") {
                        for (auto v : attr.ints())
                            kernel_shape.push_back(v);
                    } else if (attr.name() == "strides") {
                        for (auto v : attr.ints())
                            strides.push_back(v);
                    } else if (attr.name() == "dilations") {
                        for (auto v : attr.ints())
                            dilations.push_back(v);
                    } else if (attr.name() == "pads") {
                        for (auto v : attr.ints())
                            pads.push_back(v);
                    } else if (attr.name() == "group") {
                        group = attr.i();
                    }
                }

                if (inputs.size() < 2) {
                    throw std::runtime_error("Conv: need at least X and W");
                }
                Node *X = inputs[0];
                Node *W = inputs[1];
                Node *B = (inputs.size() > 2) ? inputs[2] : nullptr;

                op_node = cgraph->add_node<ConvNode<Node *>>(node_name, kernel_shape, X, W, B,
                                                             strides, dilations, pads, group);
            }
            // --- Gemm ---
            else if (op_type == "Gemm") {
                double alpha = 1.0, beta = 1.0;
                bool transA = false, transB = false;

                for (const auto &attr : node_proto.attribute()) {
                    if (attr.name() == "alpha")
                        alpha = attr.f();
                    else if (attr.name() == "beta")
                        beta = attr.f();
                    else if (attr.name() == "transA")
                        transA = attr.i();
                    else if (attr.name() == "transB")
                        transB = attr.i();
                }

                if (inputs.size() < 2) {
                    throw std::runtime_error("Gemm: need at least A and B");
                }
                Node *A = inputs[0];
                Node *B = inputs[1];
                Node *C = (inputs.size() > 2) ? inputs[2] : nullptr;

                op_node = cgraph->add_node<GemmNode<Node *>>(node_name, A, B, C, alpha, beta,
                                                             transA, transB);
            }
            // --- MatMul ---
            else if (op_type == "MatMul") {
                if (inputs.size() != 2) {
                    throw std::runtime_error("MatMul: need exactly 2 inputs");
                }
                op_node = cgraph->add_node<MatMulNode<Node *>>(node_name, inputs[0], inputs[1]);
            }
            // --- Add ---
            else if (op_type == "Add") {
                if (inputs.size() != 2) {
                    throw std::runtime_error("Add: need exactly 2 inputs");
                }
                op_node = cgraph->add_node<AddNode<Node *>>(node_name, inputs[0], inputs[1]);
            }
            // --- Mul ---
            else if (op_type == "Mul") {
                if (inputs.size() != 2) {
                    throw std::runtime_error("Mul: need exactly 2 inputs");
                }
                op_node = cgraph->add_node<MulNode<Node *>>(node_name, inputs[0], inputs[1]);
            }
            // --- Relu ---
            else if (op_type == "Relu") {
                if (inputs.size() != 1) {
                    throw std::runtime_error("Relu: need exactly 1 input");
                }
                op_node = cgraph->add_node<ReluNode<Node *>>(node_name, inputs[0]);
            } else {
                throw std::runtime_error("Unsupported operation: " + op_type);
            }

            // Добавляем текущую операцию в выходные списки входных тензоров
            for (Node *input_node : inputs) {
                auto *tensor_input = dynamic_cast<TensorNode<Node *, Tensor *> *>(input_node);
                if (tensor_input) {
                    tensor_input->add_output(op_node);
                }
            }

            // Проверяем, что узел имеет ровно один выход
            if (node_proto.output_size() != 1) {
                throw std::runtime_error("Multiple outputs not supported");
            }
            const std::string &output_name = node_proto.output(0);

            // Получаем указатель на OpNode для доступа к выходной форме
            auto *op_node_ptr = dynamic_cast<OpNode<Node *> *>(op_node);
            if (!op_node_ptr) {
                throw std::runtime_error("Node is not an OpNode");
            }

            // Вычисляем выходную форму
            Shape out_shape = op_node_ptr->get_out_shape();

            // Определяем тип данных выхода (по первому входу или FLOAT32)
            DataType out_dtype = DataType::FLOAT32;
            if (!inputs.empty()) {
                auto *input_tensor = dynamic_cast<TensorNode<Node *, Tensor *> *>(inputs[0]);
                if (input_tensor) {
                    out_dtype = input_tensor->dtype();
                }
            }

            // Создаём тензор для выхода
            Tensor *out_tensor = cgraph->add_tensor(out_shape, out_dtype);
            auto *out_tensor_node = cgraph->add_node<TensorNode<Node *, Tensor *>>(
                output_name, out_tensor->shape(), out_tensor, op_node, std::vector<Node *>{});
            tensor_map[output_name] = out_tensor_node;

            // Связываем выход операции с тензором
            op_node_ptr->set_out_tensor(out_tensor_node);
        }

        // 4. Проверяем, что все выходы графа присутствуют
        for (const auto &output : graph.output()) {
            const std::string &name = output.name();
            if (tensor_map.find(name) == tensor_map.end()) {
                // Создаём недостающий тензор (обычно это входной тензор, который является выходом
                // графа)
                Shape shape;
                const auto &type = output.type().tensor_type();
                for (const auto &dim : type.shape().dim()) {
                    if (dim.has_dim_value()) {
                        shape.push_back(static_cast<dim_t>(dim.dim_value()));
                    } else {
                        shape.push_back(0);
                    }
                }
                DataType dtype = onnx_dtype(type.elem_type());
                Tensor *tensor = cgraph->add_tensor(shape, dtype);
                auto *tensor_node = cgraph->add_node<TensorNode<Node *, Tensor *>>(
                    name, tensor->shape(), tensor, nullptr, std::vector<Node *>{});
                tensor_map[name] = tensor_node;
            }
        }

        return cgraph;
    }
};

} // namespace tensor_compiler