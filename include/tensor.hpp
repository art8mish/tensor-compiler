#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tensor_compiler {

enum class DataType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
};

size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
    case DataType::FLOAT32:
        return sizeof(float);
    case DataType::FLOAT64:
        return sizeof(double);
    case DataType::INT32:
        return sizeof(int32_t);
    case DataType::INT64:
        return sizeof(int64_t);
    default:
        return 0;
    }
}

std::string dtype_to_string(DataType dtype) {
    switch (dtype) {
    case DataType::FLOAT32:
        return "float32";
    case DataType::FLOAT64:
        return "float64";
    case DataType::INT32:
        return "int32";
    case DataType::INT64:
        return "int64";
    default:
        return "unknown";
    }
}

using dim_t = uint64_t;
using Shape = std::vector<dim_t>;

class Tensor {
    Shape shape_;
    DataType dtype_;
    std::optional<std::vector<uint8_t>> data_;

    void validate() {
        if (!std::none_of(shape_.begin(), shape_.end(), [](dim_t dim) { return dim == 0; }))
            throw std::invalid_argument("Zero dimension is not valid for shape");
    }

    void allocate_data() {
        if (data_)
            throw std::logic_error("Data is already allocated");

        size_t bytes = size() * get_dtype_size(dtype_);
        data_.emplace(bytes);
        // else
        //     data_->resize(bytes);
    }

    bool with_data() const noexcept {
        return data_.has_value();
    }

public:
    Tensor() = default;
    Tensor(Shape shape, DataType dtype) : shape_(std::move(shape)), dtype_(dtype) {
        validate();
    }

    bool empty() const noexcept {
        return !with_data() || data_->empty();
    }

    const Shape &shape() const {
        return shape_;
    }

    DataType dtype() const {
        return dtype_;
    }

    size_t size() const {
        if (shape_.empty())
            return 1;
        return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<dim_t>());
    }

    size_t bytes() const {
        return with_data() ? data_->size() : 0;
    }

    // template <typename T>
    // requires std::is_arithmetic_v<T>
    // T *data() {
    //     if (get_dtype_size(dtype_) != sizeof(T))
    //         throw std::runtime_error("Data type mismatch");
    //     return reinterpret_cast<T *>(data_.data());
    // }

    template <typename T>
        requires std::is_arithmetic_v<T>
    const T *data() const {
        if (!with_data())
            return nullptr;

        if (get_dtype_size(dtype_) != sizeof(T))
            throw std::runtime_error("Data type mismatch");
        return reinterpret_cast<const T *>(data_->data());
    }

    template <typename T>
        requires std::is_arithmetic_v<T>
    void set_data(const std::vector<T> &values) {
        if (get_dtype_size(dtype_) != sizeof(T))
            throw std::runtime_error("Data type mismatch");

        size_t elem_num = size();
        if (values.size() != elem_num)
            throw std::invalid_argument("Input size (" + std::to_string(values.size()) +
                                        ") is incompatible with tensor size (" +
                                        std::to_string(elem_num) + ")");
        if (!data_)
            allocate_data();
        std::memcpy(data_->data(), values.data(), data_->size());
    }

    template <typename T, typename It>
        requires std::is_arithmetic_v<T>
    void set_data(It begin, It end) {
        if (get_dtype_size(dtype_) != sizeof(T))
            throw std::runtime_error("Data type mismatch");

        size_t dist = std::distance(begin, end);
        size_t elem_num = size();
        if (dist != elem_num)
            throw std::invalid_argument("Input size (" + std::to_string(dist) +
                                        ") is incompatible with tensor size (" +
                                        std::to_string(elem_num) + ")");

        if (!data_)
            allocate_data();

        const uint8_t *byte_begin = reinterpret_cast<const uint8_t *>(&(*begin));
        const uint8_t *byte_end = byte_begin + data_->size();
        data_->assign(byte_begin, byte_end);
    }
};

} // namespace tensor_compiler
