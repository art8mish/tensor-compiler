#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tensor_compiler {

enum class DataType { FLOAT32, FLOAT64, INT32, INT64, INT8, UINT8, BOOL, UNDEFINED };

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
    case DataType::INT8:
    case DataType::UINT8:
        return sizeof(int8_t);
    case DataType::BOOL:
        return sizeof(bool);
    default:
        throw std::invalid_argument("Unsupported type");
    }
}

template <typename T> constexpr DataType get_dtype() {
    if constexpr (std::is_same_v<T, float>)
        return DataType::FLOAT32;
    else if constexpr (std::is_same_v<T, double>)
        return DataType::FLOAT64;
    else if constexpr (std::is_same_v<T, int32_t>)
        return DataType::INT32;
    else if constexpr (std::is_same_v<T, int64_t>)
        return DataType::INT64;
    else if constexpr (std::is_same_v<T, int8_t>)
        return DataType::INT8;
    else if constexpr (std::is_same_v<T, uint8_t>)
        return DataType::UINT8;
    else if constexpr (std::is_same_v<T, bool>)
        return DataType::BOOL;
    else {
        static_assert(!sizeof(T), "Unsupported type for tensor");
        return DataType::UNDEFINED;
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

using dim_t = int64_t;
const dim_t DYNAMIC_DIM = -1;
using Shape = std::vector<dim_t>;

class Tensor {
    Shape shape_;
    DataType dtype_;
    std::optional<std::vector<uint8_t>> data_;

    void validate() {
        validate_shape(shape_);
    }

    void validate_shape(Shape &shape) {
        if (std::any_of(shape.begin(), shape.end(),
                        [](dim_t dim) { return dim == 0 || (dim < 0 && dim != DYNAMIC_DIM); }))
            throw std::invalid_argument("Zero dimension is not valid for shape");
    }

    template <typename T> void validate_dtype() const {
        DataType input_dtype = get_dtype<T>();
        if (dtype_ != input_dtype)
            throw std::runtime_error(
                "Data type mismatch: input dtype (" + dtype_to_string(input_dtype) +
                ") is incompatible with tensor dtype (" + dtype_to_string(dtype_) + ")");
    }

    void allocate_data() {
        size_t dim_size = size();
        if (with_data())
            throw std::logic_error("Data is already allocated");

        size_t bytes = dim_size * get_dtype_size(dtype_);
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

    bool is_dynamic() const {
        return std::any_of(shape_.begin(), shape_.end(), [](dim_t d) { return d == DYNAMIC_DIM; });
    }

    DataType dtype() const {
        return dtype_;
    }

    size_t size() const {
        if (is_dynamic())
            throw std::logic_error("Dynamic tensor has no size");
        if (shape_.empty())
            return 1;

        return std::accumulate(shape_.begin(), shape_.end(), static_cast<size_t>(1),
                               std::multiplies<size_t>());
    }

    size_t bytes() const {
        return with_data() ? data_->size() : 0;
    }

    void reshape(Shape new_shape) {
        validate_shape(new_shape);

        size_t new_total_size = 1;
        bool new_is_dynamic = false;
        for (auto d : new_shape) {
            if (d == DYNAMIC_DIM)
                new_is_dynamic = true;
            else
                new_total_size *= static_cast<size_t>(d);
        }

        if (with_data() && !is_dynamic() && !new_is_dynamic) {
            if (new_total_size != size())
                throw std::invalid_argument("Reshape cannot change total element count");
        }

        shape_ = std::move(new_shape);
        validate();

        if (!with_data())
            return;

        if (is_dynamic())
            data_.reset();
        else {
            size_t new_bytes = size() * get_dtype_size(dtype_);
            data_->resize(new_bytes, 0);
        }
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

        validate_dtype<T>();
        return reinterpret_cast<const T *>(data_->data());
    }

    template <typename T>
        requires std::is_arithmetic_v<T>
    void set_data(const std::vector<T> &data) {
        if (is_dynamic())
            throw std::logic_error("Dynamic tensor can't be initialized");
        validate_dtype<T>();

        size_t data_size = data.size();
        size_t elem_num = size();
        if (data_size != elem_num)
            throw std::invalid_argument("Input size (" + std::to_string(data_size) +
                                        ") is incompatible with tensor size (" +
                                        std::to_string(elem_num) + ")");
        if (!data_)
            allocate_data();
        std::memcpy(data_->data(), data.data(), data_->size());
    }

    template <typename T, typename It>
        requires std::is_arithmetic_v<T>
    void set_data(It begin, It end) {
        if (is_dynamic())
            throw std::logic_error("Dynamic tensor can't be initialized");
        validate_dtype<T>();

        size_t dist = static_cast<size_t>(std::distance(begin, end));
        size_t elem_num = size();
        if (dist != elem_num)
            throw std::invalid_argument("Input size (" + std::to_string(dist) +
                                        ") is incompatible with tensor size (" +
                                        std::to_string(elem_num) + ")");
        if (!data_)
            allocate_data();

        T *dest = reinterpret_cast<T *>(data_->data());
        std::copy(begin, end, dest);
    }
};

} // namespace tensor_compiler
