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

#include <viz/drawable.hpp>
#include <tensor/dtype.hpp>

namespace tensor_compiler {

using dim_t = int64_t;
const dim_t DYNAMIC_DIM = -1;
using Shape = std::vector<dim_t>;

class Tensor : public Drawable {
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

    template <typename T>
    Agnode_t* draw_impl(Agraph_t* g, std::string name = "") const {
        std::string shape_str = "[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            shape_str += std::to_string(shape_[i]) + (i == shape_.size() - 1 ? "" : ", ");
        }
        shape_str += "]";

        std::string html = "<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">";
        html += "<TR><TD COLSPAN=\"4\" BGCOLOR=\"#EEEEEE\"><B>Tensor";
        if (!name.empty())
            html += " (" + name + ")";
        html += "</B></TD></TR>";
        html += "<TR><TD COLSPAN=\"4\">DType: " + 
                dtype_to_string(dtype_) + " | Shape: " + shape_str + "</TD></TR>";

        const T* tensor_data = data<T>();
        if (tensor_data != nullptr && !empty()) {
            size_t total_elements = size();
            size_t limit = 16;
            size_t to_draw = std::min(total_elements, limit);

            html += "<TR>";
            for (size_t i = 0; i < to_draw; ++i) {
                if (i > 0 && i % 4 == 0) 
                    html += "</TR><TR>";

                html += "<TD>";
                if constexpr (std::is_same_v<T, bool>)
                    html += (tensor_data[i] ? "true" : "false");
                else
                    html += std::to_string(+tensor_data[i]);
                html += "</TD>";
            }

            if (total_elements > limit) {
                html += "</TR><TR><TD COLSPAN=\"4\" BGCOLOR=\"#F5F5F5\">... and " + 
                        std::to_string(total_elements - limit) + " more</TD>";
            }
            html += "</TR>";
        } else {
            std::string status = (is_dynamic()) ? "Dynamic" : "Empty";
            html += "<TR><TD COLSPAN=\"4\">"+ status + "</TD></TR>";
        }

        html += "</TABLE>";
        std::string html_label = html;
        return Drawable::draw(g, html_label, "none");
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

    

    Agnode_t* draw(Agraph_t* g, std::string name) const {
        switch (dtype_) {
            case DataType::FLOAT32: return draw_impl<DataType_t<DataType::FLOAT32>>(g, name);
            case DataType::FLOAT64: return draw_impl<DataType_t<DataType::FLOAT64>>(g, name);
            case DataType::INT32:   return draw_impl<DataType_t<DataType::INT32>>(g, name);
            case DataType::INT64:   return draw_impl<DataType_t<DataType::INT64>>(g, name);
            case DataType::INT8:   return draw_impl<DataType_t<DataType::INT8>>(g, name);
            case DataType::UINT8:   return draw_impl<DataType_t<DataType::UINT8>>(g, name);
            case DataType::BOOL:   return draw_impl<DataType_t<DataType::BOOL>>(g, name);
            default:
                return draw_impl<int32_t>(g, name); 
        }
    }

    Agnode_t* draw(Agraph_t* g) const override {
        return draw(g, "");
    }
};

} // namespace tensor_compiler
