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

namespace tensor_compiler {

enum class DataType { FLOAT32, FLOAT64, INT32, INT64, INT8, UINT8, BOOL, UNDEFINED };


template<DataType D> struct DataType2Type;

template<> struct DataType2Type<DataType::FLOAT32> { using type = float; };
template<> struct DataType2Type<DataType::FLOAT64> { using type = double; };
template<> struct DataType2Type<DataType::INT32>   { using type = int32_t; };
template<> struct DataType2Type<DataType::INT64>   { using type = int64_t; };
template<> struct DataType2Type<DataType::INT8>    { using type = int8_t; };
template<> struct DataType2Type<DataType::UINT8>   { using type = uint8_t; };
template<> struct DataType2Type<DataType::BOOL>    { using type = bool; };

template<DataType D>
using DataType_t = typename DataType2Type<D>::type;

size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
    case DataType::FLOAT32:
        return sizeof(DataType_t<DataType::FLOAT32>);
    case DataType::FLOAT64:
        return sizeof(DataType_t<DataType::FLOAT64>);
    case DataType::INT32:
        return sizeof(DataType_t<DataType::INT32>);
    case DataType::INT64:
        return sizeof(DataType_t<DataType::INT64>);
    case DataType::INT8:
        return sizeof(DataType_t<DataType::INT8>);
    case DataType::UINT8:
        return sizeof(DataType_t<DataType::UINT8>);
    case DataType::BOOL:
        return sizeof(DataType_t<DataType::BOOL>);
    default:
        throw std::invalid_argument("Unsupported type");
    }
}

template <typename T> constexpr DataType get_dtype() {
    if constexpr (std::is_same_v<T, DataType_t<DataType::FLOAT32>>)
        return DataType::FLOAT32;
    else if constexpr (std::is_same_v<T, DataType_t<DataType::FLOAT64>>)
        return DataType::FLOAT64;
    else if constexpr (std::is_same_v<T, DataType_t<DataType::INT32>>)
        return DataType::INT32;
    else if constexpr (std::is_same_v<T, DataType_t<DataType::INT64>>)
        return DataType::INT64;
    else if constexpr (std::is_same_v<T, DataType_t<DataType::INT8>>)
        return DataType::INT8;
    else if constexpr (std::is_same_v<T, DataType_t<DataType::UINT8>>)
        return DataType::UINT8;
    else if constexpr (std::is_same_v<T, DataType_t<DataType::BOOL>>)
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
    case DataType::INT8:
        return "int8";
    case DataType::UINT8:
        return "uint8";
    case DataType::BOOL:
        return "bool";
    case DataType::UNDEFINED:
        return "undefined";
    default:
        return "unknown";
    }
}

} // namespace tensor_compiler
