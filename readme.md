
# Tensor Compiler

## Build

```shell
mkdir third_party && cd third_party
wget https://github.com/microsoft/onnxruntime/releases/download/<version>/onnxruntime-linux-x64-<version>.tgz
tar -zxvf onnxruntime-linux-x64-<version>.tgz
cd ../

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

```shell
sudo apt update
sudo apt install build-essential cmake pkg-config
sudo apt install graphviz libgraphviz-dev
sudo apt install protobuf-compiler libprotobuf-dev
sudo apt install libonnx-dev
sudo apt install libgtest-dev
```

## ONNX Format

```
onnx::ModelProto                    ← Корневой объект
├─ ir_version: int64
├─ producer_name/version: string
├─ opset_import[]: OperatorSetIdProto
│  ├─ domain: string ("", "ai.onnx.ml")
│  └─ version: int64
├─ graph: GraphProto                ← Вычислительный граф
│  ├─ name: string
│  ├─ input[]: ValueInfoProto       ← Внешние входы модели
│  ├─ output[]: ValueInfoProto      ← Внешние выходы модели
│  ├─ value_info[]: ValueInfoProto  ← Промежуточные тензоры (опционально)
│  ├─ node[]: NodeProto             ← Операции (операторы)
│  │  ├─ name: string
│  │  ├─ op_type: string ("Conv", "Relu"...)
│  │  ├─ input[]: string           ← Имена входных тензоров
│  │  ├─ output[]: string          ← Имена выходных тензоров
│  │  ├─ attribute[]: AttributeProto ← Параметры операции
│  │  │  ├─ name: string
│  │  │  ├─ type: int (AttributeType enum)
│  │  │  ├─ f: float, i: int64, s: string, t: TensorProto, floats[], ints[]...
│  │  └─ domain: string
│  └─ initializer[]: TensorProto    ← Веса (константы)
│     ├─ name: string
│     ├─ data_type: int32 (TensorProto.DataType)
│     ├─ dims[]: int64
│     ├─ float_data[], int32_data[], raw_data: bytes...
└─ metadata_props[]: StringStringEntryProto

message ValueInfoProto {
  string name = 1;
  TypeProto type = 2;  // ← ключевое поле
}

message TypeProto {
  message Tensor {
    int32 elem_type = 1;  // TensorProto.DataType
    ShapeProto shape = 2;
  }
  Tensor tensor_type = 1;
  // ... другие типы (Sequence, Map, Optional)
}

message ShapeProto {
  repeated Dimension dim = 1;
}

message Dimension {
  oneof value {
    int64 dim_value = 1;  // конкретное значение
    string dim_param = 2; // символ ("N", "batch") для динамических размеров
  }
}


message NodeProto {
  string name = 1;
  string op_type = 2;           // "Conv", "MatMul", "Relu"...
  repeated string input = 3;    // имена входных тензоров
  repeated string output = 4;   // имена выходных тензоров (обычно 1)
  repeated AttributeProto attribute = 5;
  string domain = 6;            // "" для ai.onnx, "ai.onnx.ml" для ML-операций
}

// Соответствие атрибутам ONNX Conv
attribute {
  name: "kernel_shape"  // ints: [3, 3]
  type: INTS            // enum AttributeType
  ints: 3
  ints: 3
}
attribute {
  name: "strides"       // ints: [2, 2]
  type: INTS
  ints: 2
  ints: 2
}
attribute {
  name: "pads"          // ints: [1, 1, 1, 1] (top, left, bottom, right)
  type: INTS
}
attribute {
  name: "group"         // i: 1 (int64)
  type: INT
}
```