
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