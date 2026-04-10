# Tensor Compiler

Проект компилирует **ONNX**-модель в вычислительный граф (`ComputeGraph`), строит представление в **MLIR** (Linalg/arith/tensor), затем понижает его до **LLVM IR** и генерирует **объектный файл (`.o`)** или **ассемблер (`.s`)** для нативной цели. Отдельно собирается **`kernel_driver`** — минимальный запуск скомпилированного ядра и сравнение с эталоном из Python/PyTorch.

## Поток данных

```text
ONNX → ComputeGraph → MLIR (bufferize → loops → LLVM dialect) → LLVM IR → asm/obj
```

Генерация тестовой модели: вручную `python3 end2end/gen.py` (не через CMake). Эталонные выходы для сравнения: `python3 end2end/pytorch_reference.py` (PyTorch) → `golden.bin`.

## Зависимости

- **CMake** ≥ 3.16, **C++20**
- **pkg-config**, **Graphviz** (`libgvc`, `libcgraph`)
- **Protobuf**, библиотеки **ONNX** (`onnx`, `onnx_proto`) и заголовки `onnx/onnx_pb.h`
- **LLVM/MLIR** с установленным `MLIRConfig.cmake` (типичная сборка `llvm-project` с `LLVM_ENABLE_PROJECTS=mlir`; см. ниже)
- Для **юнит-тестов**: GoogleTest
- Для **end2end** (опционально): см. [`end2end/requirements-extra.txt`](end2end/requirements-extra.txt) (`torch`, `onnx`, `numpy`)

### Сборка LLVM + MLIR (кратко)

```shell
git clone https://github.com/llvm/llvm-project.git
cd llvm-project && mkdir build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_INCLUDE_TESTS=OFF
ninja
sudo cmake --install .
```

Для `compile_commands.json` в проекте уже включено `CMAKE_EXPORT_COMPILE_COMMANDS`.

### Системные пакеты (пример Debian/Ubuntu)

```shell
sudo apt install build-essential cmake pkg-config \
  protobuf-compiler libprotobuf-dev libonnx-dev \
  libgraphviz-dev libgtest-dev
```

## Сборка проекта

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/path/to/llvm-install"
cmake --build build -j$(nproc)
```

Артефакты:

- `build/tensor-compiler` — основной компилятор
- `build/kernel_driver` — если `BUILD_KERNEL_DRIVER=ON` (по умолчанию): нужен уже существующий `end2end/models/test_model.onnx` (сгенерируйте через `gen.py` до сборки шага `kernel.o`)
- `build/tests` — юнит-тесты

Отключить драйвер ядра: `-DBUILD_KERNEL_DRIVER=OFF`.  
Отключить тесты: `-DBUILD_TESTS=OFF`.

```shell
ctest --test-dir build
```

## CLI: `tensor-compiler`

Обязательные позиционные аргументы: `<model.onnx> <выход.(s|o)>`.

| Опция | Описание |
|--------|----------|
| `-h`, `--help` | Справка |
| `-S [path]` | Дополнительно сгенерировать ассемблер. Для выхода `*.o` без пути: пишется `<stem>.s` рядом с объектником |
| `-G [path]` | PNG графа `ComputeGraph` (Graphviz). Без пути: `results/graph.png`. Если следующий аргумент — `*.onnx`, граф по умолчанию (чтобы не перепутать с именем файла) |
| `--graph=path` | Явный путь к PNG |
| `--llvm-triple=…` | Triple для LLVM `TargetMachine` |
| `--llvm-cpu=…` | CPU |
| `--llvm-features=…` | Строка subtarget features |
| `--reloc=pic\|static\|default` | Модель релокации |

Устаревший третий позиционный аргумент `object` / `assembly` всё ещё принимается; предпочтительно задавать формат по расширению (`.o` / `.s`).

Примеры:

```shell
./build/tensor-compiler model.onnx out.s
./build/tensor-compiler -S model.onnx out.o
./build/tensor-compiler -G --graph=doc/graph.png model.onnx out.o
./build/tensor-compiler --llvm-cpu=generic -S model.onnx out.o
```

## Эталон и `kernel_driver`

1. Сгенерируйте ONNX: `python3 end2end/gen.py`
2. Эталон PyTorch (те же веса, что в ONNX):  
   `python3 end2end/pytorch_reference.py --onnx end2end/models/test_model.onnx --out end2end/golden.bin`
3. Соберите объектник:  
   `./build/tensor-compiler end2end/models/test_model.onnx build/kernel.o`
4. Пересоберите `kernel_driver` (или используйте уже собранный `kernel.o` из `build/`)
5. Запуск и сравнение:

```shell
./build/kernel_driver --compare end2end/golden.bin -v
```

Формат `golden.bin`: 9 значений `float32` (выход `1×1×3×3`), порядок как у линейного массива в памяти C.

## Каталог `end2end`

| Файл | Назначение |
|------|------------|
| [`gen.py`](end2end/gen.py) | Создаёт тестовый `models/test_model.onnx` |
| [`pytorch_reference.py`](end2end/pytorch_reference.py) | Эталон через PyTorch, запись `golden.bin` |
| [`requirements-extra.txt`](end2end/requirements-extra.txt) | pip-зависимости для скриптов |

## ONNX Runtime (сторонняя ссылка)

В readme ранее упоминался архив **onnxruntime** — он не обязателен для сборки C++-части; эталон реализован через PyTorch в `pytorch_reference.py`.
