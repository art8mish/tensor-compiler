# LeNet

В документе представлено описание создание в [end2end/gen.py](end2end/gen.py) примера модель для инференса. Модель представляет близкую к оригинальной LeNet-5 в рамках поддерживаемых операторов компилятора.

Оригинальная LeNet-5 использует `AveragePool` и полносвязные слои после `Flatten`, но 
в текущем проекте компилируемое ONNX-подмножество ограничено представленными операциями: `Conv`, `Relu`, `Add`, `Mul`, `MatMul`, `Gemm`.

Изменения:

- subsampling (`S2`, `S4`) реализован как `Conv 2x2, stride=2`
- полносвязная часть (`F6`, classifier) реализована как `Conv 1x1`
- `Flatten` и `Softmax` не используются

## Архитектура реализованной сети

Вход:

- `X`: `float32`, форма `[1, 1, 28, 28]`.

Слои:

1. **C1**: `Conv(1 -> 6, k=5x5, pad=2, stride=1)` + `Relu`  
   Выход: `[1, 6, 28, 28]`
2. **S2**: `Conv(6 -> 6, k=2x2, stride=2)`  
   Выход: `[1, 6, 14, 14]`
3. **C3**: `Conv(6 -> 16, k=5x5, pad=0, stride=1)` + `Relu`  
   Выход: `[1, 16, 10, 10]`
4. **S4**: `Conv(16 -> 16, k=2x2, stride=2)`  
   Выход: `[1, 16, 5, 5]`
5. **C5**: `Conv(16 -> 120, k=5x5, pad=0, stride=1)` + `Relu`  
   Выход: `[1, 120, 1, 1]`
6. **F6**-приближение: `Conv(120 -> 84, k=1x1)` + `Relu`  
   Выход: `[1, 84, 1, 1]`
7. **Output**: `Conv(84 -> 10, k=1x1)`  
   Выход: `[1, 10, 1, 1]`

Выход трактуется как 10 логитов классов.

## Файлы для инференса

- Генерация ONNX: [end2end/gen.py](end2end/gen.py)
- PyTorch-референс для тех же весов ONNX: [end2end/pytorch_reference.py](end2end/pytorch_reference.py)
- Компилятор: [build/tcompiler](build/tcompiler)
- Драйвер инференса: [src/driver.cpp](src/driver.cpp)

## Процесс запуска

> Требуется установить зависимости из [../readme.md](../readme.md)

Ниже команды от генерации модели до запуска скомпилированной сети. 
### Сборка `build/tcompiler`

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

### Генерация ONNX и PyTorch-референс
```bash
source .venv/bin/activate
python end2end/gen.py
python end2end/pytorch_reference.py
```

Пример вывода:

```shell
Model saved: .../models/model.onnx
PyTorch output shape: (1, 10, 1, 1)
PyTorch output:
 [-0.04864708 -0.07086591  0.06295342 -0.032887    0.1736427  -0.3868878
 -0.29219002  0.02005062 -0.09745298  0.53403354]
Class index: 9
```

### Компиляция ONNX в объектный файл и запуск драйвера

```bash
mkdir -p obj
./build/tcompiler models/model.onnx -o obj/model.o
g++ -std=c++20 -fPIE -c src/driver.cpp -o obj/driver.o
g++ obj/driver.o obj/model.o -o obj/model_run -lmlir_c_runner_utils
./obj/model_run
```

Пример вывода:

```shell
Creating computation graph...
Converting graph to MLIR...
Converting MLIR to LLVM IR...
Compiling LLVM IR to object...
Object file writed: obj/model.o
Runtime: 91645 us
Input [1, 1, 28, 28] (first row): 
0 0.00127714 0.00255428 0.00383142 ... 3 0.0319285 0.0332056 0.0344828
Output [10]: 
-0.0574244 -0.0795071 0.0802596 -0.0159505 0.172787 -0.401966 -0.290615 0.0210975 -0.111618 0.581323
Class index: 9
Runtime: 1215 us
```

На вход в `src/driver.cpp` подаются тензор формы `[1, 1, 28, 28]` и значения в диапазоне `[0, 1]` по индексам `x[0,0,i,j] = (i * 28 + j) / 783.0`.

Выход сети представляет тензор `Y` формы `[1, 10, 1, 1]`. В драйвере индекс предсказанного класса получается применением `argmax` по логитам.

## Особенности

При одинаковом входе логиты в `pytorch_reference.py` и в `obj/model_run` могут незначительно отличаться. Это ожидаемое поведение для `float32` и разных backend-реализаций. Порядок суммирования в PyTorch и MLIR/LLVM в связи с неассоциативность операций с плавающей точкой в общем случае вызывает накопление небольших численных различий по слоям, особенно после `ReLU`.

Для того, чтобы избежать больших различий в логитах следует контролировать величину весов в генераторе модели.
