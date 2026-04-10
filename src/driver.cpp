#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <cstring>
#include <iostream>

namespace {

using InputMemRef = StridedMemRefType<float, 4>;
using OutputMemRef = StridedMemRefType<float, 4>;

constexpr int kOutElems = 1 * 1 * 3 * 3;

void fill_strided(InputMemRef &m, float *storage, const int64_t d0, const int64_t d1, const int64_t d2,
                  const int64_t d3) {
    m.basePtr = storage;
    m.data = storage;
    m.offset = 0;
    m.sizes[0] = d0;
    m.sizes[1] = d1;
    m.sizes[2] = d2;
    m.sizes[3] = d3;
    m.strides[3] = 1;
    m.strides[2] = d3;
    m.strides[1] = d2 * d3;
    m.strides[0] = d1 * d2 * d3;
}

} // namespace

extern "C" void _mlir_ciface_forward(InputMemRef *input, OutputMemRef *output);

int main() {
    alignas(64) float in_storage[1 * 1 * 5 * 5];
    alignas(64) float out_storage[1 * 1 * 3 * 3];

    std::memset(in_storage, 0, sizeof(in_storage));
    std::memset(out_storage, 0, sizeof(out_storage));
    in_storage[0] = 1.0f;

    InputMemRef in{};
    OutputMemRef out{};
    fill_strided(in, in_storage, 1, 1, 5, 5);
    fill_strided(out, out_storage, 1, 1, 3, 3);

    _mlir_ciface_forward(&in, &out);

    const int64_t d0 = 1, d1 = 1, d2 = 3, d3 = 3;
    std::cout << "output shape: [" << d0 << ", " << d1 << ", " << d2 << ", " << d3 << "]\n";
    std::cout << "output (row-major flat):\n  ";
    for (int i = 0; i < kOutElems; ++i) {
        std::cout << out_storage[i];
        if (i + 1 < kOutElems)
            std::cout << ' ';
    }
    std::cout << "\noutput (planes 0:0, rows x cols):\n";
    for (int h = 0; h < d2; ++h) {
        std::cout << "  ";
        for (int w = 0; w < d3; ++w) {
            const int idx = h * d3 + w;
            std::cout << out_storage[idx];
            if (w + 1 < d3)
                std::cout << ' ';
        }
        std::cout << '\n';
    }
    float s = 0.f;
    for (int i = 0; i < kOutElems; ++i)
        s += out_storage[i];
    std::cout << "output sum: " << s << '\n';

    return 0;
}
