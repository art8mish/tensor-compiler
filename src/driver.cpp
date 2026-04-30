#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <cstring>
#include <ctime>
#include <iostream>

namespace {

using ReturnMemRef = StridedMemRefType<float, 4>;
using InputMemRef = StridedMemRefType<float, 4>;

void fill_strided(InputMemRef &m, float *storage, const int64_t d0, const int64_t d1,
                  const int64_t d2, const int64_t d3) {
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

extern "C" void _mlir_ciface_forward(ReturnMemRef *output, InputMemRef *input);

int main() {
    alignas(64) float in_storage[1 * 1 * 28 * 28];
    alignas(64) float out_storage[1 * 10 * 1 * 1];

    std::memset(in_storage, 0, sizeof(in_storage));
    std::memset(out_storage, 0, sizeof(out_storage));

    for (int i = 0; i < 28; ++i)
        for (int j = 0; j < 28; ++j)
            in_storage[i * 28 + j] = static_cast<float>((i * 28 + j) / 783.0);

    const int64_t id0 = 1, id1 = 1, id2 = 28, id3 = 28;
    const int64_t in_n = id0 * id1 * id2 * id3;
    InputMemRef in{};

    const int64_t od0 = 1, od1 = 10, od2 = 1, od3 = 1;
    const int64_t out_n = od0 * od1 * od2 * od3;
    ReturnMemRef out{};
    fill_strided(in, in_storage, id0, id1, id2, id3);
    fill_strided(out, out_storage, od0, od1, od2, od3);

    std::cout << "Input [" << id0 << ", " << id1 << ", " << id2 << ", " << id3 << "] (first row): " << "\n";
    for (int j = 0; j < 28; ++j) {
        if (j)
            std::cout << ' ';
        std::cout << in_storage[j];
    }
    std::cout << "\n";

#ifndef NDEBUG
    auto start_time = std::clock();
#endif
    _mlir_ciface_forward(&out, &in);
#ifndef NDEBUG
    auto duration = std::clock() - start_time;
#endif

    float *out_data = out.data;
    std::cout << "Output [" << out_n << "]:\n";
    int argmax = 0;
    float vmax = out_data[0];
    for (int64_t i = 0; i < out_n; ++i) {
        std::cout << ' ' << out_data[i];
        if (out_data[i] > vmax) {
            vmax = out_data[i];
            argmax = static_cast<int>(i);
        }
    }
    std::cout << "\nClass index: " << argmax << "\n";

#ifndef NDEBUG
    std::cout << "Runtime: " << duration << " us" << std::endl;
#endif
    return 0;
}
