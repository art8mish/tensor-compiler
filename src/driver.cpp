#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <cstring>
#include <iostream>
#include <ctime>

namespace {

using InputMemRef = StridedMemRefType<float, 4>;
using OutputMemRef = StridedMemRefType<float, 4>;

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

extern "C" void _mlir_ciface_forward(InputMemRef *output, OutputMemRef *input);

int main() {
    alignas(64) float in_storage[1 * 1 * 5 * 5];
    alignas(64) float out_storage[1 * 1 * 3 * 3];

    std::memset(in_storage, 0, sizeof(in_storage));
    std::memset(out_storage, 0, sizeof(out_storage));
    in_storage[0] = 1.0f;

    
    const int64_t id0 = 1, id1 = 1, id2 = 5, id3 = 5;
    const int64_t in_n = id0 * id1 * id2 * id3;
    InputMemRef in{};

    const int64_t od0 = 1, od1 = 1, od2 = 3, od3 = 3;
    const int64_t out_n = od0 * od1 * od2 * od3;
    OutputMemRef out{};
    fill_strided(in, in_storage, id0, id1, id2, id3);
    fill_strided(out, out_storage, od0, od1, od2, od3);

    
    std::cout << "Input [" << id0 << ", " << id1 << ", " << id2 << ", " << id3 << "]:\n";
    for (int i = 0; i < in_n; ++i) {
        std::cout << in_storage[i];
        if (i + 1 < in_n)
            std::cout << ' ';
    }
    std::cout << "\n";

#ifndef NDEBUG
    auto start_time = std::clock();
#endif
    _mlir_ciface_forward(&out, &in);
#ifndef NDEBUG
    auto duration = std::clock() - start_time;
#endif
    
    std::cout << "Output [" << od0 << ", " << od1 << ", " << od2 << ", " << od3 << "]:\n";
    float* out_data = out.data;
    for (int i = 0; i < out_n; ++i) {
        std::cout << out_data[i];
        if (i + 1 < out_n)
            std::cout << ' ';
    } 
    std::cout << std::endl;

#ifndef NDEBUG
    std::cout << "Runtime: " << duration << " us" << std::endl;
#endif
    return 0;
}
