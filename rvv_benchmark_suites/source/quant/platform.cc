#define MLIR_CIFACE_PREFIX _mlir_ciface_quant_vl_

#define QUANT_LMUL 16

#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

#include "memref.hpp"
#include "utils.h"

DECLARE_MLIR_CIFACE(QUANT_LMUL, MemRef<float, 1> *data_memref,
                    MemRef<int8_t, 1> *quant_memref, size_t size, float scale,
                    int8_t zero_point);

#define _SIZE_N 400000

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <number>\n", argv[0]);
    return 1; // Indicate an error
  }
  int count = atoi(argv[1]);

  double start = (double)std::clock() / CLOCKS_PER_SEC;

  static const float scale = 0.1f;
  static const int8_t zero_point = 0;

  static const int64_t sizes[1] = {_SIZE_N};

  float data[_SIZE_N];
  MemRef<float, 1> data_memref(data, sizes);

  int8_t quantized_data[_SIZE_N];
  MemRef<int8_t, 1> quantized_data_memref(quantized_data, sizes);

  for (int i = 0; i < count; i++) {
    USE_MLIR_CIFACE(QUANT_LMUL, &data_memref, &quantized_data_memref, _SIZE_N,
                    scale, zero_point)
  }

  double end = (double)std::clock() / CLOCKS_PER_SEC;

  double timeElapsed = (end - start) / (double)count;

  std::cout << "Average: " << std::fixed << timeElapsed << " secs" << std::endl;
  return 0;
}
