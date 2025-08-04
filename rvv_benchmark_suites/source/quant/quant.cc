#define MLIR_CIFACE_PREFIX _mlir_ciface_quant_vl_

#ifndef QUANT_LMUL
#error "QUANT_LMUL is not defined"
#endif

#include "memref.hpp"
#include "utils.h"

DECLARE_MLIR_CIFACE(QUANT_LMUL, MemRef<float, 1> *data_memref,
                    MemRef<int8_t, 1> *quant_memref, size_t size, float scale,
                    int8_t zero_point);

#define _SIZE_N 400000

// MLIR data container configuration
__attribute((section("vdata"))) float data[_SIZE_N];
__attribute((section("vbss"))) int8_t quantized_data[_SIZE_N];

extern "C" int test() {
  static float scale = 0.1f;
  static int8_t zero_point = 0;

  static int32_t sizes[1] = {_SIZE_N};

  MemRef<float, 1> data_memref(data, sizes);
  MemRef<int8_t, 1> quantized_data_memref(quantized_data, sizes);

  USE_MLIR_CIFACE(QUANT_LMUL, &data_memref, &quantized_data_memref, _SIZE_N,
                  scale, zero_point)

  return 0;
}
