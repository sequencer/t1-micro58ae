#define MLIR_CIFACE_PREFIX _mlir_ciface_saxpy_vl_

#ifndef SAXPY_LMUL
#error "SAXPY_LMUL is not defined"
#endif

#include "memref.hpp"
#include "utils.h"

#define _SIZE_N 19999

DECLARE_MLIR_CIFACE(SAXPY_LMUL, int32_t N, MemRef<float, 1> *X,
                    MemRef<float, 1> *Y, float A)

__attribute((section("vdata"))) float dataX[_SIZE_N];
__attribute((section("vdata"))) float dataY[_SIZE_N];

extern "C" void test(void) {
  float *X = nullptr;
  const float A = 5.5;
  float *scalarY = nullptr;

  // MLIR data container configuration
  static int32_t sizesX[1] = {_SIZE_N};
  static int32_t sizesY[1] = {_SIZE_N};
  MemRef<float, 1> memrefX(dataX, sizesX);
  MemRef<float, 1> memrefY(dataY, sizesY);

  USE_MLIR_CIFACE(SAXPY_LMUL, _SIZE_N, &memrefX, &memrefY, A);
}
