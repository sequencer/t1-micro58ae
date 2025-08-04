#define MLIR_CIFACE_PREFIX _mlir_ciface_sgemm_vs_

#include "memref.hpp"
#include "utils.h"

#ifndef SGEMM_LMUL
#error "SGEMM_LMUL is not defined"
#endif

DECLARE_MLIR_CIFACE(SGEMM_LMUL, int32_t M, int32_t N, int32_t K,
                    MemRef<float, 2> *A, MemRef<float, 2> *B,
                    MemRef<float, 2> *C);

// A: MxK | B: KxN | C: MxN
// C = A · B
#define _SIZE_M 8
#define _SIZE_N 4096
#define _SIZE_K 32
__attribute((section("vdata"))) float dataA[_SIZE_M * _SIZE_K];
__attribute((section("vdata"))) float dataB[_SIZE_K * _SIZE_N];
__attribute((section("vdata"))) float dataC[_SIZE_M * _SIZE_N];

extern "C" void test(void) {

  // MLIR data container configuration
  static const int32_t sizesA[2] = {_SIZE_M, _SIZE_K};
  static const int32_t sizesB[2] = {_SIZE_K, _SIZE_N};
  static const int32_t sizesC[2] = {_SIZE_M, _SIZE_N};

  MemRef<float, 2> memrefA(dataA, sizesA);
  MemRef<float, 2> memrefB(dataB, sizesB);
  MemRef<float, 2> memrefC(dataC, sizesC);

  USE_MLIR_CIFACE(SGEMM_LMUL, _SIZE_M, _SIZE_N, _SIZE_K, &memrefA, &memrefB,
                  &memrefC);
}
