#define MLIR_CIFACE_PREFIX _mlir_ciface_sgemm_vs_

#include "memref.hpp"
#include "utils.h"

#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

#define SGEMM_LMUL 32

DECLARE_MLIR_CIFACE(SGEMM_LMUL, size_t M, size_t N, size_t K,
                    MemRef<float, 2> *A, MemRef<float, 2> *B,
                    MemRef<float, 2> *C);

// A: MxK | B: KxN | C: MxN
// C = A · B
#define _SIZE_M 8
#define _SIZE_N 4096
#define _SIZE_K 32

float dataA[_SIZE_M * _SIZE_K];
float dataB[_SIZE_K * _SIZE_N];
float dataC[_SIZE_M * _SIZE_N];

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <number>\n", argv[0]);
    return 1; // Indicate an error
  }
  int count = atoi(argv[1]);

  intptr_t sizesA[2] = {_SIZE_M, _SIZE_K};
  intptr_t sizesB[2] = {_SIZE_K, _SIZE_N};
  intptr_t sizesC[2] = {_SIZE_M, _SIZE_N};

  MemRef<float, 2> memrefA(dataA, sizesA);
  MemRef<float, 2> memrefB(dataB, sizesB);
  MemRef<float, 2> memrefC(dataC, sizesC);

  double startTime = (double)std::clock() / CLOCKS_PER_SEC;

  for (int i = 0; i < count; i++) {
    USE_MLIR_CIFACE(SGEMM_LMUL, _SIZE_M, _SIZE_N, _SIZE_K, &memrefA, &memrefB,
                    &memrefC);
  }

  double endTime = (double)std::clock() / CLOCKS_PER_SEC;
  double timeElapsed = (endTime - startTime) / (double)count;

  std::cout << "Average: " << std::fixed << timeElapsed << " secs" << std::endl;
  return 0;
}
