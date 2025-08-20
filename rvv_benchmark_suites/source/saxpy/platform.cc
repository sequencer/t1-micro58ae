#define MLIR_CIFACE_PREFIX _mlir_ciface_saxpy_vl_

#define SAXPY_LMUL 8

#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

#include "memref.hpp"
#include "utils.h"

#define _SIZE_N 19999

DECLARE_MLIR_CIFACE(SAXPY_LMUL, int32_t N, MemRef<float, 1> *X,
                    MemRef<float, 1> *Y, float A)

float dataX[_SIZE_N];
float dataY[_SIZE_N];

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <number>\n", argv[0]);
    return 1; // Indicate an error
  }
  int count = atoi(argv[1]);

  double startTime = (double)std::clock() / CLOCKS_PER_SEC;

  float *X = nullptr;
  const float A = 5.5;
  float *scalarY = nullptr;

  // MLIR data container configuration
  static int64_t sizesX[1] = {_SIZE_N};
  static int64_t sizesY[1] = {_SIZE_N};

  MemRef<float, 1> memrefX(dataX, sizesX);
  MemRef<float, 1> memrefY(dataY, sizesY);

  for (int i = 0; i < count; i++) {
    USE_MLIR_CIFACE(SAXPY_LMUL, _SIZE_N, &memrefX, &memrefY, A);
  }

  double endTime = (double)std::clock() / CLOCKS_PER_SEC;
  double timeElapsed = (endTime - startTime) / (double)count;

  std::cout << "Average: " << std::fixed << timeElapsed << " secs" << std::endl;
  return 0;
}
