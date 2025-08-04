#define MLIR_CIFACE_PREFIX _mlir_ciface_pack_vl_

#define PACK_STEP 256

#include <ctime>
#include <iomanip>
#include <iostream>

#include "memref.hpp"
#include "utils.h"

DECLARE_MLIR_CIFACE(PACK_STEP, MemRef<uint8_t, 1> *src,
                    MemRef<uint32_t, 1> *dest, int32_t src_size,
                    int32_t dest_size);

#define _SIZE_N 65536
#define _SIZE_4_N 262144

uint8_t allocSrc[_SIZE_4_N];
uint32_t allocDest[_SIZE_N];

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <number>\n", argv[0]);
    return 1; // Indicate an error
  }
  int count = atoi(argv[1]);

  // MLIR data container configuration
  static int64_t srcSizes[1] = {_SIZE_4_N};
  static int64_t destSizes[1] = {_SIZE_N};

  MemRef<uint8_t, 1> memrefSrc(allocSrc, srcSizes);
  MemRef<uint32_t, 1> memrefDest(allocDest, destSizes);

  double startTime = (double)std::clock() / CLOCKS_PER_SEC;

  for (int i = 0; i < count; i++) {
    USE_MLIR_CIFACE(PACK_STEP, &memrefSrc, &memrefDest, _SIZE_4_N, _SIZE_N);
  }

  double endTime = (double)std::clock() / CLOCKS_PER_SEC;
  double timeElapsed = (endTime - startTime) / (double)count;

  std::cout << "Average: " << std::fixed << timeElapsed << " secs" << std::endl;
  return 0;
}
