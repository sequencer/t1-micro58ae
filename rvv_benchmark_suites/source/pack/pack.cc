#define MLIR_CIFACE_PREFIX _mlir_ciface_pack_vl_

#ifndef PACK_STEP
#error "PACK_STEP is not defined"
#endif

#include "memref.hpp"
#include "utils.h"

DECLARE_MLIR_CIFACE(PACK_STEP, MemRef<uint8_t, 1> *src,
                    MemRef<uint32_t, 1> *dest, int32_t src_size,
                    int32_t dest_size);

#define _SIZE_N 65536
#define _SIZE_4_N 262144

__attribute((section("vdata"))) uint8_t allocSrc[_SIZE_4_N];
__attribute((section("vdata"))) uint32_t allocDest[_SIZE_N];

extern "C" void test(void) {
  // MLIR data container configuration
  static int32_t srcSizes[1] = {_SIZE_4_N};
  static int32_t destSizes[1] = {_SIZE_N};

  MemRef<uint8_t, 1> memrefSrc(allocSrc, srcSizes);
  MemRef<uint32_t, 1> memrefDest(allocDest, destSizes);

  USE_MLIR_CIFACE(PACK_STEP, &memrefSrc, &memrefDest, _SIZE_4_N, _SIZE_N);
}
