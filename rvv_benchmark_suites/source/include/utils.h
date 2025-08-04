#ifndef BUDDY_BENCH_UTILS_H
#define BUDDY_BENCH_UTILS_H

#ifndef MLIR_CIFACE_PREFIX
#error "MLIR_CIFACE_PREFIX is not defined"
#endif

#define JOIN(a, b) JOIN_H(a, b)
#define JOIN_H(a, b) a##b

#define DECLARE_MLIR_CIFACE(VL, ...) extern "C" void JOIN(MLIR_CIFACE_PREFIX, VL)(__VA_ARGS__);

#define USE_MLIR_CIFACE(VL, ...) JOIN(MLIR_CIFACE_PREFIX, VL)(__VA_ARGS__);

#endif
