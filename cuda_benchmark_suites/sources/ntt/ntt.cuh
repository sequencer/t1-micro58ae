#pragma once

#include <cstdint>

// array is of length n=2^l, p is a prime number
// twiddle[i] = g^i where i is an order-n element mod p
// 32bit * n <= VLEN * 8 => n <= VLEN / 4
void ntt(const int32_t *array, int32_t l, const int32_t *twiddle, int32_t p, int32_t *dst, cudaStream_t stream);
