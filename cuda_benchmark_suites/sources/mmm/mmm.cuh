#pragma once

#include <cstdint>

void mmm(const uint32_t *X, const uint32_t *Y, const uint32_t *M, int32_t n,
         uint32_t minus_M_inverse_mod_r, uint32_t *Z, cudaStream_t stream);
