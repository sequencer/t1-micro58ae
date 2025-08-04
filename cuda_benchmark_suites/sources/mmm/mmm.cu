#include <cstdio>

#include "mmm.cuh"

__global__ void mmm_kernel(const uint32_t *X, const uint32_t *Y, const uint32_t *M, int32_t n,
                           uint32_t minus_M_inverse_mod_r, uint32_t *Z_out) {
  extern __shared__ uint32_t Z_s[];  // of length (2 * n + 1)

  // Z_s is a big integer represented by n uint16_t parts, and `z_offset` represents its MSB
  // i.e. hence LSB in `z_offset` + n - 1
  // sharedZ_offset is initialized at n to allow it to shift left by n times and allow one overflow position
  int z_offset = n + 1;

  auto idx = threadIdx.x;

  auto propagate = [&]() {
    auto cur_part = Z_s[z_offset + idx];
    auto higher_part_lower = Z_s[z_offset + idx - 1] & ((1 << 16) - 1);
    __syncthreads();
    Z_s[z_offset + idx - 1] = (cur_part >> 16) + higher_part_lower;
    if (idx == n - 1) Z_s[z_offset + idx] = cur_part & ((1 << 16) - 1);  // handle LSB separately
    __syncthreads();
  };

  // initialize Z_s in a distributive way
  Z_s[2 * idx] = Z_s[2 * idx + 1] = 0;
  if (idx == n - 1) Z_s[2 * idx + 2] = 0;
  __syncthreads();

  for (int i = 0; i < n; i++) {
    auto yi = Y[n - 1 - i];
    Z_s[z_offset + idx] += yi * X[idx];
    __syncthreads();

    propagate();

    auto q = (Z_s[z_offset + n - 1] * minus_M_inverse_mod_r) & ((1 << 16) - 1);
    Z_s[z_offset + idx] += q * M[idx];
    __syncthreads();

    propagate();

    z_offset--;
  }

  propagate();

  Z_out[idx] = Z_s[z_offset + idx];
}

// X, Y and M are arrays *on device* of uint16_t zero-extended into uint32_t of length n,
// representing a big integer, with MSB first (big endian),
// R = r^n, i.e. the numbers are 4 * n digits
// X, Y < M < R
void mmm(const uint32_t *X, const uint32_t *Y, const uint32_t *M, int32_t n,
         uint32_t minus_M_inverse_mod_r, uint32_t *Z, cudaStream_t stream) {
  mmm_kernel<<<1, n, (2 * n + 1) * sizeof(uint32_t), stream>>>(
    X, Y, M, n, minus_M_inverse_mod_r, Z
  );
}
