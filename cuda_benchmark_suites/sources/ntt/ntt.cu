#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <iostream>

#include "ntt.cuh"

__device__ inline int32_t reverse_bits(int32_t n, int32_t l) {
  int32_t rev_n = 0;
  for (int i = 0; i < l; ++i) {
    if ((n >> i) & 1) {
      rev_n |= 1 << (l - 1 - i);
    }
  }
  return rev_n;
}

constexpr int MAX_THREADS = 1024;
constexpr int BUF_SIZE = 2;

__global__ void ntt_kernel(const int32_t *__restrict__ array,
                           int32_t l,
                           const int32_t *__restrict__ twiddle,
                           int32_t p,
                           int32_t *__restrict__ dst) {

  // Shared memory for data (N elements) and twiddle factors (l elements)
  extern __shared__ int32_t s_mem[];
  int32_t *s_data = s_mem;
  int32_t *s_twiddle = &s_mem[(1 << l)];
  int32_t N = 1 << l;
  int32_t jobs_per_thread = N / 2 > blockDim.x ? N / 2 / blockDim.x : 1;

  const int32_t tidx = threadIdx.x;

  // initialize twiddle coefficients
  for (int t = tidx * jobs_per_thread; t < (tidx + 1) * jobs_per_thread; t++) {
    s_twiddle[t] = twiddle[t];
  }
  // load and bit-reversal shuffle array into s_data
  for (int t = tidx * jobs_per_thread * 2; t < (tidx + 1) * jobs_per_thread * 2; t++) {
    s_data[reverse_bits(t, l)] = array[t];
  }
  __syncthreads();

  // butterfly Stages
  for (int s = 0; s <= l - 1; ++s) {
    if (jobs_per_thread == 1) {
      int t = tidx;
      // shift (l-s) highest bits by 1
      int32_t x_idx = (t << 1) - (t & ((1 << s) - 1));
      int32_t y_idx = x_idx + (1 << s);
      int32_t twiddle_idx = (t << (l - 1 - s)) & ((1 << (l - 1)) - 1);
      // Read current value into register
      int32_t x = s_data[x_idx];
      int32_t y = s_data[y_idx];
      int32_t twiddle_coef = s_twiddle[twiddle_idx];

      __syncthreads();

      int32_t new_x = (x + y * twiddle_coef) % p;
      int32_t new_y = ((x - y * twiddle_coef) % p + p) % p;  // to ensure positive result

      s_data[x_idx] = new_x;
      s_data[y_idx] = new_y;
      __syncthreads();
    } else {
      int32_t x_buf[BUF_SIZE], y_buf[BUF_SIZE];
      for (int j = 0; j < jobs_per_thread; j++) {
        int t = tidx * jobs_per_thread + j;
        int32_t x_idx = (t << 1) - (t & ((1 << s) - 1));
        int32_t y_idx = x_idx + (1 << s);
        int32_t twiddle_idx = (t << (l - 1 - s)) & ((1 << (l - 1)) - 1);
        int32_t x = s_data[x_idx];
        int32_t y = s_data[y_idx];
        int32_t twiddle_coef = s_twiddle[twiddle_idx];
        int32_t new_x = (x + y * twiddle_coef) % p;
        int32_t new_y = ((x - y * twiddle_coef) % p + p) % p;  // to ensure positive result
        x_buf[j] = new_x;
        y_buf[j] = new_y;
      }

      __syncthreads();

      for (int j = 0; j < jobs_per_thread; j++) {
        int t = tidx * jobs_per_thread + j;
        int32_t x_idx = (t << 1) - (t & ((1 << s) - 1));
        int32_t y_idx = x_idx + (1 << s);

        s_data[x_idx] = x_buf[j];
        s_data[y_idx] = y_buf[j];
      }
      __syncthreads();
    }
  }

  for (int t = tidx * jobs_per_thread * 2; t < (tidx + 1) * jobs_per_thread * 2; t++) {
    dst[t] = s_data[t];
  }
}

void ntt(const int32_t *array, int32_t l, const int32_t *twiddle, int32_t p, int32_t *dst, cudaStream_t stream) {
  const int32_t N = 1 << l;
  size_t shared_mem_size = (N + N / 2) * sizeof(int32_t);
  auto block_dim = std::min(MAX_THREADS, N / 2);
  if (N / 2 / MAX_THREADS > 2) {
    std::cerr << "arg not supported";
    exit(1);
  }
  ntt_kernel<<<1, block_dim, shared_mem_size, stream>>>(array, l, twiddle, p, dst);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }
}
