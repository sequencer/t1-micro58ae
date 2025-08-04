#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "ntt.cuh"

constexpr int l = 6;
constexpr int LEN = 1 << l;

int32_t array[LEN] = {
  9997, 6362, 7134, 11711, 5849, 9491, 5972, 4164, 5894, 11069, 7697,
  8319, 2077, 12086, 10239, 5394, 4898, 1370, 1205, 2997, 5274,
  4625, 11983, 1789, 3645, 7666, 12128, 10883, 7376, 8883, 2321,
  1889, 2026, 8059, 2741, 865, 1785, 9955, 2395, 9330, 11465, 7383,
  9649, 11285, 3647, 578, 1158, 9936, 12019, 11114, 7894, 4832,
  10148, 10363, 11388, 9122, 10758, 2642, 4171, 10586, 1194, 5280,
  3055, 9220
};
int32_t twiddle[LEN / 2] = {
  1, 7311, 5860, 3006, 4134, 5023, 3621, 2625, 8246, 8961, 1212, 563, 11567, 5728, 8785, 4821, 1479, 10938, 3195, 9545,
  6553, 6461, 9744, 11340, 5146, 5777, 10643, 9314, 1305, 4591, 3542, 2639
};
int32_t p = 12289;

#define RAW(p) thrust::raw_pointer_cast((p).data())

#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err));                                     \
        /* Consider adding exit(EXIT_FAILURE) or similar error handling */    \
    }                                                                         \
} while (0)

int main() {
  thrust::device_vector<int32_t> array_d(std::size(array));
  thrust::copy(array, array + std::size(array), array_d.begin());

  thrust::device_vector<int32_t> twiddle_d(std::size(twiddle));
  thrust::copy(twiddle, twiddle + std::size(twiddle), twiddle_d.begin());

  thrust::device_vector<int32_t> Z_d(array_d.size());

  ntt(RAW(array_d), l, RAW(twiddle_d), p, RAW(Z_d), /* stream= */0);
  CUDA_CHECK(cudaDeviceSynchronize());

  // print output
  thrust::host_vector<int> Z_h = Z_d;
  std::cout << "output = [";
  for (size_t i = 0; i < Z_h.size(); ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << Z_h[i];
  }
  std::cout << "]" << std::endl;
}
