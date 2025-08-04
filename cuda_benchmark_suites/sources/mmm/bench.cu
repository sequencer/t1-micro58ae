#include <cuda/std/chrono>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include <nvbench/nvbench.cuh>

#include "mmm.cuh"

thrust::device_vector<uint32_t> random_u32_vec_in_u16(size_t len) {
  thrust::device_vector<uint32_t> v(len);
  thrust::transform(thrust::counting_iterator<uint32_t>(0),
                    thrust::counting_iterator<uint32_t>(len),
                    v.begin(),
                    [] __device__(uint32_t idx) {
                      thrust::default_random_engine rng;
                      thrust::uniform_int_distribution<uint32_t> dist(0, (1 << 16) - 1);
                      rng.discard(idx);
                      return dist(rng);
                    });
  return v;
}

__global__ void sleep_kernel(double seconds)
{
  const auto start = cuda::std::chrono::high_resolution_clock::now();
  const auto ns =
    cuda::std::chrono::nanoseconds(static_cast<nvbench::int64_t>(seconds * 1000 * 1000 * 1000));
  const auto finish = start + ns;

  auto now = cuda::std::chrono::high_resolution_clock::now();
  while (now < finish)
  {
    now = cuda::std::chrono::high_resolution_clock::now();
  }
}

void mmm_benchmark(nvbench::state &state) {
  auto bits = state.get_int64("bits");
  auto len = bits / 16;

  thrust::device_vector<uint32_t> X_dev = random_u32_vec_in_u16(len);

  thrust::device_vector<uint32_t> Y_dev = random_u32_vec_in_u16(len);

  thrust::device_vector<uint32_t> M_dev = random_u32_vec_in_u16(len);

  thrust::device_vector<uint32_t> Z_dev(len);

  state.exec([len, &X_dev, &Y_dev, &M_dev, &Z_dev](nvbench::launch &launch) {
    mmm(
      X_dev.data().get(),
      Y_dev.data().get(),
      M_dev.data().get(),
      len,
      0xf5c9,
      Z_dev.data().get(),
      launch.get_stream()
    );
  });
}

NVBENCH_BENCH(mmm_benchmark)
.set_name("mmm")
.set_timeout(1)
.add_int64_axis("bits", {1024, 2048, 4096, 8192});