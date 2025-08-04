#include <cuda/std/chrono>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include <nvbench/nvbench.cuh>

#include <bit>

#include "ntt.cuh"

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

void mmm_benchmark(nvbench::state &state) {
  auto N = state.get_int64("N");
  auto l = std::bit_width((uint64_t) N) - 1;
  int32_t p = 12289;

  thrust::device_vector<int32_t> array_d = random_u32_vec_in_u16(N);

  thrust::device_vector<int32_t> twiddle_d = random_u32_vec_in_u16(N / 2);

  thrust::device_vector<int32_t> Z_d(N);

  state.add_element_count(l, "L");
  state.exec([&](nvbench::launch &launch) {
    ntt(
      array_d.data().get(),
      l,
      twiddle_d.data().get(),
      p,
      Z_d.data().get(),
      launch.get_stream()
    );
  });
}

NVBENCH_BENCH(mmm_benchmark)
.set_name("ntt")
.set_timeout(3)
.add_int64_axis("N", {256, 512, 1024, 2048, 4096});
