#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "mmm.cuh"

constexpr int LEN = 8;
uint32_t M[LEN] = {
  0xe3e7, 0x0682, 0xc209, 0x4cac, 0x629f, 0x6fbe, 0xd82c, 0x07cd,
};
uint32_t X[LEN] = {
  0x82e2, 0xe662, 0xf728, 0xb4fa, 0x4248, 0x5e3a, 0x0a5d, 0x2f34,
};
uint32_t Y[LEN] = {
  0xd471, 0x3d60, 0xc8a7, 0x0639, 0xeb11, 0x67b3, 0x67a9, 0xc378,
};

int main() {
  thrust::device_vector<uint32_t> X_dev(LEN);
  thrust::copy(X, X + LEN, X_dev.begin());

  thrust::device_vector<uint32_t> Y_dev(LEN);
  thrust::copy(Y, Y + LEN, Y_dev.begin());

  thrust::device_vector<uint32_t> M_dev(LEN);
  thrust::copy(M, M + LEN, M_dev.begin());

  thrust::device_vector<uint32_t> Z_dev(LEN);

  mmm(X_dev.data().get(), Y_dev.data().get(), M_dev.data().get(),
      LEN, 0xc2fb, Z_dev.data().get(), /* stream= */0);

  thrust::host_vector<int> Z_h = Z_dev;
  std::cout << "output = [";
  for (size_t i = 0; i < Z_h.size(); ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << std::hex << Z_h[i];
  }
  std::cout << "]" << std::endl;
}
