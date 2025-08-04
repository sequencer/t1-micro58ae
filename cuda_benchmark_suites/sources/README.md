# CUDA Crypto

Some cuda kernels for crypto primitives.

## Usage

Ensure cuda toolkit and cudabench are installed. The project is tested with cuda 12.8.

```console
$ cmake -B build -DCMAKE_BUILD_TYPE=Release -GNinja
$ cmake --build build
$ ./build/mmm/mmm_bench
$ ./build/ntt/ntt_bench
```

For NixOS users the project can be directly built with nix

```console
$ nix build
$ ./result/mmm_bench
$ ./result/ntt_bench
```

