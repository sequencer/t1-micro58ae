= T1 MICRO58 Artifact Evaluation

This readme guides the reviewer to reproduce the results in the T1 Paper.

For the evaluation, all software dependencies should installed.
To elaborate the baseline results, hardware dependencies should be available.

== Prerequisite

- Nix 2.28

== T1 RTL and Emulator
=== T1 RTL Generation
All RTLs has been elaborated and store in the `rtls` folder:

Reviewers can reproduce the RTL generation by executing the shell script `./rtl.sh`.
It's a wrapper to nix command to use #link("https://github.com/chipsallliance/t1")[chipsallliance/t1] to generate RTLs.

=== T1 Emulator Generation
All T1 Emulators can be compiled via Synopsys VCS with the dependency of Synopsys FusionCompiler.
These software paths:
```sh
/opt/synopsys/vcs/V-2023.12-SP2/
/opt/synopsys/fusioncompiler/V-2023.12-SP5/
```
Please make sure they are correctly installed.

Due to the limitation of commercial license, the emulator binary is not stored in the repository.
However, reviewers can reproduce the RTL generation by executing the shell script `./emulator.sh`.

We will provided emulator support with following configuration:

- `DLEN 128  VLEN 512` (`benchmark_dlen128_vlen512`)
- `DLEN 128  VLEN 1024` (`benchmark_dlen128_vlen1024`)
- `DLEN 256  VLEN 1024 w/ DRAM` (`benchmark_dlen256_vlen1024`)
- `DLEN 256  VLEN 1024 w/o DRAM` (`benchmark_dlen256_vlen1024`)
- `DLEN 512  VLEN 1024` (`benchmark_dlen512_vlen1024`)
- `DLEN 512  VLEN 2048` (`benchmark_dlen512_vlen2048`)
- `DLEN 1024 VLEN 1024` (`benchmark_dlen1024_vlen1024`)
- `DLEN 1024 VLEN 4096 w/ DRAM` (`benchmark_dlen1024_vlen4096`)
- `DLEN 1024 VLEN 4096 w/o DRAM` (`benchmark_dlen1024_vlen4096`)

== Benchmarks Generation
=== RVV Benchmarks Generation
All RVV test cases are reproducible by executing the shell script `./rvv_benchmark_suites.sh`.
It's a wrapper to compile the programs in `benchmarks` into RVV codes, all possible programs are:

- `Memset`
- `ASCII to UTF-32`
- `Byteswap`
- `Linear Normalization`
- `SAXPY`
- `Matmul`
- `NTT`
- `MMM`
- `SGEMM`
- `QUANT`
- `PACK`

We already precompiled them and stored in `rvv_benchmark_suites` folder.
They are designed to run on T1 and SpacemiT K1.

=== SVE Benchmarks Generation
All SVE test cases are reproducible by executing the shell script `./rvv_benchmark_suites.sh`.
It's a wrapper to compile the programs in `benchmarks` into SVE codes, all possible programs are:
`SGEMM`, `QUANT`, `SAXPY`, `PACK`.
We already precompiled them and stored in `sve_benchmark_suites` folder.
They are designed to run on Hisilicon KP920.

=== CUDA Benchmarks Generation
All CUDA test cases are reproducible by executing the shell script `./cuda_benchmark_suites.sh`.
It's a wrapper to compile the programs in `benchmarks` into NVIDIA GP-GPU kernels, all possible programs are:
`NTT`, `MMM`.
We already precompiled them and stored in `cuda_benchmark_suites` folder.
They are designed to run on NVIDIA 3090 and NVIDIA 5090.

== Execution

=== T1
Execution workloads in T1 is simple with shell script `./run_emulator.sh`.

```bash
# Usage: ./run_emulator.sh <config-name> <case-name>
./run_emulator.sh benchmark_dlen1024_vlen4096 pack
```

=== Hisilicon KP920
Hisilicon KP920 is accessible via Huawei cloud.

=== SpacemiT K1
SpacemiT K1 is available publicly.

=== NVIDIA 3090 & 5090
NVIDIA 3090 & 5090 is available publicly.

== Results

=== Baseline Area
There are two baseline die photos: GA102 and KP920, they has been masured and labeled in the corresponding die photos.
