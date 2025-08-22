#!/usr/bin/env bash

pushd ./cuda_benchmark_suites/sources
nix build --out-link result
mkdir -p ../bin
mkdir -p ../lib
cp /usr/lib/x86_64-linux-gnu/{libcuda.so*,libnvidia-ml*} ../lib
cp ./result/bin/*_bench ../bin/
rm result
popd
