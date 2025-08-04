#!/usr/bin/env bash

cd ./cuda_benchmark_suites/sources
nix build --out-link result
mkdir -p ../bin
cp ./result/bin/*_bench ../bin/
rm result
