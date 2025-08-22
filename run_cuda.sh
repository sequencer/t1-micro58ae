#!/usr/bin/env bash

cases=(
  "ntt"
  "mmm"
)

for case in "${cases[@]}"; do
  make ./cuda_benchmark_suites/bin/${case}_bench.elf
  printf "$case: "
  tput sc
  printf "Running"
  LD_LIBRARY_PATH=./cuda_benchmark_suites/lib result=$(./cuda_benchmark_suites/bin/${case}_bench.elf)
  tput rc
  tput el
  printf "$result\n"
  echo
done
