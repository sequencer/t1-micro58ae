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
  result=$(LD_LIBRARY_PATH=./cuda_benchmark_suites/lib ./cuda_benchmark_suites/bin/${case}_bench.elf)
  tput rc
  tput el
  echo "$result\n"
  echo
done
