#!/usr/bin/env bash

cases=(
  "sgemm"
  "saxpy"
  "quant"
  "pack"
)

for case in "${cases[@]}"; do
  make ./sve_benchmark_suites/bin/$case.elf
  printf "$case: "
  tput sc
  printf "Running"
  result=$(./sve_benchmark_suites/bin/$case.elf 100000)
  tput rc
  tput el
  printf "$result\n"
  echo
done
