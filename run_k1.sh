#!/usr/bin/env bash

cases=(
  'sgemm'
  'saxpy'
  'quant'
  'pack'
)

for ca in "${cases[@]}"; do
  make "rvv_benchmark_suites/lib_rv64/$ca.elf"
  ./rvv_benchmark_suites/lib_rv64/$ca.elf 10000
done
