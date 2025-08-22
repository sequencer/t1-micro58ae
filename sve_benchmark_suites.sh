#!/usr/bin/env bash

vendorID=$(lscpu | grep "Vendor ID" | cut -d':' -f2 | xargs)
if [[ "$vendorID" != "HiSilicon" ]]; then
  echo "Host machine does not looks like an KP920" >&2
  exit 1
fi

make sve_benchmark_suites/bin/pack.elf
sve_benchmark_suites/bin/pack.elf 10000
make sve_benchmark_suites/bin/quant.elf
sve_benchmark_suites/bin/quant.elf 10000
make sve_benchmark_suites/bin/saxpy.elf
sve_benchmark_suites/bin/saxpy.elf 10000
make sve_benchmark_suites/bin/sgemm.elf
sve_benchmark_suites/bin/sgemm.elf 10000