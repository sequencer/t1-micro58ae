#!/usr/bin/env bash

vendorID=$(lscpu | grep "Vendor ID" | cut -d':' -f2 | xargs)
if [[ "$vendorID" != "HiSilicon" ]]; then
  echo "Host machine does not looks like an KP920" >&2
  exit 1
fi

make sve_benchmark_suites/bin/pack.elf
make sve_benchmark_suites/bin/quant.elf
make sve_benchmark_suites/bin/saxpy.elf
make sve_benchmark_suites/bin/sgemm.elf
