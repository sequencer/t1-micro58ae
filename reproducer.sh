#!/usr/bin/env bash

set -o pipefail
set -o errexit

nix_add_path() {
  export PATH="$(nix build ".#$1" --no-link --print-out-paths)/bin:$PATH"
}

nix_add_path "jq.bin"
nix_add_path "util-linux.bin"
nix_add_path "python3"

readCycle() {
  jq -r '.total_cycles' ./sim_result.json
  rm ./sim_result.json
}

intToBool() {
  arg="$1"; shift

  if (( "$arg" )); then
    printf "TRUE"
  else
    printf "FALSE"
  fi
}

declare DLEN_CYCLE_DATA_TABLE
dlenReproduce() {
  echo "----------------------"
  echo "Reproducing DLEN Tests"
  echo "----------------------"

  cmpConfig=(
    "benchmark_dlen128_vlen1024_fp"
    "benchmark_dlen256_vlen1024_fp"
    "benchmark_dlen512_vlen1024_fp"
    "benchmark_dlen1024_vlen1024_fp"
  )
  cases=(
    "memset"
    "ascii_to_utf32"
    "byteswap"
    "linear_normalization"
    "saxpy_16"
    "saxpy_32"
    "sgemm_16"
    "sgemm_32"
  )


  DLEN_CYCLE_DATA_TABLE="Config,"
  declare -A header
  for case in "${cases[@]}"; do
    col=""
    if [[ $case == *_[[:digit:]]* ]]; then
      col="$(printf $case | cut -d'_' -f1)"
    else
      col="$case"
    fi

    if [[ -z "${header["$col"]}" ]]; then
      DLEN_CYCLE_DATA_TABLE+="$col,"
      header["$col"]="done"
    fi
  done
  unset header
  DLEN_CYCLE_DATA_TABLE+="\n"

  for cfg in "${cmpConfig[@]}"; do
    declare -A dataset

    for case in "${cases[@]}"; do
      EMU_TYPE=t1emu ./run_emulator.sh "$cfg" "$case"

      key=""
      if [[ $case == *_[[:digit:]]* ]]; then
        key="$(printf $case | cut -d'_' -f1)"
      else
        key="$case"
      fi

      cycle=$(readCycle)
      if [[ -z "${dataset["$key"]}" ]]; then
        dataset["$key"]="$cycle"
        continue
      fi

      if (( ${dataset["$key"]} > $cycle )); then
        dataset["$key"]="$cycle"
      fi
    done

    DLEN_CYCLE_DATA_TABLE+="$cfg,"
    for key in "${!dataset[@]}"; do
      DLEN_CYCLE_DATA_TABLE+="${dataset["$key"]},"
    done
    DLEN_CYCLE_DATA_TABLE+="\n"
  done
}


declare CHAINING_DATA_TABLE
chainingReproduce() {
  echo "-----------------------------"
  echo "Reproducing No Chaining Tests"
  echo "-----------------------------"

  chosen=benchmark_dlen256_vlen4096_fp
  case=rvv_benchmark_suites/bin/${chosen}.quant_16.elf
  make "$case"

  export EMU_TYPE=t1emu

  ./run_emulator.sh $chosen $case
  cycle_1=$(readCycle)

  ./run_emulator.sh disable_chaining $case
  cycle_2=$(readCycle)

  ./run_emulator.sh disable_memory_interleaving $case
  cycle_3=$(readCycle)

  CHAINING_DATA_TABLE="Type,Cycle,
Standard T1,$cycle_1,
Disable Chaining,$cycle_2,
Disable Memory Interleaving,$cycle_3,"

  unset EMU_TYPE
}


declare CRYPTO_DATA_TABLE
cryptoReproduce() {
  echo "------------------------"
  echo "Reproducing Crypto Bench"
  echo "------------------------"
  cmpConfig="benchmark_dlen1024_vlen16384_fp"
  cases=(
    "ntt_512"
    "ntt_1024"
    "ntt_2048"
    "ntt_4096"
    "mmm_1024"
    "mmm_2048"
    "mmm_4096"
    "mmm_8192"
  )


  freq=2.45
  CRYPTO_DATA_TABLE="Name,Cycle,Time Elapsed in Secs ($freq GHz)\n"

  for case in "${cases[@]}"; do
    DRAMSIM3_ENABLE=1 ./run_emulator.sh "$cmpConfig" "$case"
    cycle=$(readCycle)
    timeElapsed=$(python3 -c "print($cycle/($freq*1000_000_000))")
    CRYPTO_DATA_TABLE+="$case,$cycle,$timeElapsed\n"
  done
}


declare T1_KP920_CMP_DATA_TABLE
kp920CmpReproduce() {
  echo "----------------------------"
  echo "Reproducing T1-KP920 Compare"
  echo "----------------------------"
  cmpConfigs=(
    "benchmark_dlen256_vlen1024_fp,0"
    "benchmark_dlen1024_vlen4096_fp,0"
    "benchmark_dlen256_vlen1024_fp,1"
    "benchmark_dlen1024_vlen4096_fp,1"
  )
  cases=(
    "sgemm_16"
    "sgemm_32"
    "sgemm_64"
    "quant_8"
    "quant_16"
    "saxpy_8"
    "saxpy_16"
    "pack_256"
    "pack_1024"
  )
  freq=2.45

  T1_KP920_CMP_DATA_TABLE="Config,DRAM Enable,Case Name,Cycle,Time Elapsed in Secs ($freq GHz)\n"

  for config in "${cmpConfigs[@]}"; do
    declare -A dataset=()

    cfg=$(echo "$config" | cut -d',' -f1)
    dramEnable=$(echo "$config" | cut -d',' -f2)

    for case in "${cases[@]}"; do
      DRAMSIM3_ENABLE="$dramEnable" ./run_emulator.sh "$cfg" "$case"
      cycle=$(readCycle)

      key="$(printf $case | cut -d'_' -f1)"

      if [[ -z "${dataset["$key"]}" ]]; then
        dataset["$key"]=$cycle
        continue
      fi

      if (( ${dataset["$key"]} > $cycle )); then
        dataset["$key"]=$cycle
      fi
    done

    for key in "${!dataset[@]}"; do
      cycle=${dataset[$key]}
      timeElapsed=$(python3 -c "print($cycle/($freq*1000_000_000))")
      T1_KP920_CMP_DATA_TABLE+="$cfg,$(intToBool $dramEnable),$key,$cycle,$timeElapsed,\n"
    done
  done
}


declare MEMORY_SCALE_DATA_TABLE
scalabilityReproduce() {
  echo "------------------------------"
  echo "Reproducing Memory Scalability"
  echo "------------------------------"
  cmpConfigs=(
    "benchmark_dlen128_vlen512_fp"
    "benchmark_dlen256_vlen1024_fp"
    "benchmark_dlen512_vlen2048_fp"
    "benchmark_dlen1024_vlen4096_fp"
  )
  freq=2.45
  MEMORY_SCALE_DATA_TABLE="Config,Case,DRAM Enabled,Cycle,Time Elapsed in Sec($freq Ghz)\n"

  for config in "${cmpConfigs[@]}"; do
    for dramEnable in 0 1; do
      for case in sgemm_16 sgemm_32 sgemm_64; do
        DRAMSIM3_ENABLE="$dramEnable" ./run_emulator.sh "$config" "$case"
        cycle=$(readCycle)
        timeElapsed=$(python3 -c "print($cycle/($freq*1000_000_000))")

        MEMORY_SCALE_DATA_TABLE+="$config,$case,$(intToBool $dramEnable),$cycle,$timeElapsed,\n"
      done
    done
  done
}


declare X60_COMPARE_DATA_TABLE
x60CmpReproduce() {
  echo "-----------------------------"
  echo "Reproducing T1-X60 Comparison"
  echo "-----------------------------"
  cmpConfigs=(
    "benchmark_dlen128_vlen512_fp"
    "benchmark_dlen256_vlen1024_fp"
    "benchmark_dlen512_vlen2048_fp"
    "benchmark_dlen1024_vlen4096_fp"
  )
  cases=(
    "sgemm_16"
    "sgemm_32"
    "sgemm_64"
    "quant_8"
    "quant_16"
    "saxpy_8"
    "saxpy_16"
    "pack_256"
    "pack_1024"
  )
  freq=1.6

  X60_COMPARE_DATA_TABLE="Config,Case Name,Cycle,Time Elapsed in Secs ($freq GHz)\n"
  for config in "${cmpConfigs[@]}"; do
    declare -A dataset=()

    for case in "${cases[@]}"; do
      DRAMSIM3_ENABLE=1 ./run_emulator.sh "$config" "$case"
      cycle=$(readCycle)

      key=$(printf "$case" | cut -d'_' -f1)
      if [[ -z "${dataset["$key"]}" ]]; then
        dataset["$key"]=$cycle
        continue
      fi

      if (( ${dataset["$key"]} > $cycle )); then
        dataset["$key"]=$cycle
      fi
    done

    for key in ${!dataset[@]}; do
      cycle=${dataset["$key"]}
      timeElapsed=$(python3 -c "print($cycle/($freq*1000_000_000))")
      X60_COMPARE_DATA_TABLE+="$config,$key,$cycle,$timeElapsed\n"
    done
  done
}


if (( ${REP_DLEN:-1} )); then
  dlenReproduce
fi

if (( ${REP_CHAINING:-1} )); then
  chainingReproduce
fi

if (( ${REP_CRYPTO:-1} )); then
  cryptoReproduce
fi

if (( ${REP_KP920:-1} )); then
  kp920CmpReproduce
fi

if (( ${REP_SCALE:-1} )); then
  scalabilityReproduce
fi

if (( ${REP_X60:-1} )); then
  x60CmpReproduce
fi

echo "-----------------"
echo "Displaying Result"
echo "-----------------"
echo

displayCSV() {
  content="$1"; shift
  save_path="$1"; shift

  printf "$content" | column -t -s ',' | tee "$save_path"
}

mkdir -p sim_result

if (( ${REP_DLEN:-1} )); then
  echo
  echo "** Displaying *DLEN Compare Case* **"
  displayCSV "$DLEN_CYCLE_DATA_TABLE" sim_result/dlen_cmp.txt
fi

if (( ${REP_CHAINING:-1} )); then
  echo
  echo "** Displaying *No Chaining Result* **"
  displayCSV "$CHAINING_DATA_TABLE" sim_result/no_chaining.txt
fi

if (( ${REP_CRYPTO:-1} )); then
  echo
  echo "** Displaying *Crypto Bench* **"
  displayCSV "$CRYPTO_DATA_TABLE" sim_result/crypto.txt
fi

if (( ${REP_KP920:-1} )); then
  echo
  echo "** Displaying *T1-KP920 Comparison* **"
  displayCSV "$T1_KP920_CMP_DATA_TABLE" sim_result/kp920_cmp.txt
fi

if (( ${REP_SCALE:-1} )); then
  echo
  echo "** Displaying *Memory Scalability* **"
  displayCSV "$MEMORY_SCALE_DATA_TABLE" sim_result/mem_scale.txt
fi

if (( ${REP_X60:-1} )); then
  echo
  echo "** Displaying *T1-X60 Comparison* **"
  displayCSV "$X60_COMPARE_DATA_TABLE" sim_result/x60_cmp.txt
fi
