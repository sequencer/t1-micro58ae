#!/usr/bin/env bash

set -o pipefail
set -o errexit

CHOSE_DESIGN=${1:-}; shift
CHOSE_CASE=${1:-}; shift

if [[ -z "${CHOSE_DESIGN}" ]]; then
  echo "Design config not given" >&2
  exit 1
fi

if [[ ! "${CHOSE_DESIGN}" == benchmark_dlen*_vlen*_fp ]]; then
  if [[ ! "${CHOSE_DESIGN}" == disable_* ]]; then
    echo "First argument ${CHOSE_DESIGN} does not look like a design config" >&2
    exit 1
  fi
fi

if [[ -z "${CHOSE_CASE}" ]]; then
  echo "Test case not given" >&2
  exit 1
fi

JQ=$(nix build ".#jq.bin" --no-link --print-out-paths)/bin/jq
EMU_TYPE=${EMU_TYPE:-t1rocketemu}

emulator=emulators/"$CHOSE_DESIGN"/"$EMU_TYPE"/emulator
make EMU_TYPE="$EMU_TYPE" "$emulator"

declare case
if [[ ! -f "$CHOSE_CASE" ]]; then
  case=rvv_benchmark_suites/bin/"${CHOSE_DESIGN}.${CHOSE_CASE}.elf"
  make "$case"
else
  case="${CHOSE_CASE}"
fi

svRoot=$(dirname $emulator)/lib
svLibName=$(basename -s '.so' "$svRoot"/*.so)

DRAMSIM3_ENABLE=${DRAMSIM3_ENABLE:-0}
declare -a dramSim3Args
if (( $DRAMSIM3_ENABLE )); then
  mkdir -p dramsim3_output
  dramSim3Args=(
    "+t1_dramsim3_cfg=$(realpath ./rvv_benchmark_suites/dramsim3-config.ini)"
    "+t1_dramsim3_path=$(realpath dramsim3_output)"
  )
else
  dramSim3Args=("+t1_dramsim3_cfg=no")
fi


cmd="$emulator +t1_rtl_event_path=./rtl-event.jsonl +t1_elf_file=$case -sv_root $svRoot -sv_lib $svLibName ${dramSim3Args[*]}"
echo "running command: '$cmd'"
eval "$cmd" 2>/dev/null

echo
echo "CYCLE: $("$JQ" -r '.total_cycles' ./sim_result.json)"
echo
