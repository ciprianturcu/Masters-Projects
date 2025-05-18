#!/bin/bash
#
# run_exec.sh
#
# Usage:
#   For pure CUDA:
#     ./run_exec.sh pure <exe> <n> <A.bin> <B.bin> <C.bin> <blockSize> <runs> <out.csv>
#
#   For MPI+CUDA:
#     ./run_exec.sh mpi  <exe> <n> <A.bin> <B.bin> <C.bin> <blockSize> \
#                      <mode(hybrid)> <hybridMode(seq_read|par_read)> \
#                      <procs> <ppn> <runs> <out.csv> [hostsfile]
#

set -e

if [ $# -lt 1 ]; then
  echo "Usage: see header of this script"
  exit 1
fi

TYPE="$1"; shift

if [ "$TYPE" == "pure" ]; then
  if [ $# -ne 8 ]; then
    echo "Usage (pure): $0 pure <exe> <n> <A> <B> <C> <blockSize> <runs> <out.csv>"
    exit 1
  fi

  EXEC="$1"; shift
  N="$1"; shift
  A_BIN="$1"; shift
  B_BIN="$1"; shift
  C_BIN="$1"; shift
  BS="$1"; shift
  RUNS="$1"; shift
  OUT="$1"; shift

  if [ ! -x "./$EXEC" ]; then
    echo "Error: '$EXEC' not found or not executable"
    exit 1
  fi

  mkdir -p "$(dirname "$OUT")"
  echo "n,t_read,t_mult,t_write,t_total" > "$OUT"

  for run in $(seq 1 "$RUNS"); do
    echo "Pure CUDA | BS=$BS | run $run/$RUNS"
    LINE=$( "./$EXEC" "$N" "$A_BIN" "$B_BIN" "$C_BIN" "$BS" )
    echo "$LINE" >> "$OUT"
  done

  echo "Done: pure CUDA results in $OUT"

elif [ "$TYPE" == "mpi" ]; then
  if [ $# -lt 12 ]; then
    echo "Usage (mpi): $0 mpi <exe> <n> <A> <B> <C> <blockSize> <mode> <hybridMode> <procs> <ppn> <runs> <out.csv> [hostsfile]"
    exit 1
  fi

  EXEC="$1"; shift
  N="$1"; shift
  A_BIN="$1"; shift
  B_BIN="$1"; shift
  C_BIN="$1"; shift
  BS="$1"; shift
  MODE="$1"; shift
  HYBRID_MODE="$1"; shift
  PROCS="$1"; shift
  PPN="$1"; shift
  RUNS="$1"; shift
  OUT="$1"; shift
  HOSTSFILE="${1:-}"

  if [ ! -x "./$EXEC" ]; then
    echo "Error: '$EXEC' not found or not executable"
    exit 1
  fi

  # prepare mpirun with ppn
  if [ -n "$HOSTSFILE" ]; then
    MPIRUN_BASE=(mpirun -hostfile "$HOSTSFILE" -np "$PROCS" -ppn "$PPN")
  else
    MPIRUN_BASE=(mpirun -np "$PROCS" -ppn "$PPN")
  fi

  mkdir -p "$(dirname "$OUT")"
  echo "n,t_read,t_mult,t_write,t_total" > "$OUT"

  for run in $(seq 1 "$RUNS"); do
    echo "MPI+CUDA | mode=$MODE/$HYBRID_MODE | BS=$BS | ppn=$PPN | run $run/$RUNS"
    LINE=$( "${MPIRUN_BASE[@]}" "./$EXEC" \
            "$N" "$A_BIN" "$B_BIN" "$C_BIN" \
            "$BS" "$MODE" "$HYBRID_MODE" )
    echo "$LINE" >> "$OUT"
  done

  echo "Done: MPI+CUDA results in $OUT"

else
  echo "Unknown type '$TYPE'. Must be 'pure' or 'mpi'."
  exit 1
fi
