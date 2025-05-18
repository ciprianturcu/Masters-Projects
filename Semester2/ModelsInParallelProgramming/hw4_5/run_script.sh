#!/bin/bash
# run_script.sh
# Usage:
#   ./run_script.sh <exe> <n> <A.bin> <B.bin> <C.bin> <procs> <runs> <out.csv> \
#                  <read_method:seq,par> <mult_method:seq,mpi,hybrid> \
#                  [num_threads_if_hybrid] [hostsfile_if_any]
#
# BASE_DIR must be the shared filesystem path visible on all nodes.

BASE_DIR="/home/mateinfo/tchp3053"
cd "$BASE_DIR" || { echo "Error: cannot cd to $BASE_DIR"; exit 1; }

# We need at least 10 args; if hybrid, at least 11 (to get num_threads).
if [ $# -lt 10 ] || { [ "$10" = "hybrid" ] && [ $# -lt 11 ]; }; then
  echo "Usage: $0 <exe> <n> <A> <B> <C> <procs> <runs> <out.csv> <read:seq,par> <mult:seq,mpi,hybrid> [threads_if_hybrid] [hostsfile]"
  exit 1
fi

EXEC="$1"
N="$2"
A="$3"
B="$4"
C="$5"
PROCS="$6"
RUNS="$7"
OUT="$8"
READ_METHOD="$9"
MULT="${10}"

# Determine threads & hostsfile positions
if [ "$MULT" = "hybrid" ]; then
  NTHREADS="${11}"
  HOSTSFILE="${12:-}"
else
  NTHREADS=1
  HOSTSFILE="${11:-}"
fi

if [ ! -x "./${EXEC}" ]; then
  echo "Error: executable ${EXEC} not found or not executable in ${BASE_DIR}"
  exit 1
fi

# Build mpirun command
if [ -n "$HOSTSFILE" ]; then
  MPIRUN_BASE=(mpirun -hostfile "$HOSTSFILE" -np "$PROCS")
else
  MPIRUN_BASE=(mpirun -np "$PROCS")
fi

mkdir -p "$(dirname "$OUT")"
echo "n,t_reading,t_multiplication,t_writing,t_total" > "$OUT"

for i in $(seq 1 "$RUNS"); do
  echo "Run $i/$RUNS..."

  CMD=( "${MPIRUN_BASE[@]}" "./$EXEC" \
        "$N" "$A" "$B" "$C" \
        "$READ_METHOD" "$MULT" )

  if [ "$MULT" = "hybrid" ]; then
    CMD+=( "$NTHREADS" )
  fi

  echo "${CMD[@]}"
  LINE=$("${CMD[@]}")
  if [ $? -ne 0 ]; then
    echo "Error on run $i"
    exit 1
  fi
  echo "$LINE" >> "$OUT"
done

echo "Done. Results in $OUT"
