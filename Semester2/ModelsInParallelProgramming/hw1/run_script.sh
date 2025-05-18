#!/bin/bash

# Usage:
#   ./run_experiment.sh <executable> <n> <file1> <file2> <file3> <num_runs> <output_csv>
#
# Example:
#   ./run_experiment.sh ./matrix_ops 100 matrix1.bin matrix2.bin result.bin 5 results.csv
#

if [ $# -lt 7 ]; then
  echo "Usage: $0 <executable> <n> <file1> <file2> <file3> <num_runs> <output_csv>"
  exit 1
fi

EXECUTABLE=$1
MATRIX_SIZE=$2
FILE1=$3
FILE2=$4
FILE3=$5
NUM_RUNS=$6
OUTPUT_CSV=$7

# Print CSV header to the output file (only once).
# Column names match the variables: n, t_reading, t_multiplication, t_writing, t_total
echo "n,t_reading,t_multiplication,t_writing,t_total" > "$OUTPUT_CSV"

# Loop for the specified number of runs
for i in $(seq 1 $NUM_RUNS); do
  # Capture the program's single-line CSV output
  CSV_LINE=$("$EXECUTABLE" "$MATRIX_SIZE" "$FILE1" "$FILE2" "$FILE3")
  # Append that line to the output CSV
  echo "$CSV_LINE" >> "$OUTPUT_CSV"
done
