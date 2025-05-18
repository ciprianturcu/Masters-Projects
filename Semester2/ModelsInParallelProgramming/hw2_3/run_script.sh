#!/bin/bash

# Usage:
#   ./run_experiment.sh <executable> <n> <file1> <file2> <file3> <num_runs> <output_csv> <read_option> <mult_option> <num_threads>
#
# Example:
#   ./run_experiment.sh ./matrix_ops 100 matrix1.bin matrix2.bin result.bin 5 output-thread/results.csv seq par_thread 20

if [ $# -lt 10 ]; then
  echo "Usage: $0 <executable> <n> <file1> <file2> <file3> <num_runs> <output_csv> <read_option> <mult_option> <num_threads>"
  exit 1
fi

EXECUTABLE=$1
MATRIX_SIZE=$2
FILE1=$3
FILE2=$4
FILE3=$5
NUM_RUNS=$6
OUTPUT_CSV=$7
READ_OPTION=$8
MULT_OPTION=$9
NUM_THREADS=${10}

# Check if the executable exists and is executable.
if [ ! -x "$EXECUTABLE" ]; then
  echo "Error: $EXECUTABLE not found or not executable."
  exit 1
fi

# Create output directory if it does not exist.
OUTPUT_DIR=$(dirname "$OUTPUT_CSV")
mkdir -p "$OUTPUT_DIR"

# Print CSV header to the output file (only once).
# Column names: n, t_reading, t_multiplication, t_writing, t_total
echo "n,t_reading,t_multiplication,t_writing,t_total" > "$OUTPUT_CSV"

# Loop for the specified number of runs.
for i in $(seq 1 "$NUM_RUNS"); do
  echo "Run $i of $NUM_RUNS..."

  # Execute the program with all required arguments.
  CSV_LINE=$("$EXECUTABLE" "$MATRIX_SIZE" "$FILE1" "$FILE2" "$FILE3" "$READ_OPTION" "$MULT_OPTION" "$NUM_THREADS")

  # Check for errors.
  if [ $? -ne 0 ]; then
    echo "Error: Run $i encountered an error. Exiting."
    exit 1
  fi

  # Append the output CSV line.
  echo "$CSV_LINE" >> "$OUTPUT_CSV"
done

echo "Experiment completed. Results saved in $OUTPUT_CSV."
