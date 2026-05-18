#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# cd to geomr_eval_temp (created next to the script)
mkdir -p "$SCRIPT_DIR/geomr_eval_temp"
cd "$SCRIPT_DIR/geomr_eval_temp"

# The .sdf file path
SDF_PATH="$SCRIPT_DIR/samples/geom_ls_msp_eregatom_n48_ws16_rlx200.sdf"

# Directories and script paths
EVAL_DIR="$SCRIPT_DIR/geom-drugs-3dgen-evaluation"

# Built paths from sdf_path
INIT_SDF_PATH="${SDF_PATH%.sdf}_initial_structures.sdf"
OUTPUT_SDF_PATH="${SDF_PATH%.sdf}_optimized_output.sdf"

# Echo the file path to the terminal
echo "=================================================="
echo "Evaluating file: $SDF_PATH"
echo "=================================================="


echo "=================================================="
echo "Running xtb optimization on file: $SDF_PATH"
echo "=================================================="
# pipe output to file
PYTHONPATH="$EVAL_DIR" MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 python "$EVAL_DIR/scripts/energy_benchmark/xtb_optimization.py" \
  --input_sdf "$SDF_PATH" \
  --output_sdf "$OUTPUT_SDF_PATH" \
  --init_sdf "$INIT_SDF_PATH" 

echo "=================================================="
echo "Running molecule stability evaluation on file: $SDF_PATH"
echo "=================================================="
# 3. Execute with the correct PYTHONPATH and the built-in argument
# Note: "$@" is left at the end just in case you want to add extra flags later
PYTHONPATH="$EVAL_DIR" python "$EVAL_DIR/scripts/compute_molecule_stability.py" --sdf "$SDF_PATH" "$@"

echo "=================================================="
echo "Running pair geometry evaluation on file: $SDF_PATH"
echo "=================================================="

PYTHONPATH="$EVAL_DIR" python "$EVAL_DIR/scripts/energy_benchmark/compute_pair_geometry.py" \
  --init_sdf "$INIT_SDF_PATH" \
  --opt_sdf "$OUTPUT_SDF_PATH" \
  --n_subsets 5

echo "=================================================="
echo "Running RMSD evaluation on file: $SDF_PATH"
echo "=================================================="

PYTHONPATH="$EVAL_DIR" python "$EVAL_DIR/scripts/energy_benchmark/rmsd_energy.py" \
  --init_sdf "$INIT_SDF_PATH" \
  --opt_sdf "$OUTPUT_SDF_PATH" \
  --n_subsets 5