#!/bin/bash

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <ndpi_input> <inf_dir> <tissue_type> <parent_dir>"
    exit 1
fi

NDPI_INPUT="$1"

# Set the inference directory where output files will be stored
INF_DIR="$2"

# Specify the tissue type  ("sWAT" for sWAT or "pWAT" for pWAT)
TISSUE_TYPE="$3"

# Set the parent directory containing the model files
PARENT_DIR="$4"

# Set environment variables required by the Python script.
export PYTHONPATH=.
export SM_FRAMEWORK=tf.keras

# Run the pipeline
python script_notebooks/testing_WAT_inf.py --ndpi_input "$NDPI_INPUT" \
                                    --inf_dir "$INF_DIR" \
                                    --tissue_type "$TISSUE_TYPE" \
                                    --parent_dir "$PARENT_DIR"