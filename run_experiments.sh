#!/bin/bash
CONFIG_DIR="./configs/active"
COMPLETED_DIR="./configs/completed"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: Directory $CONFIG_DIR does not exist."
  exit 1
fi

if [ ! -d "$COMPLETED_DIR" ]; then
  mkdir -p "$COMPLETED_DIR"
fi

# Iterate over all .yaml files in the configs directory
for config_file in "$CONFIG_DIR"/*.yaml; do
  if [ ! -e "$config_file" ]; then
    echo "No .yaml files found in $CONFIG_DIR"
    exit 0
  fi
  
  echo ""
  echo "Running with config: $config_file"
  echo "================================================"
  echo ""
  
  python3 train.py --config "$config_file"
  
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
      echo "Warning: Training with $config_file failed with exit code $EXIT_CODE."
  else
      echo "Successfully completed: $config_file"
      mv "$config_file" "$COMPLETED_DIR/"
  fi
done

