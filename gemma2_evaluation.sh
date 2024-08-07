#!/bin/bash

#SBATCH --job-name=test_gemma2          # Job name
#SBATCH --partition=dev				    # Partition name, e.g., gpu
#SBATCH --gpus=A100_80GB:6              # Request 6 GPUs
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --mem=64G                       # Memory per node
#SBATCH --time=2:00:00                	# Time limit
#SBATCH --output=results_gemma_%j.log   # Output log file (%j expands to jobID)

ollama serve > /dev/null 2>&1 &

# Run the Python script
PYTHONWARNINGS="ignore" python3 gemma2_evaluation.py
