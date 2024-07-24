#!/bin/bash

#SBATCH --job-name=test_gemma2_27b      # Job name
#SBATCH --partition=dev				    # Partition name, e.g., gpu
#SBATCH --gres=gpu:6                    # Request 4 GPUs
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=8               # Number of CPU cores per task
#SBATCH --mem=64G                       # Memory per node
#SBATCH --time=2:00:00                	# Time limit
#SBATCH --output=results_%j.log         # Output log file (%j expands to jobID)

ollama serve > /dev/null &

# Run the Python script
python3 evaluate_prompts.py
