#!/bin/bash
#SBATCH --time=08:00:00                      # Wall time limit (hh:mm:ss)
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=4                    # CPU cores per task
#SBATCH --mem=16G                            # Memory per node

MODEL="$1"
ENV="$2"
SIZE="$3"

srun python train.py --environment "$ENV" --algorithm "$MODEL" --maze_size "$SIZE" -steps 10000000 --run_name "${1}_${2}_${3}" 