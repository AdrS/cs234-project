#!/bin/bash
#SBATCH --job-name=DDPG_HER_sparse_           # Job name
#SBATCH --output=DDPG_HER_sparse_%j.out       # Standard output and error log (%j expands to jobID)
#SBATCH --error=DDPG_HER_sparse_%j.err        # Separate error file (optional)
#SBATCH --time=08:00:00                      # Wall time limit (hh:mm:ss)
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=4                    # CPU cores per task
#SBATCH --mem=16G                            # Memory per node

srun python train.py --environment PointMazeSparse --algorithm DDPG --run_name "DDPG_HER_sparse" --her