#!/bin/bash

# PPO 2: Dense vs sparse
python compare_policies.py --control results/250309-22-01-35 --trial results/250309-22-13-11 --output_dir data/comparison-plots/
# PPO 4: Dense vs sparse
python compare_policies.py --control results/250309-22-26-46 --trial results/250309-22-38-18 --output_dir data/comparison-plots/
# PPO 8: Dense vs sparse
python compare_policies.py --control results/250309-22-49-31 --trial results/250309-23-06-31 --output_dir data/comparison-plots/
# PPO 16: Dense vs sparse
python compare_policies.py --control results/250309-23-22-29 --trial results/250309-23-33-37 --output_dir data/comparison-plots/
# PPO 32: Dense vs sparse
python compare_policies.py --control results/250309-23-45-09 --trial results/250309-23-58-01 --output_dir data/comparison-plots/

# DDPG 2: Dense vs sparse
python compare_policies.py --control results/250310-00-10-14 --trial results/250310-03-42-40 --output_dir data/comparison-plots/
# DDPG 4: Dense vs sparse
python compare_policies.py --control results/250310-06-55-24 --trial results/250310-11-50-18 --output_dir data/comparison-plots/
# DDPG 8: Dense vs sparse
python compare_policies.py --control results/250310-15-48-45 --trial results/250310-19-47-57 --output_dir data/comparison-plots/
# DDPG 16: Dense vs sparse
python compare_policies.py --control results/250311-01-22-15 --trial results/250311-04-33-35 --output_dir data/comparison-plots/
# DDPG 32: Dense vs sparse
python compare_policies.py --control results/250311-08-12-10 --trial results/250311-12-31-33 --output_dir data/comparison-plots/

# Dense 2: PPO vs DDPG
python compare_policies.py --control results/250309-22-01-35 --trial results/250310-00-10-14 --output_dir data/comparison-plots/
# Dense 4: PPO vs DDPG
python compare_policies.py --control results/250309-22-26-46 --trial results/250310-06-55-24 --output_dir data/comparison-plots/
# Dense 8: PPO vs DDPG
python compare_policies.py --control results/250309-22-49-31 --trial results/250310-15-48-45 --output_dir data/comparison-plots/
# Dense 16: PPO vs DDPG
python compare_policies.py --control results/250309-23-22-29 --trial results/250311-01-22-15 --output_dir data/comparison-plots/
# Dense 32: PPO vs DDPG
python compare_policies.py --control results/250309-23-45-09 --trial results/250311-08-12-10 --output_dir data/comparison-plots/

# Sparse 2: PPO vs DDGP
python compare_policies.py --control results/250309-22-13-11 --trial results/250310-03-42-40 --output_dir data/comparison-plots/
# Sparse 4: PPO vs DDGP
python compare_policies.py --control results/250309-22-38-18 --trial results/250310-11-50-18 --output_dir data/comparison-plots/
# Sparse 8: PPO vs DDGP
python compare_policies.py --control results/250309-23-06-31 --trial results/250310-19-47-57 --output_dir data/comparison-plots/
# Sparse 16: PPO vs DDGP
python compare_policies.py --control results/250309-23-33-37 --trial results/250311-04-33-35 --output_dir data/comparison-plots/
# Sparse 32: PPO vs DDGP
python compare_policies.py --control results/250309-23-58-01 --trial results/250311-12-31-33 --output_dir data/comparison-plots/
