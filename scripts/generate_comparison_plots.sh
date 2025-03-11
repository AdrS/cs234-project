#!/bin/bash

# PPO 2?
python compare_policies.py --control results/250309-22-01-35 --trial results/250309-22-13-11 --output_dir data/kl-divergence/
# PPO 4?
python compare_policies.py --control results/250309-22-26-46 --trial results/250309-22-38-18 --output_dir data/kl-divergence/
# PPO 8?
python compare_policies.py --control results/250309-22-49-31 --trial results/250309-23-06-31 --output_dir data/kl-divergence/
# PPO 16?
python compare_policies.py --control results/250309-23-22-29 --trial results/250309-23-33-37 --output_dir data/kl-divergence/
# PPO 32?
python compare_policies.py --control results/250309-23-45-09 --trial results/250309-23-58-01 --output_dir data/kl-divergence/

# DDPG 2?
python compare_policies.py --control results/250310-00-10-14 --trial results/250310-03-42-40 --output_dir data/kl-divergence/
# DDPG 4?
python compare_policies.py --control results/250310-06-55-24 --trial results/250310-11-50-18 --output_dir data/kl-divergence/

# TODO: compare DDPG vs PPO for the same environement and reward

# Not finished running yet
#python compare_policies.py --control results/250310-15-48-45 --trial results/250310-19-47-57 --output_dir data/kl-divergence/
