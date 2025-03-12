#!/bin/bash

set -eu

python train.py --maze_size 2 --environment PointMazeDense --algorithm PPO --steps 10000000
python train.py --maze_size 2 --environment PointMazeSparse --algorithm PPO --steps 10000000

python train.py --maze_size 4 --environment PointMazeDense --algorithm PPO --steps 10000000
python train.py --maze_size 4 --environment PointMazeSparse --algorithm PPO --steps 10000000

python train.py --maze_size 8 --environment PointMazeDense --algorithm PPO --steps 10000000
python train.py --maze_size 8 --environment PointMazeSparse --algorithm PPO --steps 10000000

python train.py --maze_size 16 --environment PointMazeDense --algorithm PPO --steps 10000000
python train.py --maze_size 16 --environment PointMazeSparse --algorithm PPO --steps 10000000

python train.py --maze_size 32 --environment PointMazeDense --algorithm PPO --steps 10000000
python train.py --maze_size 32 --environment PointMazeSparse --algorithm PPO --steps 10000000

python train.py --maze_size 2 --environment PointMazeDense --algorithm DDPG --steps 5000000
python train.py --maze_size 2 --environment PointMazeSparse --algorithm DDPG --steps 5000000

python train.py --maze_size 4 --environment PointMazeDense --algorithm DDPG --steps 5000000
python train.py --maze_size 4 --environment PointMazeSparse --algorithm DDPG --steps 5000000

python train.py --maze_size 8 --environment PointMazeDense --algorithm DDPG --steps 5000000
python train.py --maze_size 8 --environment PointMazeSparse --algorithm DDPG --steps 5000000

python train.py --maze_size 16 --environment PointMazeDense --algorithm DDPG --steps 5000000
python train.py --maze_size 16 --environment PointMazeSparse --algorithm DDPG --steps 5000000

python train.py --maze_size 32 --environment PointMazeDense --algorithm DDPG --steps 5000000
python train.py --maze_size 32 --environment PointMazeSparse --algorithm DDPG --steps 5000000
