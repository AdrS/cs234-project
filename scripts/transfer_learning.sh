#!/bin/bash

set -eu

python train.py --maze_size 2 --environment PointMazeSparse --algorithm PPO --steps 5000000 --pretraining_environment PointMazeDense --pretraining_steps 5000000
python train.py --maze_size 4 --environment PointMazeSparse --algorithm PPO --steps 5000000 --pretraining_environment PointMazeDense --pretraining_steps 5000000
python train.py --maze_size 8 --environment PointMazeSparse --algorithm PPO --steps 5000000 --pretraining_environment PointMazeDense --pretraining_steps 5000000

python train.py --maze_size 2 --environment PointMazeSparse --algorithm PPO --steps 8000000 --pretraining_environment PointMazeDense --pretraining_steps 2000000
python train.py --maze_size 4 --environment PointMazeSparse --algorithm PPO --steps 8000000 --pretraining_environment PointMazeDense --pretraining_steps 2000000
python train.py --maze_size 8 --environment PointMazeSparse --algorithm PPO --steps 8000000 --pretraining_environment PointMazeDense --pretraining_steps 2000000
