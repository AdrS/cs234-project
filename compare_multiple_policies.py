import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch as th
import types

from train import load_config, get_environment, get_agent
from torch.distributions.kl import kl_divergence
from typing import Optional, Union
from compare_policies import (
    get_action_distribution,
    add_get_action_distribution_method,
    validate_environments_match,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save plots",
    )
    return parser.parse_args()


def get_policy_diffs(trial_config, control_config, num_samples=1000):
    validate_environments_match(control_config, trial_config)
    env = get_environment(control_config)
    control_agent = get_agent(control_config, env)
    add_get_action_distribution_method(control_agent.policy)
    trial_agent = get_agent(control_config, env)
    add_get_action_distribution_method(trial_agent.policy)

    observation_space = env.observation_space
    control_policy = control_agent.policy
    trial_policy = trial_agent.policy

    def norm(x):
        return np.sqrt(x.dot(x))

    # https://stackoverflow.com/questions/2827393
    def angle_between(x, y):
        return np.degrees(
            np.arccos(np.clip(np.dot(x / norm(x), y / norm(y)), -1.0, 1.0))
        )

    norm_diffs = []
    x_diffs = []
    y_diffs = []
    direction_diffs = []
    for _ in range(num_samples):
        observation = observation_space.sample()
        control_action, _ = control_policy.predict(observation, deterministic=True)
        trial_action, _ = trial_policy.predict(observation, deterministic=True)
        norm_diffs.append(norm(trial_action) - norm(control_action))
        x_diffs.append(trial_action[0] - control_action[0])
        y_diffs.append(trial_action[1] - control_action[1])
        direction_diffs.append(angle_between(trial_action, control_action))
    return norm_diffs, x_diffs, y_diffs, direction_diffs


def get_all_policy_diffs(trial_dirs, control_dirs):
    all_norm_diffs = []
    all_x_diffs = []
    all_y_diffs = []
    all_direction_diffs = []
    for trial_dir, control_dir in zip(trial_dirs, control_dirs):
        trial_config = load_config(trial_dir)
        control_config = load_config(control_dir)
        norm_diffs, x_diffs, y_diffs, direction_diffs = get_policy_diffs(
            trial_config, control_config
        )
        all_norm_diffs.append(norm_diffs)
        all_x_diffs.append(x_diffs)
        all_y_diffs.append(y_diffs)
        all_direction_diffs.append(direction_diffs)
    return all_norm_diffs, all_x_diffs, all_y_diffs, all_direction_diffs


def plot_cdf(data, label):
    data = np.sort(data)
    p = np.arange(1, len(data) + 1) / len(data)
    plt.plot(data, p, label=label)


def plot_policy_diffs(
    trial_dirs, control_dirs, labels, title_prefix, file_prefix, output_dir
):
    all_norm_diffs, all_x_diffs, all_y_diffs, all_direction_diffs = (
        get_all_policy_diffs(trial_dirs, control_dirs)
    )

    plt.clf()
    plt.title(f"{title_prefix} Magnitude of Forces")
    plt.grid(True)
    for norm_diffs, label in zip(all_norm_diffs, labels):
        plot_cdf(norm_diffs, label)
    plt.legend()
    plt.xlabel("Difference between force magnitudes")
    plt.ylabel("Cumulative Fraction of Samples")
    plt.savefig(os.path.join(output_dir, file_prefix + "-magnitude-force.png"))

    plt.clf()
    plt.title(f"{title_prefix} X-Forces")
    plt.grid(True)
    for x_diffs, label in zip(all_x_diffs, labels):
        plot_cdf(x_diffs, label)
    plt.legend()
    plt.xlabel("Difference between x-forces")
    plt.ylabel("Cumulative Fraction of Samples")
    plt.savefig(os.path.join(output_dir, file_prefix + "-x-force.png"))

    plt.clf()
    plt.title(f"{title_prefix} Y-Forces")
    plt.grid(True)
    for y_diffs, label in zip(all_y_diffs, labels):
        plot_cdf(y_diffs, label)
    plt.legend()
    plt.xlabel("Difference between y-forces")
    plt.ylabel("Cumulative Fraction of Samples")
    plt.savefig(os.path.join(output_dir, file_prefix + "-y-force.png"))

    plt.clf()
    plt.title(f"{title_prefix} Force Directions")
    plt.grid(True)
    for direction_diffs, label in zip(all_direction_diffs, labels):
        plot_cdf(direction_diffs, label)
    plt.legend()
    plt.xlabel("Difference between force directions")
    plt.ylabel("Cumulative Fraction of Samples")
    plt.savefig(os.path.join(output_dir, file_prefix + "-force-direction.png"))


def generate_ppo_sparse_vs_dense_plots(output_dir):
    # Sparse
    trial_dirs = [
        "results/250309-22-13-11",
        "results/250309-22-38-18",
        "results/250309-23-06-31",
        "results/250309-23-33-37",
        "results/250309-23-58-01",
    ]
    # Dense
    control_dirs = [
        "results/250309-22-01-35",
        "results/250309-22-26-46",
        "results/250309-22-49-31",
        "results/250309-23-22-29",
        "results/250309-23-45-09",
    ]
    labels = [
        "Maze size 2",
        "Maze size 4",
        "Maze size 8",
        "Maze size 16",
        "Maze size 32",
    ]
    title_prefix = "PPO Sparse vs Dense"
    file_prefix = "ppo_sparse_vs_dense"
    plot_policy_diffs(
        trial_dirs, control_dirs, labels, title_prefix, file_prefix, output_dir
    )


def generate_ddpg_sparse_vs_dense_plots(output_dir):
    # Sparse
    trial_dirs = [
        "results/250310-03-42-40",
        "results/250310-11-50-18",
        "results/250310-19-47-57",
        "results/250311-04-33-35",
        "results/250311-12-31-33",
    ]
    # Dense
    control_dirs = [
        "results/250310-00-10-14",
        "results/250310-06-55-24",
        "results/250310-15-48-45",
        "results/250311-01-22-15",
        "results/250311-08-12-10",
    ]
    labels = [
        "Maze size 2",
        "Maze size 4",
        "Maze size 8",
        "Maze size 16",
        "Maze size 32",
    ]
    title_prefix = "DDPG Sparse vs Dense"
    file_prefix = "ddpg_sparse_vs_dense"
    plot_policy_diffs(
        trial_dirs, control_dirs, labels, title_prefix, file_prefix, output_dir
    )


def generate_dense_ppo_vs_ddpg_plots(output_dir):
    # DDPG
    trial_dirs = [
        "results/250310-00-10-14",
        "results/250310-06-55-24",
        "results/250310-15-48-45",
        "results/250311-01-22-15",
        "results/250311-08-12-10",
    ]
    # PPO
    control_dirs = [
        "results/250309-22-01-35",
        "results/250309-22-26-46",
        "results/250309-22-49-31",
        "results/250309-23-22-29",
        "results/250309-23-45-09",
    ]
    labels = [
        "Maze size 2",
        "Maze size 4",
        "Maze size 8",
        "Maze size 16",
        "Maze size 32",
    ]
    title_prefix = "Dense PPO vs DDPG"
    file_prefix = "dense_ppo_vs_ddpg"
    plot_policy_diffs(
        trial_dirs, control_dirs, labels, title_prefix, file_prefix, output_dir
    )


def generate_sparse_ppo_vs_ddpg_plots(output_dir):
    # DDPG
    trial_dirs = [
        "results/250310-03-42-40",
        "results/250310-11-50-18",
        "results/250310-19-47-57",
        "results/250311-04-33-35",
        "results/250311-12-31-33",
    ]
    # PPO
    control_dirs = [
        "results/250309-22-13-11",
        "results/250309-22-38-18",
        "results/250309-23-06-31",
        "results/250309-23-33-37",
        "results/250309-23-58-01",
    ]
    labels = [
        "Maze size 2",
        "Maze size 4",
        "Maze size 8",
        "Maze size 16",
        "Maze size 32",
    ]
    title_prefix = "Sparse PPO vs DDPG"
    file_prefix = "sparse_ppo_vs_ddpg"
    plot_policy_diffs(
        trial_dirs, control_dirs, labels, title_prefix, file_prefix, output_dir
    )


if __name__ == "__main__":
    args = get_args()
    generate_ppo_sparse_vs_dense_plots(args.output_dir)
    generate_ddpg_sparse_vs_dense_plots(args.output_dir)
    generate_dense_ppo_vs_ddpg_plots(args.output_dir)
    generate_sparse_ppo_vs_ddpg_plots(args.output_dir)
