import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch as th
import types

from matplotlib.patches import Rectangle
from train import load_config, get_environment, get_agent
from torch.distributions.kl import kl_divergence
from typing import Optional, Union


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control",
        type=str,
        required=True,
        help="Path to the directory for a baseline model.",
    )
    parser.add_argument(
        "--trial",
        type=str,
        required=True,
        help="Path to the directory for a model to compare with the control.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--env_seed",
        type=int,
        default=42,
        help="Seed for randomness in the environment.",
    )
    return parser.parse_args()


# Adapted from predict() in stable_baselines3/common/policies.py
def get_action_distribution(
    self,
    observation: Union[np.ndarray, dict[str, np.ndarray]],
    state: Optional[tuple[np.ndarray, ...]] = None,
    episode_start: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
    # Switch to eval mode (this affects batch norm / dropout)

    self.set_training_mode(False)
    obs_tensor, vectorized_env = self.obs_to_tensor(observation)
    with th.no_grad():
        return self.get_distribution(obs_tensor)


def add_get_action_distribution_method(policy):
    "Monkey-patches the policy to have a method for getting the distribution."
    policy.get_action_distribution = types.MethodType(get_action_distribution, policy)


def validate_environments_match(control_config, trial_config, same_rewards=False):
    def ignore_reward_type(name):
        return name.replace("Sparse", "XXX").replace("Dense", "XXX")

    if same_rewards:
        assert control_config.environment == trial_config.environment
    else:
        assert ignore_reward_type(control_config.environment) == ignore_reward_type(
            trial_config.environment
        )
    assert control_config.maze_size == trial_config.maze_size
    assert control_config.maze_seed == trial_config.maze_seed


def plot_kl_divergence_distribution(
    observation_space,
    control_policy,
    trial_policy,
    title_prefix,
    file_prefix,
    output_dir,
    num_samples=1000,
):
    samples = []
    for _ in range(num_samples):
        observation = observation_space.sample()
        policy_divergence = kl_divergence(
            control_policy.get_action_distribution(observation).distribution,
            trial_policy.get_action_distribution(observation).distribution,
        )
        assert policy_divergence.size() == th.Size([1, 2])
        samples.append(policy_divergence[0])
    plt.clf()
    plt.hist([s.mean() for s in samples], bins=100, density=True, cumulative=True)
    plt.title(f"{title_prefix} CDF of KL-Divergence")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, file_prefix + "-mean-kl-divergence.png"))

    plt.clf()
    plt.hist([s[0] for s in samples], bins=100, density=True, cumulative=True)
    plt.title(f"{title_prefix} CDF of KL-Divergence for X-force")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, file_prefix + "-x-force-kl-divergence.png"))

    plt.clf()
    plt.hist([s[1] for s in samples], bins=100, density=True, cumulative=True)
    plt.title(f"{title_prefix} CDF of KL-Divergence for Y-force")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, file_prefix + "-y-force-kl-divergence.png"))


def norm(x):
    return np.sqrt(x.dot(x))


# https://stackoverflow.com/questions/2827393
def angle_between(x, y):
    return np.degrees(np.arccos(np.clip(np.dot(x / norm(x), y / norm(y)), -1.0, 1.0)))


# TODO: swap the order so it's trial - control and make sure the labels and
# file paths use a consistent order.
def plot_action_difference_distribution(
    observation_space,
    control_policy,
    trial_policy,
    title_prefix,
    file_prefix,
    output_dir,
    num_samples=1000,
):
    norm_diffs = []
    x_diffs = []
    y_diffs = []
    direction_diffs = []
    for _ in range(num_samples):
        observation = observation_space.sample()
        control_action, _ = control_policy.predict(observation, deterministic=True)
        trial_action, _ = trial_policy.predict(observation, deterministic=True)
        norm_diffs.append(norm(control_action) - norm(trial_action))
        x_diffs.append(control_action[0] - trial_action[0])
        y_diffs.append(control_action[1] - trial_action[1])
        direction_diffs.append(angle_between(control_action, trial_action))
    plt.clf()
    plt.hist(norm_diffs, bins=100, density=True, cumulative=True)
    plt.title(f"{title_prefix} Norm of Force Delta")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, file_prefix + "-norm-force-delta.png"))

    plt.clf()
    plt.hist(x_diffs, bins=100, density=True, cumulative=True)
    plt.title(f"{title_prefix} X-Force Delta")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, file_prefix + "-x-force-delta.png"))

    plt.clf()
    plt.hist(y_diffs, bins=100, density=True, cumulative=True)
    plt.title(f"{title_prefix} Y-Force Delta")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, file_prefix + "-y-force-delta.png"))

    plt.clf()
    plt.hist(direction_diffs, bins=100, density=True, cumulative=True)
    plt.title(f"{title_prefix} Force Direction Delta (degrees)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, file_prefix + "-direction-delta.png"))


def plot_action_difference_map(
    env, control_policy, trial_policy, title_prefix, file_prefix, output_dir, seed
):
    base_observation, _ = env.reset(seed=seed)
    maze = env.env.unwrapped.maze
    XMAX = maze.map_width * maze.maze_size_scaling - maze.x_map_center
    XMIN = -maze.x_map_center
    YMAX = maze.y_map_center
    YMIN = maze.y_map_center - maze.map_length * maze.maze_size_scaling
    STEPS = 100
    norm_diffs = np.zeros((STEPS, STEPS))
    x_diffs = np.zeros((STEPS, STEPS))
    y_diffs = np.zeros((STEPS, STEPS))
    direction_diffs = np.zeros((STEPS, STEPS))
    for row in range(STEPS):
        for col in range(STEPS):
            x = XMIN + col * (XMAX - XMIN) / STEPS
            y = YMIN + row * (YMAX - YMIN) / STEPS
            # https://robotics.farama.org/envs/maze/point_maze/#observation-space
            observation = {
                "desired_goal": base_observation["desired_goal"],
                # x and y velocities are 0
                "observation": np.array([x, y, 0, 0]),
                #'achieved_goal':base_observation['achieved_goal']
                "achieved_goal": np.array([x, y]),
            }
            control_action, _ = control_policy.predict(observation, deterministic=True)
            trial_action, _ = trial_policy.predict(observation, deterministic=True)
            norm_diffs[row][col] = norm(control_action) - norm(trial_action)
            x_diffs[row][col] = control_action[0] - trial_action[0]
            y_diffs[row][col] = control_action[1] - trial_action[1]
            direction_diffs[row][col] = angle_between(control_action, trial_action)

    def point2index(x, y):
        return (
            int(STEPS * (x - XMIN) / (XMAX - XMIN)),
            int(STEPS * (y - YMIN) / (YMAX - YMIN)),
        )

    def draw_maze():
        ax = plt.gca()
        maze_map = maze.maze_map
        maze_size_scaling = maze.maze_size_scaling
        maze_height = maze.maze_height
        for i in range(maze.map_length):
            for j in range(maze.map_width):
                struct = maze_map[i][j]
                if struct != 1:
                    continue
                # See: https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom
                x1 = j * maze_size_scaling - maze.x_map_center
                x2 = x1 + maze_size_scaling
                y1 = maze.y_map_center - i * maze_size_scaling
                y2 = y1 - maze_size_scaling

                box_x, box_y = point2index(x1, y2)
                box_width = maze_size_scaling * STEPS / (XMAX - XMIN)
                box_height = maze_size_scaling * STEPS / (YMAX - YMIN)
                rect = Rectangle((box_x, box_y), box_width, box_height)
                ax.add_patch(rect)

    def plot_map(values):
        plt.imshow(values)
        draw_maze()
        plt.colorbar()
        x_goal, y_goal = point2index(
            base_observation["desired_goal"][0], base_observation["desired_goal"][1]
        )
        markersize = max(STEPS // maze.map_length, 4)
        plt.plot(x_goal, y_goal, "rx", markersize=markersize)

    plot_map(norm_diffs)
    plt.title(f"{title_prefix} Norm of Force Delta")
    plt.savefig(os.path.join(output_dir, file_prefix + "-norm-force-delta.png"))

    plt.clf()
    plot_map(x_diffs)
    plt.title(f"{title_prefix} X-Force Delta")
    plt.savefig(os.path.join(output_dir, file_prefix + "-x-force-delta.png"))

    plt.clf()
    plot_map(y_diffs)
    plt.title(f"{title_prefix} Y-Force Delta")
    plt.savefig(os.path.join(output_dir, file_prefix + "-y-force-delta.png"))

    plt.clf()
    plot_map(direction_diffs)
    plt.title(f"{title_prefix} Force Direction Delta (degrees)")
    plt.savefig(os.path.join(output_dir, file_prefix + "-direction-delta.png"))


if __name__ == "__main__":
    args = get_args()
    control_config = load_config(args.control)
    trial_config = load_config(args.trial)
    validate_environments_match(control_config, trial_config)
    env = get_environment(control_config)
    control_agent = get_agent(control_config, env)
    add_get_action_distribution_method(control_agent.policy)
    trial_agent = get_agent(control_config, env)
    add_get_action_distribution_method(trial_agent.policy)
    title_prefix = f"Maze size {control_config.maze_size}"
    file_prefix = f"{control_config.maze_size} {control_config.environment} {control_config.algorithm} vs {trial_config.environment} {trial_config.algorithm}"
    # It only makes sense to compare distributions for PPO because DDPG is
    # deterministic.
    if control_config.algorithm == "PPO" and trial_config.algorithm == "PPO":
        plot_kl_divergence_distribution(
            env.observation_space,
            control_agent.policy,
            trial_agent.policy,
            title_prefix=title_prefix,
            file_prefix=file_prefix,
            output_dir=args.output_dir,
        )
    plot_action_difference_distribution(
        env.observation_space,
        control_agent.policy,
        trial_agent.policy,
        title_prefix=title_prefix,
        file_prefix=file_prefix,
        output_dir=args.output_dir,
    )
    plot_action_difference_map(
        env,
        control_agent.policy,
        trial_agent.policy,
        title_prefix=title_prefix,
        file_prefix=file_prefix + "-map",
        output_dir=args.output_dir,
        seed=args.env_seed,
    )
