import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch as th
import types

from example import (
    load_config,
    get_environment,
    get_agent
)
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
    'Monkey-patches the policy to have a method for getting the distribution.'
    policy.get_action_distribution = types.MethodType(
        get_action_distribution, policy)

def validate_environments_match(control_config, trial_config, same_rewards=False):
    def ignore_reward_type(name):
        return name.replace('Sparse', 'XXX').replace('Dense', 'XXX')
    if same_rewards:
        assert control_config.environment == trial_config.environment
    else:
        assert (ignore_reward_type(control_config.environment) ==
            ignore_reward_type(trial_config.environment))
    assert control_config.maze_size == trial_config.maze_size
    assert control_config.maze_seed == trial_config.maze_seed

def plot_kl_divergence_distribution(
        observation_space,
        control_policy,
        trial_policy,
        title_prefix,
        file_prefix,
        output_dir,
        num_samples=1000):
    samples = []
    for _ in range(num_samples):
        observation = observation_space.sample()
        policy_divergence = kl_divergence(
            control_policy.get_action_distribution(observation).distribution,
            trial_policy.get_action_distribution(observation).distribution)
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
    # TODO: combined plot

if __name__ == '__main__':
    args = get_args()
    control_config = load_config(args.control)
    trial_config = load_config(args.trial)
    validate_environments_match(control_config, trial_config)
    env = get_environment(control_config)
    control_agent = get_agent(control_config, env)
    add_get_action_distribution_method(control_agent.policy)
    trial_agent = get_agent(control_config, env)
    add_get_action_distribution_method(trial_agent.policy)
    title_prefix = f'Maze size {control_config.maze_size}'
    file_prefix = f"{control_config.maze_size} {control_config.environment} {control_config.algorithm} vs {trial_config.environment} {trial_config.algorithm}"
    plot_kl_divergence_distribution(
        env.observation_space, control_agent.policy, trial_agent.policy,
        title_prefix=title_prefix, file_prefix=file_prefix,
        output_dir=args.output_dir)
