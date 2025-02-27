import argparse
import json
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import maze
import numpy as np
import os
import pathlib
import stable_baselines3 as sb3
import subprocess
import time
from stable_baselines3.common.callbacks import BaseCallback
from types import SimpleNamespace
from vpg import VanillaPolicyGradient


gym.register_envs(gymnasium_robotics)

algorithms_by_name = {
    "VPG": VanillaPolicyGradient,
    "A2C": sb3.A2C,
    "DDPG": sb3.DDPG,
    "PPO": sb3.PPO,
}


def get_agent(config, env):
    algorithm_constructor = algorithms_by_name.get(config.algorithm)
    if algorithm_constructor is None:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
    return algorithm_constructor("MultiInputPolicy", env, verbose=1)


def create_maze(env_name, config):
    maze_map = maze.dfs_generate(config.maze_size, config.maze_seed)
    return gym.make(env_name, render_mode="rgb_array", maze_map=maze_map)


environments_by_name = {
    "PointMazeSparse": lambda config: create_maze("PointMaze_UMaze-v3", config),
    "AntMazeSparse": lambda config: create_maze("AntMaze_UMaze-v5", config),
    "PointMazeDense": lambda config: create_maze("PointMaze_UMazeDense-v3", config),
    "AntMazeDense": lambda config: create_maze("AntMaze_UMazeDense-v5", config),
}


def get_environment(config):
    environment_constructor = environments_by_name.get(config.environment)
    if environment_constructor is None:
        raise ValueError(f"Unknown environment: {config.environment}")
    return environment_constructor(config)


# Evaluate and EvalCallback are copied from assignment 3.
def evaluate(env, policy):
    model_return = 0
    T = env.spec.max_episode_steps
    obs, _ = env.reset()
    for _ in range(T):
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        model_return += reward
        if done:
            break
    return model_return


class EvalCallback(BaseCallback):
    def __init__(self, eval_period, num_episodes, env, agent, output_dir):
        super().__init__()
        self.eval_period = eval_period
        self.num_episodes = num_episodes
        self.env = env
        # The return value of predict is a tuple where the first element is the
        # action.
        self.agent = agent
        self.policy = lambda observation: agent.predict(observation)[0]
        self.output_dir = output_dir

        # Metrics
        self.returns = []
        self.start_time = time.time()

    def get_current_metrics(self):
        return {"training_time": time.time() - self.start_time, "returns": self.returns}

    def _on_step(self):
        if self.n_calls % self.eval_period == 0:
            print(f"Evaluating after {self.n_calls} steps")
            model_returns = []
            for _ in range(self.num_episodes):
                model_returns.append(evaluate(self.env, self.policy))
            self.returns.append(np.mean(model_returns))
            # Checkpoint the results
            # TODO(adrs): only save the best model.
            save_results(self.output_dir, self.agent, self)

        # If the callback returns False, training is aborted early.
        return True


def plot_returns(returns, path):
    plt.figure()
    plt.plot(range(len(returns)), returns)
    plt.xlabel("Training Episode")
    plt.ylabel("Average Return")
    plt.title("Training Performance Over Time")
    plt.legend()
    plt.grid(True)

    plt.savefig(path)
    plt.close()


def save_results(output_dir, agent, eval_callback):
    model_path = os.path.join(output_dir, "model.zip")
    returns_path = os.path.join(output_dir, "returns.npy")
    returns_plot_path = os.path.join(output_dir, "returns.png")
    metrics_path = os.path.join(output_dir, "metrics.json")

    agent.save(model_path)
    np.save(returns_path, eval_callback.returns)
    plot_returns(eval_callback.returns, returns_plot_path)
    save_json(eval_callback.get_current_metrics(), metrics_path)


def train(config):
    dir_name = time.strftime("%y%m%d-%H-%M-%S")
    output_dir = os.path.join(config.output_dir_prefix, dir_name)
    save_config(output_dir, config)
    metrics = {}

    env = get_environment(config)
    agent = get_agent(config, env)

    eval_callback = EvalCallback(
        eval_period=config.steps // 100,
        num_episodes=10,
        env=env,
        agent=agent,
        output_dir=output_dir,
    )
    agent.learn(total_timesteps=config.steps, callback=eval_callback)
    save_results(output_dir, agent, eval_callback)


def visualize(args):
    config = load_config(args.saved_model_dir)
    env = get_environment(config)
    agent = get_agent(config, env)
    model_path = os.path.join(args.saved_model_dir, "model.zip")
    agent.load(model_path)
    vec_env = agent.get_env()
    observation = vec_env.reset()
    for _ in range(config.visualization_steps):
        action, state = agent.predict(observation, deterministic=True)
        observation, reward, done, info = vec_env.step(action)
        vec_env.render("human")


def check_for_uncommitted_changes():
    """Checks whether there are uncommitted or untracked files in the repo.

    Returns:
      (<has changed files?>, <list of changed files>).
    """
    output = (
        subprocess.check_output(["git", "status", "--porcelain"])
        .decode("utf-8")
        .strip()
    )
    changed_files = output.split("\n")
    return len(changed_files) > 0, changed_files


def get_current_git_hash():
    "Returns the commit hash of the currently checked-out commit"

    git_hash = (
        subprocess.check_output(["git", "rev-parse", "@"]).decode("utf-8").strip()
    )
    has_uncommitted_changes, uncommitted_changes = check_for_uncommitted_changes()
    if has_uncommitted_changes:
        print("WARNING: uncommitted changes or untracked files")
        print("\n".join(uncommitted_changes))
        return git_hash + " uncommitted changes"
    else:
        return git_hash


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def save_config(output_dir, config):
    git_hash = get_current_git_hash()
    config_dict = vars(config)
    config_dict["git_hash"] = git_hash

    os.makedirs(output_dir)
    config_path = os.path.join(output_dir, "config.json")
    save_json(config_dict, config_path)


def load_config(output_dir):
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r") as config_file:
        return json.load(
            config_file, object_hook=lambda fields: SimpleNamespace(**fields)
        )


def get_args():
    parser = argparse.ArgumentParser()

    # Shared arguments
    parser.add_argument(
        "--environment",
        type=str,
        choices=sorted(list(environments_by_name.keys())),
        default="PointMazeDense",
        help="What environment to simulate.",
    )
    parser.add_argument(
        "--env_seed",
        type=int,
        default=42,
        help="Seed for randomness in the environment.",
    )
    parser.add_argument("--maze_size", type=int, default=2, help="Size of the maze.")
    parser.add_argument(
        "--maze_seed", type=int, default=2025, help="Seed for generating the maze."
    )

    # Training arguments
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=sorted(list(algorithms_by_name.keys())),
        default="PPO",
        help="What reinforcement learning algorithm to use.",
    )
    parser.add_argument(
        "--steps", type=int, default=1000000, help="Number of RL steps."
    )
    parser.add_argument(
        "--output_dir_prefix",
        type=str,
        default="results",
        help="Where to save models, results, plots, etc.",
    )

    # Visualization arguments
    parser.add_argument(
        "--saved_model_dir",
        type=str,
        help="Path to the config for a trained model.",
    )
    parser.add_argument(
        "--visualization_steps",
        type=int,
        default=1000,
        help="Number of steps to show at the end.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.saved_model_dir is not None:
        visualize(args)
    else:
        train(args)
