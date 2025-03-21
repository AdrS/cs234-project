import argparse
import json
import gymnasium as gym
import gymnasium_robotics
import maze
import os
import stable_baselines3 as sb3
import subprocess
import time
import wandb
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from types import SimpleNamespace
from vpg import VanillaPolicyGradient
from stable_baselines3 import HerReplayBuffer
from wandb.integration.sb3 import WandbCallback
from naf import NAF

gym.register_envs(gymnasium_robotics)

algorithms_by_name = {
    "VPG": VanillaPolicyGradient,
    "A2C": sb3.A2C,
    "DDPG": sb3.DDPG,
    "PPO": sb3.PPO,
    "SAC": sb3.SAC,
    "TD3": sb3.TD3,
    "NAF": NAF,
}


def get_agent(config, env, tensorboard_dir=None):
    algorithm_constructor = algorithms_by_name.get(config.algorithm)
    if algorithm_constructor is None:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    if config.algorithm == "NAF":
        return NAF(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=1000000,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=tensorboard_dir,
        )

    if hasattr(config, "her") and config.her:
        return algorithm_constructor(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4, goal_selection_strategy="future"
            ),
            learning_starts=env.spec.max_episode_steps + 1,
        )
    return algorithm_constructor(
        "MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_dir
    )


def create_maze(env_name, config):
    maze_map = maze.dfs_generate(config.maze_size, config.maze_seed)
    return Monitor(gym.make(env_name, render_mode="rgb_array", maze_map=maze_map))


environments_by_name = {
    "PointMazeSparse": lambda config: create_maze("PointMaze_UMaze-v3", config),
    "AntMazeSparse": lambda config: create_maze("AntMaze_UMaze-v5", config),
    "PointMazeDense": lambda config: create_maze("PointMaze_UMazeDense-v3", config),
    "AntMazeDense": lambda config: create_maze("AntMaze_UMazeDense-v5", config),
}


def get_environment(config, environment_name_field="environment"):
    environment_name = getattr(config, environment_name_field)
    environment_constructor = environments_by_name.get(environment_name)
    if environment_constructor is None:
        raise ValueError(f"Unknown environment: {config.environment}")
    return environment_constructor(config)


class VisualizeCallback(BaseCallback):

    def __init__(self, agent, config):
        super().__init__()
        self.agent = agent
        self.config = config

    def _on_step(self):
        if self.n_calls % self.config.visualize_freq == 0:
            visualize(self.agent, self.config)
        # If the callback does not return True, then training is aborted early.
        return True


def visualize(agent, config):
    vec_env = agent.get_env()
    observation = vec_env.reset()
    for _ in range(config.visualization_steps):
        action, state = agent.predict(observation, deterministic=True)
        observation, reward, done, info = vec_env.step(action)
        vec_env.render("human")


def train_in_environment(env, eval_env, agent, config, steps, output_dir, wandb_run):
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config.max_no_improvement_evals,
        min_evals=config.min_evals,
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        callback_after_eval=stop_train_callback,
        verbose=1,
        deterministic=True,
        render=False,
    )
    callbacks = [eval_callback]
    if config.wandb:
        callbacks.append(
            WandbCallback(
                model_save_path=f"models/{wandb_run.id}",
                verbose=2,
            )
        )
    if config.visualize_freq > 0:
        callbacks.append(VisualizeCallback(agent, config))
    agent.learn(total_timesteps=steps, callback=callbacks, progress_bar=True)


def pretrain(config, tensorboard_dir, output_dir, wandb_run):
    """Pre-trains the model in the environment specified by config

    Returns the path of the saved pre-trained model.
    """
    pretraining_env = get_environment(config, "pretraining_environment")
    pretraining_eval_env = get_environment(config, "pretraining_environment")
    agent = get_agent(config, pretraining_env, tensorboard_dir)
    pretraining_output_dir = os.path.join(output_dir, "pretraining")
    train_in_environment(
        env=pretraining_env,
        eval_env=pretraining_eval_env,
        agent=agent,
        config=config,
        steps=config.pretraining_steps,
        output_dir=pretraining_output_dir,
        wandb_run=wandb_run,
    )
    return os.path.join(pretraining_output_dir, "best_model.zip")


def train(config):
    timestamp = time.strftime("%y%m%d-%H-%M-%S")
    if config.run_name:
        dir_name = f"{config.run_name}_{timestamp}"
    else:
        dir_name = timestamp
    output_dir = os.path.join(config.output_dir_prefix, dir_name)
    save_config(output_dir, config)
    tensorboard_dir = os.path.join(output_dir, "tensorboard")

    wandb_run = None
    if config.wandb:
        wandb.login()
        wandb_run = wandb.init(
            project="cs234-project",
            config=load_config_as_dict(output_dir),
            sync_tensorboard=True,
            monitor_gym=True,
        )

    env = get_environment(config)
    eval_env = get_environment(config)
    agent = get_agent(config, env, tensorboard_dir)

    if config.pretraining_environment is not None:
        agent.set_parameters(pretrain(config, tensorboard_dir, output_dir, wandb_run))

    train_in_environment(
        env=env,
        eval_env=eval_env,
        agent=agent,
        config=config,
        steps=config.steps,
        output_dir=output_dir,
        wandb_run=wandb_run,
    )

    if config.wandb:
        wandb_run.finish()


def visualize_saved_model(args):
    config = load_config(args.saved_model_dir)
    env = get_environment(config)
    agent = get_agent(config, env)
    model_path = os.path.join(args.saved_model_dir, "best_model.zip")
    # Warning: load re-creates the model from scratch, it does not update it
    # in-place! For an in-place load use set_parameters instead
    agent.set_parameters(model_path)
    visualize(agent, config)


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


def load_config_as_dict(output_dir):
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r") as config_file:
        return json.load(config_file)


def load_config(output_dir):
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r") as config_file:
        return json.load(
            config_file, object_hook=lambda fields: SimpleNamespace(**fields)
        )


def get_args():
    parser = argparse.ArgumentParser()

    # Training arguments
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
        "--eval_freq", type=int, default=10000, help="How often to evaluate the model."
    )
    parser.add_argument(
        "--visualize_freq",
        type=int,
        default=-1,
        help="How often to visualize the model.",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=50,
        help="Number of episodes to evaluate models on.",
    )

    parser.add_argument(
        "--max_no_improvement_evals",
        type=int,
        default=20,
        help="Number of evals to wait for model improvement before stopping training.",
    )
    parser.add_argument(
        "--min_evals",
        type=int,
        default=50,
        help="Minimum number of evals before stopping.",
    )
    parser.add_argument(
        "--output_dir_prefix",
        type=str,
        default="results",
        help="Where to save models, results, plots, etc.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Custom run name to prepend to the output directory.",
    )
    parser.add_argument(
        "--her",
        action="store_true",
        help="If included, use Hindsight Experience Replay (HER) during training.",
    )
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to log to Weights & Biases",
    )
    wandb_group.add_argument(
        "--nowandb",
        action="store_true",
        help="Whether to disable logging to Weights & Biases",
    )

    # Transfer learning
    parser.add_argument(
        "--pretraining_environment",
        type=str,
        choices=sorted(list(environments_by_name.keys())),
        help="Environment to pretrain the model on before doing transfer learning to --environment",
    )
    parser.add_argument(
        "--pretraining_steps",
        type=int,
        default=1000000,
        help="Number of RL steps to train in the pretraining environment before using transfer learning to train in --environment.",
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
    args = parser.parse_args()
    if args.nowandb:
        args.wandb = False
    else:
        args.wandb = True
    del args.nowandb
    return args


if __name__ == "__main__":
    args = get_args()
    if args.saved_model_dir is not None:
        visualize_saved_model(args)
    else:
        train(args)
