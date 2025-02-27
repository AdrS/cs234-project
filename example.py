import argparse
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import maze
import numpy as np
import os
import pathlib
import stable_baselines3 as sb3
import time
from stable_baselines3.common.callbacks import BaseCallback
from vpg import VanillaPolicyGradient


gym.register_envs(gymnasium_robotics)

algorithms_by_name = {
    "VPG": VanillaPolicyGradient,
    "A2C": sb3.A2C,
    "DDPG": sb3.DDPG,
    "PPO": sb3.PPO,
}

def create_maze(env_name, args):
    maze_map = maze.dfs_generate(args.maze_size, args.maze_seed)
    return gym.make(env_name, render_mode="rgb_array", maze_map=maze_map)

environments_by_name = {
    "PointMazeSparse": lambda args: create_maze("PointMaze_UMaze-v3", args),
    "AntMazeSparse": lambda args: create_maze("AntMaze_UMaze-v5", args),
    "PointMazeDense": lambda args: create_maze("PointMaze_UMazeDense-v3", args),
    "AntMazeDense": lambda args: create_maze("AntMaze_UMazeDense-v5", args),
}

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
        self.returns = []
        self.output_dir = output_dir

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
    model_path = os.path.join(output_dir, 'model.zip')
    returns_path = os.path.join(output_dir, 'returns.npy')
    returns_plot_path = os.path.join(output_dir, 'returns.png')

    os.makedirs(output_dir, exist_ok=True)
    agent.save(model_path)
    np.save(returns_path, eval_callback.returns)
    plot_returns(eval_callback.returns, returns_plot_path)

def train(render_mode, args):
    dir_name = time.strftime('%y%m%d-%H-%M-%S')
    output_dir = os.path.join(args.output_dir_prefix, dir_name)

    environment_constructor = environments_by_name.get(args.environment)
    if environment_constructor is None:
        raise ValueError(f"Unknown environment: {args.environment}")
    env = environment_constructor(args)
    algorithm_constructor = algorithms_by_name.get(args.algorithm)
    if algorithm_constructor is None:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    agent = algorithm_constructor("MultiInputPolicy", env, verbose=1)
    
    eval_callback = EvalCallback(
        eval_period=args.steps // 100,
        num_episodes=10,
        env=env,
        agent=agent,
        output_dir=output_dir
    )
    agent.learn(total_timesteps=args.steps, callback=eval_callback)
    save_results(output_dir, agent, eval_callback)

    vec_env = agent.get_env()
    observation = vec_env.reset()
    for _ in range(args.visualization_steps):
        action, state = agent.predict(observation, deterministic=True)
        observation, reward, done, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=sorted(list(algorithms_by_name.keys())),
        default="PPO",
        help="What reinforcement learning algorithm to use.",
    )
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
        "--steps", type=int, default=1000000, help="Number of RL steps."
    )
    parser.add_argument(
        "--visualization_steps", type=int, default=1000, help="Number of steps to show at the end."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show a visualization of the environment.",
    )
    parser.add_argument(
        "--output_dir_prefix",
        type=str,
        default="results",
        help="Where to save models, results, plots, etc.",
    )
    args = parser.parse_args()
    render_mode = "human" if args.visualize else None
    train(render_mode, args)
