import argparse
import gymnasium as gym
import gymnasium_robotics
import maze
import stable_baselines3 as sb3

algorithms_by_name = {
    "A2C": sb3.A2C,
    "DDPG": sb3.DDPG,
    "PPO": sb3.PPO,
}

def main(env_name, render_mode, args):
    gym.register_envs(gymnasium_robotics)
    maze_map = maze.dfs_generate(args.maze_size, args.maze_seed)
    env = gym.make(env_name, render_mode="rgb_array", maze_map=maze_map)
    model_constructor = algorithms_by_name.get(args.algorithm)
    if model_constructor is None:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    model = model_constructor("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    vec_env = model.get_env()
    observation = vec_env.reset()
    for _ in range(args.steps):
        action, state = model.predict(observation, deterministic=True)
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
        choices=["AntMaze", "PointMaze"],
        default="PointMaze",
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
        "--steps", type=int, default=1000, help="Number of steps to simulate."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show a visualization of the environment.",
    )
    args = parser.parse_args()
    full_env_names = {"PointMaze": "PointMaze_UMaze-v3", "AntMaze": "AntMaze_UMaze-v5"}
    render_mode = "human" if args.visualize else None
    main(full_env_names[args.environment], render_mode, args)
