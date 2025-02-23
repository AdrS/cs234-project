import argparse
import gymnasium as gym
import gymnasium_robotics
import maze


def main(env_name, render_mode, args):
    gym.register_envs(gymnasium_robotics)
    maze_map = maze.dfs_generate(args.maze_size, args.maze_seed)
    env = gym.make(env_name, render_mode=render_mode, maze_map=maze_map)
    observation, info = env.reset(seed=args.env_seed)
    for _ in range(args.steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset(seed=args.env_seed)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
