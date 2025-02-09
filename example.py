import argparse
import gymnasium as gym
import gymnasium_robotics


def main(env_name, render_mode, seed, num_steps):
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_name, render_mode=render_mode)
    observation, info = env.reset(seed=seed)
    for _ in range(num_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset(seed=seed)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        type=str,
        choices=["AntMaze", "PointMaze"],
        required=True,
        default="PointMaze",
        help="What environment to simulate",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the environment and RNGs"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of steps to simulate"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show a visualization of the environment",
    )
    args = parser.parse_args()
    full_env_names = {"PointMaze": "PointMaze_UMaze-v3", "AntMaze": "AntMaze_UMaze-v5"}
    render_mode = "human" if args.visualize else None
    main(full_env_names[args.environment], render_mode, args.seed, args.steps)
