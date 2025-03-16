import os
import numpy as np
import matplotlib.pyplot as plt


def get_final_mean_reward(evals_path):
    """
    Loads 'evaluations.npz' (from Stable-Baselines) and returns the mean reward
    of the final evaluation. Returns None if file not found or cannot load data.
    """
    try:
        data = np.load(evals_path)
        final_eval_rewards = data["results"][-1]
        return final_eval_rewards.mean()
    except Exception:
        return None


def parse_run_directory(dir_name):
    """
    Given a directory name like 'TD3_HER_PointMazeSparse_8_250312-01-40-41', parse out:
      - algorithm_name (e.g., 'TD3', or 'TD3 (HER)', etc.)
      - dense_or_sparse ('Dense' or 'Sparse')
      - maze_size (an integer)

    Returns (algorithm_name, dense_or_sparse, maze_size) or None if unparseable.
    """
    parts = dir_name.split("_")

    if len(parts) < 3:
        return None

    base_alg = parts[0]

    her_flag = parts[1] == "HER"

    if her_flag:
        algorithm_name = f"{base_alg} (HER)"
        if len(parts) < 4:
            return None
        reward_type = parts[2]
        try:
            maze_size = int(parts[3])
        except ValueError:
            return None
    else:
        algorithm_name = base_alg
        reward_type = parts[1]
        try:
            maze_size = int(parts[2])
        except ValueError:
            return None

    if "Dense" in reward_type:
        dense_or_sparse = "Dense"
    elif "Sparse" in reward_type:
        dense_or_sparse = "Sparse"
    else:
        return None

    return (algorithm_name, dense_or_sparse, maze_size)


def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"No '{results_dir}' directory found.")
        return

    data = {}

    for dname in os.listdir(results_dir):
        full_path = os.path.join(results_dir, dname)
        if not os.path.isdir(full_path):
            continue

        parsed = parse_run_directory(dname)
        if not parsed:
            continue

        algorithm_name, dense_or_sparse, maze_size = parsed

        evals_file = os.path.join(full_path, "evaluations.npz")
        final_reward = get_final_mean_reward(evals_file)
        if final_reward is None:
            continue

        if algorithm_name not in data:
            data[algorithm_name] = {"Dense": {}, "Sparse": {}}

        data[algorithm_name][dense_or_sparse][maze_size] = final_reward

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    for algorithm_name, reward_dict in data.items():
        dense_map = reward_dict["Dense"]
        sparse_map = reward_dict["Sparse"]

        all_sizes = sorted(set(dense_map.keys()).union(sparse_map.keys()))
        if not all_sizes:
            continue

        dense_vals = [dense_map.get(s, np.nan) for s in all_sizes]
        sparse_vals = [sparse_map.get(s, np.nan) for s in all_sizes]

        plt.figure()
        plt.title(f"{algorithm_name}: Dense vs Sparse Rewards")
        plt.xlabel("Maze Size")
        plt.ylabel("Final Mean Reward")
        plt.grid(True)

        plt.plot(all_sizes, dense_vals, label="dense reward training", linestyle="-")
        plt.plot(all_sizes, sparse_vals, label="sparse reward training", linestyle="-")

        plt.legend()

        out_path = os.path.join(plot_dir, f"{algorithm_name}_dense_vs_sparse.png")
        plt.savefig(out_path)
        plt.close()


if __name__ == "__main__":
    main()
