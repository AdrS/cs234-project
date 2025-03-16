import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_run_directory(dir_name):
    parts = dir_name.split("_")
    if len(parts) < 3:
        return None

    base_alg = parts[0]
    is_her = False

    if len(parts) >= 2 and parts[1] == "HER":
        is_her = True
        if len(parts) < 4:
            return None
        reward_part = parts[2]
        try:
            maze_size = int(parts[3])
        except ValueError:
            return None
    else:
        reward_part = parts[1]
        try:
            maze_size = int(parts[2])
        except ValueError:
            return None

    if "Dense" in reward_part:
        ds_type = "Dense"
    elif "Sparse" in reward_part:
        ds_type = "Sparse"
    else:
        return None

    return (base_alg, is_her, ds_type, maze_size)


def load_evaluations(npz_path):
    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]
    mean_rewards = results.mean(axis=1)
    return timesteps, mean_rewards


base_alg_colors = {
    "DDPG": "C0",
    "TD3": "C1",
    "SAC": "C2",
    "NAF": "C3",
    "PPO": "C4",
}


def get_color(base_alg):
    return base_alg_colors.get(base_alg, "black")


def get_linestyle(is_her):
    return "--" if is_her else "-"


def main():
    results_dir = "results"
    data_by_size = defaultdict(lambda: defaultdict(dict))

    for dname in os.listdir(results_dir):
        full_path = os.path.join(results_dir, dname)
        if not os.path.isdir(full_path):
            continue

        parsed = parse_run_directory(dname)
        if not parsed:
            continue

        base_alg, is_her, ds_type, maze_size = parsed
        npz_file = os.path.join(full_path, "evaluations.npz")
        if not os.path.isfile(npz_file):
            continue

        steps, rewards = load_evaluations(npz_file)
        if steps is None or rewards is None:
            continue

        data_by_size[maze_size][ds_type][(base_alg, is_her)] = (steps, rewards)

    os.makedirs("plots", exist_ok=True)

    for size in sorted(data_by_size.keys()):
        ds_map = data_by_size[size]

        for ds_type in ["Dense", "Sparse"]:
            if ds_type not in ds_map:
                continue
            runs_dict = ds_map[ds_type]

            if not runs_dict:
                continue

            alg_has_her = defaultdict(bool)
            alg_has_nonher = defaultdict(bool)

            for (ba, is_her), _ in runs_dict.items():
                if is_her:
                    alg_has_her[ba] = True
                else:
                    alg_has_nonher[ba] = True

            her_vs_nonher_lines = defaultdict(dict)
            for (ba, is_her), (steps, rew) in runs_dict.items():
                if alg_has_her[ba] and alg_has_nonher[ba]:
                    her_vs_nonher_lines[ba][is_her] = (steps, rew)

            if her_vs_nonher_lines:
                plt.figure()
                plt.title(f"Mean Reward by Algorithm - {ds_type} Rewards")
                plt.xlabel("Step")
                plt.ylabel("Mean Reward")
                plt.grid(True)

                for ba in sorted(her_vs_nonher_lines.keys()):
                    line_dict = her_vs_nonher_lines[ba]
                    for is_her_flag in [False, True]:
                        if is_her_flag in line_dict:
                            steps, rew = line_dict[is_her_flag]
                            c = get_color(ba)
                            style = get_linestyle(is_her_flag)
                            label_her = " (HER)" if is_her_flag else ""
                            plt.plot(
                                steps,
                                rew,
                                color=c,
                                linestyle=style,
                                label=f"{ba}{label_her}",
                            )

                plt.legend(
                    bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0
                )
                outpath = f"maze_size_{size}_{ds_type.lower()}_her_vs_nonher.png"
                outpath = os.path.join("plots", outpath)
                plt.savefig(outpath, bbox_inches="tight")
                plt.close()

            nonher_lines = []
            for (ba, is_her_flag), (steps, rew) in runs_dict.items():
                if not is_her_flag:
                    nonher_lines.append((ba, steps, rew))

            if nonher_lines:
                plt.figure()
                plt.title(f"Mean Reward by Algorithm - {ds_type} Rewards")
                plt.xlabel("Step")
                plt.ylabel("Mean Reward")
                plt.grid(True)

                nonher_lines.sort(key=lambda x: x[0])
                for ba, steps, rew in nonher_lines:
                    c = get_color(ba)
                    plt.plot(steps, rew, color=c, linestyle="-", label=ba)

                plt.legend(
                    bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0
                )
                outpath = f"maze_size_{size}_{ds_type.lower()}_non_her.png"
                outpath = os.path.join("plots", outpath)
                plt.savefig(outpath, bbox_inches="tight")
                plt.close()


if __name__ == "__main__":
    main()
