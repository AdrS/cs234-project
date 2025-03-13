import argparse
import matplotlib.pyplot as plt


def save_plot(output_path):
    plt.title("Dense vs Sparse Rewards")
    plt.xlabel("Maze Size")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.plot(
        [2, 3, 4, 5, 6, 7, 8, 16, 32],
        [
            237.92401,
            161.01132,
            99.79703,
            38.86752,
            38.36229,
            22.76335,
            22.93647,
            1.1085,
            0.00059358,
        ],
        label="Dense reward training",
    )
    plt.plot(
        [2, 3, 4, 5, 6, 7, 8, 16, 32],
        [232.04, 137.26, 74.72, 15.92, 3.78, 0.7, 0.78, 0, 0],
        label="Sparse reward training",
    )
    plt.legend()
    plt.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="images/maze_size_plot.png", help="Output image filename"
    )
    args = parser.parse_args()
    save_plot(args.output)
