import argparse
import matplotlib.pyplot as plt


def save_plot(output_path):
    plt.title("Transfer Learning")
    plt.xlabel("Maze Size")
    plt.ylabel("Mean Reward")
    plt.grid(True)

    plt.plot([2, 4, 8], [237.92401, 99.79703, 22.93647], label="Dense reward training")
    plt.plot([2, 4, 8], [232.04, 74.72, 0.78], label="Sparse reward training")
    plt.plot(
        [2, 4, 8],
        [227.95374, 81.42674, 19.24042],
        label="First 50% dense, last 50% sparse reward training",
    )
    plt.plot(
        [2, 4, 8],
        [230.79335, 117.24656, 18.27285],
        label="First 80% dense, last 20% sparse reward training",
    )
    plt.legend()
    plt.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="transfer_learning_plot.png", help="Output image filename"
    )
    args = parser.parse_args()
    save_plot(args.output)
