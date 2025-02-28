import argparse
import pandas as pd
import matplotlib.pyplot as plt


def save_plot(input_path, title, output_path):
    data = pd.read_csv(input_path)
    plt.plot(data["Step"], data["Value"])
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", help="CSV file with Step and Value columns", required=True
    )
    parser.add_argument("--output", help="Output image filename", required=True)
    parser.add_argument("--title", help="Plot title", required=True)
    args = parser.parse_args()
    save_plot(args.input, args.title, args.output)
