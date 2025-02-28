import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys


def save_plot(input_paths, labels, title, output_path):
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    for input_path, label in zip(input_paths, labels):
        data = pd.read_csv(input_path)
        plt.plot(data["Step"], data["Value"], label=label)
    plt.legend()
    plt.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        help="List of CSV files with Step and Value columns",
        nargs="*",
        required=True,
    )
    parser.add_argument(
        "--labels", help="Labels for each input", nargs="*", required=True
    )
    parser.add_argument("--output", help="Output image filename", required=True)
    parser.add_argument("--title", help="Plot title", required=True)
    args = parser.parse_args()
    if len(args.inputs) != len(args.labels):
        print("Must have the same number of inputs and labels")
        parser.print_usage()
        sys.exit()
    save_plot(args.inputs, args.labels, args.title, args.output)
