"""
Analyze the distribution of injection rates in the training dataset.

The script loads the scaled dataset and scaling parameters, identifies the
injection-rate feature, and inverse-scales it back to physical units. It keeps
only active-injection samples, converts the injection rate from kg/day to
ton/day, and counts the occurrence of each injection-rate value. The resulting
distribution is saved to a CSV file and visualized as a histogram showing the
percentage of samples in each injection-rate interval, together with the weighted
mean, weighted median and midpoint of the observed injection-rate range.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def inverse_minmax(x_scaled, x_min, x_max, a=-1.0, b=1.0):
    return x_min + (x_scaled - a) * (x_max - x_min) / (b - a)


def get_feature_column_from_scalings(
    scalings_csv: pathlib.Path,
    candidates: list[str],
):
    feature_rows = []

    with scalings_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            var = row["variable"]

            if var in ["output_WI", "target_range", "feature_range", "input_range"]:
                continue

            feature_rows.append(row)

    for i, row in enumerate(feature_rows):
        var_lower = row["variable"].lower()

        for cand in candidates:
            if cand.lower() in var_lower:
                return i, row["variable"], float(row["min"]), float(row["max"])

    print("\nAvailable feature names:")
    for i, row in enumerate(feature_rows):
        print(f"{i}: {row['variable']}")

    raise ValueError(f"Could not find feature matching: {candidates}")
def weighted_median(values, weights):
    values = np.asarray(values)
    weights = np.asarray(weights)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cumulative_weight = np.cumsum(weights)
    cutoff = 0.5 * np.sum(weights)

    return values[np.searchsorted(cumulative_weight, cutoff)]

def main(
    nn_dir: str,
    split_name: str,
    chunk_size: int,
    round_rate_digits: int,
    bins: int,
):
    nn_dir = pathlib.Path(nn_dir)

    npz_path = nn_dir / f"{split_name}set_scaled.npz"
    scalings_path = nn_dir / "scalings.csv"

    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find: {npz_path}")

    if not scalings_path.exists():
        raise FileNotFoundError(f"Could not find: {scalings_path}")

    q_col, q_name, q_min, q_max = get_feature_column_from_scalings(
        scalings_path,
        candidates=["injection_rate"],
    )

    print(f"Reading: {npz_path}")
    print(f"Using feature: {q_name}")
    print(f"Injection-rate column index: {q_col}")

    data = np.load(npz_path, mmap_mode="r")
    X = data["X"]

    n_rows = X.shape[0]
    print(f"X shape: {X.shape}")
    print(f"Processing in chunks of {chunk_size}")

    counter = Counter()
    n_active = 0
    n_total = 0

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)

        q_scaled = X[start:end, q_col]
        q = inverse_minmax(q_scaled, q_min, q_max, -1.0, 1.0)

        q = q[np.isfinite(q)]
        q = q[q > 0.0]

        q_group = np.round(q, round_rate_digits)

        counter.update(q_group.tolist())

        n_active += q_group.size
        n_total += end - start

        print(f"Processed {end}/{n_rows}", flush=True)

    if n_active == 0:
        raise RuntimeError("No active injection-rate samples found.")

    rates = np.array(sorted(counter.keys()), dtype=float)
    counts = np.array([counter[r] for r in rates], dtype=int)

    summary = pd.DataFrame(
        {
            "injection_rate_kg_day": rates,
            "injection_rate_ton_day": rates / 1000.0,
            "n_samples": counts,
            "fraction": counts / counts.sum(),
            "percent": 100.0 * counts / counts.sum(),
        }
    )

    out_csv = nn_dir / f"injection_rate_distribution_{split_name}.csv"
    summary.to_csv(out_csv, index=False)

    print("\n=== Injection-rate distribution ===")
    print(f"Dataset: {split_name}")
    print(f"Total rows processed: {n_total}")
    print(f"Active injection samples: {n_active}")
    print(f"Number of unique injection rates: {len(rates)}")
    
    print(f"Min injection rate: {rates.min() / 1000.0:.3f} ton/day")
    print(f"Max injection rate: {rates.max() / 1000.0:.3f} ton/day")
    print(f"\nSaved CSV to: {out_csv}")

    # Plot: percent of samples vs injection rate
    # ------------------------------------------------------------
    x = summary["injection_rate_ton_day"].to_numpy()
    weights = summary["n_samples"].to_numpy()

    plt.figure(figsize=(8, 5))

    plt.hist(
        x,
        bins=bins,
        weights=weights * 100.0 / weights.sum(),
        edgecolor="black",
        alpha=0.8,
        label="Training samples per interval"
        #label=f"{bins} injection-rate intervals",
    )

    mean_rate = np.average(x, weights=weights)
    median_rate = weighted_median(x, weights)

    plt.axvline(
        mean_rate,
        color="gray",
        linestyle="-.",
        linewidth=1.8,
        label=f"Mean = {mean_rate:.0f} ton/day",
    )

    plt.axvline(
        median_rate,
        color="black",
        linestyle=":",
        linewidth=2.0,
        label=f"Median = {median_rate:.0f} ton/day",
    )
    midpoint_rate = 0.5 * (x.min() + x.max())
    plt.axvline(
        midpoint_rate,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Midpoint = {midpoint_rate:.0f} ton/day",
    )
    plt.xlabel(r"Injection rate [$\mathrm{ton/day}$]")
    plt.ylabel(f"{split_name.capitalize()} samples [%]")
    plt.title(f"Distribution of injection rates in the {split_name} dataset")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = nn_dir / f"injection_rate_distribution_{split_name}.png"
    out_pdf = nn_dir / f"injection_rate_distribution_{split_name}.pdf"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


    print(f"Saved plot: {out_png}")
    print(f"Saved plot: {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn_dir", required=True)
    parser.add_argument(
        "--split_name",
        default="train",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--chunk_size", type=int, default=200000)
    parser.add_argument("--round_rate_digits", type=int, default=6)
    parser.add_argument("--bins", type=int, default=10)

    args = parser.parse_args()

    main(
        nn_dir=args.nn_dir,
        split_name=args.split_name,
        chunk_size=args.chunk_size,
        round_rate_digits=args.round_rate_digits,
        bins=args.bins,
    )