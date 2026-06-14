"""
Analyze the distribution of previous shut-in durations in the training dataset.

The script loads the scaled dataset and scaling parameters, identifies the
injection-rate and previous-shut-in-time features, and inverse-scales them back
to physical units. It keeps only active-injection samples with a previous shut-in
period, counts the occurrence of each shut-in duration, and computes summary
statistics such as the weighted mean, weighted median and midpoint of the design
range. The results are saved to a CSV file and visualized as a histogram showing
the percentage of samples in each shut-in-duration interval.
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


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cumulative_weight = np.cumsum(weights)
    cutoff = 0.5 * np.sum(weights)

    return values[np.searchsorted(cumulative_weight, cutoff)]


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


def main(
    nn_dir: str,
    split_name: str,
    chunk_size: int,
    round_digits: int,
    bins: int,
    expected_min: float,
    expected_max: float,
):
    nn_dir = pathlib.Path(nn_dir)

    npz_path = nn_dir / f"{split_name}set_scaled.npz"
    scalings_path = nn_dir / "scalings.csv"

    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find: {npz_path}")

    if not scalings_path.exists():
        raise FileNotFoundError(f"Could not find: {scalings_path}")

    # Find injection-rate column so we can keep only active-injection samples
    q_col, q_name, q_min, q_max = get_feature_column_from_scalings(
        scalings_path,
        candidates=["injection_rate"],
    )

    # Find previous shut-in time column
    shut_col, shut_name, shut_min, shut_max = get_feature_column_from_scalings(
        scalings_path,
        candidates=["previous_shutin_time", "previous_shut", "shutin"],
    )

    print(f"Reading: {npz_path}")
    print(f"Using injection-rate feature: {q_name}, column {q_col}")
    print(f"Using shut-in feature: {shut_name}, column {shut_col}")

    data = np.load(npz_path, mmap_mode="r")
    X = data["X"]

    n_rows = X.shape[0]
    print(f"X shape: {X.shape}")
    print(f"Processing in chunks of {chunk_size}")

    counter = Counter()
    n_total = 0
    n_active = 0
    n_with_previous_shutin = 0

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)

        q_scaled = X[start:end, q_col]
        shut_scaled = X[start:end, shut_col]

        q = inverse_minmax(q_scaled, q_min, q_max, -1.0, 1.0)
        previous_shutin_time = inverse_minmax(
            shut_scaled,
            shut_min,
            shut_max,
            -1.0,
            1.0,
        )

        active = np.isfinite(q) & (q > 0.0)

        # Keep only active injection samples that have a previous shut-in period
        keep = (
            active
            & np.isfinite(previous_shutin_time)
            & (previous_shutin_time > 0.0)
        )

        shut_keep = previous_shutin_time[keep]

        shut_group = np.round(shut_keep, round_digits)
        counter.update(shut_group.tolist())

        n_total += end - start
        n_active += int(np.sum(active))
        n_with_previous_shutin += int(shut_group.size)

        print(f"Processed {end}/{n_rows}", flush=True)

    if n_with_previous_shutin == 0:
        raise RuntimeError("No samples with previous_shutin_time > 0 found.")

    durations = np.array(sorted(counter.keys()), dtype=float)
    counts = np.array([counter[d] for d in durations], dtype=int)

    summary = pd.DataFrame(
        {
            "previous_shutin_time_days": durations,
            "n_samples": counts,
            "fraction": counts / counts.sum(),
            "percent": 100.0 * counts / counts.sum(),
        }
    )

    out_csv = nn_dir / f"previous_shutin_distribution_{split_name}.csv"
    summary.to_csv(out_csv, index=False)

    mean_shut = np.average(durations, weights=counts)
    median_shut = weighted_median(durations, counts)

    # Midpoint of the design interval, e.g. 7 to 42 days
    midpoint = 0.5 * (expected_min + expected_max)

    median_shift_percent = 100.0 * (median_shut - midpoint) / midpoint
    mean_shift_percent = 100.0 * (mean_shut - midpoint) / midpoint

    print("\n=== Previous shut-in time distribution ===")
    print(f"Dataset: {split_name}")
    print(f"Total rows processed: {n_total}")
    print(f"Active injection samples: {n_active}")
    print(f"Samples with previous shut-in time > 0: {n_with_previous_shutin}")
    print(f"Number of unique previous shut-in durations: {len(durations)}")
    print(f"Min previous shut-in time: {durations.min():.3f} days")
    print(f"Max previous shut-in time: {durations.max():.3f} days")
    print(f"Weighted mean: {mean_shut:.3f} days")
    print(f"Weighted median: {median_shut:.3f} days")
    print(f"Design midpoint: {midpoint:.3f} days")
    print(f"Mean shift from midpoint: {mean_shift_percent:+.2f}%")
    print(f"Median shift from midpoint: {median_shift_percent:+.2f}%")
    print(f"\nSaved CSV to: {out_csv}")

    # ------------------------------------------------------------
    # Plot: percent of samples vs previous shut-in time
    # ------------------------------------------------------------
    x = summary["previous_shutin_time_days"].to_numpy()
    weights = summary["n_samples"].to_numpy()

    plt.figure(figsize=(8, 5))

    hist_values, bin_edges, _ = plt.hist(
        x,
        bins=bins,
        weights=weights * 100.0 / weights.sum(),
        edgecolor="black",
        alpha=0.8,
        label="Training samples per interval",
    )

    plt.ylim(0, hist_values.max() * 1.25)
    
    plt.axvline(
        mean_shut,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean = {mean_shut:.1f} days",
    )

    plt.axvline(
        median_shut,
        color="black",
        linestyle=":",
        linewidth=2.0,
        label=f"Median = {median_shut:.1f} days",
    )

    plt.axvline(
        midpoint,
        color="gray",
        linestyle="-.",
        linewidth=1.8,
        label=f"Range midpoint = {midpoint:.1f} days",
    )

    plt.xlabel("Shut-in duration [days]")
    plt.ylabel(f"{split_name.capitalize()} samples [%]")
    plt.title(f"Distribution of previous shut-in durations in the {split_name} dataset")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()

    out_png = nn_dir / f"previous_shutin_distribution_{split_name}.png"
    out_pdf = nn_dir / f"previous_shutin_distribution_{split_name}.pdf"

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
    parser.add_argument("--round_digits", type=int, default=3)
    parser.add_argument("--bins", type=int, default=10)

    # The design range from data generation
    parser.add_argument("--expected_min", type=float, default=7.0)
    parser.add_argument("--expected_max", type=float, default=42.0)

    args = parser.parse_args()

    main(
        nn_dir=args.nn_dir,
        split_name=args.split_name,
        chunk_size=args.chunk_size,
        round_digits=args.round_digits,
        bins=args.bins,
        expected_min=args.expected_min,
        expected_max=args.expected_max,
    )