"""
Compare bottom-hole pressure (BHP) results from the fine-scale benchmark,
NN-based simulations and Peaceman-based simulations.

The script reads Eclipse summary files from seven simulation cases and extracts
the bottom-hole pressure for the injection well during active injection periods.
It plots the BHP response for the full simulation period and for selected
injection intervals, using different line styles for the fine-scale benchmark,
NN cases and Peaceman cases. The script also computes the maximum and mean
absolute BHP error for each coarse-scale case relative to the fine-scale
benchmark and saves these error values to a CSV file.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from ecl.summary import EclSum
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14

# ------------------------------------------------------------
# Mappe med de 7 resultatene
# ------------------------------------------------------------
DEFAULT_BASE_DIR = pathlib.Path(__file__).parent / "result5" / "integration"

if len(sys.argv) > 1:
    base_dir = pathlib.Path(sys.argv[1])
else:
    base_dir = DEFAULT_BASE_DIR


labels = [
    "Fine-scale benchmark",
    "90x90m NN",
    "52x52m NN",
    "27x27m NN",
    "90x90m Peaceman",
    "52x52m Peaceman",
    "27x27m Peaceman",
]

runs = [
    ("run_0", "8X8M_PEACEMAN_MORE_ZCELLS"),
    ("run_1", "90X90M_NN"),
    ("run_2", "52X52M_NN"),
    ("run_3", "27X27M_NN"),
    ("run_4", "90X90M_PEACEMAN"),
    ("run_5", "52X52M_PEACEMAN"),
    ("run_6", "27X27M_PEACEMAN"),
]


def find_smspec(base_dir: pathlib.Path, run_dir: str, case_name: str) -> pathlib.Path | None:
    expected = base_dir / run_dir / "output" / f"{case_name}.SMSPEC"
    if expected.exists():
        return expected

    matches = sorted(base_dir.rglob(f"{case_name}.SMSPEC"))
    if matches:
        return matches[0]

    run_path = base_dir / run_dir
    if run_path.exists():
        matches = sorted(run_path.rglob("*.SMSPEC"))
        if matches:
            return matches[0]

    return None


def read_bhp(summary_file: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    print(f"Reading {summary_file}", flush=True)

    summary = EclSum(str(summary_file))

    time = np.array(summary.get_values("TIME", report_only=True), dtype=float)
    bhp = np.array(summary.get_values("WBHP:INJ0", report_only=True), dtype=float)

    wgir = np.array(summary.get_values("WGIR:INJ0", report_only=True), dtype=float)
    injecting = wgir > 1e-8

    bhp_plot = bhp.copy()
    bhp_plot[~injecting] = np.nan

    return time, bhp_plot


def style_for_label(label: str) -> tuple[float, str, str]:
    # returnerer: linewidth, linestyle, color

    if label.startswith("Fine-scale"):
        return 2.8, "solid", "black"

    if label == "90x90m NN":
        return 2.2, "dashed", "tab:orange"

    if label == "52x52m NN":
        return 2.2, "dashed", "tab:green"

    if label == "27x27m NN":
        return 2.2, "dashed", "tab:red"

    if label == "90x90m Peaceman":
        return 2.2, "dotted", "navy"

    if label == "52x52m Peaceman":
        return 2.2, "dotted", "royalblue"

    if label == "27x27m Peaceman":
        return 2.2, "dotted", "skyblue"

    return 2.2, "solid", "gray"


def plot_period(
    base_dir: pathlib.Path,
    all_times: list[np.ndarray],
    all_bhps: list[np.ndarray],
    labels: list[str],
    period_name: str,
    start_day: float,
    end_day: float,
    output_stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))

    period_bhps = []

    for time, bhp, label in zip(all_times, all_bhps, labels):
        mask = (time >= start_day) & (time <= end_day)

        time_period = time[mask]
        bhp_period = bhp[mask]
        period_bhps.append(bhp_period)

        linewidth, linestyle, color = style_for_label(label)

        ax.plot(
            time_period,
            bhp_period,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
        )

    valid_bhp_values = [
        bhp[np.isfinite(bhp)]
        for bhp in period_bhps
        if np.isfinite(bhp).any()
    ]

    if valid_bhp_values:
        all_bhp_values = np.concatenate(valid_bhp_values)
        ymin = np.nanmin(all_bhp_values)
        ymax = np.nanmax(all_bhp_values)

        padding = 0.05 * (ymax - ymin)
        if padding == 0:
            padding = 1.0

        ax.set_ylim(ymin - padding, ymax + padding)

    ax.set_xlim(start_day, end_day)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel("Time since injection start (days)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Bottom hole pressure (bar)", fontsize=LABEL_FONTSIZE)
    #ax.set_title(period_name, fontsize=TITLE_FONTSIZE)

    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FONTSIZE)

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.4)

    ax.legend(fontsize=LEGEND_FONTSIZE)

    fig.tight_layout()

    svg_path = base_dir / f"{output_stem}.svg"
    png_path = base_dir / f"{output_stem}.png"

    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"Saved {svg_path}", flush=True)
    print(f"Saved {png_path}", flush=True)


def interpolate_ignoring_nans(
    reference_time: np.ndarray,
    time: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(time) & np.isfinite(values)

    if valid.sum() < 2:
        return np.full_like(reference_time, np.nan, dtype=float)

    return np.interp(
        reference_time,
        time[valid],
        values[valid],
        left=np.nan,
        right=np.nan,
    )


print(f"Base directory: {base_dir}", flush=True)

if not base_dir.exists():
    raise SystemExit(f"Fant ikke mappen: {base_dir}")


# ------------------------------------------------------------
# Finn og les alle 7 SMSPEC-filer først
# ------------------------------------------------------------
summary_files = []

for run_dir, case_name in runs:
    path = find_smspec(base_dir, run_dir, case_name)
    summary_files.append(path)

missing = [
    (run_dir, case_name)
    for (run_dir, case_name), path in zip(runs, summary_files)
    if path is None
]

if missing:
    print("\nMissing SMSPEC files:", flush=True)
    for run_dir, case_name in missing:
        print(f"  {run_dir}: {case_name}.SMSPEC", flush=True)

    print("\nFant disse SMSPEC-filene i mappen:", flush=True)
    for path in sorted(base_dir.rglob("*.SMSPEC")):
        print(f"  {path}", flush=True)

    raise SystemExit("Stopper fordi ikke alle 7 SMSPEC-filene ble funnet.")


all_times: list[np.ndarray] = []
all_bhps: list[np.ndarray] = []

for path, label in zip(summary_files, labels):
    assert path is not None
    time, bhp = read_bhp(path)
    all_times.append(time)
    all_bhps.append(bhp)


# ------------------------------------------------------------
# Plot hele perioden
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 9))

for time, bhp, label in zip(all_times, all_bhps, labels):
    linewidth, linestyle, color = style_for_label(label)

    ax.plot(
        time,
        bhp,
        label=label,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
    )

valid_bhp_values = [
    bhp[np.isfinite(bhp)]
    for bhp in all_bhps
    if np.isfinite(bhp).any()
]

if valid_bhp_values:
    all_bhp_values = np.concatenate(valid_bhp_values)
    ymin = np.nanmin(all_bhp_values)
    ymax = np.nanmax(all_bhp_values)

    padding = 0.05 * (ymax - ymin)
    if padding == 0:
        padding = 1.0

    ax.set_ylim(ymin - padding, ymax + padding)

ax.set_xlabel("Time since injection start (days)", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Bottom hole pressure (bar)", fontsize=LABEL_FONTSIZE)
#ax.set_title("BHP comparison: full period", fontsize=TITLE_FONTSIZE)

ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
ax.tick_params(axis="both", which="minor", labelsize=TICK_FONTSIZE)
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.4)
ax.legend(fontsize=LEGEND_FONTSIZE)
fig.tight_layout()


svg_path = base_dir / "bhp_all_7_full_period.svg"
png_path = base_dir / "bhp_all_7_full_period.png"

fig.savefig(svg_path)
fig.savefig(png_path, dpi=300)
plt.close(fig)

print(f"Saved {svg_path}", flush=True)
print(f"Saved {png_path}", flush=True)


# ------------------------------------------------------------
# Plot første og andre injeksjonsperiode separat
# ------------------------------------------------------------
plot_period(
    base_dir=base_dir,
    all_times=all_times,
    all_bhps=all_bhps,
    labels=labels,
    period_name="BHP comparison: first injection period (days 0-30)",
    start_day=0.0,
    end_day=20.0,
    output_stem="bhp_injection_period_1_days_0_30_all_7",
)

plot_period(
    base_dir=base_dir,
    all_times=all_times,
    all_bhps=all_bhps,
    labels=labels,
    period_name="BHP comparison: first injection period (days 30-75)",
    start_day=20.0,
    end_day=75.0,
    output_stem="bhp_injection_period_1_days_30_75_all_7",
)

plot_period(
    base_dir=base_dir,
    all_times=all_times,
    all_bhps=all_bhps,
    labels=labels,
    period_name="BHP comparison: second injection period",
    start_day=108.0,
    end_day=160.0,
    output_stem="bhp_injection_period_2_all_7",
)


# ------------------------------------------------------------
# CSV med feil mot fine-scale benchmark
# ------------------------------------------------------------
reference_time = all_times[0]
reference_bhp = all_bhps[0]

csv_path = base_dir / "bhp_diffs_all_7.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "label",
            "max_abs_error",
            "mean_abs_error",
            "num_valid_points",
        ]
    )

    for label, time, bhp in zip(labels[1:], all_times[1:], all_bhps[1:]):
        if len(time) == len(reference_time) and np.allclose(time, reference_time):
            bhp_on_ref_time = bhp
        else:
            bhp_on_ref_time = interpolate_ignoring_nans(reference_time, time, bhp)

        valid = np.isfinite(reference_bhp) & np.isfinite(bhp_on_ref_time)

        if valid.sum() == 0:
            max_abs_error = np.nan
            mean_abs_error = np.nan
        else:
            abs_error = np.abs(bhp_on_ref_time[valid] - reference_bhp[valid])
            max_abs_error = float(np.max(abs_error))
            mean_abs_error = float(np.mean(abs_error))

        writer.writerow(
            [
                label,
                max_abs_error,
                mean_abs_error,
                int(valid.sum()),
            ]
        )

print(f"Saved {csv_path}", flush=True)
print("\nDone.", flush=True)