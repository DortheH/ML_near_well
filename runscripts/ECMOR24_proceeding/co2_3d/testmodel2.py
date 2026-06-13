import argparse
import pathlib
import csv

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
"""
Script description: Evaluate the trained neural-network model on the scaled test dataset.

The script loads the trained model, test data and scaling parameters, predicts WI,
and converts the scaled target values back to log10(WI) and physical WI units.
It computes global test metrics, including RMSE, MAE and relative errors, and
analyzes the prediction residuals during different injection periods. The script
also generates residual histograms and a boxplot of absolute log10(WI) error as a
function of time since injection start or restart.
"""

def read_target_scaling(scalings_csv: pathlib.Path, target_name: str = "output_WI"):
    """Return (y_min, y_max, a, b), where target_range is [a, b]."""
    y_min = y_max = None
    a, b = -1.0, 1.0

    with scalings_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = row["variable"]
            if var == target_name:
                y_min = float(row["min"])
                y_max = float(row["max"])
            elif var == "target_range":
                a = float(row["min"])
                b = float(row["max"])

    if y_min is None or y_max is None:
        raise ValueError(f"Could not find {target_name} in {scalings_csv}")

    return y_min, y_max, a, b


def inverse_minmax(y_scaled, y_min, y_max, a=-1.0, b=1.0):
    return y_min + (y_scaled - a) * (y_max - y_min) / (b - a)


def rmse(y_pred, y_true):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_pred, y_true):
    return float(np.mean(np.abs(y_pred - y_true)))


def residual_histogram_physical_scaled(
    wi_true,
    wi_pred,
    xlabel,
    title,
    savepath,
    scale=1e-5,
    bins=60,
    xlim=(-0.6, 0.6),
):
    """Histogram of physical WI residuals, scaled by scale."""
    wi_true = wi_true.reshape(-1)
    wi_pred = wi_pred.reshape(-1)

    residuals = wi_pred - wi_true
    residuals_scaled = residuals / scale
    residuals_scaled = residuals_scaled[np.isfinite(residuals_scaled)]

    if residuals_scaled.size == 0:
        print(f"No finite residuals for {savepath}")
        return

    plt.figure(figsize=(7, 5))
    plt.hist(residuals_scaled, bins=bins, alpha=0.8)
    plt.axvline(0.0, linestyle="--", linewidth=1.5)

    if xlim is not None:
        plt.xlim(xlim)

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()


def get_feature_column_from_scalings(
    scalings_csv: pathlib.Path,
    candidates: list[str],
) -> tuple[int, str, float, float]:
    """
    Find feature column index in X from scalings.csv.
    Assumes feature rows in scalings.csv are in the same order as columns in X.
    """
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

    print("\nAvailable feature names in scalings.csv:")
    for i, row in enumerate(feature_rows):
        print(f"{i}: {row['variable']}")

    raise ValueError(f"Could not find feature matching: {candidates}")


def get_injection_period_masks_and_time(
    Xs: np.ndarray,
    scalings_csv: pathlib.Path,
    max_days_after_start: float = 10.0,
):
    """
    Creates masks for:
    1. First max_days_after_start days after injection start/restart.
    2. Active injection periods after max_days_after_start days.

    This includes both first injection and later restarts.
    """
    t_col, t_name, t_min, t_max = get_feature_column_from_scalings(
        scalings_csv,
        candidates=["current_injection_time"],
    )

    q_col, q_name, q_min, q_max = get_feature_column_from_scalings(
        scalings_csv,
        candidates=["injection_rate"],
    )

    shut_col, shut_name, shut_min, shut_max = get_feature_column_from_scalings(
        scalings_csv,
        candidates=["previous_shutin_time"],
    )

    current_injection_time = inverse_minmax(
        Xs[:, t_col],
        t_min,
        t_max,
        -1.0,
        1.0,
    )

    injection_rate = inverse_minmax(
        Xs[:, q_col],
        q_min,
        q_max,
        -1.0,
        1.0,
    )

    previous_shutin_time = inverse_minmax(
        Xs[:, shut_col],
        shut_min,
        shut_max,
        -1.0,
        1.0,
    )

    print("\nInjection-period analysis uses:")
    print(f"  current injection time: {t_name}")
    print(f"  injection rate        : {q_name}")
    print(f"  previous shut-in time : {shut_name}")

    active_mask = np.isfinite(injection_rate) & (injection_rate > 0.0)

    first_10_days_mask = (
        active_mask
        & np.isfinite(current_injection_time)
        & (current_injection_time >= 0.0)
        & (current_injection_time <= max_days_after_start)
    )

    later_injection_mask = (
        active_mask
        & np.isfinite(current_injection_time)
        & (current_injection_time > max_days_after_start)
    )

    restart_10_days_mask = (
        first_10_days_mask
        & np.isfinite(previous_shutin_time)
        & (previous_shutin_time > 0.0)
    )

    return (
        first_10_days_mask,
        later_injection_mask,
        restart_10_days_mask,
        current_injection_time,
        active_mask,
    )


def boxplot_abs_error_vs_time_since_start(
    y_true,
    y_pred,
    time_since_start,
    active_mask,
    savepath,
):
    """
    Boxplot of absolute error in log10(WI) grouped by time since injection start/restart.
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    time_since_start = time_since_start.reshape(-1)
    active_mask = active_mask.reshape(-1)

    abs_error = np.abs(y_pred - y_true)

    bins = [
        (0.0, 1.0),
        (1.0, 3.0),
        (3.0, 5.0),
        (5.0, 10.0),
        (10.0, 30.0),
    ]

    data = []
    labels = []

    for low, high in bins:
        mask = (
            active_mask
            & np.isfinite(time_since_start)
            & (time_since_start >= low)
            & (time_since_start < high)
        )

        values = abs_error[mask]

        if len(values) > 0:
            data.append(values)
            labels.append(f"{low:g}-{high:g}")

    if len(data) == 0:
        print("No data available for time-since-start boxplot.")
        return

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xlabel("Time since injection start/restart [days]")
    plt.ylabel(r"Absolute error in $\log_{10}(WI)$")
    plt.title(r"Absolute error vs time since injection start/restart")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()


def main(nn_dir: str, wi_log: bool = True):
    nn_dir = pathlib.Path(nn_dir)

    model_path = nn_dir / "bestmodel.keras"
    test_path = nn_dir / "testset_scaled.npz"
    scalings_path = nn_dir / "scalings.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find: {model_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Could not find: {test_path}")
    if not scalings_path.exists():
        raise FileNotFoundError(f"Could not find: {scalings_path}")

    model = tf.keras.models.load_model(model_path, compile=False)

    data = np.load(test_path)
    Xs = data["X"]
    ys = data["y"]

    yhat_s = model.predict(Xs, verbose=0)

    rmse_s = rmse(yhat_s, ys)
    mae_s = mae(yhat_s, ys)

    y_min, y_max, a, b = read_target_scaling(
        scalings_path,
        target_name="output_WI",
    )

    y_true_unscaled = inverse_minmax(ys, y_min, y_max, a, b)
    y_pred_unscaled = inverse_minmax(yhat_s, y_min, y_max, a, b)

    rmse_unscaled = rmse(y_pred_unscaled, y_true_unscaled)
    mae_unscaled = mae(y_pred_unscaled, y_true_unscaled)

    if wi_log:
        wi_true = 10.0 ** y_true_unscaled
        wi_pred = 10.0 ** y_pred_unscaled
    else:
        wi_true = y_true_unscaled
        wi_pred = y_pred_unscaled

    rmse_wi = rmse(wi_pred, wi_true)
    mae_wi = mae(wi_pred, wi_true)

    wi_min = float(np.min(wi_true))
    wi_max = float(np.max(wi_true))
    wi_range = wi_max - wi_min

    eps = 1e-30
    rel_mae = float(np.mean(np.abs(wi_pred - wi_true) / (np.abs(wi_true) + eps)))
    rel_rmse = float(
        np.sqrt(np.mean(((wi_pred - wi_true) / (np.abs(wi_true) + eps)) ** 2))
    )

    print("\n=== Test metrics ===")
    print(f"Scaled RMSE              : {rmse_s}")
    print(f"Scaled MAE               : {mae_s}")

    if wi_log:
        print(f"log10(WI) min/max        : {y_min} .. {y_max}  (range={y_max - y_min})")
        print(f"RMSE in log10(WI)        : {rmse_unscaled}")
        print(f"MAE  in log10(WI)        : {mae_unscaled}")
    else:
        print(f"Unscaled target min/max  : {y_min} .. {y_max}  (range={y_max - y_min})")
        print(f"RMSE unscaled target     : {rmse_unscaled}")
        print(f"MAE  unscaled target     : {mae_unscaled}")

    print(f"Physical WI min/max      : {wi_min} .. {wi_max}  (range={wi_range})")
    print(f"RMSE in physical WI      : {rmse_wi}")
    print(f"MAE  in physical WI      : {mae_wi}")
    print(f"Relative MAE physical WI : {rel_mae}")
    print(f"Relative RMSE physical WI: {rel_rmse}")

    # ------------------------------------------------------------
    # Masks for injection-period analysis
    # ------------------------------------------------------------
    (
        first_10_days_mask,
        later_injection_mask,
        restart_10_days_mask,
        current_injection_time,
        active_mask,
    ) = get_injection_period_masks_and_time(
        Xs,
        scalings_path,
        max_days_after_start=10.0,
    )

    wi_true_flat = wi_true.reshape(-1)
    wi_pred_flat = wi_pred.reshape(-1)

    print(
        f"\nNumber of samples within first 10 days after injection start/restart: "
        f"{np.sum(first_10_days_mask)}"
    )
    print(
        f"Number of samples within first 10 days after restart only: "
        f"{np.sum(restart_10_days_mask)}"
    )
    print(
        f"Number of samples in later injection periods after first 10 days: "
        f"{np.sum(later_injection_mask)}"
    )

    # ------------------------------------------------------------
    # Plot 1: Global physical residual histogram
    # ------------------------------------------------------------
    global_hist_path = nn_dir / "residual_histogram_physical_WI_scaled_1e-5.png"

    residual_histogram_physical_scaled(
        wi_true,
        wi_pred,
        xlabel=r"Residual $\widehat{WI}-WI$ [$10^{-5}\,\mathrm{m^4\,s/kg}$]",
        title=r"Global distribution of residuals in physical WI",
        savepath=global_hist_path,
        scale=1e-5,
        xlim=(-0.6, 0.6),
    )

    # ------------------------------------------------------------
    # Plot 2: First 10 days after injection start/restart
    # ------------------------------------------------------------
    first_10_days_hist_path = (
        nn_dir / "residual_histogram_first_10days_after_injection_start_physical_WI_scaled_1e-5.png"
    )

    if np.sum(first_10_days_mask) > 0:
        residual_histogram_physical_scaled(
            wi_true_flat[first_10_days_mask],
            wi_pred_flat[first_10_days_mask],
            xlabel=r"Residual $\widehat{WI}-WI$ [$10^{-5}\,\mathrm{m^4\,s/kg}$]",
            title=r"Residuals within first 10 days after injection start/restart",
            savepath=first_10_days_hist_path,
            scale=1e-5,
            xlim=(-0.6, 0.6),
        )
    else:
        print("No samples found within first 10 days after injection start/restart.")

    # ------------------------------------------------------------
    # Plot 3: Later active injection periods
    # ------------------------------------------------------------
    later_injection_hist_path = (
        nn_dir / "residual_histogram_later_injection_periods_physical_WI_scaled_1e-5.png"
    )

    if np.sum(later_injection_mask) > 0:
        residual_histogram_physical_scaled(
            wi_true_flat[later_injection_mask],
            wi_pred_flat[later_injection_mask],
            xlabel=r"Residual $\widehat{WI}-WI$ [$10^{-5}\,\mathrm{m^4\,s/kg}$]",
            title=r"Residuals after the first 10 days of injection",
            savepath=later_injection_hist_path,
            scale=1e-5,
            xlim=(-0.6, 0.6),
        )
    else:
        print("No samples found after the first 10 days of injection.")

    # ------------------------------------------------------------
    # Plot 4: Boxplot of absolute log error vs time since start/restart
    # ------------------------------------------------------------
    boxplot_path = nn_dir / "boxplot_abs_error_vs_injection_time_logWI.png"

    boxplot_abs_error_vs_time_since_start(
        y_true_unscaled,
        y_pred_unscaled,
        current_injection_time,
        active_mask,
        savepath=boxplot_path,
    )

    print("\nSaved plots:")
    print(f"  {global_hist_path}")
    print(f"  {first_10_days_hist_path}")
    print(f"  {later_injection_hist_path}")
    print(f"  {boxplot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nn_dir",
        required=True,
        help="Folder with bestmodel.keras, testset_scaled.npz and scalings.csv",
    )
    parser.add_argument(
        "--wi_log",
        action="store_true",
        help="Use this if the model target is log10(WI).",
    )
    args = parser.parse_args()

    main(args.nn_dir, wi_log=args.wi_log)