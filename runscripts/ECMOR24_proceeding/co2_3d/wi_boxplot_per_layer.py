import argparse
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
"""
Description: Analyze the distribution of WI across reservoir layers at a fixed radial index.

The script loads the dataset and row-to-run mapping, filters the samples at a
selected radial position, and groups the WI values by permeability layer. The WI
target is converted from log10(WI) to physical WI units and scaled by 1e-5 for
plotting. The script prints summary statistics for each layer and generates a
boxplot showing the layer-wise WI distribution.
"""

def main(dataset_dir: str, fixed_x_id: int = 5, wi_log: bool = True, batch_size: int = 100000):
    dataset_dir = pathlib.Path(dataset_dir)

    map_path = dataset_dir / "row_to_run_map.csv"
    if not map_path.exists():
        raise FileNotFoundError(f"Could not find {map_path}")

    row_map = pd.read_csv(map_path)

    required_cols = {"layer_id", "x_id"}
    missing = required_cols - set(row_map.columns)
    if missing:
        raise ValueError(f"row_to_run_map.csv is missing columns: {missing}")

    layer_ids_all = row_map["layer_id"].to_numpy()
    x_ids_all = row_map["x_id"].to_numpy()
    n_rows = len(row_map)

    ds = tf.data.Dataset.load(str(dataset_dir))

    values_by_layer = {layer: [] for layer in sorted(row_map["layer_id"].unique())}

    row_start = 0

    print(f"Loading dataset from: {dataset_dir}", flush=True)
    print(f"Filtering fixed x_id = {fixed_x_id}", flush=True)

    for batch_idx, (_, y_batch) in enumerate(ds.batch(batch_size).as_numpy_iterator()):
        y_batch = y_batch.reshape(-1)
        n_batch = len(y_batch)
        row_end = row_start + n_batch

        if row_end > n_rows:
            raise ValueError(
                f"Dataset has more rows than row_to_run_map.csv. "
                f"row_end={row_end}, n_rows={n_rows}"
            )

        layer_batch = layer_ids_all[row_start:row_end]
        x_batch = x_ids_all[row_start:row_end]

        keep = x_batch == fixed_x_id

        y_batch = y_batch[keep]
        layer_batch = layer_batch[keep]

        if wi_log:
            wi_s = 10.0 ** y_batch
        else:
            wi_s = y_batch

        # Keep WI in original unit: m^4 s/kg
        wi_values = wi_s / 1e-5
        
        valid = np.isfinite(wi_values)
        wi_values = wi_values[valid]
        layer_batch = layer_batch[valid]

        for layer in np.unique(layer_batch):
            layer = int(layer)
            values_by_layer[layer].extend(wi_values[layer_batch == layer].tolist())

        row_start = row_end

        if batch_idx % 10 == 0:
            print(f"Processed {row_start}/{n_rows} rows", flush=True)

    if row_start != n_rows:
        raise ValueError(
            f"Dataset rows and row_to_run_map.csv rows do not match. "
            f"Processed {row_start}, row map has {n_rows}."
        )

    layers = sorted(values_by_layer.keys())
    data = [np.array(values_by_layer[layer]) for layer in layers]

    for layer, vals in zip(layers, data):
        if len(vals) == 0:
            raise ValueError(f"No values found for layer {layer} at x_id={fixed_x_id}")

    print("\n=== WI distribution per layer at fixed x_id ===")
    print("Layer | n samples | min | q25 | median | q75 | max | range")
    for layer, vals in zip(layers, data):
        print(
            f"{layer:5d} | "
            f"{len(vals):9d} | "
            f"{np.min(vals):.5f} | "
            f"{np.percentile(vals, 25):.5f} | "
            f"{np.median(vals):.5f} | "
            f"{np.percentile(vals, 75):.5f} | "
            f"{np.max(vals):.5f} | "
            f"{np.max(vals) - np.min(vals):.5f}"
        )

    plt.figure(figsize=(7, 4.8))

    plt.boxplot(
        data,
        labels=[f"Layer {layer}" for layer in layers],
        showfliers=True,
        whis=(5, 95),
        flierprops=dict(
            marker="o",
            markersize=0.8,
            markerfacecolor="black",
            markeredgecolor="black",
            alpha=0.2,
        ),
    )

    out_png = f"wi_boxplot_per_layer_xid{fixed_x_id}_seconds.png"
    out_pdf = f"wi_boxplot_per_layer_xid{fixed_x_id}_seconds.pdf"
    
    plt.ylabel(r"WI [$10^{-5}\,\mathrm{m^4\,s/kg}$]")
    plt.xlabel("Reservoir layer")
    
    # Horizontal grid lines behind the boxplot
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.4, linewidth=1.0)

    plt.tight_layout()

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.show()

    print(f"\nSaved:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--x_id", type=int, default=5)
    parser.add_argument("--wi_log", action="store_true")
    parser.add_argument("--batch_size", type=int, default=10000)

    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        fixed_x_id=args.x_id,
        wi_log=args.wi_log,
        batch_size=args.batch_size,
    )