"""
Description: Plot true and predicted WI as a function of time for selected test simulations.

The script evaluates the trained neural-network model on test samples and
compares the predicted WI with the true WI from the simulation data. The results
are plotted over time, typically for selected ensemble members, radial positions
or reservoir layers, to visually assess how well the model reproduces the
time-dependent WI response during injection, shut-in and restart periods.
"""

from __future__ import annotations

import math
import pathlib
import sys
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Make parent directory importable
dirname = pathlib.Path(__file__).resolve().parent
sys.path.append(str(dirname / ".."))

from pyopmnearwell.ml import nn
from pyopmnearwell.utils import units
from runspecs import runspecs_ensemble, trainspecs
WI_AXIS_SCALE = 1e-5

def _get_history_window_days() -> float:
    return float(runspecs_ensemble["constants"].get("HISTORY_WINDOW_DAYS", 180.0))


def _get_raw_feature_indices(raw_features: np.ndarray) -> dict[str, int | None]:
    """
    Infer raw feature layout from the saved dataset.

    New layout from your current upscaler:
      0  pressure
      1  saturation
      2  radius
      3  total_injected_volume (FGIT)
      4  injection_rate (WGIR)
      5  current_injection_time
      6  previous_shutin_time
      7  previous_injection_time
      8  older_history_time
      9  time_days
      10 PI_analytical
    """
    nraw = raw_features.shape[-1]

    if nraw >= 11:
        return {
            "pressure": 0,
            "saturation": 1,
            "radius": 2,
            "total_injected_volume": 3,
            "injection_rate": 4,
            "current_injection_time": 5,
            "previous_shutin_time": 6,
            "previous_injection_time": 7,
            "older_history_time": 8,
            "time_days": 9,
            "PI_analytical": 10,
        }

    raise ValueError(
        f"Unsupported raw feature count: {nraw}. Expected >= 11 for the new dataset."
    )


def build_shaped_stencil_features(
    raw_features: np.ndarray,
    trainspecs: dict,
    stencil_size: int = 3,
) -> np.ndarray:
    """
    Rebuild exactly the same stencil features as in restructure_data().

    Input:
        raw_features shape = (nmembers, nt, nlayers, nx, nraw)

    Output:
        shaped features = (nmembers, nt, nlayers, nx, nfeat)
    """
    idx_map = _get_raw_feature_indices(raw_features)
    new_features_lst: list[np.ndarray] = []

    # Local stencil features: pressure and saturation
    for name in ["pressure", "saturation"]:
        feature = raw_features[..., idx_map[name]].copy()

        if name == "pressure":
            if trainspecs["pressure_unit"] == "bar":
                feature = feature * units.PASCAL_TO_BAR

            if trainspecs["pressure_padding"] == "zeros":
                padding_mode = "constant"
                padding_value = 0.0
            else:
                padding_mode = "edge"
                padding_value = 0.0

        else:  # saturation
            if trainspecs["saturation_padding"] == "zeros":
                padding_mode = "constant"
                padding_value = 0.0
            else:
                padding_mode = "edge"
                padding_value = 0.0

        half = math.floor(stencil_size / 2)

        if padding_mode == "constant":
            upper_features = [
                np.pad(
                    feature[:, :, : -(j + 1), ...],
                    [(0, 0) if k != 2 else ((j + 1), 0) for k in range(feature.ndim)],
                    mode=padding_mode,
                    constant_values=padding_value,
                )
                for j in range(half)
            ]
            lower_features = [
                np.pad(
                    feature[:, :, (j + 1) :, ...],
                    [(0, 0) if k != 2 else (0, (j + 1)) for k in range(feature.ndim)],
                    mode=padding_mode,
                    constant_values=padding_value,
                )
                for j in range(half)
            ]
        else:
            upper_features = [
                np.pad(
                    feature[:, :, : -(j + 1), ...],
                    [(0, 0) if k != 2 else ((j + 1), 0) for k in range(feature.ndim)],
                    mode=padding_mode,
                )
                for j in range(half)
            ]
            lower_features = [
                np.pad(
                    feature[:, :, (j + 1) :, ...],
                    [(0, 0) if k != 2 else (0, (j + 1)) for k in range(feature.ndim)],
                    mode=padding_mode,
                )
                for j in range(half)
            ]

        new_features_lst.extend(upper_features + [feature] + lower_features)

    # Global / engineered features
    new_features_lst.append(raw_features[..., idx_map["radius"]])
    new_features_lst.append(raw_features[..., idx_map["total_injected_volume"]])
    new_features_lst.append(raw_features[..., idx_map["injection_rate"]])
    new_features_lst.append(raw_features[..., idx_map["current_injection_time"]])
    new_features_lst.append(raw_features[..., idx_map["previous_shutin_time"]])
    new_features_lst.append(raw_features[..., idx_map["previous_injection_time"]])
    new_features_lst.append(raw_features[..., idx_map["older_history_time"]])

    pi_feature = raw_features[..., idx_map["PI_analytical"]]
    eps = 1e-12
    if trainspecs["WI_log"]:
        pi_safe = np.where(np.isfinite(pi_feature) & (pi_feature > 0), pi_feature, 1.0)
        new_features_lst.append(np.log10(np.maximum(pi_safe, eps)))
    else:
        new_features_lst.append(
            np.nan_to_num(pi_feature, nan=0.0, posinf=0.0, neginf=0.0)
        )

    all_features = np.stack(new_features_lst, axis=-1)

    feature_to_index = {
        "pressure_upper": 0,
        "pressure": 1,
        "pressure_lower": 2,
        "saturation_upper": 3,
        "saturation": 4,
        "saturation_lower": 5,
        "radius": 6,
        "total_injected_volume": 7,
        "injection_rate": 8,
        "current_injection_time": 9,
        "previous_shutin_time": 10,
        "previous_injection_time": 11,
        "older_history_time": 12,
        "PI_analytical": 13,
    }

    selected = all_features[..., [feature_to_index[f] for f in trainspecs["features"]]]
    return selected.astype(np.float32)


def predict_shaped(
    raw_features: np.ndarray,
    model: keras.Model,
    nn_dir: pathlib.Path,
) -> np.ndarray:
    """
    Predict WI on the full shaped raw dataset.

    Returns:
        y_pred with shape (nmembers, nt, nlayers, nx)
    """
    x_shaped = build_shaped_stencil_features(raw_features, trainspecs, stencil_size=3)
    saved_shape = x_shaped.shape

    x_flat = x_shaped.reshape(-1, saved_shape[-1])

    y_pred_scaled = nn.scale_and_evaluate(
        model,
        x_flat,
        nn_dir / "scalings.csv",
    ).numpy().reshape(saved_shape[:-1])

    if trainspecs["WI_log"]:
        y_pred = 10 ** y_pred_scaled
    else:
        y_pred = y_pred_scaled

    return y_pred

def _get_plot_mask_from_q_and_target(
    raw_features: np.ndarray,
    raw_targets: np.ndarray,
    member: int,
    layer: int,
    radius_index: int,
) -> np.ndarray:
    """
    Plot only where injection is active (Q > 0) and true WI is defined.
    """
    idx_map = _get_raw_feature_indices(raw_features)
    inj_rate_idx = idx_map["injection_rate"]

    q_layer = raw_features[member, :, layer, radius_index, inj_rate_idx]
    y_true_layer = raw_targets[member, :, layer, radius_index]

    valid = (q_layer > 0) & np.isfinite(y_true_layer) & (y_true_layer > 0)
    return valid

def plot_member_wi_vs_time(
    raw_features: np.ndarray,
    raw_targets: np.ndarray,
    model: keras.Model,
    nn_dir: pathlib.Path,
    member: int = 0,
    radius_index: int = 3,
    savepath: pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Plot true and predicted WI vs time for one member at one radius index."""
    x_shaped = build_shaped_stencil_features(raw_features, trainspecs, stencil_size=3)

    x_member = x_shaped[member]
    y_member = raw_targets[member]

    saved_shape = list(x_member.shape)
    x_flat = x_member.reshape(-1, saved_shape[-1])

    y_pred_scaled = nn.scale_and_evaluate(
        model,
        x_flat,
        nn_dir / "scalings.csv",
    ).numpy().reshape(saved_shape[:-1])

    y_pred = 10 ** y_pred_scaled if trainspecs["WI_log"] else y_pred_scaled
    y_true = y_member.copy()

    idx_map = _get_raw_feature_indices(raw_features)
    time_idx = idx_map["time_days"]
    x_values = raw_features[member, :, 0, 0, time_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.get_cmap("tab10", y_true.shape[1])
    colors = [cmap(i) for i in range(y_true.shape[1])]

    for layer, color in zip(range(y_true.shape[1]), colors):
        y_true_layer = y_true[:, layer, radius_index]
        y_pred_layer = y_pred[:, layer, radius_index]

        valid = _get_plot_mask_from_q_and_target(
            raw_features=raw_features,
            raw_targets=raw_targets,
            member=member,
            layer=layer,
            radius_index=radius_index,
        )

        y_true_plot = np.where(valid, y_true_layer / WI_AXIS_SCALE, np.nan)
        y_pred_plot = np.where(valid, y_pred_layer / WI_AXIS_SCALE, np.nan)

        ax.scatter(
            x_values,
            y_true_plot,
            color=color,
            edgecolors="black",
            linewidths=0.4,
            s=22,
            label=f"layer {layer}: WI data",
        )
        ax.plot(
            x_values,
            y_pred_plot,
            color=color,
            linewidth=2.0,
            label=f"layer {layer}: WI NN",
        )

    ax.set_xlabel("Time [days]")
    ax.set_ylabel(r"WI [$10^{-5}$ m$^4$ s/kg]")
    ax.set_title(f"WI vs time for member {member} at radius index {radius_index}")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.72, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if savepath is None:
        savepath = nn_dir / f"member_{member}_WI_vs_time_radius_{radius_index}.png"

    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
        
def plot_wi_vs_time_for_multiple_members(
    raw_features: np.ndarray,
    raw_targets: np.ndarray,
    model: keras.Model,
    nn_dir: pathlib.Path,
    radius_index: int = 3,
    members: list[int] | None = None,
    n_members: int = 10,
) -> None:
    """
    Plot WI vs time for several members and save one figure per member.
    """
    nmembers = raw_features.shape[0]

    if members is None:
        members = list(range(min(n_members, nmembers)))

    plot_dir = nn_dir / "wi_plots_new"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for member in members:
        print(f"Plotting WI vs time for member {member}")
        plot_member_wi_vs_time(
            raw_features=raw_features,
            raw_targets=raw_targets,
            model=model,
            nn_dir=nn_dir,
            member=member,
            radius_index=radius_index,
            savepath=plot_dir / f"member_{member}_WI_vs_time_radius_{radius_index}.png",
            show=False,
        )


def plot_member_wi_vs_radius(
    raw_features: np.ndarray,
    raw_targets: np.ndarray,
    model: keras.Model,
    nn_dir: pathlib.Path,
    member: int = 0,
    timestep_index: int = 10,
    savepath: pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot true and predicted WI vs radius for one member at one timestep.
    """
    y_pred = predict_shaped(raw_features, model, nn_dir)

    y_true_member = raw_targets[member]
    y_pred_member = y_pred[member]

    radius_idx = _get_raw_feature_indices(raw_features)["radius"]
    x_values = raw_features[member, timestep_index, 0, :, radius_idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.get_cmap("tab10", y_true_member.shape[1])
    colors = [cmap(i) for i in range(y_true_member.shape[1])]

    for layer, color in zip(range(y_true_member.shape[1]), colors):
        y_true_layer = y_true_member[timestep_index, layer, :]
        y_pred_layer = y_pred_member[timestep_index, layer, :]

        valid = np.isfinite(y_true_layer) & (y_true_layer > 0)

        y_true_layer_scaled = y_true_layer / WI_AXIS_SCALE
        y_pred_layer_scaled = y_pred_layer / WI_AXIS_SCALE

        ax.scatter(
            x_values[valid],
            y_true_layer_scaled[valid],
            color=color,
            edgecolors="black",
            linewidths=0.4,
            s=24,
            label=f"layer {layer}: true WI",
        )

        ax.plot(
            x_values[valid],
            y_pred_layer_scaled[valid],
            color=color,
            linewidth=2.0,
            label=f"layer {layer}: predicted WI",
        )

    ax.set_xlabel("Radius [m]")
    ax.set_ylabel(r"WI [$10^{-5}$ m$^4$ s/kg]")
    ax.set_title(f"WI vs radius for member {member}, timestep {timestep_index}")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.72, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if savepath is None:
        savepath = nn_dir / f"member_{member}_WI_vs_radius_timestep_{timestep_index}.png"

    plt.savefig(savepath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
        
if __name__ == "__main__":
    data_dir = dirname / "dataset"
    nn_dir = dirname / "nn"

    print(f"Loading dataset from: {data_dir}")
    ds = tf.data.Dataset.load(str(data_dir))

    raw_features, raw_targets = next(
        iter(ds.batch(batch_size=len(ds)).as_numpy_iterator())
    )

    print("raw_features shape:", raw_features.shape)
    print("raw_targets shape :", raw_targets.shape)
    print("valid WI count    :", np.sum(np.isfinite(raw_targets) & (raw_targets > 0)))
    print("WI min/max        :", np.nanmin(raw_targets), np.nanmax(raw_targets))

    print(f"Loading model from: {nn_dir / 'bestmodel.keras'}")
    model = keras.models.load_model(nn_dir / "bestmodel.keras")

    plot_dir = nn_dir / "wi_plots_new"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("Making WI vs time plots...")
    plot_wi_vs_time_for_multiple_members(
        raw_features=raw_features,
        raw_targets=raw_targets,
        model=model,
        nn_dir=nn_dir,
        radius_index=2,
        n_members=20,
    )

    print("Making WI vs radius plot...")
    plot_member_wi_vs_radius(
        raw_features=raw_features,
        raw_targets=raw_targets,
        model=model,
        nn_dir=nn_dir,
        member=0,
        timestep_index=10,
        savepath=plot_dir / "member_0_WI_vs_radius_timestep_10.png",
        show=False,
    )

    print(f"Done. Plots saved in: {plot_dir}")