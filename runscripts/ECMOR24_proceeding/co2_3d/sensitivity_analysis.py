"""
Run sensitivity analysis for the trained FCNN model.

The script loads the trained neural-network model and evaluates how the predicted
WI response changes when each normalized input feature is varied from -1 to 1.
For each feature, the remaining input features are held at fixed normalized
levels, producing a family of response curves. The resulting plots are used to
identify which input variables have the strongest influence on the model output
and to assess whether the learned relationships are physically reasonable.
The figure is saved both as a high-resolution PNG and as a PDF for use in the
thesis or publication material.
"""
from __future__ import annotations

import math
import pathlib
from typing import Optional
from matplotlib import colors, cm
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from pyopmnearwell.ml.analysis import plot_analysis

from pyopmnearwell.utils import plotting
from runspecs import trainspecs

NN_DIR = pathlib.Path("nn")
MODEL_PATHS = [
    NN_DIR / "bestmodel.keras",
    NN_DIR / "best_model.keras",
]

PLOT_FEATURE_NAMES = [
    "Pressure upper",
    "Pressure",
    "Pressure lower",
    "Saturation upper",
    "Saturation",
    "Saturation lower",
    "Radius",
    "Total injected volume",
    "Injection rate",
    "Time of second injection",
    "Time of second shut-in",
    "Time of first injection",
    "Time of first shut-in",
    "Analytical PI",
]


def find_model_path() -> pathlib.Path:
    """Return the first existing model path."""
    for path in MODEL_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find model. Tried: "
        + ", ".join(str(path) for path in MODEL_PATHS)
    )


def sensitivity_analysis_minus1_to1(
    model: keras.Model,
    resolution_1: int = 20,
    resolution_2: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Sensitivity analysis where every input is varied on the normalized [-1, 1] scale.

    This mirrors the assumptions in analysis.py: the model inputs are normalized, so each
    feature is swept from -1 to 1 while the other features are held fixed.
    """
    num_inputs = model.input_shape[1]

    min_values = np.full((num_inputs,), -1.0)
    max_values = np.full((num_inputs,), 1.0)

    inputs = np.zeros((num_inputs, resolution_1, resolution_2, num_inputs))
    outputs = np.zeros((num_inputs, resolution_1, resolution_2))

    # Same as analysis.py's default mode="homogeneous": each fixed input vector moves
    # evenly from all -1 to all +1. This gives cleaner, deterministic line families.
    fixed_inputs_all = np.linspace(min_values, max_values, resolution_1)

    for i in range(num_inputs):
        for j, fixed_input in enumerate(fixed_inputs_all):
            batch = np.tile(fixed_input, (resolution_2, 1))
            batch[:, i] = np.linspace(-1.0, 1.0, resolution_2)

            predictions = model.predict(batch, verbose=0)
            outputs[i, j] = predictions.reshape(-1)
            inputs[i, j] = batch

    return outputs, inputs


def plot_sensitivity_like_analysis_py(
    outputs: np.ndarray,
    inputs: np.ndarray,
    savepath: str | pathlib.Path,
    feature_names: Optional[list[str]] = None,
    legend: bool = False,
    max_columns: int = 3,
) -> None:
    """Plot with the same visual style as analysis.py, but lock x-axis to [-1, 1]."""
    if feature_names is None:
        feature_names = [rf"$x_{{{i}}}$" for i in range(inputs.shape[-1])]
    elif len(feature_names) != inputs.shape[-1]:
        raise ValueError(
            f"Expected {inputs.shape[-1]} feature names, got {len(feature_names)}."
        )

    num_rows = math.ceil(outputs.shape[0] / max_columns)
    num_columns = min(outputs.shape[0], max_columns)

    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        sharex=True,
        sharey=True,
    )
    fig.set_size_inches(w=max(num_columns * 3, 5), h=max(num_rows * 3, 5))

    axes_flat = np.atleast_1d(axes).flatten()
    cmap = plt.cm.viridis
    norm = colors.Normalize(vmin=-1, vmax=1)

    fixed_values = np.linspace(-1.0, 1.0, outputs.shape[1])
    line_colors = cmap(norm(fixed_values))
    
    for i, ax in enumerate(axes_flat):
        if i >= len(feature_names):
            break

        for j, color in enumerate(line_colors):
            ax.plot(
                inputs[i, j, :, i],
                outputs[i, j],
                color=color,
                linestyle=":",
                linewidth=1,
                alpha=1,
                label=f"{inputs[i, j, 0, i - 1]:.2f}",
            )
        ax.set_title(feature_names[i])
        ax.set_ylim(bottom=-1.0)

        yticks = ax.get_yticks()
        if -1.0 not in yticks:
            yticks = np.sort(np.append(yticks, -1.0))
            ax.set_yticks(yticks)


    # Common axis labels, like in analysis.py.
    common_ax = fig.add_subplot(111, frameon=False)
    common_ax.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    common_ax.grid(False)
    common_ax.set_xlabel("Normalized input value")
    common_ax.set_ylabel("Model Response")

    if legend:
        axes_flat[min(len(feature_names), len(axes_flat)) - 1].legend(
            title="Splits", loc="center left", bbox_to_anchor=(1, 0.5)
        )
    # Colorbar med nøyaktig samme farger som linjene
    # Lag colorbar med EKSAKT samme farger som linjene
    cmap_for_cbar = colors.ListedColormap(line_colors)

    step = fixed_values[1] - fixed_values[0]
    bounds = np.concatenate((
        [fixed_values[0] - step / 2],
        (fixed_values[:-1] + fixed_values[1:]) / 2,
        [fixed_values[-1] + step / 2]
    ))

    norm_for_cbar = colors.BoundaryNorm(bounds, cmap_for_cbar.N)
    # Lag først en vanlig colorbar bare for å få riktig plassering
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes_flat[:len(feature_names)],
        fraction=0.025,
        pad=0.015,
        aspect=25,
        ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )

    # Bruk samme akse, men tegn vår egen stiplete colorbar
    cax = cbar.ax
    cax.clear()

        # Tettere stiplete linjer i colorbar
    colorbar_values = np.linspace(-1.0, 1.0, 180)
    colorbar_colors = cmap(norm(colorbar_values))

    for value, color in zip(colorbar_values, colorbar_colors):
        cax.plot(
            [0.0, 1.0],
            [value, value],
            color=color,
            linestyle=":",
            linewidth=1,
            alpha=1,
        )

    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(-1.0, 1.0)

    cax.set_xticks([])
    cax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel("Fixed value of remaining inputs", rotation=90, labelpad=15)

    for spine in cax.spines.values():
        spine.set_visible(True)

    savepath = pathlib.Path(savepath)

    # Lagre høyoppløselig PNG som kan sendes direkte på mail
    fig.savefig(
        savepath.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    # Lagre PDF i vektorformat for rapport/publikasjon
    fig.savefig(
        savepath.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )

    # Behold også original pyopmnearwell-lagringen hvis du vil ha datafilene
    plotting.save_fig_and_data(fig, savepath)
    

def get_feature_names(num_inputs: int) -> list[str]:
    """Prefer runspecs features, but fall back to local names if needed."""
    names = list(trainspecs.get("features", FEATURE_NAMES))
    if len(names) == num_inputs:
        return names
    if len(FEATURE_NAMES) == num_inputs:
        return FEATURE_NAMES
    return [rf"$x_{{{i}}}$" for i in range(num_inputs)]


def main() -> None:
    model = keras.models.load_model(find_model_path())

    outputs, inputs = sensitivity_analysis_minus1_to1(
        model,
        resolution_1=30,  # same line count as analysis.py default
        resolution_2=200,  # smoother curves than the analysis.py default
    )

    outdir = NN_DIR / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_names = [
        "Pressure upper",
        "Pressure",
        "Pressure lower",
        "Saturation upper",
        "Saturation",
        "Saturation lower",
        "Radius",
        "Total injected volume",
        "Injection rate",
        "Time of second injection",
        "Time of second shut-in",
        "Time of first injection",
        "Time of first shut-in",
        "Analytical J",
    ]

    plot_sensitivity_like_analysis_py(
        outputs,
        inputs,
        savepath=outdir / "sensitivity_fcnn",
        feature_names=plot_names,
        legend=False,
        max_columns=3,
    )

if __name__ == "__main__":
    main()