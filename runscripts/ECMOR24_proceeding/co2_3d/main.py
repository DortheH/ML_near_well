from __future__ import annotations

import math
import pathlib
import sys
import h5py
import re
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nn import FEATURE_TO_INDEX, restructure_data
from pyopmnearwell.ml import analysis, ensemble, integration, utils
from pyopmnearwell.utils import units
from runspecs import (
    runspecs_ensemble,
    runspecs_integration_3D_and_Peaceman_1,
    runspecs_integration_3D_and_Peaceman_2,
    runspecs_integration_3D_and_Peaceman_3,
    runspecs_integration_3D_and_Peaceman_4,
    trainspecs,
)
from tensorflow import keras
from upscale import CO2_3D_upscaler
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau

dirname: pathlib.Path = pathlib.Path(__file__).parent

sys.path.append(str(dirname / ".."))
from utilstime import (
    bhp_error,
    full_ensemble,
    plot_member,
    read_and_plot_bhp,
    reload_data,
    tune_and_train,
)

# Set seaborn style.
sns.set_theme(context="paper", style="whitegrid")

SEED: int = 19123
utils.enable_determinism(SEED)

# Structure directories.
ensemble_dir: pathlib.Path = dirname / "ensemble_run"
data_dir: pathlib.Path = dirname / "dataset"
data_stencil_dir: pathlib.Path = dirname / "dataset_stencil"
nn_dir: pathlib.Path = dirname / "nn"

integration_3d_dir_1: pathlib.Path = (
    dirname / runspecs_integration_3D_and_Peaceman_1["name"]
)
integration_3d_dir_2: pathlib.Path = (
    dirname / runspecs_integration_3D_and_Peaceman_2["name"]
)

integration_3d_dir_3: pathlib.Path = (
    dirname / runspecs_integration_3D_and_Peaceman_3["name"]
)
integration_3d_dir_4: pathlib.Path = (
    dirname / runspecs_integration_3D_and_Peaceman_4["name"]
)

ensemble_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
data_stencil_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)
for integration_dir in [
    integration_3d_dir_1,
    integration_3d_dir_2,
    integration_3d_dir_3,
    integration_3d_dir_4,
]:
    integration_dir.mkdir(parents=True, exist_ok=True)

# Angle between both sides of the triangle grid.
ANGLE: float = math.pi / 3


# 
# # Run ensemble and extract data.
if False:
    print("=== BEFORE full_ensemble ===", flush=True)
    extracted_data: np.ndarray = full_ensemble(
        runspecs_ensemble,
        ensemble_dir,
        ecl_keywords=["PRESSURE", "SGAS", "FLOGASI+"],
        summary_keywords=["FGIT", "WGIR:INJ0"],
        keyword_scalings={
            "PRESSURE": units.BAR_TO_PASCAL,
        },
        seed=SEED,
        save_intermediate_data=True,
        intermediate_data_dir=ensemble_dir / "intermediate_data",
        #keep_result_files=True,
    )
    print("=== AFTER full_ensemble ===", flush=True)
    print(
        f"extracted_data shape={extracted_data.shape} dtype={extracted_data.dtype}",
        flush=True,
    )
    batch_size = 10  # Juster batch-størrelse etter hvor mye RAM du har
    num_members = extracted_data.shape[0]
    print("=== BEFORE HDF5 save ===", flush=True)
    with h5py.File(str(ensemble_dir / "features.h5"), "w") as f:
        dset = f.create_dataset("features", shape=extracted_data.shape, dtype=extracted_data.dtype, compression="gzip")
        for start in range(0, num_members, batch_size):
            end = min(start + batch_size, num_members)
            dset[start:end] = extracted_data[start:end]
            print(f"Saved batch {start}-{end}", flush=True)
    print("=== AFTER h5py save(features) ===", flush=True)
        # Les fra HDF5 i stedet for np.load
        
    print("=== BEFORE HDF5 load ===", flush=True)

    with h5py.File(str(ensemble_dir / "features.h5"), "r") as f:
        extracted_data = f["features"][:]  # eller bruk f["features"] direkte hvis du vil ha "mmap"-lignende tilgang
    print("=== AFTER HDF5 load ===", flush=True)

    print("=== BEFORE upscaler.create_ds ===", flush=True)
    upscaler: CO2_3D_upscaler = CO2_3D_upscaler(
        extracted_data, runspecs_ensemble, data_dim=5, angle=ANGLE
    )
    print("\n=== STAGE: upscaler.create_ds START ===", flush=True)
    features, targets = upscaler.create_ds(ensemble_dir, step_size_x=12, step_size_t=2, keep_xcells=142)
    # Fjern de to innerste punktene nær brønnen
    features = features[..., 2:, :]
    targets = targets[..., 2:]
    
    print(f"create_ds shapes: features={features.shape}, targets={targets.shape}", flush=True)
    print(f"create_ds dtypes: features={features.dtype}, targets={targets.dtype}", flush=True)
    print("=== STAGE: upscaler.create_ds DONE ===", flush=True)
    print("\n=== STAGE: store_dataset(raw) START ===", flush=True)
    ensemble.store_dataset(features, targets, data_dir)
    print("=== STAGE: store_dataset(raw) DONE ===", flush=True)
    print("\n=== STAGE: restructure_data START ===", flush=True)

    restructure_data(data_dir, data_stencil_dir, trainspecs, stencil_size=3)
    print("=== STAGE: restructure_data DONE ===", flush=True)



print("\n=== STAGE: tune_and_train START ===", flush=True)


history_csv = nn_dir / "training_history.csv"

callbacks = [
    CSVLogger(str(history_csv), append=False),
    EarlyStopping(
        monitor="val_loss",
        patience=40,
        min_delta=1e-6,
        verbose=1,
    ),
]

# Tune and train model.
if True:
    original_fit = keras.Model.fit

    def fit_with_callbacks(self, *args, **kwargs):
        existing_callbacks = kwargs.get("callbacks", [])
        if existing_callbacks is None:
            existing_callbacks = []

        kwargs["callbacks"] = list(existing_callbacks) + callbacks

        return original_fit(self, *args, **kwargs)

    keras.Model.fit = fit_with_callbacks

    try:
        if True:
            tune_and_train(
                trainspecs,
                data_stencil_dir,
                nn_dir,
                max_trials=10,
                lr=1e-3,
                lr_tune=1e-3,
                epochs=500,
                bs=512,
                patience=40,
                lr_patience=15,
                executions_per_trial=1,
            )
    finally:
        keras.Model.fit = original_fit

    print(f"training_history exists: {history_csv.exists()}", flush=True)
    print(f"training_history path: {history_csv}", flush=True)
print("=== STAGE: tune_and_train DONE ===", flush=True)


# Integrate into OPM.
if False:
    """integration.recompile_flow(
        nn_dir / "scalings.csv",
        runspecs_integration_3D_and_Peaceman_1["constants"]["OPM"],
        dirname / "standardwell_impl_3d.mako",
        dirname / "standardwell.hpp",
        local_feature_names=["pressure", "saturation"]
    )"""
    for integration_dir, runspecs_integration in zip(
        [
            integration_3d_dir_1,
            integration_3d_dir_2,
            integration_3d_dir_3,
            integration_3d_dir_4,
        ],
        [
            runspecs_integration_3D_and_Peaceman_1,
            runspecs_integration_3D_and_Peaceman_2,
            runspecs_integration_3D_and_Peaceman_3,
            runspecs_integration_3D_and_Peaceman_4,
        ],
    ):
        integration.run_integration(
            runspecs_integration,
            integration_dir,
            dirname / "integration.mako",
        )
   