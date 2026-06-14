# Machine-Learning in Near-Well Simulations of intermittent CO2 injections: Master thesis

This folder contains the main scripts and template files used for the 3D CO2 near-well machine-learning workflow in the master thesis. The workflow is based on ensemble reservoir simulations of intermittent CO2 injection, upscaling of near-well data, neural-network training, model evaluation and integration of the trained model into coarse-scale simulations. The main objective is to predict the effective well index (WI) from upscaled near-well simulation data using a fully connected neural network (FCNN).

The scripts in this folder were used during the thesis workflow. The large data and simulation output files are stored separately due to file-size limitations (in onedrive).

## Folder contents

### Simulation and workflow files

* `main.py`
  Main script for running the 3D CO2 near-well simulation and data-generation workflow. Start here. 

* `runspecs.py`
  Defines the simulation, data and training specifications used in the workflow, including the selected input features.

* `ensemble.mako`
  Template file used for generating ensemble simulation cases.

* `upscale.py`
  Performs the near-well upscaling and prepares the structured dataset representation used for machine learning.

* `nn.py`
  Neural-network training script.

## Model evaluation and plotting scripts

* `testmodel2.py`
  Evaluates the trained neural-network model on the scaled test dataset. It computes global test metrics, including RMSE, MAE and relative errors, and generates diagnostic plots such as residual plots, residual histograms and relative-error plots.

* `WIplot.py`
  Plots true and predicted WI as a function of time for selected test simulations, layers or radial positions. These plots are used to visually assess how well the model reproduces the time-dependent WI response.

* `wi_boxplot_per_layer.py`
  Analyzes the distribution of WI across reservoir layers at a fixed radial index. The script prints summary statistics and generates layer-wise boxplots.

* `plot_training_history.py`
  Plots the training and validation loss from the saved training-history file.

* `sensitivity_analysis.py`
  Performs sensitivity analysis for the trained FCNN model by varying each normalized input feature from -1 to 1 while holding the remaining inputs fixed. The resulting plots are used to assess the influence of each input feature on the model response.

## Dataset and input-distribution analysis

* `injection_rate_histogram.py`
  Analyzes the distribution of injection rates in the training dataset. It inverse-scales the injection-rate feature, keeps active-injection samples, and creates a histogram showing the percentage of samples in each injection-rate interval.

* `shutin_distribution_histogram.py`
  Analyzes the distribution of previous shut-in durations in the training dataset. It creates a histogram showing the percentage of samples in each shut-in-duration interval.

* `inspect_final_nn_dataset.py`
  Inspects the final neural-network dataset and reports dataset structure and feature information.

## Integration and benchmark comparison

* `integration_results_peacemanvsNN.py`
  Compares bottom-hole pressure results from the fine-scale benchmark, NN-based coarse simulations and Peaceman-based coarse simulations. The script also computes error values relative to the fine-scale benchmark. The input and outputfiles from this work is available in OneDrive.

## Data and result files

Large generated data files and simulation results are not stored in this GitHub folder. These are provided separately in a data archive in onedrive. The archive contains the generated datasets, trained model files, simulation outputs and result folders used in the thesis.

The data archive includes:

* `ensemble/`
  One remaining ensemble simulation run and intermediate simulation files.

* `dataset/`
  Raw dataset extracted from ensemble simulations.

* `dataset_stencil/`
  Final upscaled dataset used for neural-network training.

* `nn/`
  Results from model training, including scaling files, trained model outputs, training history and evaluation figures.

* `dorthe_model/`
  Best trained WI model and JSON file used for integration.

* `integration_results/`
  Integration simulations, including input, output files and results for the fine-scale benchmark, NN-based simulations and Peaceman-based simulations.

## Installations

Devcontainer
