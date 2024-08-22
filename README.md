# 2D_Cross_Correlation.py

## Overview
This Python script is designed to analyze 2D (XZ and YZ) positional data of cells tracked over time, with a focus on calculating and visualizing the spatial correlation of velocities between pairs of cells. The script facilitates the processing of raw data, data cleaning, calculation of velocity components, and the fitting of correlation functions. The ultimate goal is to extract meaningful biophysical parameters, such as the correlation length (Lcorr), that describe collective cell behavior.

## Functions

### 1. `input_csv(filename, rescale_xy=1, rescale_z=1)`
- **Description:** Loads a CSV file containing raw positional data of cells and rescales the coordinates if necessary.

### 2. `drop(file, threshold)`
- **Description:** Cleans the data by removing rows with null `TrackID`, short tracks, and outliers based on their distance from the spheroid center.
- **Parameters:**
  - `file`: DataFrame containing the raw or rescaled data.
  - `threshold`: Minimum number of time points required for a valid track.
- **Returns:** Cleaned DataFrame.

### 3. `Lcorr_fitting(r, Cvv)`
- **Description:** Fits the stretched exponential function to the correlation data using non-linear least squares optimization.
- **Parameters:**
  - `r`: Distance data.
  - `Cvv`: Correlation data.
- **Returns:** Fitted parameters `g`, `a`, `Lcorr`, and their standard deviations.

### 4. `velocity_spat(file)`
- **Description:** Calculates the velocity components for each track and normalizes them within the XZ and YZ planes.
- **Parameters:**
  - `file`: Cleaned DataFrame.
- **Returns:** DataFrame with velocity components and normalized velocities.

### 5. `correlation_spat(file, t)`
- **Description:** Computes spatial correlations of velocity components between cell pairs at a specific time point.
- **Parameters:**
  - `file`: DataFrame with velocity data.
  - `t`: Specific time point for correlation calculation.
- **Returns:** DataFrames containing correlation data for XZ and YZ planes, and pooled data.

### 6. `plot_velocity_corr(file, t, name, fitting=0)`
- **Description:** Plots the cumulative average velocity correlation as a function of distance for XZ and YZ planes, with optional fitting.
- **Parameters:**
  - `file`: DataFrame with velocity data.
  - `t`: Time point for analysis.
  - `name`: Name to label the plot.
  - `fitting`: Boolean to determine if fitting should be performed.
- **Returns:** None.

### 7. `plot_pooled_corr(file, t, name, fitting=0)`
- **Description:** Plots the pooled correlation of velocities as a function of distance and optionally fits the data to extract Lcorr.
- **Parameters:**
  - `file`: DataFrame with velocity data.
  - `t`: Time point for analysis.
  - `name`: Name to label the plot.
  - `fitting`: Boolean to determine if fitting should be performed.
- **Returns:** Lcorr and its standard deviation, or None if fitting fails.

## Usage
1. Prepare your CSV file containing cell position data.
2. Use `input_csv` to load the data and apply any necessary rescaling.
3. Clean the data using `drop`.
4. Calculate velocity components with `velocity_spat`.
5. Analyze and visualize the spatial correlations using functions like `plot_velocity_corr` or `plot_pooled_corr`.
