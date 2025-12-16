# Geometric and Statistical Analysis of Avian Skull Morphology

[![arXiv](https://img.shields.io/badge/arXiv-2511.06426-b31b1b.svg)](https://arxiv.org/abs/2511.06426)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Adobe Express - combined_grid_2x3_fixed](https://github.com/user-attachments/assets/27593bac-494d-4118-8ccd-4e1875879f2c)

This repository contains Python scripts for the geometric and statistical analysis of avian skull morphology.

* **Categories:** The scripts are categorized into (1) Geometric Analysis, (2) Statistical Analysis.
* **Geometric Analysis:** Extract geometric features from 3D avian skull models.
* **Statistical Analysis:** Perform correlation analysis and modelling on the measurements.

## Data Files

* The folder **dataset** contains the preprocessed mesh files of all 100 specimens of Darwin's finches and their relatives.
* The file **Dataset_training.xlsx** records all dimension and curvature quantities for the 50 training samples (which can be used for the training the curvature prediction model).
* The file **Dataset.xlsx** records all geometric quantities and curvature prediction results for all 100 specimens, together with relevant statistical values.


## Geometric Analysis Scripts

### 1. `fit_sphere.py`
* **Purpose:** Locate and quantify the orbit (a CONCAVE feature).
* **Action 1:** Batch-process all .stl files in a folder. 
* **Action 2:** Identify the most concave points in a specific region. 
* **Action 3:** Fit a SPHERE to these points using a robust iterative method.
* **Action 4:** Save all skull dimensions, sphere radii and centers, and orbit curvatures to an .xlsx file.

### 2. `fit_ellipsoid.py`
* **Purpose:** Locate and quantify the braincase (a CONVEX feature).
* **Action 1:** Batch-process all .stl files in a folder. 
* **Action 2:** Identify the most convex points in a specific region. 
* **Action 3:** Fit an **axis-aligned** ELLIPSOID to these points. 
* **Action 4:** Save all skull dimensions, ellipsoidal semi-axis lengths, and ellipsoid centers to an .xlsx file.

### 3. `bounding_AABB.py`
* **Purpose:** Compute an **Axis-Aligned Bounding Box (AABB)** of an input skull model. 
* **Action:** Load **one** .stl file and compute its Axis-Aligned Bounding Box (a simple, non-rotated box). 

### 4. `bounding_OBB.py`
* **Purpose:** Compute an **Oriented Bounding Box (OBB)** of an input skull model. 
* **Action:** Load **one** .stl file and compute its **Oriented Bounding Box (OBB)** (the smallest possible rotated box).


## Statistical Analysis Scripts

### 1. `correlation.py`
* **Purpose:** Perform a statistical analysis on skull dimensions and orbit geometries.
* **Action 1:** Load a specified Excel file (e.g., `Dataset.xlsx`).
* **Action 2:** Generate a 2x3 grid of scatter plots (Radius vs. Dimensions, Curvature vs. Dimensions) with linear regression lines. 
* **Action 3:** Calculate and print separate Pearson (linear) and Spearman (monotonic) correlation matrices for **radius** and **curvature** against skull dimensions. 

### 2. `correlation_specific.py`
* **Purpose:** Run a simple statistical test for a specific relationship.
* **Action 1:** Load a specified Excel file (e.g., `Dataset.xlsx`).
* **Action 2:** Build a simple linear regression model for two specified quantities.
* **Action 3:** Print the `statsmodels` summary to assess the statistical significance (P-value) of this single relationship.

### 3. `correlation_combined.py`
* **Purpose:** Perform a statistical analysis on skull dimensions, orbit geometries, and braincase geometries.
* **Action 1:** Load a specified Excel file (e.g., `Dataset.xlsx`).
* **Action 2:** Use the `tabulate` library to print neatly formatted Pearson and Spearman correlation tables.
* **Action 3:** Correlate **ellipsoid axes (a, b, c)** and **curvature** against **skull dimensions (x, y, z)** in one comprehensive table.

### 4. `modelling.py`
* **Purpose:** Create a multiple linear regression model to predict curvature from model dimensions.
* **Action 1:** Load a specified Excel file (e.g., `Dataset_training.xlsx`).
* **Action 2:** Build a multiple linear regression model: `curvature ~ length_x + width_y + height_z`.
* **Action 3:** Generate a side-by-side 3D scatter plot comparing **Actual Curvature** vs. **Model's Approximated Curvature**.
* **Action 4:** Print the full `statsmodels` summary of the linear model (R-squared, coefficients, p-values, etc.).

## Citation

If you use this code or data in your work, please cite:

Kaikwan Lau and Gary P. T. Choi, 
"[Geometric and statistical analysis of avian skull morphology.](https://arxiv.org/abs/2511.06426)"
arXiv preprint arXiv:2511.06426, 2025.

```bibtex
@article{lau2025geometric,
    author    = {Lau, Kaikwan and Choi, Gary P. T.},
    title     = {Geometric and statistical analysis of avian skull morphology},
    journal   = {arXiv preprint arXiv:2511.06426},
    year      = {2025},
    eprint    = {2511.06426},
    archivePrefix = {arXiv},
    primaryClass = {q-bio.QM}
}
```
