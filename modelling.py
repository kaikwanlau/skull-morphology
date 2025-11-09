# ===================================================================================
#
# --- Instructions ---
# This script computes a multiple linear regression model for the orbit curvature and skull dimensions.
# 1. Load the specified Excel file.
# 2. Calculate curvature (1/radius).
# 3. Build the multiple linear regression model.
# 4. Perform the curvature prediction using the model.
# 5. Plot the results.
#
# ===================================================================================


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.formula.api as smf
import numpy as np
import os

try:
    # --- 1. Load the Data ---
    file_name = 'Dataset_training.xlsx'
    df = pd.read_excel(file_name)

    # --- 2. Calculate Curvature (1/radius) ---
    print("Calculating curvature (1/radius)...")
    if 'sphere_radius' not in df.columns:
        raise KeyError("The required column 'sphere_radius' was not found.")

    if (df['sphere_radius'] == 0).any():
        print(
            "Warning: Your data contains a sphere_radius of 0. Replacing with a small number to avoid division by zero.")
        df['curvature'] = 1 / df['sphere_radius'].replace(0, 1e-9)
    else:
        df['curvature'] = 1 / df['sphere_radius']

    # --- 3. Build the Multiple Linear Regression model for Curvature ---
    print("Building the multiple linear regression model for curvature...")
    model_curvature = smf.ols("curvature ~ length_x + width_y + height_z", data=df).fit()

    # --- 4. Get the Model Predictions for Curvature ---
    df['predicted_curvature'] = model_curvature.predict(df)
    print("Model predictions for curvature have been calculated.")

    # --- 5. Create the Side-by-Side 3D Plots ---
    print("Generating comparison plots for curvature...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), subplot_kw={'projection': '3d'})
    fig.suptitle('Actual Curvature vs. Model Prediction', fontsize=20)

    all_data = np.concatenate([df['curvature'], df['predicted_curvature']])
    vmin = np.nanquantile(all_data, 0.02)
    vmax = np.nanquantile(all_data, 0.98)

    print(f"Robust color limits set (2nd-98th percentile): vmin={vmin:.4f}, vmax={vmax:.4f}")

    # --- Plot 1: Actual Curvature ---
    ax1.set_title('1. Actual Curvature', fontsize=16)
    scatter1 = ax1.scatter(df['length_x'], df['width_y'], df['height_z'],
                           c=df['curvature'], cmap='viridis', s=60, alpha=0.8,
                           vmin=vmin, vmax=vmax)  # <-- vmin/vmax are now robust
    ax1.set_xlabel('Length (x)', fontweight='bold')
    ax1.set_ylabel('Width (y)', fontweight='bold')
    ax1.set_zlabel('Height (z)', fontweight='bold')
    fig.colorbar(scatter1, ax=ax1, shrink=0.6, label='Actual Curvature (1/Radius)')

    # --- Plot 2: Model's Approximated Curvature ---
    ax2.set_title('2. Model Approximation', fontsize=16)
    scatter2 = ax2.scatter(df['length_x'], df['width_y'], df['height_z'],
                           c=df['predicted_curvature'], cmap='viridis', s=60, alpha=0.8,
                           vmin=vmin, vmax=vmax)  # <-- vmin/vmax are now robust
    ax2.set_xlabel('Length (x)', fontweight='bold')
    ax2.set_ylabel('Width (y)', fontweight='bold')
    ax2.set_zlabel('Height (z)', fontweight='bold')
    fig.colorbar(scatter2, ax=ax2, shrink=0.6, label='Predicted Curvature (1/Radius)')

    # --- 6. Save and Display the Plot ---
    output_filename = 'modelling_results.png'

    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating missing output directory: {output_dir}")
        os.makedirs(output_dir)

    plt.savefig(output_filename)
    print(f"\nComparison plot saved as '{output_filename}'")
    plt.show()

    print("\n--- Curvature Model Summary ---")
    print(model_curvature.summary())


except FileNotFoundError as e:
    if hasattr(e, 'filename') and e.filename == file_name:
        print(f"Error: The *input file* '{file_name}' was not found. Please double-check the file path.")
    else:
        print(f"Error: Could not save the *output file*. The directory may not exist or you may lack permissions: {e}")
except KeyError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
