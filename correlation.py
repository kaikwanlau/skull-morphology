# ===================================================================================
#
# This script performs a statistical analysis for skull dimensions, sphere_radius, and curvature.
# Actions:
# 1. Load the specified Excel file.
# 2. Print the entire DataFrame with all data points to the console.
# 3. Generate a single image with 6 scatter plots, each with a linear regression line.
# 4. Calculate and print 4 separate correlation matrices for comparison.
#
# ===================================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

try:
    # --- 1. Load your data from the Excel file ---
    file_name = 'Dataset.xlsx'
    df = pd.read_excel(file_name)

    # --- 2. Plot the Relationships (Radius and Curvature) with Linear Approximation ---
    dimension_columns = ['length_x', 'width_y', 'height_z']
    for col in dimension_columns:
        if col not in df.columns:
            raise KeyError(f"The required column '{col}' was not found in your Excel file.")

    # Create a figure with 2 rows and 3 columns of subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Radius and Curvature vs. Object Dimensions with Linear Approximation', fontsize=16)

    max_radius = df['sphere_radius'].max() * 1.5
    max_curvature = df['curvature'].max() * 1.5
    
    # --- Top Row: Radius Plots ---
    # Radius vs. Length
    sns.regplot(ax=axes[0, 0], data=df, x='length_x', y='sphere_radius')
    axes[0, 0].set_ylim(0, max_radius) # MODIFICATION: Set Y-axis range
    axes[0, 0].set_title('Radius vs. Length')
    axes[0, 0].set_xlabel('Length (x)')
    axes[0, 0].set_ylabel('Radius')

    # Radius vs. Width
    sns.regplot(ax=axes[0, 1], data=df, x='width_y', y='sphere_radius')
    axes[0, 1].set_ylim(0, max_radius) # MODIFICATION: Set Y-axis range
    axes[0, 1].set_title('Radius vs. Width')
    axes[0, 1].set_xlabel('Width (y)')
    axes[0, 1].set_ylabel('Radius')

    # Radius vs. Height
    sns.regplot(ax=axes[0, 2], data=df, x='height_z', y='sphere_radius')
    axes[0, 2].set_ylim(0, max_radius) # MODIFICATION: Set Y-axis range
    axes[0, 2].set_title('Radius vs. Height')
    axes[0, 2].set_xlabel('Height (z)')
    axes[0, 2].set_ylabel('Radius')

    # --- Bottom Row: Curvature Plots ---
    # Curvature vs. Length
    sns.regplot(ax=axes[1, 0], data=df, x='length_x', y='curvature')
    axes[1, 0].set_ylim(0, max_curvature) # MODIFICATION: Set Y-axis range
    axes[1, 0].set_title('Curvature vs. Length')
    axes[1, 0].set_xlabel('Length (x)')
    axes[1, 0].set_ylabel('Curvature (1/Radius)')

    # Curvature vs. Width
    sns.regplot(ax=axes[1, 1], data=df, x='width_y', y='curvature')
    axes[1, 1].set_ylim(0, max_curvature) # MODIFICATION: Set Y-axis range
    axes[1, 1].set_title('Curvature vs. Width')
    axes[1, 1].set_xlabel('Width (y)')
    axes[1, 1].set_ylabel('Curvature (1/Radius)')

    # Curvature vs. Height
    sns.regplot(ax=axes[1, 2], data=df, x='height_z', y='curvature')
    axes[1, 2].set_ylim(0, max_curvature) # MODIFICATION: Set Y-axis range
    axes[1, 2].set_title('Curvature vs. Height')
    axes[1, 2].set_xlabel('Height (z)')
    axes[1, 2].set_ylabel('Curvature (1/Radius)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # MODIFICATION: Changed save file name
    plt.savefig('combined_radius_curvature_plots.png')
    print("Combined scatter plots with linear regression saved as 'combined_radius_curvature_plots.png'")
    print("\n" + "=" * 50 + "\n")

    # --- 3. Calculate and Compare All Correlation Matrices ---

    # Analysis for RADIUS
    print("--- Analysis for: sphere_radius ---")
    pearson_radius = df[['sphere_radius'] + dimension_columns].corr(method='pearson')
    print("Pearson Correlation (Linear):")
    print(pearson_radius)
    print("-" * 25)
    spearman_radius = df[['sphere_radius'] + dimension_columns].corr(method='spearman')
    print("Spearman Correlation (Monotonic):")
    print(spearman_radius)
    print("\n" + "=" * 50 + "\n")

    # Analysis for CURVATURE
    print("--- Analysis for: curvature ---")
    pearson_curvature = df[['curvature'] + dimension_columns].corr(method='pearson')
    print("Pearson Correlation (Linear):")
    print(pearson_curvature)
    print("-" * 25)
    spearman_curvature = df[['curvature'] + dimension_columns].corr(method='spearman')
    print("Spearman Correlation (Monotonic):")
    print(spearman_curvature)

except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found. Please check the file name and path.")
except KeyError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
