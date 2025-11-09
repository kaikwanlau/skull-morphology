# ===================================================================================
#
# This script performs a combined statistical analysis for skull dimensions, sphere_radius, orbit curvature, and ellispoidal parameters.
# Actions:
# 1. Load the specified Excel file.
# 2. Calculate and print separate Pearson (linear) and Spearman (monotonic)
#    correlation matrices, formatted as tables.
#
# ===================================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate  # <-- ADDED THIS IMPORT

try:
    # --- 1. Load the data from the Excel file ---
    file_name = 'Dataset.xlsx'
    df = pd.read_excel(file_name)

    # ==============================================================================
    # --- 2. Generate Combined Correlation Table ---
    # ==============================================================================

    # Create a copy for analysis and rename 'sphere_radius' to match the table's "1/curvature"
    df_analysis = df.copy()
    df_analysis['1/curvature'] = df_analysis['sphere_radius']

    # Define the rows and columns for the final table
    # These MUST match the column names in your Excel file or the ones we just created
    row_vars = ['ellipsoid_axis_a', 'ellipsoid_axis_b', 'ellipsoid_axis_c', '1/curvature', 'curvature']
    col_vars = ['length_x', 'width_y', 'height_z', '1/curvature', 'curvature']

    # Define the display names for the final table columns
    display_cols = ['length x', 'width y', 'height z', '1/curvature', 'curvature']

    # Check if all required columns exist
    all_needed_vars = list(set(row_vars + col_vars))  # Get unique list of all vars
    missing_cols = [col for col in all_needed_vars if col not in df_analysis.columns]

    if missing_cols:
        print("--- Error: Cannot Generate Correlation Table ---")
        print("Your Excel file is missing the following required columns:")
        for col in missing_cols:
            if col != '1/curvature':  # Don't report '1/curvature' as missing, we created it
                print(f"- {col}")
    else:
        # --- Pearson Correlation ---
        print("--- Combined Pearson Correlation Matrix (Linear) ---")

        # Calculate the full Pearson correlation matrix
        corr_matrix_pearson = df_analysis[all_needed_vars].corr(method='pearson')

        # Filter the matrix to get only the rows and columns we want
        final_table_pearson = corr_matrix_pearson.loc[row_vars, col_vars]

        # Set the column and index names for a clean print
        final_table_pearson.columns = display_cols
        final_table_pearson.index.name = 'Variable'

        # Print the formatted table
        print(tabulate(final_table_pearson, headers='keys', tablefmt='psql', floatfmt=".2f"))

        # ==================================================================
        # --- Spearman Correlation (NEW) ---
        # ==================================================================
        print("\n" + "=" * 50 + "\n")  # Add a separator
        print("--- Combined Spearman Correlation Matrix (Monotonic) ---")

        # Calculate the full Spearman correlation matrix
        corr_matrix_spearman = df_analysis[all_needed_vars].corr(method='spearman')

        # Filter the matrix
        final_table_spearman = corr_matrix_spearman.loc[row_vars, col_vars]

        # Set column and index names
        final_table_spearman.columns = display_cols
        final_table_spearman.index.name = 'Variable'

        # Print the formatted Spearman table
        print(tabulate(final_table_spearman, headers='keys', tablefmt='psql', floatfmt=".2f"))


except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found. Please check the file name and path.")
except KeyError as e:
    print(f"Error: A required column was not found. {e}")
except ImportError:
    print("Error: The 'tabulate' library is required. Please install it using: pip install tabulate")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
