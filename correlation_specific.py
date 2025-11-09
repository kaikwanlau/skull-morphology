# ===================================================================================
#
# This script performs a statistical analysis for a specific relationship.
# Actions:
# 1. Load the specified Excel file.
# 2. Build a simple linear regression model for two specified quantities.
# 3. Print the `statsmodels` summary to assess the statistical significance (P-value) of this single relationship.
#
# ===================================================================================

import pandas as pd
import statsmodels.formula.api as smf

try:
    # --- 1. Load the data ---
    # Make sure this path is correct
    file_name = 'Dataset.xlsx'
    df = pd.read_excel(file_name)

    # --- 2. Run the Linear Regression ---
    model = smf.ols("curvature ~ length_x", data=df).fit()

    print("--- Regression Result ---")
    print(model.summary())

except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found. Please check the file path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
