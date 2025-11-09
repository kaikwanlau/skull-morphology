# ===================================================================================
#
# This script performs the ellipsoid fitting for skull meshes in batch.
# Actions:
# 1. Load the specified stl folders.
# 2. Perform the ellispoid fitting for the braincase (a CONVEX feature).
# 3. Display the mesh and the best-fit ellipsoid.
# 4. Generate an excel file containing the skull dimensions, ellipsoidal semi-axis lengths, and ellipsoid centers for all meshes.
#
# ===================================================================================

import trimesh
import numpy as np
import pyvista as pv
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import os
import glob
import pandas as pd


def axis_aligned_ellipsoid_loss_function(params, points):
    """
    Loss function for an axis-aligned ellipsoid.
    Calculates the sum of squared algebraic distances from each point to the surface.
    Parameters are: center (3) and semi-axis lengths (3).
    """
    center = params[:3]
    axes_lengths = params[3:6]

    if any(a <= 1e-6 for a in axes_lengths):
        return 1e9

    points_transformed = points - center
    distances = np.sum((points_transformed / axes_lengths) ** 2, axis=1)
    return np.sum((distances - 1.0) ** 2)


def fit_axis_aligned_ellipsoid_iteratively(points, max_iterations=3, outlier_std_dev=1.5):
    """
    Iteratively fits an axis-aligned ellipsoid and removes outlier points.
    """
    current_points = points.copy()
    last_successful_result = None

    for i in range(max_iterations):
        if len(current_points) < 7:
            return None, None

        center_guess = np.mean(current_points, axis=0)
        centered_points = current_points - center_guess
        axes_guess = np.std(centered_points, axis=0) * 2.0
        axes_guess[axes_guess < 1e-6] = 1e-6
        initial_guess = np.concatenate([center_guess, axes_guess])

        min_b = np.min(current_points, axis=0)
        max_b = np.max(current_points, axis=0)
        buffer = (max_b - min_b) * 0.5
        center_bounds = list(zip(min_b - buffer, max_b + buffer))
        axes_bounds = [(1e-6, None)] * 3
        optimizer_bounds = center_bounds + axes_bounds

        result = minimize(
            axis_aligned_ellipsoid_loss_function,
            initial_guess,
            args=(current_points,),
            method='L-BFGS-B',
            bounds=optimizer_bounds
        )

        if result.success:
            last_successful_result = result

        fit_params = result.x
        fit_center = fit_params[:3]
        fit_axes = fit_params[3:6]

        points_transformed = current_points - fit_center
        distances = np.sum((points_transformed / fit_axes) ** 2, axis=1)
        errors = np.abs(distances - 1.0)
        mean_error, std_error = np.mean(errors), np.std(errors)

        if std_error < 1e-6: break
        inlier_mask = errors < (mean_error + outlier_std_dev * std_error)
        if np.all(inlier_mask): break
        current_points = current_points[inlier_mask]

    if last_successful_result is None:
        return None, None

    final_result = minimize(
        axis_aligned_ellipsoid_loss_function,
        last_successful_result.x,
        args=(current_points,),
        method='L-BFGS-B',
        bounds=optimizer_bounds
    )

    if not final_result.success:
        return None, None

    return final_result.x, current_points


if __name__ == '__main__':
    # --- 1. USER-DEFINED PARAMETERS ---
    DIRECTORY_PATH = "dataset"
    ENABLE_VISUALIZATION = True
    SAVE_INDIVIDUAL_EXCEL = False

    # --- 2. ALGORITHM THRESHOLDS & SETTINGS ---
    POSTERIOR_AXIS = 'x'

    POSTERIOR_PERCENTILE = 0.40
    # --- END MODIFIED ---

    CURVATURE_RADIUS = 3
    TARGET_POINT_COUNT = 1200
    MAX_SEED_TO_CENTER_DISTANCE = 80.0

    # --- 3. SCRIPT SETUP ---
    results_list = []
    search_path = os.path.join(DIRECTORY_PATH, '*.stl')
    stl_files = glob.glob(search_path)
    if not stl_files:
        exit(f"Error: No .stl files found in {DIRECTORY_PATH}")
    print(f"Found {len(stl_files)} STL files to process.\n")

    # --- 4. MAIN PROCESSING LOOP ---
    for i, file_path in enumerate(stl_files):
        print(f"--- Processing file {i + 1}/{len(stl_files)}: {os.path.basename(file_path)} ---")
        try:
            original_mesh = trimesh.load_mesh(file_path)
            components = original_mesh.split(only_watertight=False)
            processed_mesh = sorted(components, key=lambda c: len(c.vertices), reverse=True)[0]
            processed_mesh.process(validate=True)
            box_extents = processed_mesh.bounding_box.extents
        except Exception as e:
            print(f"  -> Error loading mesh, skipping. Error: {e}")
            continue

        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map.get(POSTERIOR_AXIS)
        if axis_idx is None: exit(f"Error: POSTERIOR_AXIS must be 'x', 'y', or 'z'.")
        min_bound, max_bound = processed_mesh.bounds[:, axis_idx]

        # This calculation now uses 0.30 (30%)
        cutoff_coord = max_bound - (max_bound - min_bound) * POSTERIOR_PERCENTILE
        posterior_mask = processed_mesh.vertices[:, axis_idx] > cutoff_coord

        if not np.any(posterior_mask):
            print(
                f"  -> No vertices found in the defined posterior region (back {POSTERIOR_PERCENTILE * 100}%). Skipping.")
            continue

        print(f"  -> ROI filter activated. Selecting points in the back {POSTERIOR_PERCENTILE * 100}% of the mesh.")

        mean_curvatures = trimesh.curvature.discrete_mean_curvature_measure(processed_mesh, processed_mesh.vertices,
                                                                            radius=CURVATURE_RADIUS)
        mean_curvatures[~posterior_mask] = -np.inf
        sorted_indices = np.argsort(mean_curvatures)[::-1]
        most_convex_indices = sorted_indices[:TARGET_POINT_COUNT]
        valid_indices_mask = np.zeros(len(processed_mesh.vertices), dtype=bool)
        valid_indices_mask[most_convex_indices] = True
        seed_index = np.argmax(mean_curvatures)
        seed_point_coords = processed_mesh.vertices[seed_index]
        selected_edges = processed_mesh.edges[valid_indices_mask[processed_mesh.edges].all(axis=1)]
        if len(selected_edges) == 0:
            print(" -> No connected edges found among selected points. Skipping.")
            continue
        components = trimesh.graph.connected_components(selected_edges)
        if not components:
            print(" -> Could not form connected components. Skipping.")
            continue
        final_indices = []
        component_with_seed = next((c for c in components if seed_index in c), None)
        if component_with_seed is not None:
            final_indices = np.array(list(component_with_seed))
        else:
            print("  -> WARNING: Seed point was not in any convex component. Falling back to largest component.")
            final_indices = np.array(list(max(components, key=len)))
        if len(final_indices) < 30:
            print(f"  -> Only found {len(final_indices)} connected points. Skipping.")
            continue
        picked_points = processed_mesh.vertices[final_indices]
        print(f"  -> Selected {len(picked_points)} final connected points for fitting.")

        # --- 8. FIT AXIS-ALIGNED ELLIPSOID ---
        fit_params, final_points = fit_axis_aligned_ellipsoid_iteratively(picked_points)
        if fit_params is None:
            print("  -> Robust axis-aligned ellipsoid fit failed, skipping file.")
            continue

        fit_center = fit_params[:3]
        fit_axes_lengths = fit_params[3:6]
        fit_angles_deg = [0.0, 0.0, 0.0]

        # --- 9. SANITY CHECK ---
        distance = np.linalg.norm(seed_point_coords - fit_center)
        if distance > MAX_SEED_TO_CENTER_DISTANCE:
            print(f"  -> !!! WARNING: Fit failed sanity check. Discarding.")
            continue

        print(f"  -> Fit passed sanity check. Ellipsoid axes: {[f'{x:.2f}' for x in fit_axes_lengths]}")

        # --- 10. RECORD SUCCESSFUL RESULTS ---
        result_data = {
            'filename': os.path.basename(file_path),
            'length_x': box_extents[0],
            'width_y': box_extents[1],
            'height_z': box_extents[2],
            'ellipsoid_axis_a': fit_axes_lengths[0],
            'ellipsoid_axis_b': fit_axes_lengths[1],
            'ellipsoid_axis_c': fit_axes_lengths[2],
            'ellipsoid_center_x': fit_center[0],
            'ellipsoid_center_y': fit_center[1],
            'ellipsoid_center_z': fit_center[2],
        }
        results_list.append(result_data)

        # --- 11. SAVE INDIVIDUAL EXCEL FILE ---
        if SAVE_INDIVIDUAL_EXCEL:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = os.path.join(DIRECTORY_PATH, f"{base_name}_ellipsoid_results.xlsx")
            df_individual = pd.DataFrame([result_data])
            try:
                df_individual.to_excel(output_filename, index=False, engine='openpyxl')
                print(f"  -> Successfully saved individual results to: {output_filename}")
            except Exception as e:
                print(f"  -> An error occurred while saving the individual Excel file: {e}")

        # --- 12. VISUALIZATION ---
        if ENABLE_VISUALIZATION:
            plotter = pv.Plotter()

            # --- Add filename and progress count ---
            filename = os.path.basename(file_path)
            progress_text = f"File {i + 1} / {len(stl_files)}"

            # Add filename to the top-left (using 'black' for white backgrounds)
            plotter.add_text(filename, position='upper_left', font_size=12, color='black')
            # Add progress to the top-right
            plotter.add_text(progress_text, position='upper_right', font_size=12, color='black')
            # --- END ---

            plotter.add_mesh(processed_mesh, style='surface', opacity=0.3, color='lightgrey')

            ellipsoid_pv = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
            ellipsoid_pv.points = ellipsoid_pv.points * fit_axes_lengths
            ellipsoid_pv.points += fit_center

            plotter.add_mesh(ellipsoid_pv, style='wireframe', color='blue', line_width=2)
            plotter.add_points(final_points, color='crimson', point_size=6, render_points_as_spheres=True,
                               label='Selected Inlier Points')
            plotter.add_points(seed_point_coords, color='lime', point_size=15, render_points_as_spheres=True,
                               label='Seed Point')

            axis_colors = ['red', 'green', 'blue']
            world_axes_labels = ['X Axis', 'Y Axis', 'Z Axis']
            for i in range(3):
                axis_direction = np.zeros(3)
                axis_direction[i] = 1.0
                semi_axis_length = fit_axes_lengths[i]
                start_point = fit_center - axis_direction * semi_axis_length
                end_point = fit_center + axis_direction * semi_axis_length
                axis_line = pv.Line(start_point, end_point)
                plotter.add_mesh(axis_line, color=axis_colors[i], line_width=5,
                                 label=f"{world_axes_labels[i]} ({semi_axis_length:.2f})")

            plotter.add_legend()

            # --- Add print statement to console ---
            print(f"  -> Displaying plot for: {filename} ({progress_text}). Close the plot window to continue...")

            plotter.show()

    # --- 13. SAVE CONSOLIDATED RESULTS TO EXCEL ---
    if results_list:
        print("\n--- Saving all collected data to a single Excel file ---")
        df = pd.DataFrame(results_list)
        output_filename = os.path.join(DIRECTORY_PATH, 'measurement_ellipsoid_fitting_ALL.xlsx')
        try:
            df.to_excel(output_filename, index=False, engine='openpyxl')
            print(f"Successfully saved consolidated results to: {output_filename}")
        except Exception as e:
            print(f"An error occurred while saving the consolidated Excel file: {e}")
    else:
        print("\n--- No data was successfully processed to save to Excel. ---")

    print("\n--- All files have been processed. ---")
