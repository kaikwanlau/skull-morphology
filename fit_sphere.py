# ===================================================================================
#
# This script performs the sphere fitting for skull meshes in batch:
# Actions:
# 1. Load the specified stl folders.
# 2. Perform the sphere fitting for the orbit (a CONCAVE feature).
# 3. Display the mesh and the best-fit sphere.
# 4. Generate an excel file containing the skull dimensions, sphere radii and centers, and orbit curvatures for all meshes.
#
# ===================================================================================

import trimesh
import numpy as np
import pyvista as pv
from scipy.optimize import minimize
import os
import glob
import pandas as pd


def sphere_loss_function(params, points):
    """
    Loss function for sphere fitting. It calculates the sum of squared distances
    from each point to the sphere's surface.
    """
    center = params[:3]
    radius = params[3]
    if radius <= 0:
        return 1e9
    distances = np.linalg.norm(points - center, axis=1)
    return np.sum((distances - radius) ** 2)


def fit_sphere_iteratively(points, max_iterations=3, outlier_std_dev=2.0):
    """
    Iteratively fits a sphere and removes outlier points for better accuracy.
    This makes the final fit less sensitive to a few incorrectly selected points.
    """
    current_points = points.copy()
    for i in range(max_iterations):
        if len(current_points) < 4:
            return None, None, current_points

        initial_center = np.mean(current_points, axis=0)
        initial_radius = np.mean(np.linalg.norm(current_points - initial_center, axis=1))
        initial_guess = np.append(initial_center, initial_radius)

        result = minimize(
            sphere_loss_function,
            initial_guess,
            args=(current_points,),
            method='L-BFGS-B',
            bounds=[(None, None), (None, None), (None, None), (1e-6, None)]
        )

        fit_center, fit_radius = result.x[:3], result.x[3]

        distances_to_center = np.linalg.norm(current_points - fit_center, axis=1)
        errors = np.abs(distances_to_center - fit_radius)
        mean_error, std_error = np.mean(errors), np.std(errors)

        inlier_mask = errors < (mean_error + outlier_std_dev * std_error)

        if np.all(inlier_mask):
            break

        current_points = current_points[inlier_mask]

    final_result = minimize(sphere_loss_function, result.x, args=(current_points,), method='L-BFGS-B',
                            bounds=[(None, None), (None, None), (None, None), (1e-6, None)])
    return final_result.x[:3], final_result.x[3], current_points


if __name__ == '__main__':
    # --- 1. USER-DEFINED PARAMETERS ---

    # The folder containing your STL files
    DIRECTORY_PATH = "dataset"

    ENABLE_VISUALIZATION = True

    # --- 2. ALGORITHM THRESHOLDS & SETTINGS (MODIFIED FOR FINCH SKULL) ---

    SELECTION_METHOD = 'largest_component'
    CURVATURE_RADIUS = 2.0
    TARGET_POINT_COUNT = 400

    # Strict anatomical limits for a finch orbit
    MIN_ORBIT_RADIUS = 2.0  
    MAX_ORBIT_RADIUS = 6.0  

    # ROI parameters
    ROI_START_PERCENT = 0.30  # Ignore the back 30% (braincase)
    ROI_END_PERCENT = 0.70  # Ignore the front 30% (beak tip)

    # The number of concave points to try as seeds before giving up.
    MAX_SEED_ATTEMPTS = 15  # You can increase this to 20 for "ugly" meshes

    ENABLE_ROBUST_FIT = True
    ROBUST_FIT_OUTLIAR_STD = 2.0

    # --- 3. SCRIPT SETUP ---
    results_list = []
    search_path = os.path.join(DIRECTORY_PATH, '*.stl')
    stl_files = glob.glob(search_path)

    # --- NEW: Counters for successful and failed files ---
    successful_fit_count = 0
    failed_files_list = []
    # --- END NEW ---

    if not stl_files:
        exit(f"Error: No .stl files found in {DIRECTORY_PATH}")
    print(f"Found {len(stl_files)} STL files to process.\n")

    # --- 4. MAIN PROCESSING LOOP ---
    for i, file_path in enumerate(stl_files):
        print(f"--- Processing file {i + 1}/{len(stl_files)}: {os.path.basename(file_path)} ---")

        # This flag will be set to True once a good sphere is found for this file.
        successful_fit_found = False

        try:
            original_mesh = trimesh.load_mesh(file_path)

            components = original_mesh.split(only_watertight=False)
            if len(components) > 1:
                print(f"  -> Mesh has {len(components)} components. Keeping the largest one.")
                processed_mesh = sorted(components, key=lambda c: len(c.vertices), reverse=True)[0]
            else:
                processed_mesh = original_mesh
            processed_mesh.process(validate=True)

            box_extents = processed_mesh.bounding_box.extents

        except Exception as e:
            print(f"  -> Error loading or processing mesh, skipping. Error: {e}")
            continue

        # --- 4.5. DEFINE REGION OF INTEREST (ROI) ---
        try:
            bounds = processed_mesh.bounds  # Gets [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            x_min, x_max = bounds[0, 0], bounds[1, 0]
            x_range = box_extents[0]

            # Define a "search band" in the middle of the skull
            roi_x_min_threshold = x_min + (x_range * ROI_START_PERCENT)
            roi_x_max_threshold = x_min + (x_range * ROI_END_PERCENT)

            # Create a boolean mask for vertices within this X-axis band
            roi_mask_min = processed_mesh.vertices[:, 0] > roi_x_min_threshold
            roi_mask_max = processed_mesh.vertices[:, 0] < roi_x_max_threshold
            roi_mask = np.logical_and(roi_mask_min, roi_mask_max)

            if not np.any(roi_mask):
                print("  -> WARNING: ROI mask selected no vertices. Skull may be oriented incorrectly.")
                print("  -> Falling back to using the full mesh.")
                roi_mask = np.ones(len(processed_mesh.vertices), dtype=bool)
            else:
                print(
                    f"  -> ROI filter activated. Searching for points between X={roi_x_min_threshold:.2f} and X={roi_x_max_threshold:.2f}")
        except Exception as e:
            print(f"  -> Error creating ROI, falling back to full mesh. Error: {e}")
            roi_mask = np.ones(len(processed_mesh.vertices), dtype=bool)

        # --- 5. SELECT POINTS BASED ON CURVATURE (MODIFIED WITH ITERATIVE SEED) ---
        mean_curvatures = trimesh.curvature.discrete_mean_curvature_measure(
            processed_mesh, processed_mesh.vertices, radius=CURVATURE_RADIUS
        )

        # Apply ROI to curvature search
        curvatures_in_roi = mean_curvatures.copy()
        curvatures_in_roi[~roi_mask] = 1.0  # Set non-ROI points to a high (non-concave) value

        # Get TOP N seed points
        sorted_curvature_indices_local = np.argsort(curvatures_in_roi)
        all_vertex_indices = np.arange(len(processed_mesh.vertices))
        top_N_seed_indices = all_vertex_indices[sorted_curvature_indices_local[:MAX_SEED_ATTEMPTS]]

        # Check if we found any concave points at all
        if curvatures_in_roi[top_N_seed_indices[0]] == 1.0:
            print("  -> Error: Could not find any concave points in the ROI. Skipping file.")
            continue  # Skips to the next file

        # 1. Get the indices of all points, sorted by curvature
        sorted_indices = np.argsort(mean_curvatures)
        # 2. Create the initial mask of N most concave points
        initial_mask = np.zeros(len(processed_mesh.vertices), dtype=bool)
        initial_mask[sorted_indices[:TARGET_POINT_COUNT]] = True
        # 3. Create the *final* mask by combining with the ROI
        valid_indices_mask = np.logical_and(initial_mask, roi_mask)

        # --- Loop through the top N seed points ---
        for attempt, seed_index in enumerate(top_N_seed_indices):
            print(f"\n  -> Seed Attempt {attempt + 1}/{MAX_SEED_ATTEMPTS} (Vertex {seed_index})")
            seed_point_coords = processed_mesh.vertices[seed_index]

            # Check if this seed is part of the valid (concave + ROI) group
            if not valid_indices_mask[seed_index]:
                print(f"  -> Seed {seed_index} is not in the valid point set (concave + ROI). Trying next seed.")
                continue

            final_indices = np.array([])
            if SELECTION_METHOD == 'flood_fill':
                # ... (flood_fill logic) ...
                pass  # This logic is not being used
            else:
                # print("  -> Using 'largest_component' (improved) selection method.")
                valid_edges_mask = valid_indices_mask[processed_mesh.edges].all(axis=1)
                selected_edges = processed_mesh.edges[valid_edges_mask]

                if len(selected_edges) > 0:
                    components = trimesh.graph.connected_components(selected_edges)
                    if components:
                        # print(f"  -> Found {len(components)} component(s) in ROI. Searching for the one with the seed point...")
                        for component in components:
                            if seed_index in component:
                                # print("    -> Found component connected to the seed point.")
                                final_indices = np.array(component)
                                break
                        if len(final_indices) == 0:
                            print(
                                "    -> WARNING: Seed not in any component. Falling back to largest in ROI.")
                            final_indices = np.array(max(components, key=len))

            if len(final_indices) < 20:
                print(
                    f"  -> Only found {len(final_indices)} connected points. Trying next seed.")
                continue

            picked_points = processed_mesh.vertices[final_indices]
            print(f"  -> Selected {len(picked_points)} final connected points for fitting.")

            # --- 6. FIT SPHERE TO SELECTED POINTS ---
            fit_center, fit_radius, final_points = fit_sphere_iteratively(picked_points,
                                                                          outlier_std_dev=ROBUST_FIT_OUTLIAR_STD)
            if fit_center is None:
                print("  -> Robust fit failed. Trying next seed.")
                continue
            picked_points = final_points

            # --- 7. SANITY CHECK (MODIFIED) ---
            if not (MIN_ORBIT_RADIUS < fit_radius < MAX_ORBIT_RADIUS):
                print(f"  -> !!! WARNING: Fit failed sanity check. !!!")
                print(
                    f"  -> Fitted radius ({fit_radius:.2f}) is outside the expected range ({MIN_ORBIT_RADIUS} - {MAX_ORBIT_RADIUS}).")
                print(f"  -> This means the sphere was fit to the wrong part of the mesh. Discarding result.")
                print("  -> Trying next best seed point...")
                continue  # --- This now goes to the next seed in the for loop ---

            print(f"  -> Fit passed sanity check. Sphere radius: {fit_radius:.4f}")
            successful_fit_found = True  # We found a good one!

            # --- 8. RECORD SUCCESSFUL RESULTS ---
            results_list.append({
                'filename': os.path.basename(file_path),
                'length_x': box_extents[0],
                'width_y': box_extents[1],
                'height_z': box_extents[2],
                'sphere_radius': fit_radius,
                'sphere_center_x': fit_center[0],
                'sphere_center_y': fit_center[1],
                'sphere_center_z': fit_center[2],
                'curvature': 1 / fit_radius,
            })

            # If we don't visualize, we still need to break from the seed loop
            if not ENABLE_VISUALIZATION:
                print("  -> Fit successful. Saving result and moving to next file.")
                break  # Break from the seed loop

            # --- 9. VISUALIZATION (MODIFIED) ---
            if ENABLE_VISUALIZATION:
                plotter = pv.Plotter()
                filename = os.path.basename(file_path)
                progress_text = f"File {i + 1} / {len(stl_files)}"
                plotter.add_text(filename, position='upper_left', font_size=12, color='black')
                plotter.add_text(progress_text, position='upper_right', font_size=12, color='black')

                plotter.add_mesh(processed_mesh, style='surface', opacity=0.4, color='lightgrey')
                fit_sphere = pv.Sphere(radius=fit_radius, center=fit_center)
                plotter.add_mesh(fit_sphere, style='wireframe', color='blue', line_width=2)
                plotter.add_points(picked_points, color='crimson', point_size=6, render_points_as_spheres=True,
                                   label='Selected Points (Inliers)')

                plotter.add_points(seed_point_coords, color='lime', point_size=15, render_points_as_spheres=True,
                                   label='Seed Point')
                plotter.add_legend()

                print(f"  -> Displaying plot for: {filename} ({progress_text}). Close the plot window to continue...")
                plotter.show()

                # --- Break from seed loop after showing plot ---
                print("  -> Plot closed. Moving to next file.")
                break  # A successful fit was found and shown, break from the seed loop

        # --- This code runs after the seed loop is finished or broken ---
        if not successful_fit_found:
            print(f"  -> All {MAX_SEED_ATTEMPTS} seed attempts failed for this file. Skipping.")
            # --- NEW: Add failed file to list ---
            failed_files_list.append(os.path.basename(file_path))
            # --- END NEW ---
        else:
            # --- Increment success counter ---
            successful_fit_count += 1

    # --- 10. SAVE RESULTS TO EXCEL ---

    # --- NEW: Print final summary ---
    print("\n--- BATCH COMPLETE ---")
    print(f"--- Successfully processed {successful_fit_count} / {len(stl_files)} files. ---")

    if failed_files_list:
        print(f"\n--- {len(failed_files_list)} FILES FAILED TO FIND A FIT ---")
        for filename in failed_files_list:
            print(f"  -> {filename}")
    else:
        print("--- All files were processed successfully! ---")
    # --- END NEW ---

    if results_list:
        print("\n--- Saving all collected data to Excel ---")
        df = pd.DataFrame(results_list)
        output_filename = os.path.join(DIRECTORY_PATH, 'measurement_sphere_fitting_ALL.xlsx')
        try:
            # --- FIXED TYPO: 'open_pyxl' to 'openpyxl' ---
            df.to_excel(output_filename, index=False, engine='openpyxl')
            print(f"Successfully saved results to: {output_filename}")
        except Exception as e:
            print(f"An error occurred while saving the Excel file: {e}")
    else:
        print("\n--- No data was successfully processed to save to Excel. ---")
        print("--- Try adjusting the parameters (ROI_START_PERCENT, MIN/MAX_ORBIT_RADIUS) and re-running. ---")

    print("\n--- All files have been processed. ---")
