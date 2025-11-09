# ===================================================================================
#
# This script computes an Axis-Aligned Bounding Box (AABB) of an input skull model. 
# Actions:
# 1. Load one .stl file.
# 2. Compute its Axis-Aligned Bounding Box (a simple, non-rotated box). 
#
# ===================================================================================

import trimesh
import pyvista as pv
import os

if __name__ == '__main__':
    # --- 1. USER-DEFINED PARAMETERS ---

    # Set this to the full path of your single STL file
    FILE_PATH = "dataset/G.SeptentrionalistA.stl"

    # Set to True to visualize the 3D plot
    ENABLE_VISUALIZATION = True

    # --- 2. SCRIPT SETUP ---
    if not os.path.exists(FILE_PATH):
        exit(f"Error: File not found at {FILE_PATH}")

    print(f"--- Processing: {os.path.basename(FILE_PATH)} ---")

    # --- 3. LOAD MESH ---
    try:
        original_mesh = trimesh.load_mesh(FILE_PATH)

        components = original_mesh.split(only_watertight=False)
        if len(components) > 1:
            print(f"  -> Mesh has {len(components)} components. Keeping the largest one.")
            processed_mesh = sorted(components, key=lambda c: len(c.vertices), reverse=True)[0]
        else:
            processed_mesh = original_mesh
        processed_mesh.process(validate=True)
        print("  -> Mesh loaded successfully.")

    except Exception as e:
        print(f"  -> Error loading or processing mesh. Error: {e}")
        processed_mesh = None  # Set mesh to None to skip visualization

    # --- 4. VISUALIZATION ---
    if ENABLE_VISUALIZATION and processed_mesh:
        plotter = pv.Plotter()
        plotter.add_text("Mesh with Axis-Aligned Bounding Box (AABB)")

        plotter.add_mesh(processed_mesh, style='surface', opacity=0.4, color='lightgrey')

        # --- Start of Modification: Replacing the AABB mesh drawing ---

        # 1. Get the bounds (min_x, max_x, min_y, max_y, min_z, max_z) from trimesh
        bounds = processed_mesh.bounds.flatten()  # Flattens to [min_x, min_y, min_z], [max_x, max_y, max_z]

        # 2. Calculate the center and side lengths of the AABB
        center_x = (bounds[0] + bounds[3]) / 2.0
        center_y = (bounds[1] + bounds[4]) / 2.0
        center_z = (bounds[2] + bounds[5]) / 2.0
        center = [center_x, center_y, center_z]

        length_x = bounds[3] - bounds[0]
        length_y = bounds[4] - bounds[1]
        length_z = bounds[5] - bounds[2]

        # 3. Create a pyvista.Cube centered at the origin with the calculated side lengths
        #    Note: pyvista.Cube takes side lengths, not half-lengths.
        aabb_cube = pv.Cube(center=(0, 0, 0), x_length=length_x, y_length=length_y, z_length=length_z)

        # 4. Translate the cube to the correct center position
        aabb_cube.translate(center, inplace=True)

        # 5. Add the pyvista cube as a wireframe
        plotter.add_mesh(aabb_cube, style='wireframe', color='red', line_width=2,
                         label='Axis-Aligned Bounding Box (AABB)')

        # --- End of Modification ---

        plotter.add_legend()
        plotter.show(cpos='xy')

    elif not processed_mesh:
        print("  -> Visualization skipped because mesh could not be loaded.")

    print("\n--- Processing complete. ---")
