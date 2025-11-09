# ===================================================================================
#
# This script computes an Oriented Bounding Box (OBB) of an input skull model. 
# Actions:
# 1. Load one .stl file.
# 2. Compute its Oriented Bounding Box (the smallest possible rotated box). 
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

    # --- 3. LOAD MESH & CALCULATE OBB ---
    try:
        original_mesh = trimesh.load_mesh(FILE_PATH)

        components = original_mesh.split(only_watertight=False)
        if len(components) > 1:
            print(f"  -> Mesh has {len(components)} components. Keeping the largest one.")
            processed_mesh = sorted(components, key=lambda c: len(c.vertices), reverse=True)[0]
        else:
            processed_mesh = original_mesh
        processed_mesh.process(validate=True)

        # obb is a trimesh.primitives.Box object
        obb = processed_mesh.bounding_box_oriented
        print("  -> Mesh loaded and OBB calculated successfully.")

    except Exception as e:
        print(f"  -> Error loading or processing mesh. Error: {e}")
        processed_mesh = None

    # --- 4. VISUALIZATION ---
    if ENABLE_VISUALIZATION and processed_mesh:
        plotter = pv.Plotter()
        plotter.add_text("Mesh and Oriented Bounding Box (OBB) - Top View, Centered")

        plotter.add_mesh(processed_mesh, style='surface', opacity=0.6, color='lightgrey')

        obb_as_trimesh = trimesh.Trimesh(vertices=obb.vertices, faces=obb.faces)

        obb_as_pv_mesh = pv.wrap(obb_as_trimesh)

        obb_edges = obb_as_pv_mesh.extract_feature_edges()

        # 4. Add just the extracted edges to the plotter
        plotter.add_mesh(obb_edges, color='green', line_width=2,
                         label='Oriented Bounding Box (OBB)')

        plotter.add_legend()

        plotter.camera.SetFocalPoint(obb.centroid)

        plotter.view_xy()

        plotter.show()

    elif not processed_mesh:
        print("  -> Visualization skipped because mesh could not be loaded.")

    print("\n--- Processing complete. ---")
