#!/usr/bin/env python3
"""
extract_error_loops_pyvista.py

This script loads your main model (2x2_NEW_MN.stl) and the noisy model (2x2_NEW_MN_Noisy.stl),
computes the boolean intersection between them (which represents where the noise voxels
touch the main model), extracts the boundary edges (loops) of this intersection, and exports
them as a VTP file (which you can later convert or use directly in your SWN workflow).
"""

import pyvista as pv

# Load the main and noisy models.
main_model_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN.stl"
noisy_model_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.obj"

print("Loading models...")
main_mesh = pv.read(main_model_path)
noisy_mesh = pv.read(noisy_model_path)

# Compute the boolean intersection between the main mesh and the noisy mesh.
# This intersection should correspond to the overlapping region where the noise (error voxels) touches the main model.
print("Computing boolean intersection...")
try:
    inter_mesh = main_mesh.boolean_intersection(noisy_mesh)
except Exception as e:
    print("Boolean intersection failed:", e)
    inter_mesh = None

if inter_mesh is None or inter_mesh.n_points == 0:
    print("No intersection region found. Check your models or boolean parameters.")
else:
    print("Intersection computed. Number of points in intersection mesh:", inter_mesh.n_points)
    
    # Use PyVista's extract_feature_edges filter to obtain the boundary edges.
    # Set feature_edges=False so that we only extract the boundary edges.
    print("Extracting boundary edges (loops) from the intersection...")
    loop_edges = inter_mesh.extract_feature_edges(boundary_edges=True,
                                                  feature_edges=False,
                                                  non_manifold_edges=False)
    
    if loop_edges.n_points == 0:
        print("No boundary edges found in the intersection mesh.")
    else:
        # Save the extracted loop edges to a file.
        output_path = "error_loops.vtp"
        loop_edges.save(output_path)
        print("Extracted error loop edges saved to", output_path)

        # Optionally, visualize the result.
        p = pv.Plotter()
        p.add_mesh(main_mesh, color='lightgray', opacity=0.5, label='Main Model')
        p.add_mesh(noisy_mesh, color='salmon', opacity=0.5, label='Noisy Model')
        p.add_mesh(loop_edges, color='black', line_width=4, label='Error Loop Edges')
        p.add_legend()
        p.show()
