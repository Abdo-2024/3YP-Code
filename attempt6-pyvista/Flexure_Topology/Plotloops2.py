#!/usr/bin/env python3
"""
extract_error_loops_pyvista_distance.py

This script loads the main model (2x2_NEW_MN.stl) and the noisy model (2x2_NEW_MN_Noisy.stl),
computes a distance field on the noisy model relative to the main model using vtkImplicitPolyDataDistance,
and then extracts a contour at (or near) zero distance.
The resulting contour represents the boundary where the noise meets the main model.
The contour is saved as a VTP file for use in your SWN workflow.
"""

import pyvista as pv
import vtk
import numpy as np

# -------------------------
# Load Models
# -------------------------
# Load the main and noisy models.
main_model_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN.stl"
noisy_model_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.obj"

main_mesh = pv.read(main_model_path)
noisy_mesh = pv.read(noisy_model_path)

main_mesh = main_mesh.clean().triangulate().compute_normals()
noisy_mesh = noisy_mesh.clean().triangulate().compute_normals()

# -------------------------
# Setup Implicit Function for Main Mesh
# -------------------------
print("Setting up implicit distance function...")
implicit_distance = vtk.vtkImplicitPolyDataDistance()
# Pass the vtkPolyData from the main mesh to the implicit function.
implicit_distance.SetInput(main_mesh)

# -------------------------
# Compute Distance Field on the Noisy Mesh
# -------------------------
print("Computing distance field on the noisy model...")
points = noisy_mesh.points
# Evaluate the implicit function for each point in the noisy mesh.
distances = np.array([implicit_distance.EvaluateFunction(p) for p in points])
noisy_mesh["distance"] = distances

# Optionally, you can inspect statistics of the distance values:
print("Distance stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(distances.min(), distances.max(), distances.mean()))

# -------------------------
# Extract the Contour (Loop) Where the Distance is Zero
# -------------------------
# Adjust the isovalue slightly if the 0-level does not yield a good loop.
isovalue = -0.05
print("Extracting contour at distance = {:.4f}...".format(isovalue))
loops = noisy_mesh.contour(isosurfaces=[isovalue], scalars="distance")
if loops.n_points == 0:
    print("No contour found at the specified isovalue. Consider adjusting the threshold.")
else:
    # Convert the line data into a tube mesh.
    tube = loops.tube(radius=2.0)
    print("Tube mesh has", tube.n_points, "points and", tube.n_cells, "cells")
    # Save as both OBJ and PLY
    tube.save("error_loops_tube.obj")
    tube.save("error_loops_tube.ply")
    print("Extracted error loop tube saved to error_loops_tube.obj and error_loops_tube.ply")

    # Optional: visualize
    p = pv.Plotter()
    # Render the main model in blue (#3E4F75)
    p.add_mesh(main_mesh, color="#3E4F75", opacity=0.5, label='Main Model')
    p.add_mesh(noisy_mesh, scalars="distance", cmap="coolwarm", opacity=0.95, label='Noisy Model')
    # Render the loop tube in red (#ff0000ff)
    p.add_mesh(tube, color="#ff0000ff", label='Error Loop Tube')
    #p.add_legend()
    p.show()
