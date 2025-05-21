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
implicit_distance.SetInput(main_mesh)

# -------------------------
# Compute Distance Field on the Noisy Mesh
# -------------------------
print("Computing distance field on the noisy model...")
points = noisy_mesh.points
distances = np.array([implicit_distance.EvaluateFunction(p) for p in points])
noisy_mesh["distance"] = distances

print("Distance stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
    distances.min(), distances.max(), distances.mean()))

# -------------------------
# Extract the Contour (Loop) Where the Distance is Zero
# -------------------------
isovalue = -0.01  # adjust if necessary
print("Extracting contour at distance = {:.4f}...".format(isovalue))
loops = noisy_mesh.contour(isosurfaces=[isovalue], scalars="distance")
if loops.n_points == 0:
    print("No contour found at the specified isovalue. Consider adjusting the threshold.")
else:
    # Convert the contour (line data) into a tube mesh.
    tube = loops.tube(radius=1.0)
    print("Tube mesh has", tube.n_points, "points and", tube.n_cells, "cells")
    tube.save("error_loops_tube.obj")
    tube.save("error_loops_tube.ply")
    print("Extracted error loop tube saved to error_loops_tube.obj and error_loops_tube.ply")

    # -------------------------
    # Visualization and SVG Export
    # -------------------------
    # Use off_screen rendering so that we can export without needing an interactive window
    p = pv.Plotter(off_screen=True)
    
    # Main model rendered in blue (#3E4F75)
    p.add_mesh(main_mesh, color="#3E4F75", opacity=1, label='Main Model')
    # Optionally show the noisy mesh with distance scalars (here using a coolwarm colour map)
    p.add_mesh(noisy_mesh, scalars="distance", cmap="coolwarm", opacity=0.5, label='Noisy Model')
    # Tube (error loop) rendered in red (#ff0000ff)
    p.add_mesh(tube, color="#ff0000ff", label='Error Loop Tube')
    
    # Add an orientation axes widget (shows a small XYZ axis)
    p.show_axes()
    p.add_legend()

    # Render the scene. The window will not be displayed in interactive mode because off_screen=True.
    p.show(auto_close=False)
    
    # Use VTK's GL2PS exporter to export the scene as an SVG.
    exporter = vtk.vtkGL2PSExporter()
    exporter.SetRenderWindow(p.ren_win)
    exporter.SetFilePrefix("scene")  # This will output "scene.svg"
    exporter.SetFileFormatToSVG()      # Set the exporter mode to SVG.
    # You can adjust additional options if needed, e.g.:
    # exporter.SetSortToBSP()
    exporter.Write()
    print("Scene exported as scene.svg")
