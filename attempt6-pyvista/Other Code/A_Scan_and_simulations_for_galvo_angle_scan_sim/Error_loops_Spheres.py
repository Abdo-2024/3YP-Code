import pyvista as pv
import numpy as np
"""
This script loads a main 3D model and an extracted loop mesh, cleans and triangulates the main mesh, computes normals, and projects each loop vertex onto the mesh surface with a small offset. It then generates spheres at the projected points, merges all spheres into one mesh, combines them with the main model, and exports the resulting combined mesh to an OBJ file.
"""

# Paths to the main model and the tube (extracted loop) file.
main_model_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.obj"
tube_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/error_loops_tube.obj"
output_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/combined_model_with_spheres.obj"

# Load the main model and the tube (which contains the extracted loops).
main_mesh = pv.read(main_model_path)
tube_mesh = pv.read(tube_path)

# Clean, triangulate, and compute normals for the main mesh.
main_mesh = main_mesh.clean().triangulate().compute_normals(cell_normals=False, point_normals=True)
if "Normals" in main_mesh.point_data:
    normals = main_mesh.point_data["Normals"]
else:
    normals = np.zeros((main_mesh.n_points, 3))

# Get the curve points from the tube mesh.
curve_points = tube_mesh.points

# Project each curve vertex onto the main mesh and add a small offset.
# (This helps “attach” the curve to the surface instead of having it float freely.)
projected_curve_points = []
offset = 0.1  # Adjust the offset as needed.
for pt in curve_points:
    idx = main_mesh.find_closest_point(pt)
    proj_pt = main_mesh.points[idx]
    normal = normals[idx] if idx < len(normals) else np.array([0, 0, 1])
    new_pt = proj_pt + normal * offset
    projected_curve_points.append(new_pt)
projected_curve_points = np.array(projected_curve_points)

# Instead of creating a tube, generate a small sphere at each projected curve point.
sphere_radius = 0.2  # Adjust sphere radius as needed.
spheres = []
for pt in projected_curve_points:
    sphere = pv.Sphere(center=pt, radius=sphere_radius, theta_resolution=16, phi_resolution=16)
    spheres.append(sphere)

# Merge all the spheres into a single polydata.
spheres_merged = spheres[0]
for s in spheres[1:]:
    spheres_merged = spheres_merged.merge(s)

# Option 1: Merge the main mesh and the spheres together into one combined mesh.
combined_model = main_mesh.merge(spheres_merged)

# Save the combined mesh to an OBJ file.
combined_model.save(output_path)
print("Combined model with spheres saved to", output_path)
