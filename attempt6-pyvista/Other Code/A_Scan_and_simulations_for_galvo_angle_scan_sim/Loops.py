import pyvista as pv
import numpy as np
"""
This script loads a 3D mesh with PyVista, cleans and triangulates it, computes normals, and samples points along its surface. It projects these points back onto the mesh with a small normal offset, creates a smooth spline-based tube through those projected points, and visualises the mesh (in a semi-transparent colour) alongside the tube (in red). Optionally, it merges and saves the combined mesh to an OBJ file.
"""

# --- Your existing code to load and preprocess meshes ---
# Paths to the main model.
main_model_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.obj"
output_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/combined_model_with_spheres.obj"

# Load the main model.
main_mesh = pv.read(main_model_path)

# Clean, triangulate, and compute normals for the main mesh.
main_mesh = main_mesh.clean().triangulate().compute_normals(cell_normals=False, point_normals=True)
if "Normals" in main_mesh.point_data:
    normals = main_mesh.point_data["Normals"]
else:
    normals = np.zeros((main_mesh.n_points, 3))

# --- Generate projected curve points ---
# For demonstration, we sample some points along the mesh.
curve_points = main_mesh.points[::max(1, int(main_mesh.n_points/100))]  # sample points along the mesh

projected_curve_points = []
offset = 0.1  # Adjust the offset as needed.
for pt in curve_points:
    idx = main_mesh.find_closest_point(pt)
    proj_pt = main_mesh.points[idx]
    normal = normals[idx] if idx < len(normals) else np.array([0, 0, 1])
    new_pt = proj_pt + normal * offset
    projected_curve_points.append(new_pt)
projected_curve_points = np.array(projected_curve_points)

# --- Function to create a tube (loop) along the projected curve points ---
def create_tube_path(points, tube_radius=0.1, closed_loop=False, smooth_factor=10):
    """
    Creates a tube along a set of points using PyVista's tube filter.
    
    Parameters:
        points (np.ndarray): Array of shape (n_points, 3) with the loop's points.
        tube_radius (float): Radius of the tube.
        closed_loop (bool): If True, the tube will form a closed loop.
        smooth_factor (int): Factor to multiply by n_points for spline smoothing resolution.
        
    Returns:
        pyvista.PolyData: A polydata object representing the tube.
    """
    pts = points.copy()
    n_points = pts.shape[0]
    
    # Optionally form a closed loop by appending the first point to the end.
    if closed_loop:
        pts = np.vstack([pts, pts[0]])
        n_points += 1

    # Instead of using polyline.spline (which is unavailable), we use pv.Spline to create a smooth curve.
    smooth_polyline = pv.Spline(pts, n_points=n_points * smooth_factor)
    
    # Apply the tube filter to the smoothed polyline.
    tube = smooth_polyline.tube(radius=tube_radius)
    return tube

# --- Create the tube for the loop ---
tube_path_mesh = create_tube_path(projected_curve_points, tube_radius=0.1, closed_loop=False, smooth_factor=10)

# --- Display the meshes with specified colours ---
plotter = pv.Plotter()
# Main model sample in colour #3E4F75.
plotter.add_mesh(main_mesh, color="#3E4F75", opacity=0.5)
# Tube loops in red (#ff0000ff).
plotter.add_mesh(tube_path_mesh, color="#ff0000ff", line_width=3)
plotter.show()

# Optionally, merge the main mesh and the tube together and save the combined model.
combined_model = main_mesh.merge(tube_path_mesh)
combined_model.save(output_path)
print("Combined model with loops saved to", output_path)
