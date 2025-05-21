import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import csv

############################################################################
# 1. Load & Preprocess
############################################################################
# Read the CAD and scan meshes.
cad_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN.stl")
scan_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.obj")

# Compute normals for the CAD mesh so we can use select_enclosed_points reliably.
cad_mesh = cad_mesh.compute_normals(auto_orient_normals=True)

# Clean and decimate the scan
scan_mesh = scan_mesh.clean(tolerance=1e-5)
scan_mesh = scan_mesh.fill_holes(hole_size=5)
scan_mesh = scan_mesh.triangulate()
scan_mesh = scan_mesh.decimate(0.3)

# Also decimate the CAD mesh (if desired)
cad_mesh = cad_mesh.decimate(0.3)

############################################################################
# 2. Subsample Points
############################################################################
def random_subsample_points(mesh, max_points=40000):
    pts = mesh.points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        return pts[idx]
    else:
        return pts

cad_points   = cad_mesh.points           # CAD points (after decimation)
cad_normals  = cad_mesh.point_normals    # Corresponding normals (used only for visualization now)
scan_points  = random_subsample_points(scan_mesh, 40000)

############################################################################
# 3. KD-Tree & Distance Computation
############################################################################
cad_tree  = cKDTree(cad_points)
scan_tree = cKDTree(scan_points)

# Compute distances: scan->CAD and CAD->scan
dist_scan_to_cad, _ = cad_tree.query(scan_points)
dist_cad_to_scan, _ = scan_tree.query(cad_points)

threshold = 5.0  # Adjust this threshold as needed

# Identify "extra" points (errors on the scan) and "missing" points (errors on the CAD)
extra_idx   = np.where(dist_scan_to_cad > threshold)[0]
extra_points = scan_points[extra_idx]

############################################################################
# 8. Visualization
############################################################################
plotter = pv.Plotter()

# Add the CAD mesh in wireframe for context.
plotter.add_mesh(cad_mesh, color="green", opacity=0.5, label="CAD")
# plotter.add_mesh(scan_mesh, color="lightgrey", opacity=1, label="Scan")

# Show all error points in white.
plotter.add_points(extra_points, color="white", opacity=1, point_size=3, render_points_as_spheres=True)

plotter.show()
