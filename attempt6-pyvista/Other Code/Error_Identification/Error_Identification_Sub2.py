import numpy as np
import pyvista as pv
from sklearn.cluster import DBSCAN
import csv
import vtk
import time
start_time = time.time()
############################################################################
# 1. Load & Preprocess
############################################################################
cad_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN.stl")
scan_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.obj")

# Compute normals for CAD (if needed later)
cad_mesh = cad_mesh.compute_normals(auto_orient_normals=True)

# Clean and decimate the scan
scan_mesh = scan_mesh.clean(tolerance=1e-5)
scan_mesh = scan_mesh.fill_holes(hole_size=5)
scan_mesh = scan_mesh.triangulate()
scan_mesh = scan_mesh.decimate(0.8)

# Decimate the CAD as well
cad_mesh = cad_mesh.decimate(0.8)

############################################################################
# 2. Create a Voxel Grid Using pv.ImageData
############################################################################
# Get CAD bounding box and add a little padding.
bounds = cad_mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
padding = 1.0
xmin, xmax = bounds[0] - padding, bounds[1] + padding
ymin, ymax = bounds[2] - padding, bounds[3] + padding
zmin, zmax = bounds[4] - padding, bounds[5] + padding

# Define the number of cells in each direction (adjust as needed).
cells_x, cells_y, cells_z = 100, 100, 100
# Compute the spacing based on the number of cells (dims = cells + 1)
spacing = ((xmax - xmin) / cells_x,
           (ymax - ymin) / cells_y,
           (zmax - zmin) / cells_z)
dims = (cells_x + 1, cells_y + 1, cells_z + 1)

# Create the uniform grid using ImageData.
grid = pv.ImageData()
grid.dimensions = dims
grid.origin = (xmin, ymin, zmin)
grid.spacing = spacing
grid_points = grid.points

############################################################################
# 3. Compute Signed Distance Fields Using vtkImplicitPolyDataDistance
############################################################################
# Create implicit functions for CAD and scan.
imp_cad = vtk.vtkImplicitPolyDataDistance()
imp_cad.SetInput(cad_mesh)  # cad_mesh is a PolyData
imp_scan = vtk.vtkImplicitPolyDataDistance()
imp_scan.SetInput(scan_mesh)

# Evaluate signed distances at all grid points.
# (Points inside the mesh return negative values; outside, positive.)
d_cad = np.array([imp_cad.EvaluateFunction(p) for p in grid_points])
d_scan = np.array([imp_scan.EvaluateFunction(p) for p in grid_points])

# For comparison, you could still compute occupancy via select_enclosed_points,
# but here we focus on the continuous (distance field) measure.

############################################################################
# 4. Identify Missing Volume Voxels Based on Distance Field
############################################################################
# We assume that grid points inside the CAD should have d_cad < 0.
# For the scan, if a grid point is missing the volume, d_scan will be positive.
missing_threshold = 0.01  # adjust based on model scale and desired sensitivity

# Select grid points that are inside the CAD (d_cad < 0) but outside the scan (d_scan > missing_threshold)
missing_voxel_idx = np.where((d_cad < 0) & (d_scan > missing_threshold))[0]
missing_voxel_points = grid_points[missing_voxel_idx]
print(f"Found {len(missing_voxel_points)} missing volume sample points (distance field).")

############################################################################
# 5. Cluster the Missing Voxels with DBSCAN
############################################################################
def cluster_voxels(points, eps=spacing[0]*3.5, min_samples=20):
    if len(points) == 0:
        return np.array([]), set()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    unique = set(labels)
    return labels, unique

labels_missing, unique_missing = cluster_voxels(missing_voxel_points)
def gather_clusters(points, labels):
    clusters = {}
    for cl in set(labels):
        if cl == -1:
            continue  # ignore noise
        mask = (labels == cl)
        clusters[cl] = points[mask]
    return clusters

missing_clusters = gather_clusters(missing_voxel_points, labels_missing)

############################################################################
# 6. Fit Bounding Spheres to Each Cluster (for Visualization)
############################################################################
def fit_sphere(points):
    center = np.mean(points, axis=0)
    diffs = points - center
    radius = np.sqrt((diffs**2).sum(axis=1).max())
    return center, radius

cluster_info = []
for cl_id, pts in missing_clusters.items():
    center, radius = fit_sphere(pts)
    cluster_info.append({
        "cluster_id": cl_id,
        "center": center,
        "radius": radius,
        "points": pts
    })

############################################################################
# 7. Visualization
############################################################################
plotter = pv.Plotter()

# Show the scan mesh in light grey.
plotter.add_mesh(scan_mesh, color="lightgrey", opacity=1, label="Scan")
# Optionally, show the CAD mesh in wireframe for context.
# plotter.add_mesh(cad_mesh, style="wireframe", color="green", opacity=0.5, label="CAD")

# Visualize the missing volume voxels in white.
plotter.add_points(missing_voxel_points, color="white", opacity=0.7, point_size=3, render_points_as_spheres=True)

# Draw bounding spheres around each missing cluster in blue.
for cl in cluster_info:
    center = cl["center"]
    radius = cl["radius"]
    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=30, phi_resolution=30)
    plotter.add_mesh(sphere, color="blue", opacity=0.5)

plotter.show()

############################################################################
# 8. Export CSV (Missing Clusters Information)
############################################################################
with open("final_missing_volume_clusters.csv", "w", newline="") as csvfile:
    fieldnames = ["cluster_id", "x", "y", "z", "radius"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for cl in cluster_info:
        c = cl["center"]
        r = cl["radius"]
        row = {
            "cluster_id": cl["cluster_id"],
            "x": c[0],
            "y": c[1],
            "z": c[2],
            "radius": r,
        }
        writer.writerow(row)
end_time = time.time()
print("Execution time: {:.4f} seconds".format(end_time - start_time))