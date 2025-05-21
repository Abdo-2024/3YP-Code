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

# Clean and decimate the scan mesh.
scan_mesh = scan_mesh.clean(tolerance=1e-5)
scan_mesh = scan_mesh.fill_holes(hole_size=5)
scan_mesh = scan_mesh.triangulate()
scan_mesh = scan_mesh.decimate(0.7)

# Optionally decimate the CAD mesh.
cad_mesh = cad_mesh.decimate(0.8)

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
cad_normals  = cad_mesh.point_normals    # Normals for visualization (if needed)
scan_points  = random_subsample_points(scan_mesh, 40000)

############################################################################
# 3. KD-Tree & Distance Computation
############################################################################
cad_tree  = cKDTree(cad_points)
scan_tree = cKDTree(scan_points)

# Compute distances from scan to CAD and vice-versa.
dist_scan_to_cad, _ = cad_tree.query(scan_points)
dist_cad_to_scan, _ = scan_tree.query(cad_points)

threshold = 5.0  # Adjust this threshold as needed

# Identify "extra" points (errors on the scan) based on the threshold.
extra_idx   = np.where(dist_scan_to_cad > threshold)[0]
extra_points = scan_points[extra_idx]

############################################################################
# 4. Cluster error points and add bounding spheres
############################################################################
# Cluster the error (white dot) points using DBSCAN.
db = DBSCAN(eps=10.0, min_samples=5).fit(extra_points)
labels = db.labels_
unique_labels = set(labels) - {-1}  # Exclude noise points (label == -1)

def create_bounding_sphere(points):
    """
    Returns a PyVista sphere that approximately encloses the input points.
    This approach computes the centroid and uses the maximum distance from 
    the centroid as the sphere radius.
    """
    center = points.mean(axis=0)
    # Compute squared distances from the centroid.
    dist2 = np.sum((points - center)**2, axis=1)
    radius = np.sqrt(dist2.max())
    return pv.Sphere(radius=radius, center=center, phi_resolution=30, theta_resolution=30)

# Create a PyVista plotter before adding cluster spheres.
plotter = pv.Plotter()

for cluster_id in unique_labels:
    # Extract the points belonging to the current cluster.
    cluster_points = extra_points[labels == cluster_id]
    
    # Create a PolyData object from the cluster points.
    cluster_pdata = pv.PolyData(cluster_points)
    
    # Use the CAD mesh to determine which points in the cluster are enclosed.
    enclosed = cad_mesh.select_enclosed_points(cluster_pdata, tolerance=0.0, check_surface=True)
    
    # "SelectedPoints" array: 1 indicates the point is inside the CAD mesh.
    inside_mask = enclosed.point_data["SelectedPoints"] == 1
    num_inside = inside_mask.sum()
    num_outside = len(inside_mask) - num_inside
    
    # Choose sphere color based on whether the cluster is mostly inside or outside.
    if num_inside > num_outside:
        sphere_color = "blue"   # Subtraction error (mostly inside the CAD)
    else:
        sphere_color = "red"    # Addition error (mostly outside the CAD)
    
    # Compute an approximate bounding sphere for the cluster.
    sphere = create_bounding_sphere(cluster_points)
    
    # Add the bounding sphere to the plotter.
    plotter.add_mesh(sphere, color=sphere_color, opacity=0.4)

############################################################################
# 5. Visualization
############################################################################
# Add the CAD mesh (wireframe) for context.
# plotter.add_mesh(cad_mesh, color="lightgray", opacity=1, label="CAD")
# Optionally, add the scan mesh:
plotter.add_mesh(scan_mesh, color="lightgrey", opacity=1, label="Scan")

# Add the white error points.
plotter.add_points(extra_points, color="white", opacity=1, point_size=3, render_points_as_spheres=True)

# Render the final visualization.
plotter.show()
