import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

# ---------------------------------------------------------------------------
# 1. Load STL Files
# ---------------------------------------------------------------------------
# Adjust paths to your actual "CAD" and "noisy" STL files
cad_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN_Smooth.stl"
scan_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN+Noise.stl"  # presumably your noisy file

cad_mesh = pv.read(cad_path)
scan_mesh = pv.read(scan_path)

# ---------------------------------------------------------------------------
# 2. Repair / Clean the Noisy Mesh (optional)
# ---------------------------------------------------------------------------
# If the scan has large breaks, holes, or other geometry issues, try:
#   - clean() : merges duplicate points, removes unused vertices, etc.
#   - fill_holes() : attempts to fill larger holes in the mesh
#   - triangulate() : ensures consistent triangulation
# Adjust the arguments (like tolerance) as needed.
scan_mesh = scan_mesh.clean(tolerance=1e-5)  
scan_mesh = scan_mesh.fill_holes(hole_size=10)  # attempt to fill big holes
scan_mesh = scan_mesh.triangulate()          # ensure triangulated faces

# ---------------------------------------------------------------------------
# 3. Convert Each Mesh to a Point Cloud
# ---------------------------------------------------------------------------
# Option A: Directly use the existing vertices
#   (Depending on your STL, the distribution might be irregular.)
cad_points = cad_mesh.points
scan_points = scan_mesh.points

# Option B: Uniformly sample points across the surface
#   (This helps if you want an evenly distributed point cloud.)
num_samples = 20000  # number of points to sample on each mesh
#cad_cloud = cad_mesh.sample(n_points=num_samples)
#scan_cloud = scan_mesh.sample(n_points=num_samples)

#cad_points = cad_cloud.points
#scan_points = scan_cloud.points

# ---------------------------------------------------------------------------
# 4. Compute Distances in Both Directions
# ---------------------------------------------------------------------------
# We'll build KD-Trees to quickly find nearest distances:
cad_kdtree = cKDTree(cad_points)
scan_kdtree = cKDTree(scan_points)

# Distances from each scan point --> nearest CAD point
distances_scan_to_cad, _ = cad_kdtree.query(scan_points)

# Distances from each CAD point --> nearest scan point
distances_cad_to_scan, _ = scan_kdtree.query(cad_points)

# ---------------------------------------------------------------------------
# 5. Identify Outliers (Errors) Above a Threshold
# ---------------------------------------------------------------------------
error_threshold = 0.2  # user-defined, tune based on your units/tolerance

# Indices of "extra geometry" (scan > CAD threshold)
extra_idx = np.where(distances_scan_to_cad > error_threshold)[0]
extra_points = scan_points[extra_idx]

# Indices of "missing geometry" (CAD > scan threshold)
missing_idx = np.where(distances_cad_to_scan > error_threshold)[0]
missing_points = cad_points[missing_idx]

# ---------------------------------------------------------------------------
# 6. (Optional) Clustering of Outlier Points (e.g. DBSCAN)
# ---------------------------------------------------------------------------
# We'll cluster the "extra" points and the "missing" points separately
# so you can see if the errors form large connected regions or random scatter.

def cluster_outliers(points, eps=0.5, min_samples=10):
    """
    Run DBSCAN clustering on a set of 3D outlier points.
    Returns labels (array) for each point and the unique cluster IDs.
    """
    if len(points) == 0:
        return np.array([]), []
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    unique_labels = set(labels)
    return labels, unique_labels

# Cluster the "extra" points
labels_extra, unique_extra = cluster_outliers(extra_points, eps=0.5, min_samples=10)
# Cluster the "missing" points
labels_missing, unique_missing = cluster_outliers(missing_points, eps=0.5, min_samples=10)

def label_to_color(label):
    """
    Map a cluster label (int) to an RGB color.
    Noise points have label = -1, so color them black or gray.
    """
    if label == -1:  # "noise"
        return (0.2, 0.2, 0.2)
    # Simple deterministic approach: use label % 3 for different color channels
    # Or pick from a predefined palette
    palette = [
        (1, 0, 0),   # red
        (0, 1, 0),   # green
        (0, 0, 1),   # blue
        (1, 1, 0),   # yellow
        (1, 0, 1),   # magenta
        (0, 1, 1),   # cyan
        (1, 0.5, 0), # orange
        (0.5, 0, 0.5),
    ]
    return palette[label % len(palette)]

# ---------------------------------------------------------------------------
# 7. Visualize in PyVista
# ---------------------------------------------------------------------------
plotter = pv.Plotter()

# Add the main meshes with some transparency so you can see the spheres
plotter.add_mesh(cad_mesh, color="lightgray", opacity=0.5, show_edges=False, label="CAD")
plotter.add_mesh(scan_mesh, color="lightgreen", opacity=0.5, show_edges=False, label="SCAN")

# Helper function to add spheres for a set of points
def add_spheres_for_points(points, labels, radius=0.1, is_extra=True):
    """
    points: Nx3 array
    labels: array of cluster labels
    is_extra: if True => "extra" geometry, if False => "missing" geometry
    """
    for i, pt in enumerate(points):
        sphere = pv.Sphere(radius=radius, center=pt)
        # color by cluster label or type of error
        color = label_to_color(labels[i]) if labels.size > 0 else (1, 0, 0)
        if not is_extra:
            # For "missing" geometry, we can shift color or do something different
            # or just use the same color approach
            pass
        plotter.add_mesh(sphere, color=color)

# Add spheres for "extra" geometry
add_spheres_for_points(extra_points, labels_extra, radius=0.05, is_extra=True)

# Add spheres for "missing" geometry
add_spheres_for_points(missing_points, labels_missing, radius=0.05, is_extra=False)

plotter.add_legend()
plotter.show()
