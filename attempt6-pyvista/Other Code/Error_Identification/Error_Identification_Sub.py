import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import csv

############################################################################
# 1. Load & Preprocess
############################################################################
# Read the CAD and scan meshes.
cad_mesh = pv.read("2x2_MN_Smooth.stl")
scan_mesh = pv.read("2x2_MN+Noise.stl")

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
def random_subsample_points(mesh, max_points=30000):
    pts = mesh.points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        return pts[idx]
    else:
        return pts

cad_points   = cad_mesh.points           # CAD points (after decimation)
cad_normals  = cad_mesh.point_normals    # Corresponding normals (used only for visualization now)
scan_points  = random_subsample_points(scan_mesh, 30000)

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

missing_idx   = np.where(dist_cad_to_scan > threshold)[0]
missing_points = cad_points[missing_idx]

############################################################################
# 4. DBSCAN Clustering
############################################################################
def cluster_outliers(points, eps=10, min_samples=15):
    if len(points) == 0:
        return np.array([]), set()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    unique = set(labels)
    return labels, unique

labels_extra, _   = cluster_outliers(extra_points, eps=10, min_samples=15)
labels_missing, _ = cluster_outliers(missing_points, eps=10, min_samples=15)

def gather_clusters(points, labels):
    clusters = {}
    for cl in set(labels):
        if cl == -1:
            continue  # ignore noise
        mask = (labels == cl)
        clusters[cl] = points[mask]
    return clusters

extra_clusters   = gather_clusters(extra_points, labels_extra)
missing_clusters = gather_clusters(missing_points, labels_missing)

############################################################################
# 5. Fit a Bounding Sphere to Each Cluster
############################################################################
def fit_sphere(points):
    center = np.mean(points, axis=0)
    diffs = points - center
    r = np.sqrt((diffs**2).sum(axis=1).max())
    return center, r

############################################################################
# 6. Classification by Inside/Outside Test Using the CAD Wire Mesh
############################################################################
def classify_cluster_by_inside_test(cluster_points, cad_mesh):
    """
    Create a PolyData object from the cluster points, then use
    select_enclosed_points to determine which points are inside the CAD mesh.
    If more than 50% of the points are inside, label the cluster as a 'subtraction';
    otherwise, label it as an 'addition'.
    """
    pd_cluster = pv.PolyData(cluster_points)
    enclosed = cad_mesh.select_enclosed_points(pd_cluster, tolerance=0.0)
    inside_flags = enclosed.point_data["SelectedPoints"]  # 1 if inside, 0 if outside
    
    # Protect against an empty array
    if len(inside_flags) == 0:
        return "addition"
    
    if (np.sum(inside_flags) / len(inside_flags)) > 0.3:
        return "subtraction"
    else:
        return "addition"

############################################################################
# 7. Process Clusters with the New Classification
############################################################################
def process_clusters(cluster_dict, cad_mesh):
    results = []
    for cl_id, pts in cluster_dict.items():
        center, radius = fit_sphere(pts)
        label = classify_cluster_by_inside_test(pts, cad_mesh)
        results.append({
            "cluster_id": cl_id,
            "center": center,
            "radius": radius,
            "label": label,
            "points": pts
        })
    return results

extra_info   = process_clusters(extra_clusters, cad_mesh)
missing_info = process_clusters(missing_clusters, cad_mesh)
all_clusters = extra_info + missing_info

############################################################################
# 8. Visualization
############################################################################
plotter = pv.Plotter()

# Add the scan mesh for context.
# plotter.add_mesh(scan_mesh, color="lightgrey", opacity=1, label="Scan")
# Show the CAD mesh in wireframe for context.
plotter.add_mesh(cad_mesh, color="green", opacity=0.5, label="CAD")


# Show all error points in one color (black) regardless of classification.
plotter.add_points(extra_points, color="white", opacity=1, point_size=3, render_points_as_spheres=True)


# Draw the bounding spheres for each cluster using different colors for each label.
for cluster_data in all_clusters:
    center = cluster_data["center"]
    radius = cluster_data["radius"]
    label  = cluster_data["label"]

    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=30, phi_resolution=30)
    # For visualization, you might choose a color scheme (e.g., red for additions, blue for subtractions)
    color = "red" if label == "addition" else "blue"
    plotter.add_mesh(sphere, color=color, opacity=0.5)

plotter.show()

############################################################################
# 9. Export Cluster Information to CSV
############################################################################
with open("final_combined_spheres.csv", "w", newline="") as csvfile:
    fieldnames = ["cluster_id", "x", "y", "z", "radius", "label"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for cluster_data in all_clusters:
        c = cluster_data["center"]
        writer.writerow({
            "cluster_id": cluster_data["cluster_id"],
            "x": c[0],
            "y": c[1],
            "z": c[2],
            "radius": cluster_data["radius"],
            "label": cluster_data["label"]
        })
