import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import csv

############################################################################
# 1. Load & Preprocess
############################################################################
cad_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN_Smooth.stl")
scan_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN+Noise.stl")

# Compute normals for CAD so we can do the dot-product classification.
# 'auto_orient_normals=True' tries to consistently orient them outward, but
# make sure your mesh is relatively closed or manifold.
cad_mesh = cad_mesh.compute_normals(auto_orient_normals=True)

# Clean and decimate the scan
scan_mesh = scan_mesh.clean(tolerance=1e-5)
scan_mesh = scan_mesh.fill_holes(hole_size=5)
scan_mesh = scan_mesh.triangulate()
scan_mesh = scan_mesh.decimate(0.3)

# Decimate the CAD as well
cad_mesh = cad_mesh.decimate(0.3)

############################################################################
# 2. Subsample
############################################################################
def random_subsample_points(mesh, max_points=40000):
    pts = mesh.points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        return pts[idx]
    else:
        return pts

cad_points = cad_mesh.points           # after decimation
cad_normals = cad_mesh.point_normals   # parallel array of normals
scan_points = random_subsample_points(scan_mesh, 40000)

############################################################################
# 3. KD-Tree & Distances
############################################################################
cad_tree = cKDTree(cad_points)
scan_tree = cKDTree(scan_points)

dist_scan_to_cad, idx_scan = cad_tree.query(scan_points)  # scan->cad distances
dist_cad_to_scan, idx_cad  = scan_tree.query(cad_points)  # cad->scan distances

threshold = 5.0  # mm scale; adjust as needed

# Potential "extras" (bulges) are scan points far from CAD
extra_idx = np.where(dist_scan_to_cad > threshold)[0]
extra_points = scan_points[extra_idx]

# Potential "missing" (cavities) are cad points far from scan
missing_idx = np.where(dist_cad_to_scan > threshold)[0]
missing_points = cad_points[missing_idx]

############################################################################
# 4. DBSCAN
############################################################################
def cluster_outliers(points, eps=5, min_samples=10):
    if len(points) == 0:
        return np.array([]), set()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    unique = set(labels)
    return labels, unique

labels_extra, unique_extra = cluster_outliers(extra_points, eps=15, min_samples=15)
labels_missing, unique_missing = cluster_outliers(missing_points, eps=15, min_samples=15)

############################################################################
# 5. Gather clusters into dictionaries
############################################################################
def gather_clusters(points, labels):
    clusters = {}
    for cl in set(labels):
        if cl == -1:
            continue  # noise
        mask = (labels == cl)
        clusters[cl] = points[mask]
    return clusters

extra_clusters = gather_clusters(extra_points, labels_extra)
missing_clusters = gather_clusters(missing_points, labels_missing)

############################################################################
# 6. Fit bounding sphere (kept for size/color visualization)
############################################################################
def fit_sphere(points):
    center = np.mean(points, axis=0)
    diffs = points - center
    r = np.sqrt((diffs**2).sum(axis=1).max())
    return center, r

############################################################################
# 7. Normal-based final classification
############################################################################
def process_clusters(cluster_dict):
    """
    For each cluster:
      1) Fit bounding sphere (just for reporting or color-coding).
      2) Use the normal-based approach to decide 'addition' or 'subtraction'.
         - For each point in the cluster, find nearest CAD point -> dot product
           with CAD normal -> collect signs.
         - If average dot > 0 => addition, else => subtraction.
    """
    results = []
    for cl_id, pts in cluster_dict.items():
        # bounding sphere
        center, radius = fit_sphere(pts)

        # normal-based classification
        dot_signs = []
        for p in pts:
            # find nearest CAD point
            idx_nearest = cad_tree.query(p)[1]
            nearest_pt = cad_points[idx_nearest]
            normal = cad_normals[idx_nearest]
            vec = p - nearest_pt
            dot_val = np.dot(vec, normal)
            dot_signs.append(dot_val)

        avg_dot = np.mean(dot_signs)
        if avg_dot > 0:
            label = "addition"     # outward from CAD
        else:
            label = "subtraction"  # inward / cavity

        results.append({
            "cluster_id": cl_id,
            "center": center,
            "radius": radius,
            "label": label,
            "points": pts
        })
    return results

extra_info = process_clusters(extra_clusters)
missing_info = process_clusters(missing_clusters)

# Combine
all_clusters = extra_info + missing_info

############################################################################
# 8. Color Gradients for additions vs. subtractions
############################################################################
add_radii = [c["radius"] for c in all_clusters if c["label"] == "addition"]
sub_radii = [c["radius"] for c in all_clusters if c["label"] == "subtraction"]

r_min_add = min(add_radii) if add_radii else 0.0
r_max_add = max(add_radii) if add_radii else 1.0
r_min_sub = min(sub_radii) if sub_radii else 0.0
r_max_sub = max(sub_radii) if sub_radii else 1.0
"""
def color_from_radius(r, r_min, r_max, is_addition=True):
    if r_max == r_min:
        t = 0.0
    else:
        t = (r - r_min) / (r_max - r_min)

    if is_addition:
        # from (1, 0, 1) => (0, 0, 1)
        start = np.array([1, 0, 1])
        end   = np.array([0, 0, 1])
    else:
        # from (1, 1, 0) => (1, 0, 0)
        start = np.array([1, 1, 0])
        end   = np.array([1, 0, 0])

    c = start + t * (end - start)
    return tuple(c)
"""
############################################################################
# 9. Visualization
############################################################################
plotter = pv.Plotter()

# Show the main meshes
plotter.add_mesh(scan_mesh, color="lightgrey", opacity=1, label="Scan")

# We can still color the "extra" points black, "missing" white, if you want
plotter.add_points(extra_points, color="black", opacity=0.7, point_size=3, render_points_as_spheres=True)
#plotter.add_points(missing_points, color="white", opacity=0.7, point_size=3, render_points_as_spheres=True)

# Draw the bounding spheres with color gradients
for cluster_data in all_clusters:
    center = cluster_data["center"]
    radius = cluster_data["radius"]
    label  = cluster_data["label"]

    sphere = pv.Sphere(radius=radius, center=center,
                       theta_resolution=30, phi_resolution=30)
    if label == "addition":
        color = 'red'
        # color_from_radius(radius, r_min_add, r_max_add, is_addition=True)
    else:
        color = 'blue'
        #color_from_radius(radius, r_min_sub, r_max_sub, is_addition=False)

    plotter.add_mesh(sphere, color=color, opacity=0.5)

plotter.show()

############################################################################
# 10. Export CSV
############################################################################
with open("final_combined_spheres.csv", "w", newline="") as csvfile:
    fieldnames = ["cluster_id", "x", "y", "z", "radius", "label"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for cluster_data in all_clusters:
        c = cluster_data["center"]
        r = cluster_data["radius"]
        row = {
            "cluster_id": cluster_data["cluster_id"],
            "x": c[0],
            "y": c[1],
            "z": c[2],
            "radius": r,
            "label": cluster_data["label"]
        }
        writer.writerow(row)
