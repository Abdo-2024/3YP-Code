import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import csv

# --- VTK imports to build the SDF ---
import vtk

############################################################################
# 1. Load & Preprocess
############################################################################
cad_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/1_Needle.stl")
scan_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN+Noise.stl")

# Clean and decimate the scan
scan_mesh = scan_mesh.clean(tolerance=1e-5)
scan_mesh = scan_mesh.fill_holes(hole_size=5)
scan_mesh = scan_mesh.triangulate()
scan_mesh = scan_mesh.decimate(0.8)

# Decimate the CAD as well
cad_mesh = cad_mesh.decimate(0.8)

############################################################################
# 2. Subsample
############################################################################
def random_subsample_points(mesh, max_points=15000):
    pts = mesh.points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        return pts[idx]
    else:
        return pts

cad_points = random_subsample_points(cad_mesh, 15000)
scan_points = random_subsample_points(scan_mesh, 15000)

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

labels_extra, unique_extra   = cluster_outliers(extra_points,   eps=15, min_samples=15)
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
# 6. Bounding Sphere (kept for visualization)
############################################################################
def fit_sphere(points):
    center = np.mean(points, axis=0)
    diffs = points - center
    r = np.sqrt((diffs**2).sum(axis=1).max())
    return center, r

############################################################################
# 7. Build the Signed Distance Field from CAD (NEW / MODIFIED)
############################################################################
def build_sdf_from_pyvista_mesh(mesh):
    """
    Convert a PyVista mesh to a vtkPolyData, then build a vtkImplicitPolyDataDistance.
    If your CAD mesh isn't watertight, the sign might be invalid.
    """
    # Ensure we have a polydata object
    pd = mesh.cast_to_poly_points()  # or mesh.polydata, but cast_to_polydata is safer
    sdf = vtk.vtkImplicitPolyDataDistance()
    sdf.SetInput(pd)
    return sdf

cad_sdf = build_sdf_from_pyvista_mesh(cad_mesh)

############################################################################
# 8. SDF-based Classification (NEW / MODIFIED)
############################################################################
def process_clusters(cluster_dict):
    """
    For each cluster:
      1) Fit bounding sphere (just for reporting or color-coding).
      2) Use the SDF -> EvaluateFunction() on each point.
         - If average signed distance > 0 => 'addition' (outside).
         - If average signed distance < 0 => 'subtraction' (inside).
    """
    results = []
    for cl_id, pts in cluster_dict.items():
        center, radius = fit_sphere(pts)

        # Evaluate the SDF for each point in this cluster
        dvals = []
        for pt in pts:
            dist_val = cad_sdf.EvaluateFunction(pt)  # + => outside, - => inside
            dvals.append(dist_val)

        avg_dist = np.mean(dvals)
        if avg_dist > 0:
            label = "addition"
        else:
            label = "subtraction"

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

all_clusters = extra_info + missing_info

############################################################################
# 9. Visualization
############################################################################
plotter = pv.Plotter()
plotter.add_mesh(scan_mesh, color="lightgrey", opacity=1, label="Scan")

# We'll color "extra" points black, "missing" points white
plotter.add_points(extra_points,   color="black", opacity=0.7, point_size=2, render_points_as_spheres=False)
plotter.add_points(missing_points, color="white", opacity=0.7, point_size=2, render_points_as_spheres=False)

for cluster_data in all_clusters:
    center = cluster_data["center"]
    radius = cluster_data["radius"]
    label  = cluster_data["label"]

    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=30, phi_resolution=30)
    color  = "red" if label == "addition" else "blue"
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
