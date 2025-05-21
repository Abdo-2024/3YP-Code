import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import csv

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

# Adjust eps/min_samples to your geometry
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
# 6. Paraboloid Fit (Logic from Error_Identification4)
############################################################################
def fit_paraboloid_3d(points):
    """
    1) Compute PCA => define local coords (u,v,w)
    2) Fit w = a*u^2 + b*v^2 + c*u*v + d*u + e*v + f
    3) Return: (a,b,c,d,e,f), plus the transform info
    """
    from sklearn.decomposition import PCA
    mean_pt = points.mean(axis=0)
    centered = points - mean_pt

    pca = PCA(n_components=3)
    pca.fit(centered)
    # localZ => pca.components_[2] (smallest variance), so (u,v) is the plane
    localX = pca.components_[0]
    localY = pca.components_[1]
    localZ = pca.components_[2]

    T = np.vstack([localX, localY, localZ])  # shape (3,3)
    uvw = centered.dot(T.T)
    u = uvw[:,0]
    v = uvw[:,1]
    w = uvw[:,2]

    # Build design matrix: w = a*u^2 + b*v^2 + c*u*v + d*u + e*v + f
    M = np.column_stack((u**2, v**2, u*v, u, v, np.ones_like(u)))
    params, _, _, _ = np.linalg.lstsq(M, w, rcond=None)
    return params, (mean_pt, T)

def paraboloid_curvature_sign(params):
    """
    Hessian => [[2a, c],
                [ c, 2b]]
    +1 => bowl (concave up), -1 => dome (concave down), 0 => saddle/flat
    """
    a,b,c,d,e,f = params
    H = np.array([[3*a, c],
                  [  c, 4*b]])
    eigvals = np.linalg.eigvals(H)
    if np.all(eigvals > 0):
        return +1  # bowl
    elif np.all(eigvals <= 0):
        return -1  # dome
    else:
        return 0    # saddle or ambiguous

############################################################################
# 7. Fit bounding sphere + final classification
############################################################################
def fit_sphere(points):
    center = np.mean(points, axis=0)
    diffs = points - center
    r = np.sqrt((diffs**2).sum(axis=1).max())
    return center, r

# We want to store the final results in a single structure:
#   [
#       {
#          "center": (x,y,z),
#          "radius": R,
#          "type": "addition" or "subtraction",
#          "points": Nx3 array, # optional, if needed
#          ...
#       },
#       ...
#   ]
# We'll handle "extra_clusters" with default "addition", "missing_clusters" with default "subtraction"

def process_clusters(cluster_dict, default_label):
    """
    1) Fit bounding sphere.
    2) Paraboloid fit => shape_sign => override default if needed
    shape_sign = +1 => 'subtraction', -1 => 'addition', 0 => fallback
    """
    results = []
    for cl_id, pts in cluster_dict.items():
        # bounding sphere
        center, radius = fit_sphere(pts)

        # paraboloid
        params, transform_info = fit_paraboloid_3d(pts)
        shape_sign = paraboloid_curvature_sign(params)
        if shape_sign == +1:
            shape_label = "subtraction"
        elif shape_sign == -1:
            shape_label = "addition"
        else:
            # fallback to default
            shape_label = default_label

        results.append({
            "cluster_id": cl_id,
            "center": center,
            "radius": radius,
            "shape_sign": shape_sign,
            "label": shape_label,  # final classification
            "points": pts
            
        })
    return results

extra_info = process_clusters(extra_clusters, default_label="addition")
missing_info = process_clusters(missing_clusters, default_label="subtraction")

all_clusters = extra_info + missing_info

############################################################################
# 8. Color Gradients from Error_Identification3
############################################################################
# We want separate radius-based gradients for additions vs. subtractions:
# Additions: red -> yellow
# Subtractions: blue -> majenta

add_radii = [c["radius"] for c in all_clusters if c["label"] == "addition"]
sub_radii = [c["radius"] for c in all_clusters if c["label"] == "subtraction"]

r_min_add = min(add_radii) if add_radii else 0.0
r_max_add = max(add_radii) if add_radii else 1.0
r_min_sub = min(sub_radii) if sub_radii else 0.0
r_max_sub = max(sub_radii) if sub_radii else 1.0

def color_from_radius(r, r_min, r_max, is_addition=True):
    if r_max == r_min:
        t = 0.0
    else:
        t = (r - r_min) / (r_max - r_min)

    if is_addition:
        # from (0.7,0.9,1.0) => (0,0,0.7)
        start = np.array([1, 0, 1])
        end   = np.array([0, 0, 1])
    else:
        # from (1,1,0) => (1,0,0)
        start = np.array([1, 1, 0])
        end   = np.array([1, 0, 0])

    c = start + t * (end - start)
    return tuple(c)

############################################################################
# 9. Visualization
############################################################################
plotter = pv.Plotter()

# Show the main meshes
# plotter.add_mesh(cad_mesh, color="lightyellow", opacity=0.2, label="CAD")
plotter.add_mesh(scan_mesh, color="lightgrey", opacity=1, label="Scan")

# We'll color "extra" points grey, "missing" points blue (like #3)
plotter.add_points(extra_points, color="black", opacity=0.7, point_size=2, render_points_as_spheres=False)
plotter.add_points(missing_points, color="white", opacity=0.7, point_size=2, render_points_as_spheres=False)

# Draw the bounding spheres with the color gradients
for cluster_data in all_clusters:
    center = cluster_data["center"]
    radius = cluster_data["radius"]
    label  = cluster_data["label"]

    sphere = pv.Sphere(radius=radius, center=center,
                       theta_resolution=30, phi_resolution=30)
    if label == "addition":
        color = color_from_radius(radius, r_min_add, r_max_add, is_addition=True)
    else:  # "subtraction"
        color = color_from_radius(radius, r_min_sub, r_max_sub, is_addition=False)

    plotter.add_mesh(sphere, color=color, opacity=0.5)

plotter.show()

############################################################################
# 10. Export CSV
############################################################################
with open("final_combined_spheres.csv", "w", newline="") as csvfile:
    fieldnames = ["cluster_id", "x", "y", "z", "radius", "label", "shape_sign"]
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
#           "shape_sign": cluster_data["shape_sign"]
        }
        writer.writerow(row)
