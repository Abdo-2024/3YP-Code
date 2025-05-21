import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import csv

############################################################################
# 1. Load & Preprocess
############################################################################
cad_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN_Smooth.stl")
scan_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN+Noise.stl")

# Clean and decimate the scan
scan_mesh = scan_mesh.clean(tolerance=1e-5)
scan_mesh = scan_mesh.fill_holes(hole_size=5)
scan_mesh = scan_mesh.triangulate()
scan_mesh = scan_mesh.decimate(0.8)

# Decimate the CAD as well
cad_mesh = cad_mesh.decimate(0.8)

# Optional: if you want the CAD watertight for other checks:
# cad_mesh = cad_mesh.clean()
# cad_mesh = cad_mesh.fill_holes(hole_size=5)
# cad_mesh = cad_mesh.triangulate()

############################################################################
# 2. Subsample
############################################################################
def random_subsample_points(mesh, max_points=20000):
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

dist_scan_to_cad, idx_scan = cad_tree.query(scan_points)  # scan->cad
dist_cad_to_scan, idx_cad  = scan_tree.query(cad_points)  # cad->scan

threshold = 5.0  # mm scale, tweak as needed

# Potential "extras" (bulges) are scan points far from CAD
extra_idx = np.where(dist_scan_to_cad > threshold)[0]
extra_points = scan_points[extra_idx]

# Potential "missing" (cavities) are cad points far from scan
missing_idx = np.where(dist_cad_to_scan > threshold)[0]
missing_points = cad_points[missing_idx]

############################################################################
# 4. DBSCAN
############################################################################
def cluster_outliers(points, eps=5.0, min_samples=20):
    if len(points) == 0:
        return np.array([]), set()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    unique = set(labels)
    return labels, unique

# You can adjust eps for your geometry
labels_extra, unique_extra = cluster_outliers(extra_points, eps=15, min_samples=20)
labels_missing, unique_missing = cluster_outliers(missing_points, eps=15, min_samples=20)

############################################################################
# 5. Prepare clusters
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
# 6. Local Paraboloid Fit
############################################################################
def fit_paraboloid_3d(points):
    """
    1) Compute PCA => define local coords (u,v,w)
    2) Fit w = a*u^2 + b*v^2 + c*u*v + d*u + e*v + f
    3) Return: (a,b,c,d,e,f), local transform for future analysis
    """
    # Step A: PCA to find local axes
    # We'll center points at the mean
    mean_pt = points.mean(axis=0)
    centered = points - mean_pt

    pca = PCA(n_components=3)
    pca.fit(centered)
    # Columns of pca.components_ are the principal axes in descending variance
    # We'll define new coords = centered.dot(axes.T)
    # This transforms from global (X,Y,Z) to local (u,v,w).
    # Let's define localZ as the axis with the *smallest* variance => pca.components_[2]
    # so that the shape is "spread" mostly in the u,v plane.
    localX = pca.components_[0]
    localY = pca.components_[1]
    localZ = pca.components_[2]

    # Transform all points
    T = np.vstack([localX, localY, localZ])  # shape (3,3)
    uvw = centered.dot(T.T)                  # shape (n,3)

    u = uvw[:,0]
    v = uvw[:,1]
    w = uvw[:,2]

    # Step B: Fit w(u,v) = a*u^2 + b*v^2 + c*u*v + d*u + e*v + f
    # We can build a design matrix M:
    # each row = [u^2, v^2, u*v, u, v, 1]
    # so that M * [a,b,c,d,e,f]^T = w
    M = np.column_stack((u**2, v**2, u*v, u, v, np.ones_like(u)))
    # Solve least squares: params = (M^T M)^(-1) M^T w
    params, _, _, _ = np.linalg.lstsq(M, w, rcond=None)
    # params = [a,b,c,d,e,f]

    return params, (mean_pt, T)

def paraboloid_curvature_sign(params):
    """
    From w(u,v) = a*u^2 + b*v^2 + c*u*v + ...
    Hessian = [[2a, c],
               [ c, 2b]]
    We'll check eigenvalues: if both > 0 => "positive definite" => bowl
                            if both < 0 => "negative definite" => dome
                            else => saddle/flat
    Returns +1 for bowl, -1 for dome, 0 for saddle/flat
    """
    a, b, c, d, e, f = params
    H = np.array([[2*a, c],
                  [  c, 2*b]])
    eigvals = np.linalg.eigvals(H)
    if np.all(eigvals > 0):
        return +1  # concave up (bowl)
    elif np.all(eigvals < 0):
        return -1  # concave down (dome)
    else:
        return 0    # saddle or ambiguous

############################################################################
# 7. Classify Clusters (Using Paraboloid Shape)
############################################################################
# We’ll store sphere or bounding data for each cluster, plus an "addition"/"subtraction" label.

def fit_sphere(points):
    center = np.mean(points, axis=0)
    diffs = points - center
    r = np.sqrt((diffs**2).sum(axis=1).max())
    return center, r

cluster_results = []  # each entry => {center, radius, shape, label, cluster_id, points}

def process_clusters(cluster_dict, default_label):
    """
    cluster_dict: dict {cluster_id -> Nx3 points}
    default_label: "addition" or "subtraction", used only if shape is ambiguous
    """
    out = []
    for cl_id, pts in cluster_dict.items():
        # Fit sphere for bounding
        center, radius = fit_sphere(pts)
        # Fit paraboloid
        params, (mean_pt, T) = fit_paraboloid_3d(pts)
        shape_sign = paraboloid_curvature_sign(params)

        # By default, let's define:
        #  shape_sign = +1 => "bowl" => we interpret as "subtraction"
        #  shape_sign = -1 => "dome" => "addition"
        #  shape_sign = 0 => ambiguous => fallback to default_label
        if shape_sign == +1:
            shape_label = "subtraction"
        elif shape_sign == -1:
            shape_label = "addition"
        else:
            shape_label = default_label

        out.append({
            "cluster_id": cl_id,
            "center": center,
            "radius": radius,
            "shape_sign": shape_sign,
            "label": shape_label,
            "points": pts
        })
    return out

# Process "extra" clusters (initial guess => addition)
extra_info = process_clusters(extra_clusters, default_label="addition")
# Process "missing" clusters (initial guess => subtraction)
missing_info = process_clusters(missing_clusters, default_label="subtraction")

cluster_results = extra_info + missing_info

############################################################################
# 8. Visualization
############################################################################
plotter = pv.Plotter()

# Add main meshes
# plotter.add_mesh(cad_mesh, color="lightyellow", opacity=0.3)
plotter.add_mesh(scan_mesh, color="lightblue", opacity=0.8)

# Show outlier points (just to see them)
plotter.add_points(extra_points, color="grey", opacity=1, point_size=3, render_points_as_spheres=True)
plotter.add_points(missing_points, color="blue", opacity=1, point_size=3, render_points_as_spheres=True)

# For each cluster, draw the bounding sphere in a color
# based on shape_label.
def color_by_label(label):
    if label == "addition":
        return (1, 0, 0)  # red
    elif label == "subtraction":
        return (0, 0, 1)  # blue
    else:
        return (0.5, 0.5, 0.5)

for cr in cluster_results:
    c = cr["center"]
    r = cr["radius"]
    label = cr["label"]
    sph = pv.Sphere(radius=r, center=c, theta_resolution=24, phi_resolution=24)
    plotter.add_mesh(sph, color=color_by_label(label), opacity=0.5)

plotter.show()

############################################################################
# 9. Export CSV
############################################################################
with open("cluster_spheres_paraboloid.csv", "w", newline="") as csvfile:
    fieldnames = ["cluster_id", "x", "y", "z", "radius", "label", "shape_sign"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for cr in cluster_results:
        row = {
            "cluster_id": cr["cluster_id"],
            "x": cr["center"][0],
            "y": cr["center"][1],
            "z": cr["center"][2],
            "radius": cr["radius"],
            "label": cr["label"],
            "shape_sign": cr["shape_sign"]
        }
        writer.writerow(row)
