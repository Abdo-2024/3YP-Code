import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import csv

# ---------------------------------------------------------------------------
# 1. Load STL
# ---------------------------------------------------------------------------
cad_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN_Smooth.stl")
scan_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN+Noise.stl")

# ---------------------------------------------------------------------------
# 2. Clean & Decimate
# ---------------------------------------------------------------------------
scan_mesh = scan_mesh.clean(tolerance=1e-5)
scan_mesh = scan_mesh.fill_holes(hole_size=10)
scan_mesh = scan_mesh.triangulate()

cad_mesh = cad_mesh.decimate(0.8)   # reduce triangles by 80% (example)
scan_mesh = scan_mesh.decimate(0.8)

# ---------------------------------------------------------------------------
# 3. Sub-sample if needed
# ---------------------------------------------------------------------------
def random_subsample_points(mesh, max_points=20000):
    pts = mesh.points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        return pts[idx]
    else:
        return pts

cad_points = random_subsample_points(cad_mesh, 20000)
scan_points = random_subsample_points(scan_mesh, 20000)

# ---------------------------------------------------------------------------
# 4. KD-tree & Distances
# ---------------------------------------------------------------------------
cad_tree = cKDTree(cad_points)
scan_tree = cKDTree(scan_points)

dist_scan_to_cad, idx_scan = cad_tree.query(scan_points)
dist_cad_to_scan, idx_cad  = scan_tree.query(cad_points)

# ---------------------------------------------------------------------------
# 5. Identify outliers (Additions vs. Subtractions)
#    Tweak threshold to match typical deviations for your part
# ---------------------------------------------------------------------------
threshold = 11 # in mm (example; set appropriately)

extra_idx = np.where(dist_scan_to_cad > threshold)[0]   # "additions" from scan
missing_idx = np.where(dist_cad_to_scan > threshold)[0] # "subtractions" from CAD

extra_points = scan_points[extra_idx]
missing_points = cad_points[missing_idx]

# ---------------------------------------------------------------------------
# 6. Cluster each set of outliers
# ---------------------------------------------------------------------------
def cluster_outliers(points, eps=2.0, min_samples=10):
    """
    Cluster a set of 3D points using DBSCAN.
    eps = the max radius for neighbors
    min_samples = how many neighbors to form a cluster
    """
    if len(points) == 0:
        return np.array([]), set()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    unique_labels = set(labels)
    return labels, unique_labels

labels_extra, unique_extra = cluster_outliers(extra_points, eps=15, min_samples=20)
labels_missing, unique_missing = cluster_outliers(missing_points, eps=15, min_samples=20)

# ---------------------------------------------------------------------------
# 7. Optionally "fit a sphere" for each cluster
# ---------------------------------------------------------------------------
def fit_sphere_to_points(pts):
    """
    Simple sphere fit: center = average of points,
    radius = max distance from center (encloses all points).
    This is a naive bounding sphere, not minimal bounding sphere.
    """
    center = np.mean(pts, axis=0)
    # radius = maximum distance from center
    diffs = pts - center
    dist_sq = np.einsum('ij,ij->i', diffs, diffs)
    radius = np.sqrt(dist_sq.max())
    return center, radius

# We'll gather sphere data in a list of dicts
sphere_data = []

# A toggle to enable/disable drawing spheres in PyVista
DRAW_FITTED_SPHERES = True

# For coloring by radius, we’ll create two separate gradients:
#  - Additions: from Yellow (small) to Red (large)
#  - Subtractions: from LightBlue (small) to DarkBlue (large)
def color_from_radius(r, r_min, r_max, is_addition=True):
    """
    Returns an (R,G,B) color by interpolating between two colors
    based on the sphere's radius r. 
    For additions: (yellow -> red).
    For subtractions: (lightblue -> darkblue).
    """
    if r_max == r_min:
        t = 0.0
    else:
        t = (r - r_min) / (r_max - r_min)  # fraction in [0,1]

    if is_addition:
        # from (1,1,0) [yellow] to (1,0,0) [red]
        # color = start + t*(end - start)
        start = np.array([1.0, 1.0, 0.0])
        end   = np.array([1.0, 0.0, 0.0])
    else:
        # from (0.7,0.9,1.0) [lightblue] to (0.0,0.0,0.7) [darker blue]
        start = np.array([0.7, 0.9, 1.0])
        end   = np.array([0.0, 0.0, 0.7])

    c = start + t*(end - start)
    return tuple(c)

# Compute spheres for "extra" clusters (additions)
extra_spheres = []
for cluster_label in unique_extra:
    if cluster_label == -1:
        # DBSCAN label -1 => "noise"
        continue
    pts_in_cluster = extra_points[labels_extra == cluster_label]
    center, radius = fit_sphere_to_points(pts_in_cluster)
    extra_spheres.append((center, radius))

# Compute spheres for "missing" clusters (subtractions)
missing_spheres = []
for cluster_label in unique_missing:
    if cluster_label == -1:
        continue
    pts_in_cluster = missing_points[labels_missing == cluster_label]
    center, radius = fit_sphere_to_points(pts_in_cluster)
    missing_spheres.append((center, radius))

# We'll map each sphere's radius onto a color gradient.
# 1) Find min & max radius in 'extra_spheres'
# 2) Same for 'missing_spheres'
radii_extra   = [s[1] for s in extra_spheres]   # second element is radius
radii_missing = [s[1] for s in missing_spheres]

r_min_extra = min(radii_extra) if len(radii_extra) else 0.0
r_max_extra = max(radii_extra) if len(radii_extra) else 1.0

r_min_missing = min(radii_missing) if len(radii_missing) else 0.0
r_max_missing = max(radii_missing) if len(radii_missing) else 1.0

# ---------------------------------------------------------------------------
# 8. PyVista Visualization
# ---------------------------------------------------------------------------
plotter = pv.Plotter()

# Show the main meshes (or one of them) in the scene
# plotter.add_mesh(cad_mesh, color="lightyellow", opacity=0.1)  # CAD reference
plotter.add_mesh(scan_mesh, color="lightblue", opacity=0.8)  # scanned

# Optionally still show outlier points (just to see them)
plotter.add_points(extra_points, color="grey", opacity=1, point_size=3, render_points_as_spheres=True)
plotter.add_points(missing_points, color="blue", opacity=1, point_size=3, render_points_as_spheres=True)

# Draw the fitted spheres
if DRAW_FITTED_SPHERES:
    for (center, radius) in extra_spheres:
        sphere = pv.Sphere(radius=radius, center=center, theta_resolution=30, phi_resolution=30)
        color = color_from_radius(radius, r_min_extra, r_max_extra, is_addition=True)
        plotter.add_mesh(sphere, color=color, opacity=0.7)

    for (center, radius) in missing_spheres:
        sphere = pv.Sphere(radius=radius, center=center, theta_resolution=30, phi_resolution=30)
        color = color_from_radius(radius, r_min_missing, r_max_missing, is_addition=False)
        plotter.add_mesh(sphere, color=color, opacity=0.7)

plotter.show()

# ---------------------------------------------------------------------------
# 9. Export to CSV
#    We'll save each fitted sphere to a row with:
#       x, y, z, radius, type => "addition" or "subtraction"
# ---------------------------------------------------------------------------
all_spheres = []
for (center, radius) in extra_spheres:
    all_spheres.append({
        "x": center[0],
        "y": center[1],
        "z": center[2],
        "radius": radius,
        "type": "addition"
    })

for (center, radius) in missing_spheres:
    all_spheres.append({
        "x": center[0],
        "y": center[1],
        "z": center[2],
        "radius": radius,
        "type": "subtraction"
    })

with open("cluster_spheres.csv", "w", newline="") as csvfile:
    fieldnames = ["x", "y", "z", "radius", "type"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_spheres:
        writer.writerow(row)
