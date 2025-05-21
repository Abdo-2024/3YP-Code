import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import csv
import time 
start_time = time.time()
"""
This script loads a CAD and a noisy scan mesh, preprocesses and decimates them, and subsamples scan points. It removes points near the CAD surface, then identifies “extra” and “missing” error points by distance thresholds. DBSCAN clusters these outliers, and a fully vectorised winding number computation classifies cluster points as inside or outside the CAD volume. Each cluster is labeled (addition, subtraction, tie) based on majority classification, and minimal bounding spheres are fitted. The scan mesh with error points and spheres is visualised in PyVista, and cluster summaries (centres, radii, labels, point counts, in/out ratios) are exported to CSV, with execution time reported.
"""

############################################################################
# 1. Load & Preprocess
############################################################################
cad_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN.stl")
scan_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.obj")

cad_mesh = cad_mesh.compute_normals(auto_orient_normals=True)

scan_mesh = scan_mesh.clean(tolerance=1e-10)
scan_mesh = scan_mesh.fill_holes(hole_size=5)
scan_mesh = scan_mesh.triangulate()
scan_mesh = scan_mesh.decimate(0.8)

# Further decimate the CAD mesh to reduce face count => speeds up winding calculations
cad_mesh = cad_mesh.triangulate()
cad_mesh = cad_mesh.decimate(0.8)  # experiment with 0.5, 0.7, Bigger = More SPEEEEEED but lower accuracy etc.

############################################################################
# 2. Subsample points from the scan
############################################################################
def random_subsample_points(mesh, max_points=20000):
    pts = mesh.points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        return pts[idx]
    return pts

scan_points = random_subsample_points(scan_mesh, 20000)
cad_points  = cad_mesh.points

############################################################################
# 3. Remove points near/on the CAD surface
############################################################################
surface_tolerance = 0.1
cad_tree = cKDTree(cad_points)
dists, _ = cad_tree.query(scan_points)
on_surface_idx = np.where(dists < surface_tolerance)[0]
if len(on_surface_idx) > 0:
    print(f"Removing {len(on_surface_idx)} points near the CAD surface...")
    scan_points = np.delete(scan_points, on_surface_idx, axis=0)

############################################################################
# 4. Distance-based identification of "extra" vs "missing" errors
############################################################################
scan_tree = cKDTree(scan_points)
dist_scan_to_cad, _ = cad_tree.query(scan_points)
dist_cad_to_scan, _ = scan_tree.query(cad_points)

threshold = 30
extra_idx   = np.where(dist_scan_to_cad > threshold)[0]
extra_points = scan_points[extra_idx]

missing_idx   = np.where(dist_cad_to_scan > threshold)[0]
missing_points = cad_points[missing_idx]

def cluster_outliers(points, eps=5, min_samples=15):
    if len(points) == 0:
        return np.array([]), set()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    unique = set(labels)
    return labels, unique

labels_extra, _   = cluster_outliers(extra_points, eps=7, min_samples=15)
labels_missing, _ = cluster_outliers(missing_points, eps=7, min_samples=15)

def gather_clusters(points, labels):
    clusters = {}
    for cl in set(labels):
        if cl == -1:
            continue
        mask = (labels == cl)
        clusters[cl] = points[mask]
    return clusters

extra_clusters   = gather_clusters(extra_points, labels_extra)
missing_clusters = gather_clusters(missing_points, labels_missing)

############################################################################
# 5. Vectorized winding number (fully batched)
############################################################################
cad_faces = cad_mesh.faces.reshape((-1, 4))[:, 1:4]
triangles = cad_points[cad_faces]  # shape: (F, 3, 3) => each face is a 3D triangle

def winding_numbers(points, triangles, batch_size=2500):
    """
    *Fully* vectorized + batched computation for all 'points'.
    Returns array of shape (len(points),).
    """
    # We'll expand in memory-friendly chunks
    N = points.shape[0]
    F = triangles.shape[0]
    w = np.empty(N, dtype=np.float32)

    for start in range(0, N, batch_size):
        end = start + batch_size
        batch = points[start:end]  # shape (B, 3)
        B = len(batch)

        # shape (1, F, 3, 3) minus shape (B, 1, 1, 3) => (B, F, 3, 3)
        T = triangles[np.newaxis, :, :, :]
        P = batch[:, np.newaxis, np.newaxis, :]
        diff = T - P

        a = diff[:, :, 0, :]
        b = diff[:, :, 1, :]
        c = diff[:, :, 2, :]

        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)
        norm_c = np.linalg.norm(c, axis=-1)

        dot_ab = np.sum(a * b, axis=-1)
        dot_bc = np.sum(b * c, axis=-1)
        dot_ca = np.sum(c * a, axis=-1)

        cross_bc = np.cross(b, c)
        numerator = np.sum(a * cross_bc, axis=-1)  # signed

        denominator = (norm_a * norm_b * norm_c +
                       dot_ab * norm_c +
                       dot_bc * norm_a +
                       dot_ca * norm_b)

        # shape (B, F)
        Omega = 2.0 * np.arctan2(numerator, denominator)
        total_angle = np.sum(Omega, axis=1)
        w_batch = total_angle / (4.0 * np.pi)
        w[start:end] = w_batch

    return w

############################################################################
# 6. Single-pass classification for ALL cluster points
############################################################################
# Merge all cluster points into one array, but keep track of which cluster they belong to.
# We do this for extra and missing in one pass, or you can do them separately if you prefer.

# (a) Gather everything into a single list
all_cluster_points = []
all_cluster_ids    = []
all_labels_type    = []  # e.g. "extra" or "missing"
cluster_map = {}         # (type, cluster_id) -> index range in the global array

def flatten_clusters(cluster_dict, type_str):
    """
    Convert {cluster_id: points_array} into appended arrays
    and keep track of mapping for unscrambling later.
    """
    start_index = len(all_cluster_points)
    for cid, arr in cluster_dict.items():
        all_cluster_points.append(arr)
        # Fill cluster IDs
        all_cluster_ids.extend([cid]*len(arr))
        # Fill the label type
        all_labels_type.extend([type_str]*len(arr))
    end_index = len(all_cluster_points)
    return start_index, end_index

start_extra, end_extra   = flatten_clusters(extra_clusters, "extra")
start_missing, end_missing = flatten_clusters(missing_clusters, "missing")

# Combine everything into one big Nx3 array
if len(all_cluster_points) == 0:
    # No clusters at all
    all_points_array = np.empty((0, 3), dtype=np.float32)
else:
    all_points_array = np.vstack(all_cluster_points)

all_cluster_ids = np.array(all_cluster_ids, dtype=int)   # which cluster each point belongs to
all_labels_type = np.array(all_labels_type, dtype=object)  # "extra"/"missing"

print(f"Total cluster points: {len(all_points_array)}")

# (b) Compute winding in one pass
if len(all_points_array) > 0:
    w_values = winding_numbers(all_points_array, triangles, batch_size=2500)
else:
    w_values = np.empty(0, dtype=np.float32)

############################################################################
inside_threshold = 0.99  # I have ## around where this number has an impact (analyze_cluster)
############################################################################

# (c) Assign each cluster's points as inside or outside
# We'll build results in new data structures
cluster_results_extra   = {}
cluster_results_missing = {}

def analyze_clusters(cluster_dict, type_str):
    """
    For each cluster in 'cluster_dict', figure out which subset of all_points_array belongs to it,
    then find how many are inside vs outside, and how we label that cluster.
    """
    results = {}
    for cid, points_array in cluster_dict.items():
        # Indices in all_points_array that belong to (type_str, cid)
        mask = (all_labels_type == type_str) & (all_cluster_ids == cid)
        cluster_w = w_values[mask]
        
        # If no points, skip
        if mask.sum() == 0:
            results[cid] = {
                "inside_points": 0,
                "outside_points": 0,
                "label": "empty",
                "ratio_in": 0.0,
            }
            continue
############################################################################
        inside_mask = (cluster_w >= inside_threshold)
        nin  = np.count_nonzero(inside_mask)
        nout = mask.sum() - nin
############################################################################
        if nin > nout:
            label = "subtraction"
        elif nout > nin:
            label = "addition"
        else:
            label = "tie"
        ratio_in = nin / float(nin + nout) if (nin + nout) > 0 else 0
        
        results[cid] = {
            "inside_points": nin,
            "outside_points": nout,
            "label": label,
            "ratio_in": ratio_in,
        }
    return results

results_extra   = analyze_clusters(extra_clusters,   "extra")
results_missing = analyze_clusters(missing_clusters, "missing")

############################################################################
# 7. Fit spheres to inside or outside subset ONLY
############################################################################
def fit_sphere(points):
    if len(points) == 0:
        return np.array([0,0,0]), 0.0
    center = np.mean(points, axis=0)
    diffs = points - center
    r = np.sqrt((diffs**2).sum(axis=1).max())
    return center, r

def build_sphere_info(cluster_dict, cluster_analysis, type_str):
    """
    cluster_analysis has inside/outside counts and final label.
    We'll pick the actual points from the global array 
    to fit the sphere only to the relevant subset.
    """
    cluster_info = []
    for cid, pts in cluster_dict.items():
        info = cluster_analysis[cid]
        label = info["label"]
        # gather the relevant subset of points from the big array
        mask = (all_labels_type == type_str) & (all_cluster_ids == cid)

        if label == "Missing":
            # fit sphere to only inside points
            inside_mask = (w_values[mask] >= inside_threshold)
            subset_pts = pts[inside_mask]  # or the sub-index from the big array
        elif label == "Extra":
            # fit sphere to only outside points
            inside_mask = (w_values[mask] >= inside_threshold)
            subset_pts = pts[~inside_mask]
        else:
            # tie or empty => just fit to entire cluster
            subset_pts = pts

        center, radius = fit_sphere(subset_pts)
        out = {
            "cluster_id": cid,
            "label": label,
            "center": center,
            "radius": radius,
            "points_in": info["inside_points"],
            "points_out": info["outside_points"],
            "ratio_in": info["ratio_in"],
        }
        cluster_info.append(out)
    return cluster_info

extra_info   = build_sphere_info(extra_clusters,   results_extra,   "Extra")
missing_info = build_sphere_info(missing_clusters, results_missing, "Missing")
all_clusters = extra_info + missing_info
"""
############################################################################
# 8. Visualization
############################################################################
plotter = pv.Plotter()
# plotter.add_mesh(cad_mesh, color="white", opacity=0.5, label="CAD Mesh")
plotter.add_mesh(scan_mesh, color="lightgray", opacity=1)


# Visualize inside vs. outside scan points.

plotter.add_points(extra_points, color="black", point_size=2, render_points_as_spheres=True) 
# plotter.add_points(missing_points, color="blue", point_size=5, render_points_as_spheres=True, label="Inside Scan Points")

# Uncomment to view the following
# Plot the "extra" points 
#if extra_points.size:
#    plotter.add_points(
#        extra_points,
#        color="red",
#        point_size=5,
#        render_points_as_spheres=True,
#        label="Extra Points"
#    )

# (Plot the missing_points)
#plotter.add_points(
#    missing_points,
#    color="blue",
#    point_size=5,
#    render_points_as_spheres=True,
#    label="Missing Points"
#)

# Draw bounding spheres for each cluster in different colors and add labels
for cdata in all_clusters:
    c = cdata["center"]
    r = cdata["radius"]
    lbl = cdata["label"]
    sphere = pv.Sphere(radius=r, center=c, theta_resolution=15, phi_resolution=15)
    if lbl == "addition":
        color = "red"
        sphere_label = "Extra 'Blobs'"
    elif lbl == "subtraction":
        color = "blue"
        sphere_label = "Missing 'Blobs'"
    elif lbl == "tie":
        color = "green"  # Ties are now drawn in green
        sphere_label = "Not sure 'Blobs'"
    else:
        color = "green"
        sphere_label = "Undefined"
        
    # Add the sphere with a legend label just add ,label=sphere_label
    plotter.add_mesh(sphere, color=color, opacity=0.3)
    
# Optionally, add a text label at the center of the sphere
#    plotter.add_point_labels(np.array([c]), [sphere_label], font_size=10, text_color='black')

# Define a custom legend with label-color pairs
legend_entries = [
    ("Scan", "lightgray"),
    ("Scan Points", "black"),
    ("Extra 'Blobs'", "red"),
    ("Missing 'Blobs'", "blue"),
    ("Not sure 'Blobs'", "green")
]

# Add the custom legend (key) to the top right corner with a white background
plotter.add_legend(labels=legend_entries, bcolor="white", loc="upper right")
plotter.show()

"""
############################################################################
# 9. Export to CSV
############################################################################
with open("clusters_with_in_out_ratio.csv", "w", newline="") as csvfile:
    fieldnames = [
        "cluster_id",
        "x","y","z",
        "radius",
        "label",
        "points_in",
        "points_out",
        "in_out_ratio"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_clusters:
        writer.writerow({
            "cluster_id": row["cluster_id"],
            "x": row["center"][0],
            "y": row["center"][1],
            "z": row["center"][2],
            "radius": row["radius"],
            "label": row["label"],
            "points_in": row["points_in"],
            "points_out": row["points_out"],
            "in_out_ratio": row["ratio_in"]
        })
        
end_time = time.time()
print("Execution time: {:.4f} seconds".format(end_time - start_time))
