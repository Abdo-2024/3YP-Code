import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import csv

############################################################################
# 1. Load & Preprocess
############################################################################
cad_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN_Smooth.stl")
scan_mesh = pv.read("2x2_MN+Noise.stl")

# Compute normals for CAD to help ensure outward-facing normals.
cad_mesh = cad_mesh.compute_normals(auto_orient_normals=True)

# Clean and decimate the scan mesh.
scan_mesh = scan_mesh.clean(tolerance=1e-5)
scan_mesh = scan_mesh.fill_holes(hole_size=5)
scan_mesh = scan_mesh.triangulate()
scan_mesh = scan_mesh.decimate(0.3)

# Decimate the CAD mesh (and be sure it's triangulated).
cad_mesh = cad_mesh.triangulate()
cad_mesh = cad_mesh.decimate(0.3)

############################################################################
# 2. Subsample & Filter Points from the Scan
############################################################################
def random_subsample_points(mesh, max_points=9500):
    pts = mesh.points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        return pts[idx]
    else:
        return pts

scan_points = random_subsample_points(scan_mesh, 9500)

# Build KD-Tree for the CAD so we can remove points that lie on/very near the surface.
cad_points  = cad_mesh.points
cad_tree  = cKDTree(cad_points)

surface_tolerance = 1  # Adjust as needed for “very close”
dist_scan_to_cad, _ = cad_tree.query(scan_points)
# Indices of points that are "on or near" the CAD surface
on_surface_indices = np.where(dist_scan_to_cad < surface_tolerance)[0]

if len(on_surface_indices) > 0:
    print(f"Filtering out {len(on_surface_indices)} scan points near the CAD surface.")
    # Remove these "on-surface" points
    scan_points = np.delete(scan_points, on_surface_indices, axis=0)

############################################################################
# 3. Distance-Based Error Identification & Clustering
############################################################################
# Recompute the KD-Tree with the filtered scan points.
scan_tree = cKDTree(scan_points)

# Compute distances: scan->CAD and CAD->scan
dist_scan_to_cad, _ = cad_tree.query(scan_points)  # Distances for each scan point to nearest CAD point
dist_cad_to_scan, _ = scan_tree.query(cad_points)  # Distances for each CAD point to nearest scan point

threshold = 10  # example threshold for “error”
extra_idx   = np.where(dist_scan_to_cad > threshold)[0]
extra_points = scan_points[extra_idx]

missing_idx   = np.where(dist_cad_to_scan > threshold)[0]
missing_points = cad_points[missing_idx]

def cluster_outliers(points, eps=20, min_samples=10):
    if len(points) == 0:
        return np.array([]), set()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    unique = set(labels)
    return labels, unique

labels_extra, _   = cluster_outliers(extra_points, eps=20, min_samples=10)
labels_missing, _ = cluster_outliers(missing_points, eps=20, min_samples=10)

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
# 4. Fit Spheres & Classify “Addition” vs “Subtraction”
############################################################################
def fit_sphere(points):
    """
    Simple bounding sphere: center = centroid, radius = max distance from center.
    This is not the minimal bounding sphere in a strict sense, but it is quick to compute.
    """
    center = np.mean(points, axis=0)
    diffs = points - center
    r = np.sqrt((diffs**2).sum(axis=1).max())
    return center, r

def classify_cluster_by_inside_test(cluster_points, cad_mesh):
    """
    Use PyVista's select_enclosed_points for a quick inside/outside test.
    If > 50% of the points are inside the CAD, call it 'subtraction',
    else call it 'addition.'
    Also return how many are in vs out for that cluster.
    """
    pd_cluster = pv.PolyData(cluster_points)
    enclosed = cad_mesh.select_enclosed_points(pd_cluster, tolerance=0.0)
    inside_flags = enclosed.point_data["SelectedPoints"]  # 1 if inside, 0 if outside
    
    if len(inside_flags) == 0:
        # If PyVista returns no flags, treat as entirely outside.
        return "addition", 0, len(cluster_points)
    
    inside_count = np.sum(inside_flags) 
    outside_count = len(inside_flags) - inside_count 
    # majority
    label = "subtraction" if inside_count > outside_count else "addition"
    return label, inside_count, outside_count

def process_clusters(cluster_dict, cad_mesh):
    results = []
    for cl_id, pts in cluster_dict.items():
        center, radius = fit_sphere(pts)
        label, nin, nout = classify_cluster_by_inside_test(pts, cad_mesh)
        results.append({
            "cluster_id": cl_id,
            "center": center,
            "radius": radius,
            "label": label,
            "points": pts,
            "points_in": nin,
            "points_out": nout
        })
    return results

extra_info   = process_clusters(extra_clusters, cad_mesh)
missing_info = process_clusters(missing_clusters, cad_mesh)
all_clusters = extra_info + missing_info

############################################################################
# 5. Winding Number Functions (Signed)
############################################################################
def compute_solid_angle(p, triangle):
    """
    Signed solid angle subtended by a triangle (3x3 array) at point p.
    """
    a = triangle[0] - p
    b = triangle[1] - p
    c = triangle[2] - p

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    norm_c = np.linalg.norm(c)

    dot_ab = np.dot(a, b)
    dot_bc = np.dot(b, c)
    dot_ca = np.dot(c, a)

    cross_bc = np.cross(b, c)
    numerator = np.dot(a, cross_bc)  # Signed, do NOT use abs.

    denominator = (norm_a * norm_b * norm_c +
                   dot_ab * norm_c +
                   dot_bc * norm_a +
                   dot_ca * norm_b)

    Omega = 2.0 * np.arctan2(numerator, denominator)
    return Omega

def winding_number_at_point(p, vertices, faces):
    """
    Sum the signed solid angles to get the generalized winding number at p.
    """
    total_angle = 0.0
    for face in faces:
        triangle = vertices[face]
        total_angle += compute_solid_angle(p, triangle)
    return total_angle / (4.0 * np.pi)

def vectorized_winding_numbers_batched(query_points, vertices, faces, batch_size=500):
    """
    Vectorized + batched winding number calculation.
    """
    triangles = vertices[faces]  # shape (F, 3, 3)
    N = query_points.shape[0]
    F = triangles.shape[0]
    winding = np.empty(N, dtype=np.float32)
    
    for i in range(0, N, batch_size):
        batch = query_points[i:i+batch_size]  # shape (B, 3)
        B = batch.shape[0]

        p = batch[:, np.newaxis, np.newaxis, :]  # (B, 1, 1, 3)
        T = triangles[np.newaxis, :, :, :]       # (1, F, 3, 3)
        diff = T - p                             # (B, F, 3, 3)

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
        numerator = np.sum(a * cross_bc, axis=-1)  # Signed

        denominator = (norm_a * norm_b * norm_c +
                       dot_ab * norm_c +
                       dot_bc * norm_a +
                       dot_ca * norm_b)

        Omega = 2.0 * np.arctan2(numerator, denominator)
        total_angle = np.sum(Omega, axis=1)
        winding_batch = total_angle / (4.0 * np.pi)
        winding[i:i+B] = winding_batch

    return winding

############################################################################
# 6. Inside/Outside Classification for Scan Points (Vectorized)
############################################################################
cad_faces = cad_mesh.faces.reshape((-1, 4))[:, 1:4]
threshold_winding = 0.5  # Points with winding > 0.5 -> inside

print("Computing winding numbers for remaining scan points (vectorized, batched)...")
winding_vals = vectorized_winding_numbers_batched(scan_points, cad_mesh.points, cad_faces, batch_size=500)

inside_scan = scan_points[winding_vals >= threshold_winding]
outside_scan = scan_points[winding_vals < threshold_winding]

############################################################################
# 7. Visualization
############################################################################
plotter = pv.Plotter()

# Show the CAD mesh as a (semi-)transparent surface or wireframe.
plotter.add_mesh(cad_mesh, color="white", opacity=0.6, label="CAD Mesh")

# Show "extra" error points in white, if desired.
if len(extra_points) > 0:
    plotter.add_points(extra_points, color="black", point_size=5, render_points_as_spheres=True, label="Extra Points")

# Visualize inside vs. outside scan points.
if inside_scan.size:
    plotter.add_points(
        inside_scan, 
        color="blue", 
        point_size=5, 
        render_points_as_spheres=True, 
        label="Inside Scan Points"
    )
if outside_scan.size:
    plotter.add_points(
        outside_scan, 
        color="red", 
        point_size=5, 
        render_points_as_spheres=True, 
        label="Outside Scan Points"
    )

# Draw bounding spheres for each cluster using different colors for each label.
for cluster_data in all_clusters:
    center = cluster_data["center"]
    radius = cluster_data["radius"]
    label  = cluster_data["label"]
    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=15, phi_resolution=15)
    color = "red" if label == "addition" else "blue" 
    plotter.add_mesh(sphere, color=color, opacity=0.3)

plotter.add_legend()
plotter.show()

############################################################################
# 8. Export Cluster Information (Including #in / #out) to CSV
############################################################################
with open("final_combined_spheres.csv", "w", newline="") as csvfile:
    fieldnames = [
        "cluster_id", 
        "x", "y", "z", 
        "radius", 
        "label", 
        "points_in", 
        "points_out"
    ]
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
            "label": cluster_data["label"],
            "points_in": cluster_data["points_in"],
            "points_out": cluster_data["points_out"]
        })
