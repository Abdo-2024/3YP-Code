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

# Compute normals for CAD. Using auto_orient_normals=True helps ensure outward-facing normals.
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
# 2. Subsample Points from the Scan
############################################################################
def random_subsample_points(mesh, max_points=15000):
    pts = mesh.points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        return pts[idx]
    else:
        return pts

scan_points  = random_subsample_points(scan_mesh, 15000)

############################################################################
# 3. (Existing) Error Identification via KD-Tree & DBSCAN Clustering
############################################################################
cad_points  = cad_mesh.points
cad_tree  = cKDTree(cad_points)
scan_tree = cKDTree(scan_points)

dist_scan_to_cad, _ = cad_tree.query(scan_points)
dist_cad_to_scan, _ = scan_tree.query(cad_points)

threshold = 5.0  # example threshold
extra_idx   = np.where(dist_scan_to_cad > threshold)[0]
extra_points = scan_points[extra_idx]

missing_idx   = np.where(dist_cad_to_scan > threshold)[0]
missing_points = cad_points[missing_idx]

def cluster_outliers(points, eps=15, min_samples=15):
    if len(points) == 0:
        return np.array([]), set()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    unique = set(labels)
    return labels, unique

labels_extra, _   = cluster_outliers(extra_points, eps=15, min_samples=15)
labels_missing, _ = cluster_outliers(missing_points, eps=15, min_samples=15)

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

def fit_sphere(points):
    center = np.mean(points, axis=0)
    diffs = points - center
    r = np.sqrt((diffs**2).sum(axis=1).max())
    return center, r

def classify_cluster_by_inside_test(cluster_points, cad_mesh):
    """
    Use PyVista's select_enclosed_points for a quick inside/outside test.
    """
    pd_cluster = pv.PolyData(cluster_points)
    enclosed = cad_mesh.select_enclosed_points(pd_cluster, tolerance=0.0)
    inside_flags = enclosed.point_data["SelectedPoints"]  # 1 if inside, 0 if outside
    if len(inside_flags) == 0:
        return "addition"
    if (np.sum(inside_flags) / len(inside_flags)) > 0.5:
        return "subtraction"
    else:
        return "addition"

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
# 4. Correct Solid Angle & Winding Number Functions
############################################################################

def compute_solid_angle(p, triangle):
    """
    Compute the signed solid angle subtended by a triangle (3x3 array) at point p.
    IMPORTANT: Do NOT take absolute value of the cross product if you need a signed angle.
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

    # Signed cross product
    cross_bc = np.cross(b, c)
    numerator = np.dot(a, cross_bc)  # <-- Removed np.abs(...)

    denominator = (norm_a * norm_b * norm_c +
                   dot_ab * norm_c +
                   dot_bc * norm_a +
                   dot_ca * norm_b)

    Omega = 2.0 * np.arctan2(numerator, denominator)
    return Omega

def winding_number_at_point(p, vertices, faces):
    """
    Compute the generalized winding number at point p by summing signed solid angles.
    """
    total_angle = 0.0
    for face in faces:
        triangle = vertices[face]
        Omega = compute_solid_angle(p, triangle)
        total_angle += Omega
    return total_angle / (4.0 * np.pi)

def vectorized_winding_numbers_batched(query_points, vertices, faces, batch_size=1000):
    """
    Vectorized + batched generalized winding number computation.
    Removes the absolute value for a proper signed angle.
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
        # Again, for a signed angle, do NOT take absolute value here:
        numerator = np.sum(a * cross_bc, axis=-1)
        
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
# 5. Inside/Outside Classification for Scan Points (Vectorized)
############################################################################
# Make sure the CAD is triangulated; faces are stored as [nVerts, i0, i1, i2, ...].
cad_faces = cad_mesh.faces.reshape((-1, 4))[:, 1:4]
threshold_winding = 0.5  # or a stricter cutoff like 0.99, depending on your data

print("Computing winding numbers for scan points (vectorized, batched)...")
winding_vals = vectorized_winding_numbers_batched(scan_points, cad_mesh.points, cad_faces, batch_size=500)

inside_scan = scan_points[winding_vals > threshold_winding]
outside_scan = scan_points[winding_vals <= threshold_winding]

############################################################################
# 6. Visualization
############################################################################
plotter = pv.Plotter()

# Add the CAD mesh (wireframe) for context.
plotter.add_mesh(cad_mesh, color="lightgreen", opacity=0.7, label="CAD Mesh")

# Show the extra error points in white.
plotter.add_points(extra_points, color="white", opacity=1, point_size=3, render_points_as_spheres=True)

# Visualize inside and outside scan points 
if inside_scan.size:
    plotter.add_points(
        inside_scan, 
        color="blue", 
        opacity=1, 
        point_size=5, 
        render_points_as_spheres=True, 
        label="Inside Scan Points"
    )
if outside_scan.size:
    plotter.add_points(
        outside_scan, 
        color="red", 
        opacity=1, 
        point_size=5, 
        render_points_as_spheres=True, 
        label="Outside Scan Points"
    )

plotter.add_legend()
plotter.show()

############################################################################
# 7. Export Cluster Information to CSV
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
