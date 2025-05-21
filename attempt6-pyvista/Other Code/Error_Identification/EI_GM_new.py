import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

############################################################################
# 1. Load & Convert the CAD Mesh into a Point Cloud
############################################################################

cad_mesh = pv.read("2x2_MN_Smooth.stl")

# (Optional) Triangulate, clean, fill holes. 
# Even though we won't rely on the built-in normals, 
# cleaning can remove duplicate points, etc.
cad_mesh = cad_mesh.triangulate().clean()
cad_mesh = cad_mesh.fill_holes(1000)
cad_mesh = cad_mesh.clean()

# Extract the CAD points (our "point cloud")
cad_points = cad_mesh.points

############################################################################
# 2. Compute Normals for the CAD Point Cloud (Nearest-Neighbor Cross-Product)
############################################################################

# Build a KD-tree on the CAD points
cad_tree = cKDTree(cad_points)

# We'll store the new "CAD normals" here
cad_normals = np.zeros_like(cad_points)

for i in range(len(cad_points)):
    # Query the 3 nearest points (including the point itself)
    dists, idxs = cad_tree.query(cad_points[i], k=3)
    
    # If we don't have at least 3 distinct points, skip
    if len(idxs) < 3:
        continue
    
    p0 = cad_points[i]
    p1 = cad_points[idxs[1]]
    p2 = cad_points[idxs[2]]
    
    # Cross product of two vectors from p0
    v1 = p1 - p0
    v2 = p2 - p0
    n = np.cross(v1, v2)
    
    # Normalize if not negligible
    length = np.linalg.norm(n)
    if length > 1e-12:
        n /= length
    
    cad_normals[i] = n

# Create a PolyData for the CAD point cloud and attach these computed normals
cad_pdata = pv.PolyData(cad_points)
cad_pdata["Normals"] = cad_normals

############################################################################
# 3. Load & Preprocess the Scan Mesh
############################################################################

scan_mesh = pv.read("2x2_MN+Noise.stl")
scan_mesh = scan_mesh.clean(tolerance=1e-7)
scan_mesh = scan_mesh.fill_holes(hole_size=20)
scan_mesh = scan_mesh.triangulate().clean()
scan_mesh = scan_mesh.decimate(0.3)  # optional for performance

def random_subsample_points(mesh, max_points=40000):
    pts = mesh.points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        return pts[idx]
    else:
        return pts

scan_points = random_subsample_points(scan_mesh, 40000)

############################################################################
# 4. Identify "Extra" Error Points from the Scan
############################################################################

scan_tree = cKDTree(scan_points)

# Distances from scan->CAD (using the same CAD points from cad_mesh)
dist_scan_to_cad, _ = cad_tree.query(scan_points)
threshold = 5.0
extra_idx = np.where(dist_scan_to_cad > threshold)[0]
extra_points = scan_points[extra_idx]

############################################################################
# 5. Cluster the Extra Points & Compute Their Local Normals
############################################################################

db = DBSCAN(eps=10.0, min_samples=5).fit(extra_points)
labels = db.labels_
unique_labels = set(labels) - {-1}

def create_bounding_sphere(points):
    center = points.mean(axis=0)
    dist2 = np.sum((points - center)**2, axis=1)
    radius = np.sqrt(dist2.max())
    return pv.Sphere(radius=radius, center=center, phi_resolution=30, theta_resolution=30)

# Approximate normals for each "extra" point (just like before)
dot_tree = cKDTree(extra_points)
dot_normals = np.zeros_like(extra_points)

for i in range(len(extra_points)):
    dists, idxs = dot_tree.query(extra_points[i], k=3)
    if len(idxs) < 3:
        continue
    p0 = extra_points[i]
    p1 = extra_points[idxs[1]]
    p2 = extra_points[idxs[2]]
    v1 = p1 - p0
    v2 = p2 - p0
    n = np.cross(v1, v2)
    length = np.linalg.norm(n)
    if length > 1e-12:
        n /= length
    dot_normals[i] = n

dots_pdata = pv.PolyData(extra_points)
dots_pdata["Normals"] = dot_normals

############################################################################
# 6. (Optional) Resultant Normals for Each Cluster
############################################################################

def compute_cluster_resultant_normal(cluster_indices):
    cluster_ns = dot_normals[cluster_indices]
    if len(cluster_ns) == 0:
        return np.array([0, 0, 0])
    avg_normal = cluster_ns.mean(axis=0)
    length = np.linalg.norm(avg_normal)
    if length > 1e-12:
        avg_normal /= length
    else:
        avg_normal = np.array([0, 0, 1.0])
    return avg_normal

def create_arrow_at_centroid(centroid, normal, factor=10.0):
    single_point = pv.PolyData(centroid.reshape(1, 3))
    single_point["Normals"] = normal.reshape(1, 3)
    arrow = single_point.glyph(orient="Normals", scale=False, factor=factor)
    return arrow

############################################################################
# 7. Visualization
############################################################################

plotter = pv.Plotter()

# -- A) Show the CAD mesh as a point cloud with "hand-computed" normals --
# Create arrow glyphs for the CAD point cloud
cad_arrows = cad_pdata.glyph(orient="Normals", scale=False, factor=10.0)
plotter.add_mesh(cad_pdata, color="green", point_size=3, render_points_as_spheres=True)
plotter.add_mesh(cad_arrows, color="blue", label="CAD Cloud Normals")

# -- B) Show the clusters with bounding spheres + resultant normals --
for cluster_id in unique_labels:
    cluster_mask = (labels == cluster_id)
    cluster_indices = np.where(cluster_mask)[0]
    cluster_points = extra_points[cluster_mask]
    
    # (1) Create bounding sphere
    sphere = create_bounding_sphere(cluster_points)
    plotter.add_mesh(sphere, color="red", opacity=0.3)
    
    # (2) Add an arrow for the "resultant normal"
    centroid = cluster_points.mean(axis=0)
    resultant_normal = compute_cluster_resultant_normal(cluster_indices)
    arrow = create_arrow_at_centroid(centroid, resultant_normal, factor=30.0)
    plotter.add_mesh(arrow, color="magenta")

# -- C) Show all "extra" points with their local normals --
arrows = dots_pdata.glyph(orient="Normals", scale=False, factor=10.0)
plotter.add_points(extra_points, color="white", point_size=5, render_points_as_spheres=True)
plotter.add_mesh(arrows, color="black", opacity=0.2, label="Dot Normals")

# Done
plotter.show()
