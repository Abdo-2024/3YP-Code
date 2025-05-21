import numpy as np
import pyvista as pv
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

############################################################################
# 1. Load & Preprocess
############################################################################
cad_mesh_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/STL/2x2_MN_Smooth.stl"
scan_mesh_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/STL/2x2_MN+Noise.stl"

cad_mesh = pv.read(cad_mesh_path)
scan_mesh = pv.read(scan_mesh_path)

# Process the CAD mesh
cad_mesh = cad_mesh.compute_normals(auto_orient_normals=True)
cad_mesh = cad_mesh.triangulate().decimate(0.3)

# Process the scanned mesh
scan_mesh = scan_mesh.clean(tolerance=1e-15).fill_holes(hole_size=2).triangulate().decimate(0.8)

############################################################################
# 2. Fast inside/outside classification with select_enclosed_points
############################################################################
select = scan_mesh.select_enclosed_points(cad_mesh)
inside_out_array = select["SelectedPoints"]  # 1 for inside, 0 for outside

############################################################################
# 3. Remove points near/on the CAD surface
############################################################################
# Build a KDTree from the CAD mesh points (not the scan points)
scan_points = scan_mesh.points
cad_tree = cKDTree(cad_mesh.points)
dists, _ = cad_tree.query(scan_points)
surface_tolerance = 0.3
mask = dists >= surface_tolerance  # keep points not near the CAD surface
print(f"Removing {np.count_nonzero(~mask)} points near the CAD surface...")

filtered_scan_points = scan_points[mask]
filtered_inout = inside_out_array[mask]

############################################################################
# 4. Cluster (“blob”) detection with DBSCAN on filtered scan points
############################################################################
# Adjust eps and min_samples as needed for your data
db = DBSCAN(eps=10, min_samples=10)
labels = db.fit_predict(filtered_scan_points)
unique_clusters = np.unique(labels)

############################################################################
# 5. Majority inside vs. outside per cluster and sphere fitting
############################################################################
def fit_sphere(points):
    """
    Returns (center, radius) for a sphere fitted to 'points'
    using the mean as the center and the maximum distance as the radius.
    """
    if len(points) == 0:
        return np.array([0, 0, 0]), 0.0
    center = np.mean(points, axis=0)
    diffs = points - center
    radius = np.sqrt((diffs**2).sum(axis=1).max())
    return center, radius

cluster_info = []
for cid in unique_clusters:
    if cid == -1:
        # DBSCAN labels noise as -1
        continue
    cluster_mask = (labels == cid)
    cluster_pts = filtered_scan_points[cluster_mask]
    cluster_inout = filtered_inout[cluster_mask]
    
    if len(cluster_pts) == 0:
        continue

    # New labeling logic: if any point in the cluster is outside, label as "addition"
    # Otherwise, label as "subtraction"
    if np.any(cluster_inout == 0):
         label = "addition"
    else:
         label = "subtraction"

    center, radius = fit_sphere(cluster_pts)
    
    cluster_info.append({
        "cluster_id": cid,
        "points_in_cluster": len(cluster_pts),
        "inside_count": np.count_nonzero(cluster_inout == 1),
        "outside_count": np.count_nonzero(cluster_inout == 0),
        "label": label,
        "center": center,
        "radius": radius
    })

############################################################################
# 5b. Merge touching spheres (only if they have the same label)
############################################################################
def merge_spheres(cluster_list):
    """
    For spheres with the same label, merge any that are touching.
    Two spheres are considered touching if the distance between centers 
    is less than or equal to the sum of their radii.
    Returns a new list of merged sphere descriptors.
    """
    merged_list = []
    # Group spheres by label.
    groups = {}
    for sphere in cluster_list:
        lab = sphere["label"]
        groups.setdefault(lab, []).append(sphere)
    
    for lab, spheres in groups.items():
        n = len(spheres)
        # Use a simple union-find to group touching spheres.
        parent = list(range(n))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union(i, j):
            ri = find(i)
            rj = find(j)
            if ri != rj:
                parent[rj] = ri
        centers = [s["center"] for s in spheres]
        radii = [s["radius"] for s in spheres]
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d <= (radii[i] + radii[j]):
                    union(i, j)
        # Build connected components
        components = {}
        for i in range(n):
            root = find(i)
            components.setdefault(root, []).append(spheres[i])
        # For each component, compute a merged sphere.
        for comp in components.values():
            comp_centers = np.array([s["center"] for s in comp])
            comp_radii = np.array([s["radius"] for s in comp])
            # Use the centroid as a simple merged center.
            union_center = comp_centers.mean(axis=0)
            # The merged radius is set to cover every individual sphere.
            union_radius = 0
            for s in comp:
                d = np.linalg.norm(union_center - s["center"]) + s["radius"]
                union_radius = max(union_radius, d)
            merged_list.append({
                "label": lab,
                "center": union_center,
                "radius": union_radius,
                "points_in_cluster": sum(s["points_in_cluster"] for s in comp),
                "inside_count": sum(s["inside_count"] for s in comp),
                "outside_count": sum(s["outside_count"] for s in comp)
            })
    return merged_list

merged_spheres = merge_spheres(cluster_info)

############################################################################
# 6. Visualization in PyVista using merged spheres
############################################################################
p = pv.Plotter()

# Display the full scanned mesh in light gray (optional)
p.add_mesh(scan_mesh, color="lightgray", opacity=0.8, show_edges=False, label="Scan Mesh")

# Draw a sphere for each merged blob, colored by its label:
for info in merged_spheres:
    c = info["center"]
    r = info["radius"]
    lbl = info["label"]
    
    sphere = pv.Sphere(radius=r, center=c, theta_resolution=15, phi_resolution=15)
    
    if lbl == "addition":
        color = "red"
    elif lbl == "subtraction":
        color = "blue"
    else:
        color = "green"
    
    p.add_mesh(sphere, color=color, opacity=0.3)
    # Optionally, add a text label at the sphere center:
    # p.add_point_labels([c], [f"{lbl}"], font_size=12)

# Add a legend
p.add_legend([
    ["CAD Mesh", "white"],
    ["Scan Mesh", "lightgray"],
    ["Addition Blob", "red"],
    ["Subtraction Blob", "blue"],
    ["Tie Blob", "green"]
], bcolor="white")

p.show()

############################################################################
# 7. (Optional) Output merged sphere summary
############################################################################
for info in merged_spheres:
    lab = info["label"]
    ni  = info["inside_count"]
    no  = info["outside_count"]
    print(f"Merged Sphere: label={lab}, inside={ni}, outside={no}, center={info['center']}, radius={info['radius']:.2f}")
